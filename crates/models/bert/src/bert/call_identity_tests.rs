//! Test-only observation of the actual embedding-lookup inputs.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

const MAX_ROWS: usize = 4096;
const MAX_SEQUENCES: usize = 128;

#[derive(Debug, PartialEq, Eq)]
enum ForwardInputs {
    Single {
        ids: Vec<u32>,
        mask: Vec<u32>,
        types: Vec<u32>,
    },
    Packed {
        ids: Vec<u32>,
        types: Vec<u32>,
        offsets: Vec<usize>,
    },
}

#[derive(Default)]
struct Capture {
    event: Option<ForwardInputs>,
    invalid: bool,
}

thread_local! {
    static ACTIVE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

struct Scope(PhantomData<Rc<()>>);

impl Scope {
    fn begin() -> Result<Self, &'static str> {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active.is_some() {
                return Err("nested observation");
            }
            *active = Some(Capture::default());
            Ok(Self(PhantomData))
        })
    }

    fn finish(self) -> Result<ForwardInputs, &'static str> {
        ACTIVE.with(|active| {
            let capture = active.borrow_mut().take().ok_or("missing observation")?;
            if capture.invalid {
                return Err("invalid or repeated observation");
            }
            capture.event.ok_or("missing forward")
        })
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.borrow_mut().take();
        });
    }
}

pub(super) fn record_single(ids: &[u32], mask: &[u32], types: &[u32]) {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(capture) = active.as_mut() else {
            return;
        };
        if capture.event.is_some()
            || ids.is_empty()
            || ids.len() > MAX_ROWS
            || ids.len() != mask.len()
            || ids.len() != types.len()
        {
            capture.invalid = true;
            return;
        }
        capture.event = Some(ForwardInputs::Single {
            ids: ids.to_vec(),
            mask: mask.to_vec(),
            types: types.to_vec(),
        });
    });
}

pub(super) fn record_packed(ids: &[u32], types: &[u32], offsets: &[usize]) {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(capture) = active.as_mut() else {
            return;
        };
        if capture.event.is_some()
            || ids.is_empty()
            || ids.len() > MAX_ROWS
            || ids.len() != types.len()
            || offsets.len() < 2
            || offsets.len() > MAX_SEQUENCES + 1
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&ids.len())
            || offsets.windows(2).any(|pair| pair[0] >= pair[1])
        {
            capture.invalid = true;
            return;
        }
        capture.event = Some(ForwardInputs::Packed {
            ids: ids.to_vec(),
            types: types.to_vec(),
            offsets: offsets.to_vec(),
        });
    });
}

use crate::bert_pooling_fixture as fixture;

use super::*;
use lattice_tokenizer::TokenizedInput;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ChangingTokenizer(AtomicUsize);

impl ChangingTokenizer {
    fn input(text: &str, call: usize) -> TokenizedInput {
        let varying = if call.is_multiple_of(2) { 5 } else { 6 };
        let ids = match text {
            "left" => vec![2, varying, 3],
            "right" => vec![2, 8, 3],
            "long" => {
                let mut ids = vec![2];
                ids.extend(std::iter::repeat_n(varying, 17));
                ids.push(3);
                ids
            }
            _ => panic!("unexpected test input"),
        };
        let real_length = ids.len();
        TokenizedInput {
            input_ids: ids,
            attention_mask: vec![1; real_length],
            token_type_ids: (0..real_length).map(|i| (i % 2) as u32).collect(),
            real_length,
            pre_truncation_len: real_length + usize::from(text == "long") * 3,
        }
    }
}

impl Tokenizer for ChangingTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedInput {
        let call = self.0.fetch_add(1, Ordering::SeqCst);
        Self::input(text, call)
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        let call = self.0.fetch_add(1, Ordering::SeqCst);
        let mut inputs: Vec<_> = texts.iter().map(|text| Self::input(text, call)).collect();
        let width = inputs
            .iter()
            .map(|input| input.real_length)
            .max()
            .unwrap_or(0);
        for input in &mut inputs {
            input.input_ids.resize(width, 0);
            input.attention_mask.resize(width, 0);
            input.token_type_ids.resize(width, 0);
        }
        inputs
    }

    fn vocab_size(&self) -> usize {
        10
    }

    fn max_seq_len(&self) -> usize {
        64
    }
}

fn model() -> (tempfile::TempDir, BertModel) {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    fixture::write_checkpoint(&checkpoint).unwrap();
    let mut model = BertModel::from_directory(&checkpoint).unwrap();
    model.tokenizer = Box::new(ChangingTokenizer(AtomicUsize::new(0)));
    (dir, model)
}

fn assert_vectors(vectors: &[Vec<f32>], count: usize) {
    assert_eq!(vectors.len(), count);
    for vector in vectors {
        assert_eq!(vector.len(), fixture::DIMENSIONS);
        assert!(vector.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn singleton_capture_follows_same_call_after_position_clamp() {
    let (_dir, model) = model();
    let preflight = model.tokenizer().tokenize("long");
    assert_eq!(preflight.input_ids[1], 5);
    let scope = Scope::begin().unwrap();
    let output = model.encode_batch(&["long"]).unwrap();
    let captured = scope.finish().unwrap();
    assert_vectors(&output, 1);
    let mut ids = vec![2];
    ids.extend([6; 15]);
    assert_eq!(
        captured,
        ForwardInputs::Single {
            ids,
            mask: vec![1; 16],
            types: (0..16).map(|i| (i % 2) as u32).collect(),
        }
    );
}

#[test]
fn packed_capture_follows_same_batch_order_and_clamps_each_sequence() {
    let (_dir, model) = model();
    assert_eq!(model.tokenizer().tokenize("left").input_ids, [2, 5, 3]);
    let scope = Scope::begin().unwrap();
    let output = model.encode_batch(&["left", "long", "right"]).unwrap();
    let captured = scope.finish().unwrap();
    assert_vectors(&output, 3);
    let mut ids = vec![2, 6, 3, 2];
    ids.extend([6; 15]);
    ids.extend([2, 8, 3]);
    let mut types = vec![0, 1, 0];
    types.extend((0..16).map(|i| (i % 2) as u32));
    types.extend([0, 1, 0]);
    assert_eq!(
        captured,
        ForwardInputs::Packed {
            ids,
            types,
            offsets: vec![0, 3, 19, 22],
        }
    );
}

#[test]
fn packed_capture_preserves_more_than_sixteen_items() {
    let (_dir, model) = model();
    let texts = vec!["left"; 17];
    let scope = Scope::begin().unwrap();
    let output = model.encode_batch(&texts).unwrap();
    let captured = scope.finish().unwrap();
    assert_vectors(&output, 17);
    let ForwardInputs::Packed { ids, offsets, .. } = captured else {
        panic!("packed path required");
    };
    assert_eq!(offsets, (0..=17).map(|i| i * 3).collect::<Vec<_>>());
    assert_eq!(ids, [2, 5, 3].repeat(17));

    let scope = Scope::begin().unwrap();
    assert!(model.encode_batch(&[]).unwrap().is_empty());
    assert_eq!(scope.finish(), Err("missing forward"));
}

#[test]
fn observation_scope_rejects_nested_missing_and_repeated_and_cleans_up() {
    let scope = Scope::begin().unwrap();
    assert!(Scope::begin().is_err());
    assert_eq!(scope.finish(), Err("missing forward"));
    let scope = Scope::begin().unwrap();
    record_single(&[1], &[1], &[0]);
    record_single(&[1], &[1], &[0]);
    assert_eq!(scope.finish(), Err("invalid or repeated observation"));
    let panic = std::panic::catch_unwind(|| {
        let _scope = Scope::begin().unwrap();
        panic!("test cleanup");
    });
    assert!(panic.is_err());
    fn early_error() -> Result<(), &'static str> {
        let _scope = Scope::begin()?;
        Err("test error")
    }
    assert_eq!(early_error(), Err("test error"));
    let scope = Scope::begin().unwrap();
    record_single(
        &vec![1; MAX_ROWS + 1],
        &vec![1; MAX_ROWS + 1],
        &vec![0; MAX_ROWS + 1],
    );
    assert_eq!(scope.finish(), Err("invalid or repeated observation"));
    assert!(Scope::begin().is_ok());
}

#[test]
fn observations_are_scoped_to_each_thread() {
    let scope = Scope::begin().unwrap();
    let worker = std::thread::spawn(|| {
        let scope = Scope::begin().unwrap();
        record_single(&[2], &[1], &[0]);
        scope.finish().unwrap()
    });
    record_single(&[1], &[1], &[0]);
    assert_eq!(
        scope.finish().unwrap(),
        ForwardInputs::Single {
            ids: vec![1],
            mask: vec![1],
            types: vec![0]
        }
    );
    assert_eq!(
        worker.join().unwrap(),
        ForwardInputs::Single {
            ids: vec![2],
            mask: vec![1],
            types: vec![0]
        }
    );
}
