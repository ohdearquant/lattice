//! Measure held-out adapter-domain separation and feedback on exported embeddings.
//! Run with mixture,inference-hook,serde. Arguments: vector JSON and optional arm
//! (`real`, `shuffled`, `balance`, `loop`, `cost`; default runs all).

use lattice_fann::{Activation, BackpropTrainer, Network, NetworkBuilder, Trainer, TrainingConfig};
use lattice_inference::mixture::AdapterRouter;
use lattice_tune::lora::router_update::{
    DiagonalFisher, FeedbackEvent, PreferenceSignal, ReplayBuffer, RouterUpdateConfig,
    update_router,
};
use serde::Deserialize;
use std::{collections::HashSet, error::Error, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Data {
    model: String,
    dimension: usize,
    mrl: bool,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    prompt: String,
    label: usize,
    split: String,
    vector: Vec<f32>,
    embedding_seconds: f64,
}

fn gate(dim: usize, output: Activation) -> Result<Network> {
    Ok(NetworkBuilder::new()
        .input(dim)
        .hidden(16, Activation::Tanh)
        .output(2, output)
        .build_with_seed(42)?)
}

fn shuffle<T>(values: &mut [T], mut state: u64) {
    for i in (1..values.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.swap(i, (state % (i as u64 + 1)) as usize);
    }
}

fn majority(labels: &[usize]) -> f64 {
    let zero = labels.iter().filter(|&&label| label == 0).count();
    zero.max(labels.len() - zero) as f64 / labels.len() as f64
}

fn fit(data: &Data, shuffled: bool) -> Result<f64> {
    let train: Vec<&Row> = data.rows.iter().filter(|r| r.split == "train").collect();
    let test: Vec<&Row> = data.rows.iter().filter(|r| r.split == "test").collect();
    let mut train_labels: Vec<usize> = train.iter().map(|r| r.label).collect();
    let mut test_labels: Vec<usize> = test.iter().map(|r| r.label).collect();
    if shuffled {
        shuffle(&mut train_labels, 7183);
        shuffle(&mut test_labels, 9919);
    }
    let inputs: Vec<Vec<f32>> = train.iter().map(|r| r.vector.clone()).collect();
    let targets: Vec<Vec<f32>> = train_labels
        .iter()
        .map(|&label| {
            let mut target = vec![0.0; 2];
            target[label] = 1.0;
            target
        })
        .collect();
    let mut network = gate(data.dimension, Activation::Softmax)?;
    let config = TrainingConfig {
        learning_rate: 0.03,
        max_epochs: 100,
        target_error: 0.0,
        batch_size: 16,
        seed: Some(42),
        ..TrainingConfig::default()
    };
    let trained = BackpropTrainer::new().train(&mut network, &inputs, &targets, &config)?;
    let mut correct = 0;
    for (row, label) in test.iter().zip(test_labels.iter()) {
        let scores = network.forward(&row.vector)?;
        correct += usize::from(usize::from(scores[1] > scores[0]) == *label);
    }
    let accuracy = correct as f64 / test.len() as f64;
    println!(
        "shuffled={shuffled} correct={correct}/{} accuracy={accuracy:.6} epochs={}",
        test.len(),
        trained.epochs_trained
    );
    Ok(accuracy)
}

fn feedback(data: &Data) -> Result<()> {
    let context = &data
        .rows
        .iter()
        .find(|r| r.split == "test")
        .ok_or("no test row")?
        .vector;
    let network = gate(data.dimension, Activation::Linear)?;
    let bytes = network.to_bytes();
    let pool = vec!["domain-0".to_string(), "domain-1".to_string()];
    let before = AdapterRouter::new(network).route(context, &pool, 1)?;
    let selected = usize::from(before[0].0 == pool[1]);
    let desired = 1 - selected;
    let events = vec![
        FeedbackEvent {
            context_vector: context.clone(),
            preferred_adapter_idx: desired,
            adapter_id: pool[desired].clone(),
            signal: PreferenceSignal::Positive,
        };
        16
    ];
    let config = RouterUpdateConfig {
        learning_rate: 0.1,
        epochs: 40,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
        replay_mix_fraction: 0.0,
        ..RouterUpdateConfig::default()
    };
    let delta = update_router(
        &bytes,
        &events,
        &mut ReplayBuffer::new(32),
        &mut DiagonalFisher::new(0, 0.99)?,
        &config,
    )?;
    let after =
        AdapterRouter::new(Network::from_bytes(&delta.network_bytes)?).route(context, &pool, 1)?;
    let unchanged = AdapterRouter::new(Network::from_bytes(&bytes)?).route(context, &pool, 1)?;
    println!(
        "loop before={before:?} after={after:?} events_consumed={} replay_accuracy={:?}",
        delta.events_consumed, delta.replay_accuracy
    );
    assert_eq!(before, unchanged, "no-update counterfactual changed");
    assert_ne!(before, after, "feedback did not change selection");
    assert_eq!(after[0].0, pool[desired]);
    assert_eq!(delta.events_consumed, events.len());
    assert_eq!(delta.replay_accuracy, Some(1.0));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("expected vector JSON path")?;
    let arm = args.get(2).map(String::as_str).unwrap_or("all");
    if !["all", "real", "shuffled", "balance", "loop", "cost"].contains(&arm) {
        return Err("unknown arm".into());
    }
    let data: Data = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut seen = HashSet::new();
    for row in &data.rows {
        if row.label > 1
            || !["train", "test"].contains(&row.split.as_str())
            || row.vector.len() != data.dimension
            || row.vector.iter().any(|v| !v.is_finite())
            || !seen.insert(row.prompt.to_lowercase())
        {
            return Err("invalid or duplicate row".into());
        }
    }
    let labels: Vec<usize> = data
        .rows
        .iter()
        .filter(|r| r.split == "test")
        .map(|r| r.label)
        .collect();
    for split in ["train", "test"] {
        for label in 0..2 {
            if data
                .rows
                .iter()
                .filter(|r| r.split == split && r.label == label)
                .count()
                < 30
            {
                return Err("need at least 30 rows per domain per split".into());
            }
        }
    }
    let floor = majority(&labels);
    println!(
        "model={} dimension={} mrl={} heldout_counts=[{},{}] majority={floor:.6}",
        data.model,
        data.dimension,
        data.mrl,
        labels.iter().filter(|&&l| l == 0).count(),
        labels.iter().filter(|&&l| l == 1).count()
    );
    if arm == "all" || arm == "balance" {
        assert_eq!(majority(&[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]), 0.7);
        assert_eq!(majority(&[0, 1, 1, 1]), 0.75);
        assert!(floor >= 0.5);
        println!("balance checks passed");
    }
    if arm == "all" || arm == "real" {
        assert!(
            fit(&data, false)? > floor,
            "real labels do not beat majority floor"
        );
    }
    if arm == "all" || arm == "shuffled" {
        let accuracy = fit(&data, true)?;
        let tolerance = 3.0 * (0.25 / labels.len() as f64).sqrt();
        println!(
            "shuffled acceptance: |accuracy-0.5| <= {tolerance:.6} (balanced corpus required)"
        );
        assert_eq!(
            floor, 0.5,
            "control interval requires balanced heldout labels"
        );
        assert!(
            (accuracy - 0.5).abs() <= tolerance,
            "shuffled control outside chance interval"
        );
    }
    if arm == "all" || arm == "loop" {
        feedback(&data)?;
    }
    if arm == "all" || arm == "cost" {
        let row = data.rows.get(1).ok_or("missing warm embedding sample")?;
        let mut network = gate(data.dimension, Activation::Softmax)?;
        network.forward(&row.vector)?;
        let start = Instant::now();
        std::hint::black_box(network.forward(std::hint::black_box(&row.vector))?);
        let seconds = start.elapsed().as_secs_f64();
        assert!(seconds > 0.0 && row.embedding_seconds > 0.0);
        println!(
            "warm_embedding_seconds={} single_gate_forward_seconds={seconds} cold_embedding_seconds={}",
            row.embedding_seconds, data.rows[0].embedding_seconds
        );
    }
    Ok(())
}
