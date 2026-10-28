use super::*;
use std::cell::Cell;
use std::io::{Cursor, Error, ErrorKind};

const GOOD: &[u8] = br#"{"hidden_size":384}"#;
const BAD: &[u8] = br#"{"hidden_size":768}"#;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn reader(bytes: &'static [u8]) -> Result<Box<dyn Read>, InferenceError> {
    Ok(Box::new(Cursor::new(bytes)))
}

fn list_entries(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect()
}

#[test]
fn pending_and_malformed_pins_refuse_before_fetch_or_publication() {
    let dir = tempfile::tempdir().unwrap();
    let called = Cell::new(false);
    for pin in [None, Some("not-a-sha256")] {
        let result = materialize_with(dir.path(), pin, false, || {
            called.set(true);
            reader(GOOD)
        });
        assert!(result.is_err());
        assert!(!called.get());
        assert!(list_entries(dir.path()).is_empty());
    }
    let names: Vec<_> = CONFIG_SHA256.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        [
            "bge-small-en-v1.5",
            "bge-base-en-v1.5",
            "bge-large-en-v1.5",
            "multilingual-e5-small",
            "multilingual-e5-base",
            "all-minilm-l6-v2",
            "paraphrase-multilingual-minilm-l12-v2"
        ]
    );
    for (name, pin) in CONFIG_SHA256 {
        assert!(canonical_model_name(name).is_ok());
        if let Some(pin) = pin {
            assert!(
                pin.len() == 64
                    && pin
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            );
        }
    }
}

#[test]
fn verified_config_is_created_and_existing_verified_file_works_offline() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("config.json");
    let hash = digest(GOOD);
    let result = materialize_with(dir.path(), Some(&hash), false, || {
        assert!(
            !target.exists(),
            "target must remain absent until verification"
        );
        reader(GOOD)
    })
    .unwrap();
    assert_eq!(result, ConfigMaterialization::Created);
    assert_eq!(fs::read(&target).unwrap(), GOOD);
    assert_eq!(list_entries(dir.path()), vec![target.clone()]);
    let result =
        materialize_with(dir.path(), Some(&hash), true, || panic!("offline fetch")).unwrap();
    assert_eq!(result, ConfigMaterialization::AlreadyVerified);
    assert_eq!(fs::read(&target).unwrap(), GOOD);
}

#[test]
fn mismatched_existing_or_download_never_overwrites_or_publishes() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("config.json");
    let hash = digest(GOOD);
    let result = materialize_with(dir.path(), Some(&hash), false, || reader(BAD));
    assert!(matches!(
        result,
        Err(InferenceError::ChecksumMismatch { .. })
    ));
    assert!(list_entries(dir.path()).is_empty());
    fs::write(&target, BAD).unwrap();
    let result = materialize_with(dir.path(), Some(&hash), false, || panic!("existing fetch"));
    assert!(matches!(
        result,
        Err(InferenceError::ChecksumMismatch { .. })
    ));
    assert_eq!(fs::read(target).unwrap(), BAD);
}

#[test]
fn offline_missing_config_never_fetches() {
    let dir = tempfile::tempdir().unwrap();
    let result = materialize_with(dir.path(), Some(&digest(GOOD)), true, || {
        panic!("offline fetch")
    });
    assert!(result.is_err());
    assert!(list_entries(dir.path()).is_empty());
}

struct PartialReader {
    first: bool,
}
impl Read for PartialReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.first {
            self.first = false;
            buf[0] = b'{';
            Ok(1)
        } else {
            Err(Error::new(
                ErrorKind::UnexpectedEof,
                "simulated interrupted body",
            ))
        }
    }
}

#[test]
fn interrupted_and_oversize_bodies_leave_no_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let hash = digest(GOOD);
    let partial = materialize_with(dir.path(), Some(&hash), false, || {
        Ok(Box::new(PartialReader { first: true }))
    });
    assert!(partial.is_err());
    assert!(list_entries(dir.path()).is_empty());
    let over = materialize_with(dir.path(), Some(&hash), false, || {
        Ok(Box::new(std::io::repeat(b'x')))
    });
    assert!(over.is_err());
    assert!(list_entries(dir.path()).is_empty());
}

#[test]
fn competing_writer_is_never_overwritten() {
    for (winner, succeeds) in [(GOOD, true), (BAD, false)] {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("config.json");
        let hash = digest(GOOD);
        let result = materialize_with(dir.path(), Some(&hash), false, || {
            fs::write(&target, winner).unwrap();
            reader(GOOD)
        });
        if succeeds {
            assert_eq!(result.unwrap(), ConfigMaterialization::AlreadyVerified);
        } else {
            assert!(matches!(
                result,
                Err(InferenceError::ChecksumMismatch { .. })
            ));
        }
        assert_eq!(fs::read(&target).unwrap(), winner);
        assert_eq!(list_entries(dir.path()), vec![target]);
    }
}

#[test]
fn missing_model_directory_refuses_before_fetch() {
    let dir = tempfile::tempdir().unwrap();
    let absent = dir.path().join("missing");
    let result = materialize_with(&absent, Some(&digest(GOOD)), false, || {
        panic!("missing directory fetch")
    });
    assert!(result.is_err());
    assert!(!absent.exists());
}

#[test]
fn legacy_offline_cache_without_config_remains_a_valid_hit() {
    let root = tempfile::tempdir().unwrap();
    let model_dir = root.path().join("all-minilm-l6-v2");
    fs::create_dir(&model_dir).unwrap();
    fs::write(model_dir.join("model.safetensors"), b"test weights").unwrap();
    fs::write(model_dir.join("vocab.txt"), b"test vocab").unwrap();
    let expected = super::super::ExpectedChecksums {
        model_safetensors: Some("058255880913fc0dc1dea54fbd018719894cc4a9b512eaa586447d91dc2da104"),
        tokenizer: Some((
            "vocab.txt",
            "e6aaecc4138726212adc9899b54fa7cd09aaad6ba28be0d26346620361f8ae26",
        )),
    };
    let result = super::super::ensure_model_files_inner_with_checksums(
        "all-minilm-l6-v2",
        root.path(),
        true,
        expected,
    );
    assert_eq!(result.unwrap(), model_dir);
    assert!(!model_dir.join("config.json").exists());
}

#[test]
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn actual_http_reader_is_verified_before_publication() {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0u8; 1024];
        let _ = stream.read(&mut request).unwrap();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            GOOD.len()
        )
        .unwrap();
        stream.write_all(GOOD).unwrap();
    });
    let dir = tempfile::tempdir().unwrap();
    let hash = digest(GOOD);
    let url = format!("http://{address}/config.json");
    let result = materialize_with(dir.path(), Some(&hash), false, || fetch_config(&url));
    server.join().unwrap();
    assert_eq!(result.unwrap(), ConfigMaterialization::Created);
    assert_eq!(fs::read(dir.path().join("config.json")).unwrap(), GOOD);
}

#[test]
fn public_api_refuses_each_pending_pin_without_creating_files() {
    let root = tempfile::tempdir().unwrap();
    for (name, pin) in CONFIG_SHA256 {
        if pin.is_none() {
            let result = materialize_pinned_config(name, root.path());
            assert!(
                matches!(result, Err(InferenceError::Download(message)) if message.contains("pinned config SHA-256"))
            );
        }
    }
    assert!(list_entries(root.path()).is_empty());
}

#[test]
#[cfg(unix)]
fn existing_symlink_is_refused_and_preserved() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("user-config.json");
    fs::write(&original, GOOD).unwrap();
    let target = dir.path().join("config.json");
    symlink(&original, &target).unwrap();
    let result = materialize_with(dir.path(), Some(&digest(GOOD)), false, || {
        panic!("symlink fetch")
    });
    assert!(result.is_err());
    assert!(
        fs::symlink_metadata(&target)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read(original).unwrap(), GOOD);
}
