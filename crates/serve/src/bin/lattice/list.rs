use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use lattice_model_io::{default_cache_dir, download::has_cached_model_files};

#[derive(Debug, PartialEq, Eq)]
enum Status {
    Complete,
    Incomplete,
    Link(PathBuf),
    Dangling(PathBuf),
}

#[derive(Debug, PartialEq, Eq)]
struct Entry {
    name: OsString,
    status: Status,
    bytes: Option<u64>,
    links: u64,
    parts: u64,
}

fn path_error(path: &Path, error: io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("{}: {error}", path.display()))
}

fn directory_size(root: &Path) -> io::Result<(u64, u64, u64)> {
    let mut directories = vec![root.to_path_buf()];
    let mut bytes = 0u64;
    let mut links = 0u64;
    let mut parts = 0u64;
    while let Some(directory) = directories.pop() {
        let entries = fs::read_dir(&directory).map_err(|error| path_error(&directory, error))?;
        for entry in entries {
            let entry = entry.map_err(|error| path_error(&directory, error))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(|error| path_error(&path, error))?;
            if metadata.file_type().is_symlink() {
                links = links
                    .checked_add(1)
                    .ok_or_else(|| path_error(&path, io::Error::other("symlink count overflow")))?;
            } else if metadata.is_dir() {
                directories.push(path);
            } else if metadata.is_file() {
                bytes = bytes.checked_add(metadata.len()).ok_or_else(|| {
                    path_error(&path, io::Error::other("file size total overflow"))
                })?;
                if entry.file_name().as_encoded_bytes().ends_with(b".part") {
                    parts = parts.checked_add(1).ok_or_else(|| {
                        path_error(&path, io::Error::other("partial file count overflow"))
                    })?;
                }
            }
        }
    }
    Ok((bytes, links, parts))
}

fn scan(root: &Path) -> io::Result<Vec<Entry>> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(path_error(root, error)),
    };
    let mut models = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| path_error(root, error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| path_error(&path, error))?;
        let name = entry.file_name();
        if metadata.file_type().is_symlink() {
            let target = fs::read_link(&path).map_err(|error| path_error(&path, error))?;
            let status = match fs::metadata(&path) {
                Ok(_) => Status::Link(target),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Status::Dangling(target),
                Err(error) => return Err(path_error(&path, error)),
            };
            models.push(Entry {
                name,
                status,
                bytes: None,
                links: 0,
                parts: 0,
            });
        } else if metadata.is_dir() {
            let (bytes, links, parts) = directory_size(&path)?;
            let status = if has_cached_model_files(&path) {
                Status::Complete
            } else {
                Status::Incomplete
            };
            models.push(Entry {
                name,
                status,
                bytes: Some(bytes),
                links,
                parts,
            });
        }
    }
    models.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(models)
}

fn write_listing(root: &Path, output: &mut impl Write) -> io::Result<()> {
    let entries = scan(root)?;
    writeln!(output, "Cache: {}", root.display())?;
    if entries.is_empty() {
        return Ok(());
    }
    writeln!(output, "NAME\tSTATUS\tBYTES\tLINKS\tPARTS\tTARGET")?;
    for entry in entries {
        let bytes = entry
            .bytes
            .map_or_else(|| "-".to_owned(), |v| v.to_string());
        match entry.status {
            Status::Complete => writeln!(
                output,
                "{:?}\tcomplete\t{bytes}\t{}\t{}\t-",
                entry.name, entry.links, entry.parts
            )?,
            Status::Incomplete => writeln!(
                output,
                "{:?}\tincomplete\t{bytes}\t{}\t{}\t-",
                entry.name, entry.links, entry.parts
            )?,
            Status::Link(target) => {
                writeln!(output, "{:?}\tlink\t-\t-\t-\t{:?}", entry.name, target)?
            }
            Status::Dangling(target) => {
                writeln!(output, "{:?}\tdangling\t-\t-\t-\t{:?}", entry.name, target)?
            }
        }
    }
    Ok(())
}

pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let root = default_cache_dir()?;
    write_listing(&root, &mut io::stdout().lock())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn missing_cache_prints_only_its_location() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing");
        let mut output = Vec::new();
        write_listing(&missing, &mut output).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            format!("Cache: {}\n", missing.display())
        );
        assert!(!missing.exists());
    }

    #[test]
    fn lists_sorted_cache_directories_with_recursive_logical_sizes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let complete = root.join("z-complete");
        let incomplete = root.join("a-incomplete");
        fs::create_dir(&complete).unwrap();
        fs::create_dir(&incomplete).unwrap();
        fs::create_dir(complete.join("nested")).unwrap();
        fs::write(complete.join("model.safetensors"), b"123").unwrap();
        fs::write(complete.join("vocab.txt"), b"12").unwrap();
        fs::write(complete.join("nested/data"), b"1234").unwrap();
        fs::write(complete.join("interrupted.part"), b"12345").unwrap();
        fs::write(complete.join("nested/another.part"), b"").unwrap();
        fs::write(incomplete.join("config.json"), b"123456").unwrap();
        fs::write(incomplete.join("incomplete.part"), b"x").unwrap();
        fs::write(root.join("not-a-directory"), b"ignored").unwrap();

        let entries = scan(root).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "a-incomplete");
        assert_eq!(entries[0].status, Status::Incomplete);
        assert_eq!(entries[0].bytes, Some(7));
        assert_eq!(entries[0].parts, 1);
        assert_eq!(entries[1].name, "z-complete");
        assert_eq!(entries[1].status, Status::Complete);
        assert_eq!(entries[1].bytes, Some(14));
        assert_eq!(entries[1].links, 0);
        assert_eq!(entries[1].parts, 2);

        let mut output = Vec::new();
        write_listing(root, &mut output).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"a-incomplete\"\tincomplete\t7\t0\t1\t-"));
        assert!(text.contains("\"z-complete\"\tcomplete\t14\t0\t2\t-"));
    }

    #[test]
    fn invalid_cache_root_error_names_the_path() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("not-a-directory");
        fs::write(&root, b"file").unwrap();
        let error = scan(&root).unwrap_err();
        assert!(error.to_string().contains(&root.display().to_string()));
    }

    #[test]
    fn tokenizer_json_is_a_cache_hit_but_a_tokenizer_alone_is_not() {
        let temp = tempfile::tempdir().unwrap();
        let ready = temp.path().join("ready");
        let partial = temp.path().join("partial");
        fs::create_dir(&ready).unwrap();
        fs::create_dir(&partial).unwrap();
        fs::write(ready.join("model.safetensors"), b"model").unwrap();
        fs::write(ready.join("tokenizer.json"), b"{}").unwrap();
        fs::write(partial.join("tokenizer.json"), b"{}").unwrap();
        let entries = scan(temp.path()).unwrap();
        assert_eq!(entries[0].status, Status::Incomplete);
        assert_eq!(entries[1].status, Status::Complete);
    }

    #[cfg(unix)]
    #[test]
    fn reports_links_without_traversing_their_contents() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("cache");
        let outside = temp.path().join("outside");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("large"), vec![1; 8192]).unwrap();
        let model = root.join("model");
        fs::create_dir(&model).unwrap();
        fs::write(model.join("data"), b"123").unwrap();
        symlink(&outside, model.join("nested-directory-link")).unwrap();
        symlink(outside.join("large"), model.join("nested-file-link.part")).unwrap();
        symlink(&outside, root.join("shortcut")).unwrap();
        let missing = temp.path().join("missing");
        symlink(&missing, root.join("dangling")).unwrap();

        let entries = scan(&root).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "dangling");
        assert_eq!(entries[0].status, Status::Dangling(missing));
        assert_eq!(entries[0].bytes, None);
        assert_eq!(entries[1].name, "model");
        assert_eq!(entries[1].bytes, Some(3));
        assert_eq!(entries[1].links, 2);
        assert_eq!(entries[1].parts, 0);
        assert_eq!(entries[2].name, "shortcut");
        assert_eq!(entries[2].status, Status::Link(outside));
        assert_eq!(entries[2].bytes, None);
    }

    #[cfg(unix)]
    #[test]
    fn marker_symlinks_follow_the_shared_presence_predicate() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("cache");
        let model = root.join("model");
        fs::create_dir_all(&model).unwrap();
        let target = temp.path().join("outside");
        fs::write(&target, b"external bytes").unwrap();
        symlink(&target, model.join("model.safetensors")).unwrap();
        fs::write(model.join("vocab.txt"), b"a").unwrap();
        let entries = scan(&root).unwrap();
        assert_eq!(entries[0].status, Status::Complete);
        assert_eq!(entries[0].bytes, Some(1));
        assert_eq!(entries[0].links, 1);
    }

    #[cfg(unix)]
    #[test]
    fn link_target_errors_name_the_link_path() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("loop");
        symlink("loop", &path).unwrap();
        let error = scan(temp.path()).unwrap_err();
        assert!(error.to_string().contains(&path.display().to_string()));
    }
}
