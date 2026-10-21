use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Maps a worker thread name to the lengths of the chunks it tokenized, one
/// entry per chunk execution, so the total execution count is preserved no
/// matter how the pool schedules chunks onto threads.
static CHUNKS: OnceLock<Mutex<HashMap<String, Vec<usize>>>> = OnceLock::new();

fn chunks() -> &'static Mutex<HashMap<String, Vec<usize>>> {
    CHUNKS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn record(thread_name: &str, chunk_len: usize) {
    if let Ok(mut map) = chunks().lock() {
        map.entry(thread_name.to_string())
            .or_default()
            .push(chunk_len);
    }
}

/// Removes and returns every entry whose thread name starts with `prefix`.
pub(super) fn take_matching(prefix: &str) -> Vec<(String, Vec<usize>)> {
    let mut taken = Vec::new();
    if let Ok(mut map) = chunks().lock() {
        let matching: Vec<String> = map
            .keys()
            .filter(|name| name.starts_with(prefix))
            .cloned()
            .collect();
        for name in matching {
            if let Some(lengths) = map.remove(&name) {
                taken.push((name, lengths));
            }
        }
    }
    taken
}

/// Removes every entry whose thread name starts with `prefix`, discarding it.
pub(super) fn discard_matching(prefix: &str) {
    let _ = take_matching(prefix);
}
