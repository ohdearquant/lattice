use std::cell::RefCell;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    File,
    Directory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Event {
    pub(super) kind: Kind,
    pub(super) path: PathBuf,
}

struct Trace {
    events: Vec<Event>,
    fail: Box<dyn FnMut(&Event) -> bool>,
}

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

pub(super) fn observe(kind: Kind, path: &Path) -> Result<(), RouterArtifactError> {
    TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        let Some(trace) = trace.as_mut() else {
            return Ok(());
        };
        let event = Event {
            kind,
            path: path.to_path_buf(),
        };
        trace.events.push(event.clone());
        if (trace.fail)(&event) {
            return Err(write_error(path, "injected synchronization failure"));
        }
        Ok(())
    })
}

pub(super) fn events() -> Vec<Event> {
    TRACE.with(|trace| trace.borrow().as_ref().unwrap().events.clone())
}

pub(super) fn run<T>(
    fail: impl FnMut(&Event) -> bool + 'static,
    operation: impl FnOnce() -> T,
) -> (T, Vec<Event>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TRACE.with(|trace| {
                trace.borrow_mut().take();
            });
        }
    }
    TRACE.with(|trace| {
        assert!(
            trace.borrow().is_none(),
            "nested sync probes are unsupported"
        );
        *trace.borrow_mut() = Some(Trace {
            events: Vec::new(),
            fail: Box::new(fail),
        });
    });
    let _reset = Reset;
    let result = operation();
    (result, events())
}
