//! Emission-recording seam for tests: records begin/end pairing without
//! touching the real macOS FFI path, so pairing can be asserted on every
//! platform/feature combination `cargo test` runs under.
use super::{Label, Scope};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Begin(Label),
    End(Label),
}

thread_local! {
    static LOG: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn clear() {
    LOG.with(|log| log.borrow_mut().clear());
}

pub(crate) fn events() -> Vec<Event> {
    LOG.with(|log| log.borrow().clone())
}

pub(crate) struct RecordingInterval(Option<Label>);

impl RecordingInterval {
    pub(crate) fn begin(label: Label) -> Self {
        LOG.with(|log| log.borrow_mut().push(Event::Begin(label)));
        RecordingInterval(Some(label))
    }

    /// Mirrors [`super::interval_in`]'s scope gating: `Scope::NotDecode`
    /// records nothing at begin or drop, exactly like the real FFI path's
    /// `Interval::not_recording()`.
    pub(crate) fn begin_in(scope: Scope, label: Label) -> Self {
        match scope {
            Scope::Decode => Self::begin(label),
            Scope::NotDecode => RecordingInterval(None),
        }
    }
}

impl Drop for RecordingInterval {
    fn drop(&mut self) {
        if let Some(label) = self.0 {
            LOG.with(|log| log.borrow_mut().push(Event::End(label)));
        }
    }
}
