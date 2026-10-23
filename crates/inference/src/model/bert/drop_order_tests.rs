use std::sync::atomic::{AtomicU8, Ordering};

static DROP_ORDER: AtomicU8 = AtomicU8::new(0);

struct DropTracker {
    name: &'static str,
    expected_position: u8,
}

impl Drop for DropTracker {
    fn drop(&mut self) {
        let position = DROP_ORDER.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            position, self.expected_position,
            "Field '{}' dropped in wrong order: expected position {}, got position {}. \
                 This means the struct field drop-order invariant that BertModel relies on \
                 is violated. The transmute in BertModel::from_directory is UNSOUND.",
            self.name, self.expected_position, position
        );
    }
}

#[test]
fn test_struct_field_drop_order() {
    // Verify that Rust drops struct fields in declaration order.
    // BertModel relies on `weights` being dropped before `_safetensors`.
    // If this test fails, the transmute in BertModel::from_directory is unsound.
    DROP_ORDER.store(0, Ordering::SeqCst);

    struct FieldOrderMirror {
        _first: DropTracker,
        _second: DropTracker,
        _third: DropTracker,
    }

    {
        let _s = FieldOrderMirror {
            _first: DropTracker {
                name: "first (config analog)",
                expected_position: 0,
            },
            _second: DropTracker {
                name: "second (weights analog)",
                expected_position: 1,
            },
            _third: DropTracker {
                name: "third (_safetensors analog)",
                expected_position: 2,
            },
        };
    }
    // After the block, all three fields have been dropped in declaration order.
    assert_eq!(
        DROP_ORDER.load(Ordering::SeqCst),
        3,
        "Not all fields were dropped"
    );
}
