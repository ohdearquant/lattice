use super::*;

const P: u32 = 7;

fn table(rows: usize, hidden: usize, base: f32) -> Vec<f32> {
    (0..rows * hidden).map(|v| base + v as f32).collect()
}

#[test]
fn slots_are_placeholder_positions_in_order() {
    assert_eq!(placeholder_slots(&[1, P, 2, P, P, 3], P), vec![1, 3, 4]);
    assert!(placeholder_slots(&[1, 2, 3], P).is_empty());
}

#[test]
fn rows_land_at_slots_in_order_and_text_rows_are_untouched() {
    let hidden = 3;
    let ids = [1, P, 2, P, P, 3];
    let mut embeds = table(ids.len(), hidden, 100.0);
    let text = embeds.clone();
    let rows = table(3, hidden, 0.0);
    let slots = placeholder_slots(&ids, P);
    scatter_rows(&slots, &rows, hidden, &mut embeds).unwrap();
    for (pos, chunk) in embeds.chunks_exact(hidden).enumerate() {
        let expected = match slots.iter().position(|&s| s == pos) {
            Some(i) => &rows[i * hidden..(i + 1) * hidden],
            None => &text[pos * hidden..(pos + 1) * hidden],
        };
        assert_eq!(chunk, expected, "row {pos}");
    }
}

#[test]
fn no_slots_and_no_rows_is_a_no_op() {
    let mut embeds = table(2, 4, 1.0);
    let before = embeds.clone();
    scatter_rows(&[], &[], 4, &mut embeds).unwrap();
    assert_eq!(embeds, before);
}

#[test]
fn refusals_never_modify_embeds() {
    let hidden = 2;
    let rows = table(2, hidden, 0.0);
    let cases: [(&[usize], &[f32], usize, usize); 7] = [
        (&[0, 1], &rows, 0, 8),            // zero width
        (&[0, 1], &rows, hidden, 7),       // embeds not whole rows
        (&[0], &rows, hidden, 8),          // one slot, two rows
        (&[0, 1, 2], &rows, hidden, 8),    // three slots, two rows
        (&[1, 1], &rows, hidden, 8),       // repeated slot
        (&[2, 1], &rows, hidden, 8),       // decreasing slots
        (&[0, 4], &rows, hidden, 8),       // slot past the end (4 rows)
    ];
    for (i, (slots, rows, width, len)) in cases.into_iter().enumerate() {
        let mut embeds = table(len, 1, 50.0);
        let before = embeds.clone();
        assert!(
            scatter_rows(slots, rows, width, &mut embeds).is_err(),
            "case {i} must refuse"
        );
        assert_eq!(embeds, before, "case {i} modified embeds");
    }
}
