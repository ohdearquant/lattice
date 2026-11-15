use std::ops::Range;

use lattice_compute::forward::cpu::rms_norm;

use super::ForwardObserver;

pub(super) struct FinalRowObserver {
    pub(super) hidden_size: usize,
}

impl ForwardObserver for FinalRowObserver {
    type Stamp = ();
    type Output = Vec<f32>;

    #[inline]
    fn start(&self) {}

    fn final_norm(&self, hidden: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) {
        debug_assert_eq!(hidden_size, self.hidden_size);
        let start = hidden.len() - self.hidden_size;
        rms_norm(&mut hidden[start..], weight, self.hidden_size, eps);
    }

    fn finish(self, hidden: &[f32]) -> Self::Output {
        hidden[hidden.len() - self.hidden_size..].to_vec()
    }
}

pub(super) struct FinalRowsObserver<'a> {
    pub(super) segments: &'a [usize],
    pub(super) hidden_size: usize,
}

impl ForwardObserver for FinalRowsObserver<'_> {
    type Stamp = ();
    type Output = Vec<Vec<f32>>;

    #[inline]
    fn start(&self) {}

    fn final_norm(&self, hidden: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) {
        debug_assert_eq!(hidden_size, self.hidden_size);
        for range in self.final_row_ranges() {
            rms_norm(&mut hidden[range], weight, self.hidden_size, eps);
        }
    }

    fn finish(self, hidden: &[f32]) -> Self::Output {
        self.final_row_ranges()
            .map(|range| hidden[range].to_vec())
            .collect()
    }
}

impl FinalRowsObserver<'_> {
    /// Element range of each segment's final row in the packed hidden state.
    fn final_row_ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        let hidden_size = self.hidden_size;
        self.segments.iter().scan(0usize, move |end, &length| {
            *end += length;
            Some((*end - 1) * hidden_size..*end * hidden_size)
        })
    }
}
