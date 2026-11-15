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
        let start = hidden.len() - hidden_size;
        rms_norm(&mut hidden[start..], weight, hidden_size, eps);
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
        let mut end = 0usize;
        for &length in self.segments {
            end += length;
            rms_norm(
                &mut hidden[(end - 1) * hidden_size..end * hidden_size],
                weight,
                hidden_size,
                eps,
            );
        }
    }

    fn finish(self, hidden: &[f32]) -> Self::Output {
        let mut end = 0usize;
        self.segments
            .iter()
            .map(|&length| {
                end += length;
                hidden[(end - 1) * self.hidden_size..end * self.hidden_size].to_vec()
            })
            .collect()
    }
}
