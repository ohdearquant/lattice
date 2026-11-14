use super::ForwardObserver;

pub(super) struct FinalRowObserver {
    pub(super) hidden_size: usize,
}

impl ForwardObserver for FinalRowObserver {
    type Stamp = ();
    type Output = Vec<f32>;

    #[inline]
    fn start(&self) {}

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
