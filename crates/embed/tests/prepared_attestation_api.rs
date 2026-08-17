#![cfg(feature = "native")]

use lattice_embed::{
    CheckpointAttestor, EmbedError, MAX_ATTESTATION_REPORT_BYTES, MIN_ATTESTATION_REPORT_BYTES,
    OpaqueAttestationReport,
};

struct FixedAttestor {
    report: Vec<u8>,
}

impl CheckpointAttestor for FixedAttestor {
    fn begin(&mut self, _file_count: u64) -> lattice_embed::Result<()> {
        Ok(())
    }

    fn begin_file(
        &mut self,
        _logical_path: &[u8],
        _declared_len: u64,
    ) -> lattice_embed::Result<()> {
        Ok(())
    }

    fn chunk(&mut self, _bytes: &[u8]) -> lattice_embed::Result<()> {
        Ok(())
    }

    fn end_file(&mut self) -> lattice_embed::Result<()> {
        Ok(())
    }

    fn finish(self) -> lattice_embed::Result<OpaqueAttestationReport> {
        OpaqueAttestationReport::try_from_bytes(self.report)
    }
}

fn finish<A: CheckpointAttestor>(attestor: A) -> lattice_embed::Result<OpaqueAttestationReport> {
    attestor.finish()
}

#[test]
fn attestation_report_accepts_exact_closed_bounds() {
    let minimum =
        OpaqueAttestationReport::try_from_bytes(vec![0x11; MIN_ATTESTATION_REPORT_BYTES]).unwrap();
    assert_eq!(minimum.as_bytes(), &[0x11]);

    let maximum = finish(FixedAttestor {
        report: vec![0x22; MAX_ATTESTATION_REPORT_BYTES],
    })
    .unwrap();
    assert_eq!(maximum.as_bytes().len(), MAX_ATTESTATION_REPORT_BYTES);
    assert!(maximum.as_bytes().iter().all(|byte| *byte == 0x22));
}

#[test]
fn attestation_report_rejects_outside_closed_bounds() {
    for bytes in [
        Vec::new(),
        vec![0x33; MAX_ATTESTATION_REPORT_BYTES.saturating_add(1)],
    ] {
        let length = bytes.len();
        let error = OpaqueAttestationReport::try_from_bytes(bytes).unwrap_err();
        assert!(matches!(
            error,
            EmbedError::AttestationReportSize {
                length: actual,
                min: MIN_ATTESTATION_REPORT_BYTES,
                max: MAX_ATTESTATION_REPORT_BYTES,
            } if actual == length
        ));
    }
}
