#![cfg(feature = "native")]

use lattice_embed::{
    AttestationAlgorithm, CheckpointAttestor, EmbedError,
    MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES, MIN_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES,
    SupplementaryAttestationEvidence,
};

const TEST_DIGEST: [u8; 32] = [0x77; 32];

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

    fn finish(self) -> lattice_embed::Result<SupplementaryAttestationEvidence> {
        SupplementaryAttestationEvidence::try_new(
            AttestationAlgorithm::Sha256V1,
            TEST_DIGEST,
            self.report,
        )
    }
}

fn finish<A: CheckpointAttestor>(
    attestor: A,
) -> lattice_embed::Result<SupplementaryAttestationEvidence> {
    attestor.finish()
}

#[test]
fn attestation_report_accepts_exact_closed_bounds() {
    let minimum = SupplementaryAttestationEvidence::try_new(
        AttestationAlgorithm::Sha256V1,
        TEST_DIGEST,
        vec![0x11; MIN_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES],
    )
    .unwrap();
    assert_eq!(minimum.as_bytes(), &[0x11]);

    let maximum = finish(FixedAttestor {
        report: vec![0x22; MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES],
    })
    .unwrap();
    assert_eq!(
        maximum.as_bytes().len(),
        MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES
    );
    assert!(maximum.as_bytes().iter().all(|byte| *byte == 0x22));
}

#[test]
fn attestation_report_rejects_outside_closed_bounds() {
    for bytes in [
        Vec::new(),
        vec![0x33; MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES.saturating_add(1)],
    ] {
        let length = bytes.len();
        let error = SupplementaryAttestationEvidence::try_new(
            AttestationAlgorithm::Sha256V1,
            TEST_DIGEST,
            bytes,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            EmbedError::AttestationReportSize {
                length: actual,
                min: MIN_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES,
                max: MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES,
            } if actual == length
        ));
    }
}
