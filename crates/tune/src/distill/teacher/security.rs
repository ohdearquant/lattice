//! Endpoint policy for teacher-model connections.
//!
//! The policy can require TLS, restrict hosts, and carry certificate or local
//! weight-integrity expectations. See `docs/distill.md` for the exact checks
//! performed today and the verification that remains the HTTP client's job.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Endpoint security configuration
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EndpointSecurity {
    /// Require TLS for remote connections
    pub require_tls: bool,

    /// Expected SHA-256 certificate fingerprint in hexadecimal.
    pub expected_cert_fingerprint: Option<String>,

    /// Endpoint host allowlist.
    pub allowed_domains: Option<Vec<String>>,

    /// Expected SHA-256 checksum for local model weights.
    pub model_checksum: Option<String>,
}

impl EndpointSecurity {
    /// Create default security settings with TLS required
    pub fn default_secure() -> Self {
        Self {
            require_tls: true,
            expected_cert_fingerprint: None,
            allowed_domains: Some(vec![
                "api.anthropic.com".to_string(),
                "api.openai.com".to_string(),
                "generativelanguage.googleapis.com".to_string(),
            ]),
            model_checksum: None,
        }
    }

    /// Create settings for local models (no TLS required)
    pub fn for_local() -> Self {
        Self {
            require_tls: false,
            expected_cert_fingerprint: None,
            allowed_domains: Some(vec!["localhost".to_string(), "127.0.0.1".to_string()]),
            model_checksum: None,
        }
    }

    /// Require a specific certificate fingerprint
    pub fn with_cert_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.expected_cert_fingerprint = Some(fingerprint.into());
        self
    }

    /// Set expected model checksum (for local models)
    pub fn with_model_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.model_checksum = Some(checksum.into());
        self
    }

    /// Add an allowed domain
    pub fn allow_domain(mut self, domain: impl Into<String>) -> Self {
        let domains = self.allowed_domains.get_or_insert_with(Vec::new);
        domains.push(domain.into());
        self
    }

    /// Verify an endpoint against security settings
    ///
    /// Returns Ok(()) if the endpoint passes all checks, Err with reason otherwise.
    pub fn verify_endpoint(&self, endpoint: &str) -> Result<(), String> {
        let endpoint_lower = endpoint.to_lowercase();

        if self.require_tls && !endpoint_lower.starts_with("https://") {
            return Err(format!(
                "TLS required but endpoint uses insecure protocol: {endpoint}"
            ));
        }

        if let Some(ref allowed) = self.allowed_domains {
            let domain = Self::extract_domain(endpoint);
            if !allowed.iter().any(|d| domain == d.as_str()) {
                return Err(format!(
                    "Domain '{domain}' not in allowed list: {allowed:?}"
                ));
            }
        }

        Ok(())
    }

    /// Extract domain from URL
    pub(crate) fn extract_domain(url: &str) -> &str {
        let url = url.strip_prefix("https://").unwrap_or(url);
        let url = url.strip_prefix("http://").unwrap_or(url);
        url.split('/')
            .next()
            .unwrap_or(url)
            .split(':')
            .next()
            .unwrap_or(url)
    }

    /// Validate the configured certificate fingerprint's format.
    pub fn validate_cert_fingerprint(&self, _actual_fingerprint: &str) -> Result<(), String> {
        if let Some(ref expected) = self.expected_cert_fingerprint {
            // A live client must compare the format-validated value to its peer certificate.
            if expected.len() != 64 {
                return Err(
                    "Certificate fingerprint should be 64 hex characters (SHA-256)".to_string(),
                );
            }
            if !expected.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err("Certificate fingerprint should be hex characters only".to_string());
            }
        }
        Ok(())
    }

    /// Verify model checksum (for local models)
    pub fn verify_model_checksum(&self, actual_checksum: &str) -> Result<(), String> {
        if let Some(ref expected) = self.model_checksum
            && expected != actual_checksum
        {
            return Err(format!(
                "Model checksum mismatch: expected {expected}, got {actual_checksum}"
            ));
        }
        Ok(())
    }
}
