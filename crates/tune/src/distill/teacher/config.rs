//! Teacher model configuration and builder.

use super::TeacherProvider;
use super::security::EndpointSecurity;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for a teacher model
///
/// Specifies which model to use for generating soft labels during distillation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TeacherConfig {
    /// Model provider
    pub provider: TeacherProvider,

    /// Model identifier (e.g., "claude-3-5-sonnet-20241022")
    pub model_id: String,

    /// API endpoint (None = use provider default)
    pub endpoint: Option<String>,

    /// API key environment variable name
    pub api_key_env: String,

    /// Temperature for generation (0.0 - 1.0)
    pub temperature: f32,

    /// Maximum tokens in response
    pub max_tokens: usize,

    /// Request timeout in milliseconds
    pub timeout_ms: u64,

    /// Number of retries on failure
    pub max_retries: usize,

    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,

    /// System prompt for intent classification
    pub system_prompt: Option<String>,

    /// Endpoint security configuration
    pub security: EndpointSecurity,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self::claude_sonnet()
    }
}

impl TeacherConfig {
    /// Create a config for Claude 3.5 Sonnet
    pub fn claude_sonnet() -> Self {
        Self {
            provider: TeacherProvider::Claude,
            model_id: "claude-sonnet-4-20250514".to_string(),
            endpoint: None,
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            temperature: 0.3,
            max_tokens: 1024,
            timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            system_prompt: Some(Self::default_system_prompt()),
            security: EndpointSecurity::default_secure(),
        }
    }

    /// Create a config for Claude 3 Haiku (faster, cheaper)
    pub fn claude_haiku() -> Self {
        Self {
            provider: TeacherProvider::Claude,
            model_id: "claude-3-5-haiku-20241022".to_string(),
            endpoint: None,
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            temperature: 0.3,
            max_tokens: 1024,
            timeout_ms: 15000,
            max_retries: 3,
            retry_delay_ms: 500,
            system_prompt: Some(Self::default_system_prompt()),
            security: EndpointSecurity::default_secure(),
        }
    }

    /// Create a config for GPT-4
    pub fn gpt4() -> Self {
        Self {
            provider: TeacherProvider::OpenAI,
            model_id: "gpt-4-turbo-preview".to_string(),
            endpoint: None,
            api_key_env: "OPENAI_API_KEY".to_string(),
            temperature: 0.3,
            max_tokens: 1024,
            timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            system_prompt: Some(Self::default_system_prompt()),
            security: EndpointSecurity::default_secure(),
        }
    }

    /// Create a config for Gemini Pro
    pub fn gemini_pro() -> Self {
        Self {
            provider: TeacherProvider::Gemini,
            model_id: "gemini-pro".to_string(),
            endpoint: None,
            api_key_env: "GOOGLE_API_KEY".to_string(),
            temperature: 0.3,
            max_tokens: 1024,
            timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            system_prompt: Some(Self::default_system_prompt()),
            security: EndpointSecurity::default_secure(),
        }
    }

    /// Create a config for a local model
    pub fn local(model_id: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            provider: TeacherProvider::Local,
            model_id: model_id.into(),
            endpoint: Some(endpoint.into()),
            api_key_env: String::new(),
            temperature: 0.3,
            max_tokens: 1024,
            timeout_ms: 60000,
            max_retries: 2,
            retry_delay_ms: 500,
            system_prompt: Some(Self::default_system_prompt()),
            security: EndpointSecurity::for_local(),
        }
    }

    /// Create a builder for custom configuration
    pub fn builder() -> TeacherConfigBuilder {
        TeacherConfigBuilder::new()
    }

    /// Get the default system prompt for intent classification
    pub fn default_system_prompt() -> String {
        r#"You are an intent classification system. Given a conversation context and a new message, classify the intent of the new message.

Output a JSON object with probability scores for each intent class. Scores should sum to approximately 1.0.

Intent classes:
- continuation: Natural conversation continuation
- topic_shift: User is changing the topic
- explicit_query: Direct question or request for information
- person_lookup: Looking up information about a person/contact
- health_check: Health or wellness related inquiry
- task_status: Checking on task or todo status

Example output:
{"continuation": 0.7, "topic_shift": 0.1, "explicit_query": 0.15, "person_lookup": 0.02, "health_check": 0.02, "task_status": 0.01}

Respond ONLY with the JSON object, no other text."#.to_string()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.model_id.is_empty() {
            return Err("model_id cannot be empty".to_string());
        }
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(format!(
                "temperature must be between 0.0 and 2.0, got {}",
                self.temperature
            ));
        }
        if self.max_tokens == 0 {
            return Err("max_tokens must be > 0".to_string());
        }
        if self.timeout_ms == 0 {
            return Err("timeout_ms must be > 0".to_string());
        }

        // Validate endpoint security if custom endpoint is provided
        if let Some(ref endpoint) = self.endpoint {
            self.security.verify_endpoint(endpoint)?;
        }

        Ok(())
    }

    /// Verify the teacher endpoint before use
    ///
    /// This performs additional security checks that may require network access:
    /// - TLS certificate validation (if fingerprint specified)
    /// - Model checksum verification (for local models)
    ///
    /// Call this before starting distillation to ensure the endpoint is trusted.
    pub fn verify_endpoint(&self) -> Result<(), String> {
        // Get the actual endpoint
        let endpoint = self.get_endpoint();
        self.security.verify_endpoint(&endpoint)?;

        // If certificate fingerprint is specified, validate format
        // (actual cert verification happens at connection time)
        self.security.validate_cert_fingerprint("")?;

        Ok(())
    }

    /// Get the effective endpoint (default or custom)
    pub fn get_endpoint(&self) -> String {
        if let Some(ref ep) = self.endpoint {
            ep.clone()
        } else {
            match self.provider {
                TeacherProvider::Claude => "https://api.anthropic.com/v1".to_string(),
                TeacherProvider::OpenAI => "https://api.openai.com/v1".to_string(),
                TeacherProvider::Gemini => {
                    "https://generativelanguage.googleapis.com/v1".to_string()
                }
                TeacherProvider::Local => "http://localhost:11434".to_string(),
                TeacherProvider::Custom(_) => "".to_string(),
            }
        }
    }

    /// Set security configuration
    pub fn with_security(mut self, security: EndpointSecurity) -> Self {
        self.security = security;
        self
    }

    /// Get the full model name for display
    pub fn display_name(&self) -> String {
        format!("{}:{}", self.provider, self.model_id)
    }
}

/// Builder for TeacherConfig
#[derive(Debug, Clone)]
pub struct TeacherConfigBuilder {
    config: TeacherConfig,
}

impl TeacherConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: TeacherConfig::claude_sonnet(),
        }
    }

    /// Set the provider
    pub fn provider(mut self, provider: TeacherProvider) -> Self {
        self.config.provider = provider;
        self
    }

    /// Set the model ID
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.config.model_id = model_id.into();
        self
    }

    /// Set the API endpoint
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.endpoint = Some(endpoint.into());
        self
    }

    /// Set the API key environment variable
    pub fn api_key_env(mut self, env_var: impl Into<String>) -> Self {
        self.config.api_key_env = env_var.into();
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = tokens;
        self
    }

    /// Set timeout in milliseconds
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.config.timeout_ms = ms;
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set retry delay in milliseconds
    pub fn retry_delay_ms(mut self, ms: u64) -> Self {
        self.config.retry_delay_ms = ms;
        self
    }

    /// Set the system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Clear the system prompt
    pub fn no_system_prompt(mut self) -> Self {
        self.config.system_prompt = None;
        self
    }

    /// Set security configuration
    pub fn security(mut self, security: EndpointSecurity) -> Self {
        self.config.security = security;
        self
    }

    /// Build the configuration
    pub fn build(self) -> TeacherConfig {
        self.config
    }
}

impl Default for TeacherConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
