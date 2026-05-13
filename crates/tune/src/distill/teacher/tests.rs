use super::*;

#[test]
fn test_teacher_config_defaults() {
    let config = TeacherConfig::default();
    assert_eq!(config.provider, TeacherProvider::Claude);
    assert!(config.validate().is_ok());
}

#[test]
fn test_teacher_config_builder() {
    let config = TeacherConfig::builder()
        .provider(TeacherProvider::OpenAI)
        .model_id("gpt-4")
        .temperature(0.5)
        .max_tokens(2048)
        .build();

    assert_eq!(config.provider, TeacherProvider::OpenAI);
    assert_eq!(config.model_id, "gpt-4");
    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.max_tokens, 2048);
}

#[test]
fn test_teacher_config_validation() {
    let mut config = TeacherConfig::default();

    // Valid config
    assert!(config.validate().is_ok());

    // Invalid temperature
    config.temperature = 3.0;
    assert!(config.validate().is_err());
    config.temperature = 0.3;

    // Invalid model_id
    config.model_id = String::new();
    assert!(config.validate().is_err());
}

#[test]
fn test_teacher_provider_display() {
    assert_eq!(TeacherProvider::Claude.to_string(), "claude");
    assert_eq!(TeacherProvider::OpenAI.to_string(), "openai");
    assert_eq!(
        TeacherProvider::Custom("my-model".to_string()).to_string(),
        "custom:my-model"
    );
}

#[test]
fn test_preset_configs() {
    let claude = TeacherConfig::claude_sonnet();
    assert_eq!(claude.provider, TeacherProvider::Claude);
    assert!(claude.validate().is_ok());

    let gpt = TeacherConfig::gpt4();
    assert_eq!(gpt.provider, TeacherProvider::OpenAI);
    assert!(gpt.validate().is_ok());

    let gemini = TeacherConfig::gemini_pro();
    assert_eq!(gemini.provider, TeacherProvider::Gemini);
    assert!(gemini.validate().is_ok());
}

// Endpoint Security Tests

#[test]
fn test_endpoint_security_default_secure() {
    let security = EndpointSecurity::default_secure();
    assert!(security.require_tls);
    assert!(security.allowed_domains.is_some());
}

#[test]
fn test_endpoint_security_for_local() {
    let security = EndpointSecurity::for_local();
    assert!(!security.require_tls);
    assert!(
        security
            .allowed_domains
            .unwrap()
            .contains(&"localhost".to_string())
    );
}

#[test]
fn test_endpoint_verification_tls_required() {
    let security = EndpointSecurity {
        require_tls: true,
        allowed_domains: None,
        ..Default::default()
    };

    // HTTPS should pass
    assert!(security.verify_endpoint("https://api.example.com").is_ok());

    // HTTP should fail
    assert!(security.verify_endpoint("http://api.example.com").is_err());
}

#[test]
fn test_endpoint_verification_domain_whitelist() {
    let security = EndpointSecurity {
        require_tls: false,
        allowed_domains: Some(vec![
            "api.anthropic.com".to_string(),
            "api.openai.com".to_string(),
        ]),
        ..Default::default()
    };

    // Allowed domains should pass
    assert!(
        security
            .verify_endpoint("https://api.anthropic.com/v1")
            .is_ok()
    );
    assert!(
        security
            .verify_endpoint("https://api.openai.com/v1")
            .is_ok()
    );

    // Disallowed domains should fail
    assert!(
        security
            .verify_endpoint("https://evil.example.com/v1")
            .is_err()
    );
}

#[test]
fn test_endpoint_verification_no_restrictions() {
    let security = EndpointSecurity {
        require_tls: false,
        allowed_domains: None,
        ..Default::default()
    };

    // Any endpoint should pass
    assert!(security.verify_endpoint("http://any.example.com").is_ok());
    assert!(security.verify_endpoint("https://any.example.com").is_ok());
}

#[test]
fn test_cert_fingerprint_validation() {
    // Valid 64-char hex fingerprint
    let security = EndpointSecurity::default()
        .with_cert_fingerprint("a94a8fe5ccb19ba61c4c0873d391e987982fbbd3a94a8fe5ccb19ba61c4c0873");
    assert!(security.validate_cert_fingerprint("").is_ok());

    // Invalid length
    let security = EndpointSecurity::default().with_cert_fingerprint("tooshort");
    assert!(security.validate_cert_fingerprint("").is_err());

    // Invalid characters
    let security = EndpointSecurity::default()
        .with_cert_fingerprint("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz");
    assert!(security.validate_cert_fingerprint("").is_err());
}

#[test]
fn test_model_checksum_verification() {
    let expected = "abc123def456";
    let security = EndpointSecurity::default().with_model_checksum(expected);

    // Matching checksum should pass
    assert!(security.verify_model_checksum(expected).is_ok());

    // Mismatched checksum should fail
    assert!(security.verify_model_checksum("wrong_checksum").is_err());
}

#[test]
fn test_teacher_config_with_invalid_custom_endpoint() {
    let config = TeacherConfig::builder()
        .endpoint("http://insecure.example.com") // HTTP, not HTTPS
        .security(EndpointSecurity::default_secure())
        .build();

    // Should fail validation because TLS is required but endpoint is HTTP
    assert!(config.validate().is_err());
}

#[test]
fn test_teacher_config_with_valid_custom_endpoint() {
    let security = EndpointSecurity::default_secure().allow_domain("custom.example.com");

    let config = TeacherConfig::builder()
        .endpoint("https://custom.example.com/v1")
        .security(security)
        .build();

    // Should pass validation
    assert!(config.validate().is_ok());
}

#[test]
fn test_local_config_security() {
    let config = TeacherConfig::local("llama2", "http://localhost:11434");

    // Local config should allow HTTP localhost
    assert!(config.validate().is_ok());
}

#[test]
fn test_get_endpoint_defaults() {
    let claude = TeacherConfig::claude_sonnet();
    assert!(claude.get_endpoint().contains("anthropic.com"));

    let gpt = TeacherConfig::gpt4();
    assert!(gpt.get_endpoint().contains("openai.com"));

    let gemini = TeacherConfig::gemini_pro();
    assert!(gemini.get_endpoint().contains("googleapis.com"));
}

#[test]
fn test_extract_domain() {
    assert_eq!(
        EndpointSecurity::extract_domain("https://api.example.com/v1"),
        "api.example.com"
    );
    assert_eq!(
        EndpointSecurity::extract_domain("http://localhost:8080/api"),
        "localhost"
    );
    assert_eq!(
        EndpointSecurity::extract_domain("api.example.com/path"),
        "api.example.com"
    );
}
