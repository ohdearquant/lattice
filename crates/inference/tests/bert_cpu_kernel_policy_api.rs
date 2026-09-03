use lattice_inference::{BertCpuKernelPolicy, BertCpuKernelProfile};

#[test]
fn captured_profile_is_stable_and_policy_retains_it_exactly() {
    let first = BertCpuKernelProfile::capture();
    let second = BertCpuKernelProfile::capture();
    assert_eq!(first, second);

    let automatic = BertCpuKernelPolicy::Auto;
    assert_eq!(automatic.pinned_profile(), None);

    let pinned = BertCpuKernelPolicy::Pinned(first);
    assert_eq!(pinned.pinned_profile(), Some(&first));
}

#[test]
fn captured_profile_reports_the_closed_host_capability_set() {
    let profile = BertCpuKernelProfile::capture();
    assert_eq!(profile.target_architecture(), std::env::consts::ARCH);
    assert!(profile.scalar_enabled());

    #[cfg(target_arch = "x86_64")]
    {
        assert_eq!(profile.avx2_enabled(), is_x86_feature_detected!("avx2"));
        assert_eq!(profile.fma_enabled(), is_x86_feature_detected!("fma"));
        assert_eq!(
            profile.avx512f_enabled(),
            is_x86_feature_detected!("avx512f")
        );
        assert!(!profile.neon_enabled());
    }

    #[cfg(target_arch = "aarch64")]
    {
        assert!(!profile.avx2_enabled());
        assert!(!profile.fma_enabled());
        assert!(!profile.avx512f_enabled());
        assert!(profile.neon_enabled());
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        assert!(!profile.avx2_enabled());
        assert!(!profile.fma_enabled());
        assert!(!profile.avx512f_enabled());
        assert!(!profile.neon_enabled());
    }
}
