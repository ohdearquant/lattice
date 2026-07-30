use super::*;

impl MetalQwen35State {
    // ===================================================================
    // Dispatch helpers
    // ===================================================================

    /// Dispatch f16-weight matmul: C[M,N] = A[M,K] @ B_half[N,K]^T.
    ///
    /// A is f32 (activations), B is f16 (weights), C is f32 (output).
    /// The MSL kernel loads weight tiles as `half` and widens to `float`
    /// before accumulation, maintaining f32 precision in the dot product.
    /// Dispatch optimized GEMV for M=1 decode (one threadgroup per output element).
    /// Uses gemv_decode_m1: float4/half4 vectorized loads + simdgroup reduction.
    pub(super) fn dispatch_matmul_half(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let params = GemmParams {
            m,
            n,
            k,
            lda: k,
            ldb: k,
            ldc: n,
        };
        enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<GemmParams>() as u64,
            &params as *const GemmParams as *const _,
        );
        // One threadgroup per output element N, 256 threads per group.
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch Q8_0 GEMV for M=1 decode. 2 rows per threadgroup, 128 threads.
    /// Uses int8 × f32 direct multiply + simd_sum — no dequantization step.
    fn dispatch_matmul_q8(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,       // activation [M,K] float (M=1 for decode)
        qw: &Q4WeightBuf, // Q8_0 packed weights [N, K/32 * 34]
        y: &Buffer,       // output [M,N] float
        _m: u32,          // always 1 for decode (ignored)
        n: u32,
        k: u32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q8);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(&qw.buffer), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(2) as u64, 1, 1),
            MTLSize::new(32, 4, 1), // 128 threads: 32 lanes × 4 simdgroups
        );
    }

    /// Q8 matmul with batch support and buffer offsets.
    /// M=1 uses GEMV (decode hot path), M>1 uses GEMM (batch prefill).
    fn dispatch_gemm_q8(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64, // byte offset into x
        qw: &Q4WeightBuf,
        y: &Buffer,
        y_offset: u64, // byte offset into y
        m: u32,
        n: u32,
        k: u32,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        assert!(
            k > 0 && k.is_multiple_of(32),
            "dispatch_gemm_q8 requires K non-zero and divisible by 32, got {k}"
        );
        if m <= 1 {
            // GEMV decode path (M=1)
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q8);
            enc.set_buffer(0, Some(x), x_offset);
            enc.set_buffer(1, Some(&qw.buffer), 0);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                MTLSize::new(32, 4, 1),
            );
        } else if let Some(tiled) = self.engine.pipelines.gemm_q8_tiled.as_ref() {
            // Tiled simdgroup-matrix GEMM (Apple7+, BM=64 × BN=32).
            // Buffer bindings: buf(0)=QW, buf(1)=X, buf(2)=Y.
            enc.set_compute_pipeline_state(tiled);
            enc.set_buffer(0, Some(&qw.buffer), 0);
            enc.set_buffer(1, Some(x), x_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        } else {
            // Naive fallback GEMM. Buffer bindings: buf(0)=X, buf(1)=QW, buf(2)=Y.
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemm_q8);
            enc.set_buffer(0, Some(x), x_offset);
            enc.set_buffer(1, Some(&qw.buffer), 0);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, m.div_ceil(4) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        }
    }

    pub(super) fn dispatch_matmul_q4(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,       // activation [1,K] float
        qw: &Q4WeightBuf, // Q4 packed weights [N, (K/32) * 20]; payload at qw.payload_offset
        y: &Buffer,       // output [1,N] float
        _m: u32,
        n: u32,
        k: u32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q4);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(&qw.buffer), qw.payload_offset);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(2) as u64, 1, 1), // gemv_q4_decode writes NR=2 rows/threadgroup
            MTLSize::new(32, 4, 1),
        );
    }

    fn dispatch_gemm_q4(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        qw: &Q4WeightBuf,
        y: &Buffer,
        y_offset: u64,
        m: u32,
        n: u32,
        k: u32,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        assert!(
            k > 0 && k.is_multiple_of(32),
            "dispatch_gemm_q4 requires K to be non-zero and divisible by 32, got {k}"
        );

        if m == 1 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q4);
            enc.set_buffer(0, Some(x), x_offset);
            enc.set_buffer(1, Some(&qw.buffer), qw.payload_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                MTLSize::new(32, 4, 1),
            );
        } else if let Some(tiled) = self.engine.pipelines.gemm_q4_tiled.as_ref() {
            enc.set_compute_pipeline_state(tiled);
            enc.set_buffer(0, Some(&qw.buffer), qw.payload_offset);
            enc.set_buffer(1, Some(x), x_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        } else {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemm_q4);
            enc.set_buffer(0, Some(&qw.buffer), qw.payload_offset);
            enc.set_buffer(1, Some(x), x_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, m.div_ceil(4) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        }
    }

    pub(super) fn dispatch_matmul(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        qw: &Q4WeightBuf,
        y: &Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        match self.engine.quant_format {
            QuantFormat::Q8_0 => self.dispatch_matmul_q8(enc, x, qw, y, m, n, k),
            QuantFormat::Q4_0 => self.dispatch_matmul_q4(enc, x, qw, y, m, n, k),
        }
    }

    /// Dispatch a Q3 GEMV/GEMM (ADR-072 P1, #420 Stage 2), mirroring
    /// `dispatch_gemm_q4` exactly: `gemv_q3_decode` for M=1 (the decode-path
    /// kernel that matters — decode is weight-bandwidth-bound), the tiled
    /// simdgroup-matrix `gemm_q3_tiled` for M>1.
    ///
    /// Unlike Q4/Q8, there is no naive fallback GEMM for M>1 — Stage 2 scope
    /// is `gemv_q3_decode` + `gemm_q3_tiled` only (see #420 design note §3),
    /// so `gemm_q3_tiled` must be present (Apple7+) whenever this is called
    /// with M>1. Callers that need to support non-Apple7 prefill would need
    /// a naive `gemm_q3` kernel, out of Stage 2 scope.
    ///
    /// # Errors
    /// Returns `Err` if `m > 1` and `gemm_q3_tiled` is unavailable (device
    /// is not Apple7+, or the tiled kernel failed to compile) — Stage 2
    /// has no naive Q3 GEMM fallback to fall back to (mirrors how
    /// `mmap_q3_weight` propagates its fail-closed checks as `Result`
    /// rather than panicking).
    ///
    /// # Panics
    /// Panics if `k` is zero or not a multiple of 32 (a caller
    /// programming error, not a runtime/capability condition).
    #[allow(dead_code)] // wiring to real MLP dispatch is deferred past Stage 2
    fn dispatch_gemm_q3(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        qw: &Q3WeightBuf,
        y: &Buffer,
        y_offset: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), String> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        assert!(
            k > 0 && k.is_multiple_of(32),
            "dispatch_gemm_q3 requires K to be non-zero and divisible by 32, got {k}"
        );

        if m == 1 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q3);
            enc.set_buffer(0, Some(x), x_offset);
            enc.set_buffer(1, Some(&qw.buffer), qw.payload_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                MTLSize::new(32, 4, 1),
            );
        } else {
            let tiled = self
                .engine
                .pipelines
                .gemm_q3_tiled
                .as_ref()
                .ok_or_else(|| {
                    "dispatch_gemm_q3 called with M>1 but gemm_q3_tiled is unavailable \
                 (device is not Apple7+, or the tiled kernel failed to compile); \
                 Stage 2 has no naive Q3 GEMM fallback"
                        .to_string()
                })?;
            enc.set_compute_pipeline_state(tiled);
            enc.set_buffer(0, Some(&qw.buffer), qw.payload_offset);
            enc.set_buffer(1, Some(x), x_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        }
        Ok(())
    }

    pub(super) fn dispatch_gdn_chunked_prefill_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        linear_idx: usize,
        weights: GdnChunkedWeights<'_>,
        params: GdnChunkParams,
    ) {
        self.dispatch_gdn_chunk_materialize_c32(enc, linear_idx, weights, params);
        // MAJ-2 fix: conv_buf update runs in a separate dispatch AFTER the all-chunks
        // materialize completes, so chunk 0's read and the last chunk's write cannot race.
        self.dispatch_gdn_chunk_conv_buf_update_c32(enc, linear_idx, params);
        self.dispatch_gdn_chunk_solve_c32(enc, params);

        for chunk in 0..params.num_chunks {
            let mut cp = params;
            cp.active_chunk = chunk;
            self.dispatch_gdn_chunk_residual_output_c32(enc, linear_idx, cp);
            self.dispatch_gdn_chunk_state_update_c32(enc, linear_idx, cp);
        }

        self.dispatch_gdn_chunk_norm_silu_c32(enc, weights.norm_weight, params);
    }

    fn dispatch_gdn_chunk_materialize_c32(
        &self,
        enc: &ComputeCommandEncoderRef,
        linear_idx: usize,
        weights: GdnChunkedWeights<'_>,
        params: GdnChunkParams,
    ) {
        let sc = &self.session.activations.gdn_chunk;
        let p_bytes = std::mem::size_of::<GdnChunkParams>() as u64;
        enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_chunk_materialize_c32);
        enc.set_buffer(
            0,
            Some(self.session.gdn_gpu_state.layer(linear_idx).conv_buffer()),
            0,
        );
        enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), 0);
        enc.set_buffer(2, Some(weights.conv1d_weight), 0);
        enc.set_buffer(3, Some(&self.session.activations.hidden), 0);
        enc.set_buffer(4, Some(weights.in_proj_b), 0);
        enc.set_buffer(5, Some(weights.in_proj_a), 0);
        enc.set_buffer(6, Some(weights.a_log), 0);
        enc.set_buffer(7, Some(weights.dt_bias), 0);
        enc.set_buffer(8, Some(&sc.q), 0);
        enc.set_buffer(9, Some(&sc.k), 0);
        enc.set_buffer(10, Some(&sc.v), 0);
        enc.set_buffer(11, Some(&sc.beta_log_alpha), 0);
        enc.set_bytes(12, p_bytes, &params as *const GdnChunkParams as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(params.num_chunks as u64, params.num_value_heads as u64, 1),
            MTLSize::new(32, 4, 1),
        );
    }

    fn dispatch_gdn_chunk_conv_buf_update_c32(
        &self,
        enc: &ComputeCommandEncoderRef,
        linear_idx: usize,
        params: GdnChunkParams,
    ) {
        let p_bytes = std::mem::size_of::<GdnChunkParams>() as u64;
        enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_chunk_conv_buf_update_c32);
        enc.set_buffer(
            0,
            Some(self.session.gdn_gpu_state.layer(linear_idx).conv_buffer()),
            0,
        );
        enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), 0);
        enc.set_bytes(2, p_bytes, &params as *const GdnChunkParams as *const _);
        // Grid: (1, num_value_heads, 1) — single threadgroup for the serial conv-buf update.
        enc.dispatch_thread_groups(
            MTLSize::new(1, params.num_value_heads as u64, 1),
            MTLSize::new(32, 4, 1),
        );
    }

    fn dispatch_gdn_chunk_solve_c32(&self, enc: &ComputeCommandEncoderRef, params: GdnChunkParams) {
        let sc = &self.session.activations.gdn_chunk;
        let p_bytes = std::mem::size_of::<GdnChunkParams>() as u64;
        enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_chunk_solve_c32);
        enc.set_buffer(0, Some(&sc.q), 0);
        enc.set_buffer(1, Some(&sc.k), 0);
        enc.set_buffer(2, Some(&sc.v), 0);
        enc.set_buffer(3, Some(&sc.beta_log_alpha), 0);
        enc.set_buffer(4, Some(&sc.gamma), 0);
        enc.set_buffer(5, Some(&sc.gamma_end), 0);
        enc.set_buffer(6, Some(&sc.kkt), 0);
        enc.set_buffer(7, Some(&sc.qk_l), 0);
        enc.set_buffer(8, Some(&sc.w), 0);
        enc.set_buffer(9, Some(&sc.u), 0);
        enc.set_buffer(10, Some(&sc.k_right), 0);
        enc.set_bytes(11, p_bytes, &params as *const GdnChunkParams as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(params.num_chunks as u64, params.num_value_heads as u64, 1),
            MTLSize::new(32, 4, 1),
        );
    }

    fn dispatch_gdn_chunk_residual_output_c32(
        &self,
        enc: &ComputeCommandEncoderRef,
        linear_idx: usize,
        params: GdnChunkParams,
    ) {
        let sc = &self.session.activations.gdn_chunk;
        let vd = params.value_dim as u64;
        let num_v_tiles = vd.div_ceil(8);
        let p_bytes = std::mem::size_of::<GdnChunkParams>() as u64;
        enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_chunk_residual_output_c32);
        enc.set_buffer(
            0,
            Some(self.session.gdn_gpu_state.layer(linear_idx).s_matrix()),
            0,
        );
        enc.set_buffer(1, Some(&sc.q), 0);
        enc.set_buffer(2, Some(&sc.w), 0);
        enc.set_buffer(3, Some(&sc.u), 0);
        enc.set_buffer(4, Some(&sc.gamma), 0);
        enc.set_buffer(5, Some(&sc.qk_l), 0);
        enc.set_buffer(6, Some(&sc.r), 0);
        enc.set_buffer(7, Some(&sc.raw_out), 0);
        enc.set_bytes(8, p_bytes, &params as *const GdnChunkParams as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(num_v_tiles, params.num_value_heads as u64, 1),
            MTLSize::new(32, 8, 1),
        );
    }

    fn dispatch_gdn_chunk_state_update_c32(
        &self,
        enc: &ComputeCommandEncoderRef,
        linear_idx: usize,
        params: GdnChunkParams,
    ) {
        let sc = &self.session.activations.gdn_chunk;
        let kd = params.key_dim as u64;
        let vd = params.value_dim as u64;
        let p_bytes = std::mem::size_of::<GdnChunkParams>() as u64;
        enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_chunk_state_update_c32);
        enc.set_buffer(
            0,
            Some(self.session.gdn_gpu_state.layer(linear_idx).s_matrix()),
            0,
        );
        enc.set_buffer(1, Some(&sc.r), 0);
        enc.set_buffer(2, Some(&sc.k_right), 0);
        enc.set_buffer(3, Some(&sc.gamma_end), 0);
        enc.set_bytes(4, p_bytes, &params as *const GdnChunkParams as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(
                kd.div_ceil(16),
                vd.div_ceil(16),
                params.num_value_heads as u64,
            ),
            MTLSize::new(16, 16, 1),
        );
    }

    fn dispatch_gdn_chunk_norm_silu_c32(
        &self,
        enc: &ComputeCommandEncoderRef,
        norm_weight: &Buffer,
        params: GdnChunkParams,
    ) {
        let sc = &self.session.activations.gdn_chunk;
        let p_bytes = std::mem::size_of::<GdnChunkParams>() as u64;
        enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_chunk_norm_silu_c32);
        enc.set_buffer(0, Some(&sc.raw_out), 0);
        enc.set_buffer(1, Some(&self.session.activations.gdn_z), 0);
        enc.set_buffer(2, Some(norm_weight), 0);
        enc.set_buffer(3, Some(&self.session.activations.gdn_z), 0);
        enc.set_bytes(4, p_bytes, &params as *const GdnChunkParams as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(params.n_tokens as u64, params.num_value_heads as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    pub(super) fn dispatch_gemm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        qw: &Q4WeightBuf,
        y: &Buffer,
        y_offset: u64,
        m: u32,
        n: u32,
        k: u32,
    ) {
        match self.engine.quant_format {
            QuantFormat::Q8_0 => self.dispatch_gemm_q8(enc, x, x_offset, qw, y, y_offset, m, n, k),
            QuantFormat::Q4_0 => self.dispatch_gemm_q4(enc, x, x_offset, qw, y, y_offset, m, n, k),
        }
    }

    pub(super) fn dispatch_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        gamma: &Buffer,
        row_len: u32,
        num_rows: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.rms_norm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gamma), 0);
        enc.set_bytes(2, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(3, 4, &num_rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        let wg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), MTLSize::new(wg, 1, 1));
    }

    pub(super) fn dispatch_per_head_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        gamma: &Buffer,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.per_head_rms_norm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gamma), 0);
        enc.set_bytes(2, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        let wg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), MTLSize::new(wg, 1, 1));
    }

    /// `mrope_override`, when supplied (Qwen3.5 vision M-RoPE, ADR-069 MP3),
    /// replaces the 1-D `engine.rope_cos`/`rope_sin` table indexed by
    /// `position` with a caller-supplied single-token cos/sin row (indexed
    /// at offset 0) — the interleaved-axis row built by
    /// `Qwen35VisionRequest::build_mrope_tables`/`build_decode_cos_sin`.
    /// The rotation kernel and its stride-half arithmetic are unchanged
    /// either way; only the cos/sin row source differs, mirroring
    /// `cpu_f16::full_attention_step_f16`'s `mrope_cos_sin` parameter.
    pub(super) fn dispatch_partial_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        num_heads: u32,
        head_dim: u32,
        half_rope_dim: u32,
        position: u32,
        mrope_override: Option<(&Buffer, &Buffer)>,
    ) {
        let (cos_buf, sin_buf, pos_offset) = match mrope_override {
            Some((cos_buf, sin_buf)) => (cos_buf, sin_buf, 0u32),
            None => (&self.engine.rope_cos, &self.engine.rope_sin, position),
        };
        enc.set_compute_pipeline_state(&self.engine.pipelines.partial_rope);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(cos_buf), 0);
        enc.set_buffer(2, Some(sin_buf), 0);
        enc.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(5, 4, &half_rope_dim as *const u32 as *const _);
        enc.set_bytes(6, 4, &pos_offset as *const u32 as *const _);
        let total_pairs = num_heads * half_rope_dim;
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(total_pairs as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_decode_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        k_cache: &Buffer,
        v_cache: &Buffer,
        cache_len: u32,
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        q_dim: u32,
        kv_dim: u32,
        scale: f32,
    ) {
        const PARTITION_TOKENS: u32 = METAL_FLASH_PARTITION_TOKENS as u32;
        const DIRECT_THRESHOLD: u32 = 512; // use direct path for short caches

        validate_flash_decode_shape(
            head_dim as usize,
            num_q_heads as usize,
            num_kv_heads as usize,
            q_dim as usize,
            kv_dim as usize,
        )
        .expect("invalid Metal FlashAttention decode shape");

        let kv_f16 = self.use_kv_f16;

        // Common buffer setup (same layout for both direct and partial kernels)
        let set_common_bufs = |enc: &ComputeCommandEncoderRef, out_or_partials: &Buffer| {
            enc.set_buffer(0, Some(&self.session.activations.q_separated), 0);
            enc.set_buffer(1, Some(k_cache), 0);
            enc.set_buffer(2, Some(v_cache), 0);
            enc.set_buffer(3, Some(out_or_partials), 0);
            enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
            enc.set_bytes(5, 4, &head_dim as *const u32 as *const _);
            enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
            enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
            enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
            enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
            enc.set_bytes(10, 4, &scale as *const f32 as *const _);
        };

        if cache_len <= DIRECT_THRESHOLD {
            // Direct grouped flash decode: one threadgroup per KV head (H1+H2+H4+H5)
            if kv_f16 {
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention_f16);
            } else {
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
            }
            set_common_bufs(enc, &self.session.activations.attn_out);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            if self.path_proof_enabled {
                self.path_proof
                    .decode_attn_direct
                    .fetch_add(1, Ordering::Relaxed);
            }
        } else {
            // Partitioned flash decode (H3): partial kernel + reduce kernel.
            // Split KV cache into PARTITION_TOKENS-token chunks for better occupancy.
            let num_partitions = cache_len.div_ceil(PARTITION_TOKENS);

            // Partial pass: one TG per (KV head, partition)
            if kv_f16 {
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attn_partial_f16);
            } else {
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attn_partial);
            }
            set_common_bufs(enc, &self.session.activations.attn_partials);
            enc.set_bytes(11, 4, &PARTITION_TOKENS as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv_heads as u64, num_partitions as u64, 1),
                MTLSize::new(256, 1, 1),
            );
            if self.path_proof_enabled {
                self.path_proof
                    .decode_attn_split_partial
                    .fetch_add(1, Ordering::Relaxed);
            }

            // Reduce pass: one TG per KV head, combines all partitions.
            // decode_attention_flash_reduce reads f32 attn_partials, not KV — no f16 variant.
            enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attn_reduce);
            enc.set_buffer(0, Some(&self.session.activations.attn_partials), 0);
            enc.set_buffer(1, Some(&self.session.activations.attn_out), 0);
            enc.set_bytes(2, 4, &num_q_heads as *const u32 as *const _);
            enc.set_bytes(3, 4, &num_kv_heads as *const u32 as *const _);
            enc.set_bytes(4, 4, &num_partitions as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            if self.path_proof_enabled {
                self.path_proof
                    .decode_attn_split_reduce
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub(super) fn dispatch_sigmoid_gate(&self, enc: &ComputeCommandEncoderRef, count: u32) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
        enc.set_buffer(0, Some(&self.session.activations.attn_out), 0);
        enc.set_buffer(1, Some(&self.session.activations.gate_z), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    pub(super) fn dispatch_scatter_q_gate(
        &self,
        enc: &ComputeCommandEncoderRef,
        num_heads: u32,
        head_dim: u32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
        enc.set_buffer(0, Some(&self.session.activations.q), 0);
        enc.set_buffer(1, Some(&self.session.activations.q_separated), 0);
        enc.set_buffer(2, Some(&self.session.activations.gate_z), 0);
        enc.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        let total = num_heads * head_dim;
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(total as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    pub(super) fn dispatch_scatter_q_gate_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        num_tokens: u32,
        num_heads: u32,
        head_dim: u32,
    ) {
        let total = num_tokens * num_heads * head_dim;
        enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate_batch);
        enc.set_buffer(0, Some(&self.session.activations.q), 0);
        enc.set_buffer(1, Some(&self.session.activations.q_separated), 0);
        enc.set_buffer(2, Some(&self.session.activations.gate_z), 0);
        enc.set_bytes(3, 4, &num_tokens as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(5, 4, &head_dim as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(total as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    pub(super) fn dispatch_per_head_rms_norm_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        gamma: &Buffer,
        num_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        let total_groups = num_tokens * num_heads;
        enc.set_compute_pipeline_state(&self.engine.pipelines.per_head_rms_norm_batch);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gamma), 0);
        enc.set_bytes(2, 4, &num_tokens as *const u32 as *const _);
        enc.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(5, 4, &eps as *const f32 as *const _);
        let wg = 256u64;
        enc.dispatch_thread_groups(
            MTLSize::new(total_groups as u64, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_partial_rope_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        num_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        half_rope_dim: u32,
        base_pos: u32,
    ) {
        let total_pairs = num_tokens * num_heads * half_rope_dim;
        enc.set_compute_pipeline_state(&self.engine.pipelines.partial_rope_batch);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(&self.engine.rope_cos), 0);
        enc.set_buffer(2, Some(&self.engine.rope_sin), 0);
        enc.set_bytes(3, 4, &num_tokens as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(5, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(6, 4, &half_rope_dim as *const u32 as *const _);
        enc.set_bytes(7, 4, &base_pos as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(total_pairs as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_copy_kv_cache_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        k_src: &Buffer,
        v_src: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        num_tokens: u32,
        kv_dim: u32,
        base_pos: u32,
    ) {
        let total = num_tokens * kv_dim;
        if self.use_kv_f16 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_kv_cache_batch_f16);
        } else {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_kv_cache_batch);
        }
        enc.set_buffer(0, Some(k_src), 0);
        enc.set_buffer(1, Some(v_src), 0);
        enc.set_buffer(2, Some(k_cache), 0);
        enc.set_buffer(3, Some(v_cache), 0);
        enc.set_bytes(4, 4, &num_tokens as *const u32 as *const _);
        enc.set_bytes(5, 4, &kv_dim as *const u32 as *const _);
        enc.set_bytes(6, 4, &base_pos as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(total as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
        if self.path_proof_enabled {
            self.path_proof
                .prefill_kv_batch
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_prefill_attention_batched(
        &self,
        enc: &ComputeCommandEncoderRef,
        k_cache: &Buffer,
        v_cache: &Buffer,
        base_pos: u32,
        num_tokens: u32,
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        q_dim: u32,
        kv_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        validate_flash_decode_shape(
            head_dim as usize,
            num_q_heads as usize,
            num_kv_heads as usize,
            q_dim as usize,
            kv_dim as usize,
        )?;
        let cache_len_total = base_pos.checked_add(num_tokens).ok_or_else(|| {
            "prefill_attention_batched: base_pos + num_tokens overflow".to_string()
        })?;
        if self.use_kv_f16 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.prefill_attention_batched_f16);
        } else {
            enc.set_compute_pipeline_state(&self.engine.pipelines.prefill_attention_batched);
        }
        enc.set_buffer(0, Some(&self.session.activations.q_separated), 0);
        enc.set_buffer(1, Some(k_cache), 0);
        enc.set_buffer(2, Some(v_cache), 0);
        enc.set_buffer(3, Some(&self.session.activations.attn_out), 0);
        enc.set_bytes(4, 4, &base_pos as *const u32 as *const _);
        enc.set_bytes(5, 4, &num_tokens as *const u32 as *const _);
        enc.set_bytes(6, 4, &cache_len_total as *const u32 as *const _);
        enc.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(8, 4, &num_q_heads as *const u32 as *const _);
        enc.set_bytes(9, 4, &num_kv_heads as *const u32 as *const _);
        enc.set_bytes(10, 4, &q_dim as *const u32 as *const _);
        enc.set_bytes(11, 4, &kv_dim as *const u32 as *const _);
        enc.set_bytes(12, 4, &scale as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv_heads as u64, num_tokens as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        if self.path_proof_enabled {
            self.path_proof
                .prefill_attn_batched
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    pub(super) fn dispatch_silu_mul(&self, enc: &ComputeCommandEncoderRef, count: u32) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
        enc.set_buffer(0, Some(&self.session.activations.gate), 0);
        enc.set_buffer(1, Some(&self.session.activations.up), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    // Dense decode fused: data[0..count] = silu(data[0..count]) * data[count..2*count]
    pub(super) fn dispatch_silu_mul_fused(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &Buffer,
        count: u32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul_fused);
        enc.set_buffer(0, Some(data), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    // Variant of dispatch_gemm that adds qw_extra_offset bytes to the weight buffer binding.
    // Used by batch Dense paths to access gate (offset=0) or up (offset=gate_byte_size)
    // within the fused gate_up_proj buffer without modifying the MSL kernels.
    pub(super) fn dispatch_gemm_at(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        qw: &Q4WeightBuf,
        qw_extra_offset: u64,
        y: &Buffer,
        y_offset: u64,
        m: u32,
        n: u32,
        k: u32,
    ) {
        match self.engine.quant_format {
            QuantFormat::Q8_0 => self.dispatch_gemm_q8_at(
                enc,
                x,
                x_offset,
                qw,
                qw_extra_offset,
                y,
                y_offset,
                m,
                n,
                k,
            ),
            QuantFormat::Q4_0 => self.dispatch_gemm_q4_at(
                enc,
                x,
                x_offset,
                qw,
                qw_extra_offset,
                y,
                y_offset,
                m,
                n,
                k,
            ),
        }
    }

    fn dispatch_gemm_q8_at(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        qw: &Q4WeightBuf,
        qw_extra_offset: u64,
        y: &Buffer,
        y_offset: u64,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let wq_offset = qw.payload_offset + qw_extra_offset;
        if m == 0 || n == 0 {
            return;
        }
        assert!(
            k > 0 && k.is_multiple_of(32),
            "dispatch_gemm_q8_at requires K non-zero and divisible by 32, got {k}"
        );
        if m <= 1 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q8);
            enc.set_buffer(0, Some(x), x_offset);
            enc.set_buffer(1, Some(&qw.buffer), wq_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                MTLSize::new(32, 4, 1),
            );
        } else if let Some(tiled) = self.engine.pipelines.gemm_q8_tiled.as_ref() {
            // Tiled path: buf(0)=QW (with offset), buf(1)=X, buf(2)=Y.
            enc.set_compute_pipeline_state(tiled);
            enc.set_buffer(0, Some(&qw.buffer), wq_offset);
            enc.set_buffer(1, Some(x), x_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        } else {
            // Naive fallback. Buffer bindings: buf(0)=X, buf(1)=QW, buf(2)=Y.
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemm_q8);
            enc.set_buffer(0, Some(x), x_offset);
            enc.set_buffer(1, Some(&qw.buffer), wq_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, m.div_ceil(4) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        }
    }

    fn dispatch_gemm_q4_at(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        qw: &Q4WeightBuf,
        qw_extra_offset: u64,
        y: &Buffer,
        y_offset: u64,
        m: u32,
        n: u32,
        k: u32,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        assert!(
            k > 0 && k.is_multiple_of(32),
            "dispatch_gemm_q4_at requires K to be non-zero and divisible by 32, got {k}"
        );
        let wq_offset = qw.payload_offset + qw_extra_offset;
        if m == 1 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q4);
            enc.set_buffer(0, Some(x), x_offset);
            enc.set_buffer(1, Some(&qw.buffer), wq_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                MTLSize::new(32, 4, 1),
            );
        } else if let Some(tiled) = self.engine.pipelines.gemm_q4_tiled.as_ref() {
            enc.set_compute_pipeline_state(tiled);
            enc.set_buffer(0, Some(&qw.buffer), wq_offset);
            enc.set_buffer(1, Some(x), x_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        } else {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemm_q4);
            enc.set_buffer(0, Some(&qw.buffer), wq_offset);
            enc.set_buffer(1, Some(x), x_offset);
            enc.set_buffer(2, Some(y), y_offset);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, m.div_ceil(4) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        }
    }

    pub(super) fn dispatch_copy(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        count: u32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.copy);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    pub(super) fn dispatch_copy_offset(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        count: u32,
        dst_offset: u32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        enc.set_bytes(3, 4, &dst_offset as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    /// Copy `count` f32 elements from `src` into the KV cache buffer `dst` at element
    /// offset `dst_offset`. Selects the f16 narrowing kernel when `use_kv_f16` is set.
    pub(super) fn dispatch_copy_offset_kv(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        count: u32,
        dst_offset: u32,
    ) {
        if self.use_kv_f16 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset_f16);
        } else {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
        }
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        enc.set_bytes(3, 4, &dst_offset as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
        if self.path_proof_enabled {
            self.path_proof
                .decode_kv_copy
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Fused residual add + RMS norm.
    /// residual_out = base + delta; normed_out = rms_norm(residual_out) * (1+gamma)
    /// Replaces 4 dispatches (copy+add+copy+rms_norm) with 1.
    pub(super) fn dispatch_fused_residual_add_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        base: &Buffer,
        delta: &Buffer,
        residual_out: &Buffer,
        normed_out: &Buffer,
        gamma: &Buffer,
        row_len: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.fused_residual_add_norm);
        enc.set_buffer(0, Some(base), 0);
        enc.set_buffer(1, Some(delta), 0);
        enc.set_buffer(2, Some(residual_out), 0);
        enc.set_buffer(3, Some(normed_out), 0);
        enc.set_buffer(4, Some(gamma), 0);
        enc.set_bytes(5, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(6, 4, &eps as *const f32 as *const _);
        let wg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(wg, 1, 1));
    }

    /// Fused copy-then-RMS-norm for decode pre-norm: saves `src` to `residual_out`
    /// and normalizes `src` in-place.  Replaces `dispatch_copy` + `dispatch_rms_norm`.
    pub(super) fn dispatch_copy_and_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        residual_out: &Buffer,
        gamma: &Buffer,
        row_len: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.copy_and_rms_norm);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(residual_out), 0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_bytes(3, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        let wg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(wg, 1, 1));
    }

    /// Batch copy-then-RMS-norm for prefill: copies `src` rows to `residual_out`
    /// and normalizes `src` in-place.  Multi-row version.
    pub(super) fn dispatch_copy_and_rms_norm_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        residual_out: &Buffer,
        gamma: &Buffer,
        row_len: u32,
        num_rows: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.copy_and_rms_norm_batch);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(residual_out), 0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_bytes(3, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_rows as *const u32 as *const _);
        enc.set_bytes(5, 4, &eps as *const f32 as *const _);
        let wg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), MTLSize::new(wg, 1, 1));
    }

    /// Batch fused residual-add + RMS-norm for prefill: `residual_out = base + delta`,
    /// `normed_out = rms_norm(residual_out)`.  Multi-row version.
    pub(super) fn dispatch_fused_residual_add_norm_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        base: &Buffer,
        delta: &Buffer,
        residual_out: &Buffer,
        normed_out: &Buffer,
        gamma: &Buffer,
        row_len: u32,
        num_rows: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.fused_residual_add_norm_batch);
        enc.set_buffer(0, Some(base), 0);
        enc.set_buffer(1, Some(delta), 0);
        enc.set_buffer(2, Some(residual_out), 0);
        enc.set_buffer(3, Some(normed_out), 0);
        enc.set_buffer(4, Some(gamma), 0);
        enc.set_bytes(5, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(6, 4, &num_rows as *const u32 as *const _);
        enc.set_bytes(7, 4, &eps as *const f32 as *const _);
        let wg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), MTLSize::new(wg, 1, 1));
    }

    /// Fused add-into-residual + copy-to-hidden for decode end-of-layer.
    /// `residual[i] += src[i]; dst[i] = residual[i]`.
    /// Replaces `dispatch_add` + `dispatch_copy`.
    pub(super) fn dispatch_add_and_copy(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        residual: &Buffer,
        dst: &Buffer,
        count: u32,
    ) {
        enc.set_compute_pipeline_state(&self.engine.pipelines.add_and_copy);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(residual), 0);
        enc.set_buffer(2, Some(dst), 0);
        enc.set_bytes(3, 4, &count as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }

    // -----------------------------------------------------------------------
    // lm_head two-stage block-top-k dispatch helpers (issue #171)
    // -----------------------------------------------------------------------

    /// Stage 1: fused lm_head GEMV + block-local exact argmax/top-k.
    /// Writes compact `(logit, token_id)` candidates into `topk_scratch_a`,
    /// `local_k` per tile. Appended to the same encoder as the caller's
    /// other dispatches — no second command-buffer round-trip.
    ///
    /// `hidden_offset` is the byte offset into
    /// `self.session.activations.hidden` of the single post-RMSNorm token
    /// row to project (0 for decode; the last-token offset for prefill).
    ///
    /// DISPATCH GEOMETRY: the kernel owns vocab rows on tg_pos.x, so x
    /// groups must be `ceil(vocab_size / LM_HEAD_ROWS_PER_TG)`.
    /// Under-dispatch leaves rows unwritten, which has previously
    /// corrupted lm_head outputs silently in this codebase.
    ///
    /// `slot` is the precompiled Stage-1 pipeline index for `local_k`
    /// (`LM_HEAD_LOCAL_KS[slot] == local_k`). Callers must obtain both via
    /// `resolve_block_local_k`, which is the single place that validates
    /// `local_k` against `LM_HEAD_LOCAL_KS` and falls back to the exact
    /// full-logit path on an unsupported value — by the time `slot`
    /// reaches this function it is already a valid array index, so this
    /// dispatch-encoding helper (mid command-buffer, nothing sane to
    /// "return an error" to) has no unsupported-LOCAL_K case left to
    /// guard against.
    ///
    /// Returns the candidate-group count (Stage 2's `candidate_groups`).
    pub(super) fn dispatch_lm_head_block_stage1_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        cfg: &Qwen35Config,
        hidden_offset: u64,
        local_k: u32,
        slot: usize,
    ) -> u32 {
        let vocab_size = cfg.vocab_size as u32;
        let row_groups = vocab_size.div_ceil(LM_HEAD_ROWS_PER_TG);

        // Fail closed rather than silently overrunning topk_scratch_a: for every
        // vocab_size, ceil(vocab/1024)*256 (the existing allocation, sized for the
        // older fixed-256-per-group scheme) is provably >= ceil(vocab/256)*local_k
        // (this Stage-1's worst case, local_k<=64) because 1024 = 4*256 and
        // 256*8/1024 == 64*8/256 — the two schemes reserve the same 2 bytes/vocab-row
        // asymptotically, and the coarser grouping's ceiling only adds headroom. This
        // assert documents and guards that invariant instead of leaving it implicit.
        let candidate_capacity = self.session.activations.topk_scratch_a.length() / 8;
        let candidates_written = row_groups as u64 * local_k as u64;
        assert!(
            candidates_written <= candidate_capacity,
            "lm_head block-topk Stage-1 would write {candidates_written} candidates \
             ({row_groups} groups * local_k {local_k}) but topk_scratch_a only holds \
             {candidate_capacity} — buffer-sizing invariant violated"
        );

        match self.engine.quant_format {
            QuantFormat::Q8_0 => {
                enc.set_compute_pipeline_state(&self.engine.pipelines.lm_head_block_topk_f16[slot]);
                enc.set_buffer(0, Some(&self.session.activations.hidden), hidden_offset);
                enc.set_buffer(1, Some(&self.engine.embed_tokens), 0);
                enc.set_buffer(2, Some(&self.session.activations.topk_scratch_a), 0);
                enc.set_bytes(3, 4, &vocab_size as *const u32 as *const _);
            }
            QuantFormat::Q4_0 => {
                let qw = &self.engine.embed_tokens_q8;
                enc.set_compute_pipeline_state(&self.engine.pipelines.lm_head_block_topk_q4[slot]);
                enc.set_buffer(0, Some(&self.session.activations.hidden), hidden_offset);
                enc.set_buffer(1, Some(&qw.buffer), qw.payload_offset);
                enc.set_buffer(2, Some(&self.session.activations.topk_scratch_a), 0);
                enc.set_bytes(3, 4, &vocab_size as *const u32 as *const _);
            }
        }
        enc.dispatch_thread_groups(
            MTLSize::new(row_groups as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        row_groups
    }

    /// Stage 2: reduces the compact per-tile candidates Stage 1 already
    /// wrote into `topk_scratch_a` to the global result. Reuses the exact
    /// same `argmax_merge` / `topk_merge_pass` kernels as the existing
    /// `dispatch_topk_enc` seam below — the only difference is that the
    /// input candidates come from Stage 1's fused GEMV instead of a
    /// full-logits first pass over a materialized `[vocab_size]` buffer.
    /// Appended to the same encoder; no second command-buffer round-trip.
    ///
    /// Returns 0 if the final `local_k` candidates are in
    /// `topk_scratch_a`, 1 for `topk_scratch_b`.
    pub(super) fn dispatch_block_topk_merge_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        candidate_groups: u32,
        local_k: u32,
    ) -> u8 {
        if local_k == 1 {
            enc.set_compute_pipeline_state(&self.engine.pipelines.argmax_merge);
            enc.set_buffer(0, Some(&self.session.activations.topk_scratch_a), 0);
            enc.set_buffer(1, Some(&self.session.activations.topk_scratch_b), 0);
            enc.set_bytes(2, 4, &candidate_groups as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
            return 1;
        }

        let mut current_groups = candidate_groups;
        let mut which: u8 = 0;
        while current_groups > 1 {
            let fan_in: u32 = 16u32.min(current_groups);
            let out_groups = current_groups.div_ceil(fan_in);
            let (in_buf, out_buf) = if which == 0 {
                (
                    &self.session.activations.topk_scratch_a,
                    &self.session.activations.topk_scratch_b,
                )
            } else {
                (
                    &self.session.activations.topk_scratch_b,
                    &self.session.activations.topk_scratch_a,
                )
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.topk_merge_pass);
            enc.set_buffer(0, Some(in_buf), 0);
            enc.set_buffer(1, Some(out_buf), 0);
            enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
            enc.set_bytes(3, 4, &local_k as *const u32 as *const _);
            enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(out_groups as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            current_groups = out_groups;
            which = 1 - which;
        }
        which
    }

    // -----------------------------------------------------------------------
    // GPU Top-K dispatch helpers
    // -----------------------------------------------------------------------

    /// Dispatch top-k kernels into `enc` (same command buffer as the logits GEMV).
    ///
    /// Runs first-pass + iterative merge passes.  All dispatches are in the
    /// same encoder so the GPU executes them in order without extra synchronisation.
    ///
    /// Returns 0 if the final result is in `topk_scratch_a`, 1 for `topk_scratch_b`.
    /// The caller reads from that buffer after `wait_until_completed()`.
    pub(super) fn dispatch_topk_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        vocab_size: u32,
        k: u32,
    ) -> u8 {
        // compact_route must be set before dispatch; CpuFallback means compact_topk=0.
        debug_assert!(
            self.session.compact_route != GpuTopkRoute::CpuFallback,
            "dispatch_topk_enc called with CpuFallback route"
        );
        if k == 1 {
            // Dedicated argmax: two passes, no sorting.
            let groups = vocab_size.div_ceil(1024);
            enc.set_compute_pipeline_state(&self.engine.pipelines.argmax_first);
            enc.set_buffer(0, Some(&self.session.activations.logits), 0);
            enc.set_buffer(1, Some(&self.session.activations.topk_scratch_a), 0);
            enc.set_bytes(2, 4, &vocab_size as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(1024, 1, 1));

            enc.set_compute_pipeline_state(&self.engine.pipelines.argmax_merge);
            enc.set_buffer(0, Some(&self.session.activations.topk_scratch_a), 0);
            enc.set_buffer(1, Some(&self.session.activations.topk_scratch_b), 0);
            enc.set_bytes(2, 4, &groups as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));

            return 1; // result in scratch_b[0]
        }

        // k > 1: hierarchical k=50 SIMD-group tournament (no bitonic sort).
        debug_assert_eq!(k, 50, "only HierarchicalK50 route is supported for k>1");
        debug_assert_eq!(self.session.compact_route, GpuTopkRoute::HierarchicalK50);

        let tile = 1024u32;
        let first_pass_groups = vocab_size.div_ceil(tile);

        enc.set_compute_pipeline_state(&self.engine.pipelines.topk_select50_first);
        enc.set_buffer(0, Some(&self.session.activations.logits), 0);
        enc.set_buffer(1, Some(&self.session.activations.topk_scratch_a), 0);
        enc.set_bytes(2, 4, &vocab_size as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(first_pass_groups as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        let mut current_groups = first_pass_groups;
        let mut which: u8 = 0;

        while current_groups > 1 {
            let fan_in: u32 = 16u32.min(current_groups);
            let out_groups = current_groups.div_ceil(fan_in);

            let (in_buf, out_buf) = if which == 0 {
                (
                    &self.session.activations.topk_scratch_a,
                    &self.session.activations.topk_scratch_b,
                )
            } else {
                (
                    &self.session.activations.topk_scratch_b,
                    &self.session.activations.topk_scratch_a,
                )
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.topk_select50_merge);
            enc.set_buffer(0, Some(in_buf), 0);
            enc.set_buffer(1, Some(out_buf), 0);
            enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
            enc.set_bytes(3, 4, &fan_in as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(out_groups as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            current_groups = out_groups;
            which = 1 - which;
        }

        which
    }
}
