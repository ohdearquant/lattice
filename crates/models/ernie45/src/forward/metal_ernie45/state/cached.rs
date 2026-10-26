//! Explicit cache ownership and embedding-driven forwards.

use super::super::MetalErnie45KvCache;
use super::*;

pub(in super::super) struct CacheStorage {
    k: Buffer,
    v: Buffer,
    owner: Arc<()>,
}

#[cfg(all(test, feature = "f16"))]
pub(in super::super) type TestKvRows = (Vec<f32>, Vec<f32>);

#[cfg(all(test, feature = "f16"))]
pub(super) struct TestTraceBuffers {
    layers: Vec<(Buffer, Buffer)>,
}

impl MetalErnie45State {
    /// Allocate an empty, device-resident cache bound to this exact decoder.
    ///
    /// # Errors
    /// Rejects zero capacity, capacity exceeding this state's allocation, integer
    /// overflow or device allocation failure. Buffers never grow during decode.
    pub fn new_kv_cache(&self, capacity: usize) -> Result<MetalErnie45KvCache, InferenceError> {
        if capacity == 0 || capacity > self.max_seq_len {
            return Err(invalid("cache capacity is outside the state capacity"));
        }
        let kv_dim = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let per_layer = elements("cache layer", capacity, kv_dim)?;
        let count = elements("cache", self.cfg.num_hidden_layers, per_layer)?;
        Ok(MetalErnie45KvCache {
            len: 0,
            capacity,
            layers: self.cfg.num_hidden_layers,
            kv_dim,
            storage: CacheStorage {
                k: allocate(&self._device, count, "ernie45.cache_k")?,
                v: allocate(&self._device, count, "ernie45.cache_v")?,
                owner: Arc::clone(&self.cache_owner),
            },
        })
    }

    fn check_cache(&self, cache: &MetalErnie45KvCache) -> Result<(), InferenceError> {
        if cache.layers != self.cfg.num_hidden_layers
            || cache.kv_dim != self.cfg.num_key_value_heads * self.cfg.head_dim
            || cache.capacity == 0
            || cache.capacity > self.max_seq_len
            || cache.len > cache.capacity
            || !Arc::ptr_eq(&cache.storage.owner, &self.cache_owner)
        {
            return Err(invalid("cache shape or owning decoder does not match"));
        }
        Ok(())
    }

    /// Fill an empty cache from `[positions.len(), hidden_size]` embeddings.
    ///
    /// Positions are token-major `[T,H,W]` triples, independent of cache length.
    /// Causality follows token order. `logits` must hold exactly `vocab_size`
    /// values and receives only the final token's output. K/V buffers store f32
    /// rows without conversion, and previously allocated scratch is reused.
    ///
    /// # Errors
    /// Returns `InvalidInput` for empty or excessive input, a nonempty or foreign
    /// cache, shape mismatch or non-finite embeddings/angles. Execution failures
    /// return an inference error. Errors preserve output and the live cache
    /// length; incomplete rows outside that length must not be observed.
    pub fn kv_prefill(
        &mut self,
        embeds: &[f32],
        positions: &[[u32; 3]],
        cache: &mut MetalErnie45KvCache,
        logits: &mut [f32],
    ) -> Result<(), InferenceError> {
        self.reset_counts();
        self.check_cache(cache)?;
        if !cache.is_empty() {
            return Err(invalid("kv prefill requires an empty cache"));
        }
        if positions.is_empty() || positions.len() > cache.capacity {
            return Err(invalid("kv prefill sequence is outside cache capacity"));
        }
        self.embedded_forward(embeds, positions, Some(cache), false, logits)?;
        cache.len = positions.len();
        Ok(())
    }

    /// Append one embedding row using its explicit `[T,H,W]` position.
    ///
    /// Appends post-RoPE K and unrotated V at the old cache length, then attends
    /// over all live rows including the new row. `logits` is exactly one
    /// vocabulary row. No host readback of cache contents occurs during decode.
    ///
    /// # Errors
    /// Returns `InvalidInput` for an empty, full or foreign cache, shape mismatch
    /// or non-finite inputs/angles, and an inference error for failed execution.
    /// Errors leave logits, the live cache prefix and its length unchanged;
    /// the unpublished append row may have been written and is overwritten on retry.
    pub fn kv_decode_step(
        &mut self,
        embeds: &[f32],
        position: [u32; 3],
        cache: &mut MetalErnie45KvCache,
        logits: &mut [f32],
    ) -> Result<(), InferenceError> {
        self.reset_counts();
        self.check_cache(cache)?;
        if cache.is_empty() {
            return Err(invalid("kv decode cache is empty; run kv_prefill first"));
        }
        if cache.len == cache.capacity {
            return Err(invalid("kv decode cache is full"));
        }
        self.embedded_forward(embeds, &[position], Some(cache), true, logits)?;
        cache.len += 1;
        Ok(())
    }

    fn reset_counts(&mut self) {
        #[cfg(all(test, feature = "f16"))]
        {
            self.last_counts = [0; 7];
            self.pending_counts.set([0; 7]);
        }
    }

    fn write_positions(&mut self, positions: &[[u32; 3]]) -> Result<(), InferenceError> {
        let half = self.cfg.head_dim / 2;
        let count = positions.len() * half;
        // SAFETY: callers validate the row count against max_seq_len. These two
        // separate, private shared allocations each hold max_seq_len * half f32s.
        // &mut self excludes submission, and every previous command was waited.
        let (cos, sin) = unsafe {
            (
                std::slice::from_raw_parts_mut(self.position_cos.contents().cast::<f32>(), count),
                std::slice::from_raw_parts_mut(self.position_sin.contents().cast::<f32>(), count),
            )
        };
        for (token, position) in positions.iter().enumerate() {
            let mut lane = 0;
            for (axis, &width) in self.cfg.rope_scaling.mrope_section.iter().enumerate() {
                for _ in 0..width {
                    let angle = position[axis] as f32 * self.inv_freq[lane];
                    let (sn, cs) = angle.sin_cos();
                    if !sn.is_finite() || !cs.is_finite() {
                        return Err(invalid("position produces non-finite RoPE tables"));
                    }
                    cos[token * half + lane] = cs;
                    sin[token * half + lane] = sn;
                    lane += 1;
                }
            }
        }
        Ok(())
    }

    fn embedded_forward(
        &mut self,
        embeds: &[f32],
        positions: &[[u32; 3]],
        cache: Option<&MetalErnie45KvCache>,
        decode: bool,
        logits: &mut [f32],
    ) -> Result<(), InferenceError> {
        let rows = positions.len();
        if rows == 0 || rows > self.max_seq_len {
            return Err(invalid("embedded sequence is outside state capacity"));
        }
        let count = elements("embeddings", rows, self.cfg.hidden_size)?;
        if embeds.len() != count {
            return Err(invalid(
                "embeds must have the exact [seq_len,hidden_size] shape",
            ));
        }
        if logits.len() != self.cfg.vocab_size {
            return Err(invalid("cached logits must have exactly vocab_size values"));
        }
        if embeds.iter().any(|value| !value.is_finite()) {
            return Err(invalid("embeds contain a non-finite value"));
        }
        self.write_positions(positions)?;
        #[cfg(all(test, feature = "f16"))]
        self.poison_destinations();
        // SAFETY: count is bounded by the private shared activation capacity.
        // Prior commands completed, &mut self excludes concurrent access, and
        // the caller cannot alias this private allocation through safe APIs.
        unsafe {
            std::ptr::copy_nonoverlapping(
                embeds.as_ptr(),
                self.activations.hidden.contents().cast::<f32>(),
                count,
            );
        }
        objc::rc::autoreleasepool(|| {
            // SAFETY: these are the documented nullable, autoreleased Metal
            // selectors. Nil is checked before borrowing; both references stay
            // inside this pool until completion, including the output readback.
            let command = unsafe {
                let raw: *mut metal::MTLCommandBuffer =
                    objc::Message::send_message(&*self.queue, self.command_buffer_selector, ())
                        .map_err(|error| runtime(format!("command creation failed: {error}")))?;
                if raw.is_null() {
                    return Err(runtime("command buffer allocation failed"));
                }
                <metal::CommandBufferRef as metal::foreign_types::ForeignTypeRef>::from_ptr(raw)
            };
            // SAFETY: the command is live, the nullable result is checked, and
            // the encoder borrow cannot escape the surrounding autorelease pool.
            let encoder = unsafe {
                let raw: *mut metal::MTLComputeCommandEncoder =
                    objc::Message::send_message(command, self.compute_encoder_selector, ())
                        .map_err(|error| runtime(format!("encoder creation failed: {error}")))?;
                if raw.is_null() {
                    return Err(runtime("compute encoder allocation failed"));
                }
                <ComputeCommandEncoderRef as metal::foreign_types::ForeignTypeRef>::from_ptr(raw)
            };
            let s = rows as u32;
            let h = self.cfg.hidden_size as u32;
            let q = (self.cfg.num_attention_heads * self.cfg.head_dim) as u32;
            let kv = (self.cfg.num_key_value_heads * self.cfg.head_dim) as u32;
            let a = &self.activations;
            for (layer, weights) in self.weights.layers.iter().enumerate() {
                self.project_qkv(encoder, weights, s);
                self.rope_tables(
                    encoder,
                    &a.q,
                    s,
                    self.cfg.num_attention_heads as u32,
                    &self.position_cos,
                    &self.position_sin,
                );
                self.rope_tables(
                    encoder,
                    &a.k,
                    s,
                    self.cfg.num_key_value_heads as u32,
                    &self.position_cos,
                    &self.position_sin,
                );
                #[cfg(all(test, feature = "f16"))]
                if let Some(trace) = &self.trace_buffers {
                    let (k, v) = &trace.layers[layer];
                    self.copy(encoder, &a.k, k, s * kv);
                    self.copy(encoder, &a.v, v, s * kv);
                }
                if let Some(cache) = cache {
                    let offset = (layer * cache.capacity + if decode { cache.len } else { 0 })
                        * cache.kv_dim;
                    self.copy_into_cache(encoder, &a.k, &cache.storage.k, offset, s * kv);
                    self.copy_into_cache(encoder, &a.v, &cache.storage.v, offset, s * kv);
                    if decode {
                        self.cached_attention(encoder, cache, layer);
                    } else {
                        self.attention(encoder, s, q, kv);
                    }
                } else {
                    self.attention(encoder, s, q, kv);
                }
                self.finish_layer(encoder, weights, s);
            }
            self.copy(encoder, &a.hidden, &a.normed, s * h);
            self.norm(encoder, &a.normed, &self.weights.final_norm, s, h);
            self.last_logits(encoder, rows);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(runtime(format!(
                    "command buffer ended with {:?}",
                    command.status()
                )));
            }
            // SAFETY: the completed command initialized this private, aligned
            // vocab-sized prefix. &mut self prevents another submission or write.
            let output = unsafe {
                std::slice::from_raw_parts(a.logits.contents().cast::<f32>(), self.cfg.vocab_size)
            };
            if output.iter().any(|value| !value.is_finite()) {
                return Err(runtime("cached forward produced non-finite logits"));
            }
            logits.copy_from_slice(output);
            #[cfg(all(test, feature = "f16"))]
            {
                self.last_counts = self.pending_counts.get();
            }
            Ok(())
        })
    }

    fn last_logits(&self, enc: &ComputeCommandEncoderRef, rows: usize) {
        let h = self.cfg.hidden_size as u32;
        let vocab = self.cfg.vocab_size as u32;
        enc.set_compute_pipeline_state(&self.pipelines.matmul);
        enc.set_buffer(
            0,
            Some(&self.activations.normed),
            ((rows - 1) * self.cfg.hidden_size * size_of::<f32>()) as u64,
        );
        bind(enc, 1, &self.weights.lm_head);
        bind(enc, 2, &self.activations.logits);
        scalar(enc, 3, &1u32);
        scalar(enc, 4, &vocab);
        scalar(enc, 5, &h);
        enc.dispatch_thread_groups(
            MTLSize::new(u64::from(vocab).div_ceil(16), 1, 1),
            MTLSize::new(16, 16, 1),
        );
        #[cfg(all(test, feature = "f16"))]
        self.record(0);
    }

    fn copy_into_cache(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        offset: usize,
        count: u32,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.copy);
        bind(enc, 0, src);
        enc.set_buffer(1, Some(dst), (offset * size_of::<f32>()) as u64);
        scalar(enc, 2, &count);
        element_dispatch(enc, count);
    }

    fn cached_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        cache: &MetalErnie45KvCache,
        layer: usize,
    ) {
        let layer_offset = (layer * cache.capacity * cache.kv_dim * size_of::<f32>()) as u64;
        enc.set_compute_pipeline_state(&self.pipelines.decode_attention);
        bind(enc, 0, &self.activations.q);
        enc.set_buffer(1, Some(&cache.storage.k), layer_offset);
        enc.set_buffer(2, Some(&cache.storage.v), layer_offset);
        bind(enc, 3, &self.activations.attention);
        scalar(enc, 4, &((cache.len + 1) as u32));
        scalar(enc, 5, &((cache.kv_dim / 4) as u32));
        scalar(enc, 6, &(1.0 / (self.cfg.head_dim as f32).sqrt()));
        let groups = self.cfg.num_attention_heads / self.cfg.num_key_value_heads;
        enc.dispatch_thread_groups(
            MTLSize::new(self.cfg.num_key_value_heads as u64, 1, 1),
            MTLSize::new((groups * 32) as u64, 1, 1),
        );
        #[cfg(all(test, feature = "f16"))]
        self.record(3);
    }

    #[cfg(all(test, feature = "f16"))]
    pub(in super::super) fn prefill_embeds_for_test(
        &mut self,
        embeds: &[f32],
        positions: &[[u32; 3]],
        logits: &mut [f32],
    ) -> Result<(), InferenceError> {
        self.reset_counts();
        self.embedded_forward(embeds, positions, None, false, logits)
    }

    #[cfg(all(test, feature = "f16"))]
    pub(in super::super) fn prefill_embeds_trace_for_test(
        &mut self,
        embeds: &[f32],
        positions: &[[u32; 3]],
        logits: &mut [f32],
    ) -> Result<Vec<TestKvRows>, InferenceError> {
        self.reset_counts();
        let rows = positions.len();
        if rows == 0 || rows > self.max_seq_len {
            return Err(invalid("trace sequence is outside state capacity"));
        }
        let kv_dim = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let count = elements("trace layer rows", rows, kv_dim)?;
        let unwritten = vec![f32::NAN; count];
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.cfg.num_hidden_layers)
            .map_err(|error| runtime(format!("trace allocation failed: {error}")))?;
        for _ in 0..self.cfg.num_hidden_layers {
            layers.push((
                upload(&self._device, &unwritten, "ernie45.trace_k")?,
                upload(&self._device, &unwritten, "ernie45.trace_v")?,
            ));
        }
        self.trace_buffers = Some(TestTraceBuffers { layers });
        let result = self.embedded_forward(embeds, positions, None, false, logits);
        let trace = self
            .trace_buffers
            .take()
            .ok_or_else(|| runtime("trace buffers were not retained"))?;
        result?;
        let mut snapshots = Vec::new();
        for (k, v) in &trace.layers {
            // SAFETY: each separate trace allocation holds count aligned f32s.
            // The uncached command copied scratch before each layer reused it,
            // completed before returning, and this borrow excludes resubmission.
            let (keys, values) = unsafe {
                (
                    std::slice::from_raw_parts(k.contents().cast::<f32>(), count),
                    std::slice::from_raw_parts(v.contents().cast::<f32>(), count),
                )
            };
            if keys.iter().chain(values).any(|value| !value.is_finite()) {
                return Err(runtime("trace contains unwritten or non-finite layer rows"));
            }
            snapshots.push((keys.to_vec(), values.to_vec()));
        }
        Ok(snapshots)
    }

    #[cfg(all(test, feature = "f16"))]
    pub(in super::super) fn last_layer_rows_for_test(&self, rows: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(rows > 0 && rows <= self.max_seq_len);
        let count = rows * self.cfg.num_key_value_heads * self.cfg.head_dim;
        // SAFETY: the private buffers hold at least count aligned f32s. Test
        // callers use completed forwards; this shared borrow prevents resubmission.
        unsafe {
            (
                std::slice::from_raw_parts(self.activations.k.contents().cast::<f32>(), count)
                    .to_vec(),
                std::slice::from_raw_parts(self.activations.v.contents().cast::<f32>(), count)
                    .to_vec(),
            )
        }
    }
}

#[cfg(all(test, feature = "f16"))]
impl MetalErnie45KvCache {
    pub(in super::super) fn replace_value_row_for_test(
        &mut self,
        layer: usize,
        token: usize,
        values: &[f32],
    ) -> Result<(), InferenceError> {
        if layer >= self.layers || token >= self.len {
            return Err(invalid("test value row must be a live cache row"));
        }
        if values.len() != self.kv_dim || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "test value row must contain exactly kv_dim finite values",
            ));
        }
        let offset = (layer * self.capacity + token) * self.kv_dim;
        // SAFETY: construction checked the full layer/capacity/width product,
        // and the live-row checks bound this one row. All cache-writing calls
        // wait for completion, &mut self excludes concurrent access, and safe
        // callers cannot alias this private allocation through values.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.storage.v.contents().cast::<f32>().add(offset),
                self.kv_dim,
            );
        }
        Ok(())
    }

    pub(in super::super) fn layer_rows_for_test(&self, layer: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(layer < self.layers);
        let count = self.len * self.kv_dim;
        let offset = layer * self.capacity * self.kv_dim;
        // SAFETY: only initialized live rows are read; the private allocations
        // have layers * capacity * kv_dim aligned f32s. A shared cache borrow
        // prevents the mutable borrow needed by either cache-writing entry.
        unsafe {
            (
                std::slice::from_raw_parts(
                    self.storage.k.contents().cast::<f32>().add(offset),
                    count,
                )
                .to_vec(),
                std::slice::from_raw_parts(
                    self.storage.v.contents().cast::<f32>().add(offset),
                    count,
                )
                .to_vec(),
            )
        }
    }
}
