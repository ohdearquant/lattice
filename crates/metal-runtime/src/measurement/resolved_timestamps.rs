//! Internal owned storage for resolved Metal timestamp samples.

use lattice_inference_core::InferenceError;
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, CommandQueueRef, CounterSampleBuffer,
    CounterSampleBufferDescriptor, CounterSampleBufferRef, Device, MTLCommandBufferStatus,
    MTLCounterSamplingPoint, MTLStorageMode, NSRange,
};

use crate::{BufferPolicy, UploadCopy, ZeroLengthPolicy};

fn invalid(message: &str) -> InferenceError {
    InferenceError::Inference(format!("Metal timestamp readback: {message}"))
}

fn buffer_error(message: String) -> InferenceError {
    InferenceError::Inference(format!("Metal timestamp readback: {message}"))
}

fn checked_range(
    base_addr: usize,
    buffer_bytes: u64,
    resolved_count: usize,
    first: usize,
    count: usize,
) -> Result<(usize, usize), InferenceError> {
    const WIDTH: usize = std::mem::size_of::<u64>();
    if base_addr == 0 || count == 0 {
        return Err(invalid("null buffer or empty range"));
    }
    let last = first
        .checked_add(count)
        .ok_or_else(|| invalid("sample range overflow"))?;
    if last > resolved_count {
        return Err(invalid("range was not resolved"));
    }
    let offset = first
        .checked_mul(WIDTH)
        .ok_or_else(|| invalid("offset overflow"))?;
    let bytes = count
        .checked_mul(WIDTH)
        .ok_or_else(|| invalid("length overflow"))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| invalid("range overflow"))?;
    if end > isize::MAX as usize
        || u64::try_from(end).map_err(|_| invalid("range conversion"))? > buffer_bytes
    {
        return Err(invalid("range exceeds buffer"));
    }
    let start_addr = base_addr
        .checked_add(offset)
        .ok_or_else(|| invalid("address overflow"))?;
    base_addr
        .checked_add(end)
        .ok_or_else(|| invalid("address overflow"))?;
    if start_addr % std::mem::align_of::<u64>() != 0 {
        return Err(invalid("misaligned timestamp address"));
    }
    Ok((offset, count))
}

fn check_reuse(completed: bool) -> Result<(), InferenceError> {
    if !completed {
        return Err(invalid("previous producer has not completed successfully"));
    }
    Ok(())
}

fn mark_resolve_attempt(attempted: &mut bool) -> bool {
    !std::mem::replace(attempted, true)
}

fn resolve_state(
    same_producer: bool,
    producer_status: MTLCommandBufferStatus,
    first_attempt: bool,
) -> Result<(), InferenceError> {
    if !same_producer || producer_status != MTLCommandBufferStatus::Completed || !first_attempt {
        return Err(invalid(
            "producer mismatch, incomplete producer, or resolve already attempted",
        ));
    }
    Ok(())
}

fn read_state(
    same_producer: bool,
    producer_status: MTLCommandBufferStatus,
    same_resolver: bool,
    resolver_status: MTLCommandBufferStatus,
    distinct_commands: bool,
    resolved_count: Option<usize>,
) -> Result<usize, InferenceError> {
    if !same_producer
        || producer_status != MTLCommandBufferStatus::Completed
        || !same_resolver
        || resolver_status != MTLCommandBufferStatus::Completed
        || !distinct_commands
    {
        return Err(invalid(
            "producer or resolver mismatch or incomplete command buffer",
        ));
    }
    resolved_count.ok_or_else(|| invalid("no timestamp range resolved"))
}

fn timestamp_samples(
    device: &Device,
    capacity: usize,
) -> Result<CounterSampleBuffer, InferenceError> {
    if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return Err(invalid("stage-boundary counter sampling is unavailable"));
    }
    let counter_sets = device.counter_sets();
    let mut timestamps = counter_sets.iter().filter(|set| set.name() == "timestamp");
    let timestamp = timestamps
        .next()
        .ok_or_else(|| invalid("timestamp counter set is unavailable"))?;
    if timestamps.next().is_some() {
        return Err(invalid("timestamp counter set is ambiguous"));
    }
    let sample_count =
        u64::try_from(capacity).map_err(|_| invalid("sample capacity conversion"))?;
    let descriptor = CounterSampleBufferDescriptor::new();
    descriptor.set_counter_set(timestamp);
    descriptor.set_sample_count(sample_count);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    device
        .new_counter_sample_buffer_with_descriptor(&descriptor)
        .map_err(|error| buffer_error(format!("counter allocation failed: {error}")))
}

/// Private destination and its normally retaining command buffer.
///
/// The destination never leaves this type. In production, only the owned resolver writes it.
#[doc(hidden)]
pub struct ResolvedTimestamps {
    destination: Buffer,
    samples: CounterSampleBuffer,
    producer: CommandBuffer,
    queue: CommandQueue,
    resolver: Option<CommandBuffer>,
    resolve_attempted: bool,
    capacity: usize,
    resolved_count: Option<usize>,
}

impl ResolvedTimestamps {
    /// Allocate initialized shared storage and one normally retaining command buffer.
    pub fn new(queue: &CommandQueueRef, capacity: usize) -> Result<Self, InferenceError> {
        if capacity == 0 {
            return Err(invalid("empty sample capacity"));
        }
        let words = capacity
            .checked_mul(2)
            .ok_or_else(|| invalid("sample capacity overflow"))?;
        let mut zeroes = Vec::new();
        zeroes
            .try_reserve_exact(words)
            .map_err(|_| invalid("sample initialization allocation failed"))?;
        zeroes.resize(words, 0.0_f32);
        let device = queue.device().to_owned();
        let destination = crate::upload(
            &device,
            &zeroes,
            "resolved timestamp samples",
            BufferPolicy {
                zero_length: ZeroLengthPolicy::Exact,
                invalid: buffer_error,
                runtime: buffer_error,
            },
            UploadCopy::Direct,
        )?;
        let samples = timestamp_samples(&device, capacity)?;
        Ok(Self {
            destination,
            samples,
            producer: queue.new_command_buffer().to_owned(),
            queue: queue.to_owned(),
            resolver: None,
            resolve_attempted: false,
            capacity,
            resolved_count: None,
        })
    }

    /// Retain the same normally retaining command buffer for ordinary encoding.
    pub fn command_buffer(&self) -> CommandBuffer {
        self.producer.clone()
    }

    /// Borrow the internally created standard timestamp sample buffer for attachments.
    pub fn sample_buffer(&self) -> &CounterSampleBufferRef {
        self.samples.as_ref()
    }

    /// Resolve samples after the exact producer has completed, using a separate command buffer.
    pub fn resolve_after_completion(
        &mut self,
        producer: &CommandBufferRef,
        first: usize,
        count: usize,
    ) -> Result<CommandBuffer, InferenceError> {
        let first_attempt = mark_resolve_attempt(&mut self.resolve_attempted);
        resolve_state(
            std::ptr::eq(producer, self.producer.as_ref()),
            producer.status(),
            first_attempt && self.resolver.is_none(),
        )?;
        if count == 0 || count > self.capacity {
            return Err(invalid("invalid sample count"));
        }
        let end = first
            .checked_add(count)
            .ok_or_else(|| invalid("sample range overflow"))?;
        let end_u64 = u64::try_from(end).map_err(|_| invalid("sample range conversion"))?;
        if end_u64 > self.samples.sample_count() {
            return Err(invalid("sample range exceeds counter buffer"));
        }
        let location = first
            .try_into()
            .map_err(|_| invalid("sample offset conversion"))?;
        let length = count
            .try_into()
            .map_err(|_| invalid("sample count conversion"))?;
        let resolver = self.queue.new_command_buffer().to_owned();
        if std::ptr::eq(producer, resolver.as_ref()) {
            return Err(invalid("resolver must differ from producer"));
        }
        let encoder = resolver.new_blit_command_encoder();
        encoder.resolve_counters(
            self.samples.as_ref(),
            NSRange::new(location, length),
            &self.destination,
            0,
        );
        encoder.end_encoding();
        self.resolver = Some(resolver.clone());
        self.resolved_count = Some(count);
        resolver.commit();
        resolver.wait_until_completed();
        if resolver.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid("resolver did not complete successfully"));
        }
        Ok(resolver)
    }

    /// Reuse storage only after its previous command buffer completed successfully.
    pub fn reset(&mut self, queue: &CommandQueueRef) -> Result<(), InferenceError> {
        check_reuse(
            self.producer.status() == MTLCommandBufferStatus::Completed
                && self
                    .resolver
                    .as_ref()
                    .is_none_or(|resolver| resolver.status() == MTLCommandBufferStatus::Completed),
        )?;
        if !std::ptr::eq(queue.device(), self.destination.device()) {
            return Err(invalid("queue and destination devices differ"));
        }
        let device = queue.device().to_owned();
        let samples = timestamp_samples(&device, self.capacity)?;
        self.producer = queue.new_command_buffer().to_owned();
        self.queue = queue.to_owned();
        self.samples = samples;
        self.resolver = None;
        self.resolve_attempted = false;
        self.resolved_count = None;
        Ok(())
    }
}

/// Copy a resolved range after its exact producer and resolver have both completed.
#[doc(hidden)]
pub fn read_resolved_timestamps(
    producer: &CommandBufferRef,
    resolver: &CommandBufferRef,
    result: &mut ResolvedTimestamps,
    first: usize,
    count: usize,
) -> Result<Vec<u64>, InferenceError> {
    let resolved_count = read_state(
        std::ptr::eq(producer, result.producer.as_ref()),
        producer.status(),
        result
            .resolver
            .as_ref()
            .is_some_and(|stored| std::ptr::eq(resolver, stored.as_ref())),
        resolver.status(),
        !std::ptr::eq(producer, resolver),
        result.resolved_count,
    )?;
    if result.destination.storage_mode() != MTLStorageMode::Shared {
        return Err(invalid("destination is not shared storage"));
    }
    let pointer = result.destination.contents().cast::<u8>();
    let (offset, count) = checked_range(
        pointer as usize,
        result.destination.length(),
        resolved_count,
        first,
        count,
    )?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("timestamp output allocation failed"))?;
    values.resize(count, 0_u64);
    let source = pointer.wrapping_add(offset).cast::<u64>();
    // SAFETY: the exact producer and resolver completed successfully, and this type
    // alone owns the shared destination and schedules writes; &mut prevents reuse during the
    // copy. The resolved range, allocation length, address arithmetic and u64
    // alignment were checked above. Both buffers remain live and do not overlap.
    unsafe { std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), count) };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{check_reuse, checked_range, mark_resolve_attempt, read_state, resolve_state};

    /// Temporary diagnostic for #2646 F2; remove or convert after the follow-up ruling.
    #[cfg(all(test, target_os = "macos", feature = "metal-gpu", feature = "signpost"))]
    #[test]
    #[ignore = "requires a Metal device with stage-boundary timestamp sampling"]
    fn post_completion_resolve_control_keeps_sampler_and_producer_alive() {
        let _gpu_guard = crate::measurement::gpu_test_lock();

        fn run_probe() -> Result<Vec<u64>, String> {
            const GROUPS: usize = 49;
            const SAMPLES: usize = GROUPS * 2;
            let device = metal::Device::system_default().ok_or("Metal device required")?;
            let queue = device.new_command_queue();
            let library = device.new_library_with_source(
                "#include <metal_stdlib>\nusing namespace metal; kernel void touch(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { uint v = out[i]; for (uint j = 0; j < 256; j++) { v = v * 1664525u + 1013904223u; } out[i] = v; }",
                &metal::CompileOptions::new(),
            ).map_err(|error| format!("compile control kernel: {error}"))?;
            let function = library
                .get_function("touch", None)
                .map_err(|error| format!("control kernel: {error}"))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| format!("control pipeline: {error}"))?;
            let initial = [1_u32; 256];
            let output = device.new_buffer_with_data(
                initial.as_ptr().cast(),
                std::mem::size_of_val(&initial) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let mut producer = super::ResolvedTimestamps::new(&queue, SAMPLES)
                .map_err(|error| error.to_string())?;
            let producer_command = producer.command_buffer();
            if !std::ptr::eq(producer_command.as_ref(), producer.producer.as_ref())
                || producer.samples.sample_count() != SAMPLES as u64
            {
                return Err("producer identity or sample count mismatch".into());
            }
            for group in 0..GROUPS {
                let pass = metal::ComputePassDescriptor::new();
                pass.set_dispatch_type(metal::MTLDispatchType::Serial);
                let attachment = metal::ComputePassSampleBufferAttachmentDescriptor::new();
                attachment.set_sample_buffer(producer.samples.as_ref());
                attachment.set_start_of_encoder_sample_index((group * 2) as u64);
                attachment.set_end_of_encoder_sample_index((group * 2 + 1) as u64);
                pass.sample_buffer_attachments()
                    .set_object_at(0, Some(&attachment));
                let installed = pass
                    .sample_buffer_attachments()
                    .object_at(0)
                    .ok_or("missing attachment")?;
                if !installed
                    .sample_buffer()
                    .is_some_and(|samples| std::ptr::eq(samples, producer.samples.as_ref()))
                    || installed.start_of_encoder_sample_index() != (group * 2) as u64
                    || installed.end_of_encoder_sample_index() != (group * 2 + 1) as u64
                {
                    return Err(format!("attachment mismatch for group {group}"));
                }
                let encoder = producer_command.compute_command_encoder_with_descriptor(pass);
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&output), 0);
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new(4, 1, 1),
                    metal::MTLSize::new(64, 1, 1),
                );
                encoder.end_encoding();
            }
            producer_command.commit();
            producer_command.wait_until_completed();
            if producer_command.status() != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("producer status {:?}", producer_command.status()));
            }

            // This private test keeps the completed producer and its sampler alive;
            // the second owning result only resolves that same, now-idle sampler.
            let mut resolver = super::ResolvedTimestamps::new(&queue, SAMPLES)
                .map_err(|error| error.to_string())?;
            resolver.samples = producer.samples.clone();
            let resolver_command = resolver.command_buffer();
            if std::ptr::eq(producer_command.as_ref(), resolver_command.as_ref())
                || !std::ptr::eq(resolver_command.as_ref(), resolver.producer.as_ref())
                || !std::ptr::eq(producer.samples.as_ref(), resolver.samples.as_ref())
            {
                return Err("resolver or sampler identity mismatch".into());
            }
            let encoder = resolver_command.new_blit_command_encoder();
            encoder.resolve_counters(
                resolver.samples.as_ref(),
                metal::NSRange::new(0, SAMPLES as u64),
                &producer.destination,
                0,
            );
            encoder.end_encoding();
            producer.resolver = Some(resolver_command.clone());
            producer.resolve_attempted = true;
            producer.resolved_count = Some(SAMPLES);
            resolver_command.commit();
            resolver_command.wait_until_completed();
            if resolver_command.status() != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("resolver status {:?}", resolver_command.status()));
            }
            let raw = super::read_resolved_timestamps(
                producer_command.as_ref(),
                resolver_command.as_ref(),
                &mut producer,
                0,
                SAMPLES,
            )
            .map_err(|error| error.to_string())?;
            // Both owning results and command buffers remain live until after the copy.
            drop((producer, producer_command, resolver, resolver_command));
            Ok(raw)
        }

        let outcome = run_probe();
        match &outcome {
            Ok(raw) => eprintln!("[GPU_TIMESTAMP_POST_COMPLETION_PROBE] groups=49 raw={raw:?}"),
            Err(error) => {
                eprintln!("[GPU_TIMESTAMP_POST_COMPLETION_PROBE] groups=49 error={error}")
            }
        }
        let raw = outcome.expect("post-completion probe failed; see recorded error");
        assert_eq!(raw.len(), 98, "raw vector printed above");
        let mut previous_end = 0;
        for (group, pair) in raw.chunks_exact(2).enumerate() {
            assert!(
                pair[0] != 0
                    && pair[1] != u64::MAX
                    && pair[0] < pair[1]
                    && (group == 0 || pair[0] >= previous_end),
                "invalid samples for group {group}: {pair:?}; full raw vector printed above"
            );
            previous_end = pair[1];
        }
    }

    #[test]
    fn accepts_exact_resolved_range() {
        assert_eq!(checked_range(0x1000, 24, 3, 1, 2).unwrap(), (8, 2));
    }

    #[test]
    fn rejects_empty_unresolved_and_misaligned_ranges() {
        assert!(checked_range(0x1000, 24, 3, 1, 0).is_err());
        assert!(checked_range(0x1000, 24, 2, 1, 2).is_err());
        assert!(checked_range(0x1000, 23, 3, 1, 2).is_err());
        assert!(checked_range(0, 24, 3, 0, 1).is_err());
        assert!(checked_range(0x1001, 24, 3, 0, 1).is_err());
    }

    #[test]
    fn rejects_integer_address_and_pointer_range_overflow() {
        assert!(checked_range(0x1000, u64::MAX, usize::MAX, usize::MAX, 1).is_err());
        assert!(checked_range(0x1000, u64::MAX, usize::MAX, 0, usize::MAX).is_err());
        assert!(checked_range(0x1000, u64::MAX, usize::MAX, usize::MAX / 8 + 1, 1).is_err());
        assert!(checked_range(0x1000, u64::MAX, usize::MAX, 0, usize::MAX / 8 + 1).is_err());
        assert!(checked_range(usize::MAX - 7, 16, 2, 0, 2).is_err());
        assert!(checked_range(0x1000, u64::MAX, usize::MAX, (isize::MAX as usize) / 8, 2).is_err());
    }

    #[test]
    fn rejects_reuse_until_previous_producer_completed() {
        assert!(check_reuse(false).is_err());
        assert!(check_reuse(true).is_ok());
    }

    #[test]
    fn rejects_wrong_incomplete_failed_and_duplicate_resolve_states() {
        use metal::MTLCommandBufferStatus::{Completed, Error, Scheduled};
        assert!(resolve_state(false, Completed, true).is_err());
        assert!(resolve_state(true, Scheduled, true).is_err());
        assert!(resolve_state(true, Error, true).is_err());
        assert!(resolve_state(true, Completed, false).is_err());
        assert!(resolve_state(true, Completed, true).is_ok());
    }

    #[test]
    fn failed_resolve_attempt_cannot_be_retried() {
        use metal::MTLCommandBufferStatus::Completed;
        let mut attempted = false;
        assert!(resolve_state(false, Completed, mark_resolve_attempt(&mut attempted)).is_err());
        assert!(attempted);
        assert!(resolve_state(true, Completed, mark_resolve_attempt(&mut attempted)).is_err());
    }

    #[test]
    fn rejects_wrong_incomplete_failed_or_unresolved_read_states() {
        use metal::MTLCommandBufferStatus::{Completed, Error, Scheduled};
        assert!(read_state(false, Completed, true, Completed, true, Some(1)).is_err());
        assert!(read_state(true, Scheduled, true, Completed, true, Some(1)).is_err());
        assert!(read_state(true, Error, true, Completed, true, Some(1)).is_err());
        assert!(read_state(true, Completed, false, Completed, true, Some(1)).is_err());
        assert!(read_state(true, Completed, true, Scheduled, true, Some(1)).is_err());
        assert!(read_state(true, Completed, true, Error, true, Some(1)).is_err());
        assert!(read_state(true, Completed, true, Completed, false, Some(1)).is_err());
        assert!(read_state(true, Completed, true, Completed, true, None).is_err());
        assert_eq!(
            read_state(true, Completed, true, Completed, true, Some(3)).unwrap(),
            3
        );
    }
}
