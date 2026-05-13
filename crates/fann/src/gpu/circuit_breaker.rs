//! Circuit breaker for memory pressure handling
//!
//! Prevents cascade failures by temporarily rejecting requests
//! when too many allocation failures occur.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// < 60% usage - normal operation
    None,
    /// 60-70% usage - start monitoring
    Low,
    /// 70-80% usage - reduce allocations
    Medium,
    /// 80-90% usage - aggressive cleanup
    High,
    /// > 90% usage - critical, reject new allocations
    Critical,
}

impl MemoryPressure {
    /// Calculate pressure from usage ratio (0.0 - 1.0)
    pub fn from_ratio(ratio: f32) -> Self {
        if ratio < 0.6 {
            MemoryPressure::None
        } else if ratio < 0.7 {
            MemoryPressure::Low
        } else if ratio < 0.8 {
            MemoryPressure::Medium
        } else if ratio < 0.9 {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        }
    }

    /// How aggressively to clean up (0.0 - 1.0)
    pub fn cleanup_aggressiveness(&self) -> f32 {
        match self {
            MemoryPressure::None => 0.1,
            MemoryPressure::Low => 0.3,
            MemoryPressure::Medium => 0.5,
            MemoryPressure::High => 0.7,
            MemoryPressure::Critical => 1.0,
        }
    }

    /// Whether new allocations should be blocked
    pub fn should_block_allocations(&self) -> bool {
        matches!(self, MemoryPressure::Critical)
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Normal operation - all requests allowed
    Closed,
    /// Failures exceeded threshold - rejecting requests
    Open,
    /// Testing if service recovered - allowing limited requests
    HalfOpen,
}

/// Circuit breaker for GPU memory operations
///
/// State machine:
/// ```text
/// Closed ──(failures >= threshold)──> Open
///    ^                                  │
///    │                                  │ (after timeout)
///    │                                  v
///    └────────(success)───────────── HalfOpen
///                                      │
///                                      │ (failure)
///                                      v
///                                    Open
/// ```
pub struct CircuitBreaker {
    state: Mutex<CircuitBreakerState>,
    failure_threshold: usize,
    recovery_timeout: Duration,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure: Mutex<Option<Instant>>,
    last_state_change: Mutex<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    ///
    /// # Arguments
    /// * `failure_threshold` - Number of failures before opening
    /// * `recovery_timeout` - Time to wait before testing recovery
    pub fn new(failure_threshold: usize, recovery_timeout: Duration) -> Self {
        Self {
            state: Mutex::new(CircuitBreakerState::Closed),
            failure_threshold,
            recovery_timeout,
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure: Mutex::new(None),
            last_state_change: Mutex::new(Instant::now()),
        }
    }

    /// Check if request should be allowed
    ///
    /// Returns false (fail-safe) if locks are poisoned.
    pub fn allow_request(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            // Lock poisoned - fail safe by denying request
            return false;
        };

        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if recovery timeout has passed
                let Ok(last_change) = self.last_state_change.lock() else {
                    return false;
                };
                if last_change.elapsed() >= self.recovery_timeout {
                    *state = CircuitBreakerState::HalfOpen;
                    drop(last_change);
                    if let Ok(mut lsc) = self.last_state_change.lock() {
                        *lsc = Instant::now();
                    }
                    true // Allow one request to test
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Only allow limited requests in half-open state
                true
            }
        }
    }

    /// Record a successful operation
    ///
    /// Silently ignores poisoned locks (stats best-effort).
    pub fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if *state == CircuitBreakerState::HalfOpen {
            // Recovery successful - close the circuit
            *state = CircuitBreakerState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
            if let Ok(mut lsc) = self.last_state_change.lock() {
                *lsc = Instant::now();
            }
        }
    }

    /// Record a failed operation
    ///
    /// Silently ignores poisoned locks (stats best-effort).
    pub fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut last_failure) = self.last_failure.lock() {
            *last_failure = Some(Instant::now());
        }

        let Ok(mut state) = self.state.lock() else {
            return;
        };

        match *state {
            CircuitBreakerState::Closed => {
                if count as usize >= self.failure_threshold {
                    *state = CircuitBreakerState::Open;
                    if let Ok(mut lsc) = self.last_state_change.lock() {
                        *lsc = Instant::now();
                    }
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Recovery failed - open the circuit again
                *state = CircuitBreakerState::Open;
                if let Ok(mut lsc) = self.last_state_change.lock() {
                    *lsc = Instant::now();
                }
            }
            CircuitBreakerState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Get current state
    ///
    /// Returns Open (fail-safe) if lock is poisoned.
    pub fn state(&self) -> CircuitBreakerState {
        self.state
            .lock()
            .map(|s| *s)
            .unwrap_or(CircuitBreakerState::Open)
    }

    /// Reset the circuit breaker
    ///
    /// Silently ignores poisoned locks (best-effort reset).
    pub fn reset(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = CircuitBreakerState::Closed;
        }
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        if let Ok(mut last_failure) = self.last_failure.lock() {
            *last_failure = None;
        }
        if let Ok(mut lsc) = self.last_state_change.lock() {
            *lsc = Instant::now();
        }
    }

    /// Get failure count
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Get success count
    pub fn success_count(&self) -> u64 {
        self.success_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pressure() {
        assert_eq!(MemoryPressure::from_ratio(0.5), MemoryPressure::None);
        assert_eq!(MemoryPressure::from_ratio(0.65), MemoryPressure::Low);
        assert_eq!(MemoryPressure::from_ratio(0.75), MemoryPressure::Medium);
        assert_eq!(MemoryPressure::from_ratio(0.85), MemoryPressure::High);
        assert_eq!(MemoryPressure::from_ratio(0.95), MemoryPressure::Critical);
    }

    #[test]
    fn test_circuit_breaker_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30));

        // Should allow requests when closed
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitBreakerState::Closed);

        // Record some failures
        cb.record_failure();
        cb.record_failure();
        assert!(cb.allow_request()); // Still closed

        // Third failure opens the circuit
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));

        // Trigger open state
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        // Wait for recovery timeout
        std::thread::sleep(Duration::from_millis(15));

        // Should transition to half-open
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);

        // Success should close it
        cb.record_success();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));

        // Trigger open state
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        // Wait for recovery timeout
        std::thread::sleep(Duration::from_millis(15));

        // Transition to half-open
        cb.allow_request();
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);

        // Failure in half-open should re-open
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
    }
}
