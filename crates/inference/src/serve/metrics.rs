//! Minimal in-process Prometheus text-exposition-format metrics registry
//! (issue #583). Deliberately dependency-free: `lattice_serve`'s HTTP
//! surface exposes O(10) distinct series at a scrape interval measured in
//! seconds, so a hand-rolled `HashMap<label-tuple, count>` behind a
//! `std::sync::Mutex` is simpler to audit than pulling in the `prometheus`
//! crate for this scope, and keeps this leaf-adjacent module free of a new
//! external dependency.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Latency histogram bucket upper bounds, in seconds. Mirrors the default
/// buckets most Prometheus client libraries ship, covering sub-millisecond
/// health checks up to a generous worst-case generation request.
const LATENCY_BUCKETS_SECONDS: [f64; 13] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Per-route latency accumulator. `bucket_counts[i]` holds the number of
/// observations that fell in `(LATENCY_BUCKETS_SECONDS[i - 1],
/// LATENCY_BUCKETS_SECONDS[i]]` (or `(0, LATENCY_BUCKETS_SECONDS[0]]` for
/// `i == 0`) -- NOT yet cumulative; [`ServeMetrics::render`] computes the
/// running total Prometheus's `le` bucket semantics require. An observation
/// exceeding every finite bucket bound increments no `bucket_counts` slot but
/// still counts toward `count`/`sum_seconds`, which is exactly what the
/// implicit `le="+Inf"` bucket must report.
#[derive(Default)]
struct RouteLatency {
    bucket_counts: [u64; LATENCY_BUCKETS_SECONDS.len()],
    sum_seconds: f64,
    count: u64,
}

impl RouteLatency {
    fn observe(&mut self, seconds: f64) {
        self.sum_seconds += seconds;
        self.count += 1;
        for (i, bound) in LATENCY_BUCKETS_SECONDS.iter().enumerate() {
            if seconds <= *bound {
                self.bucket_counts[i] += 1;
                break;
            }
        }
    }
}

/// Process-wide metrics registry for one `lattice_serve` instance. Cheap to
/// share: every field is a `Mutex`/atomic behind a shared reference, so
/// callers hold this behind an `Arc` (see `AppState::metrics` in
/// `lattice_serve.rs`) and every clone observes the same counters.
#[derive(Default)]
pub struct ServeMetrics {
    /// (method, route, status) -> request count.
    requests_total: Mutex<HashMap<(String, String, u16), u64>>,
    /// route -> latency histogram.
    route_latency: Mutex<HashMap<String, RouteLatency>>,
    prompt_tokens_total: AtomicU64,
    completion_tokens_total: AtomicU64,
    /// OpenAI-style error code -> count.
    errors_total: Mutex<HashMap<String, u64>>,
}

impl ServeMetrics {
    /// Records one completed HTTP request: increments the (method, route,
    /// status) counter and observes `dur_seconds` into that route's latency
    /// histogram.
    pub fn record_request(&self, method: &str, route: &str, status: u16, dur_seconds: f64) {
        let mut requests = self
            .requests_total
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *requests
            .entry((method.to_string(), route.to_string(), status))
            .or_insert(0) += 1;
        drop(requests);

        let mut latencies = self
            .route_latency
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        latencies
            .entry(route.to_string())
            .or_default()
            .observe(dur_seconds);
    }

    /// Adds to the prompt/completion token counters. A no-op call with both
    /// zero (the common case for non-generation routes) is harmless.
    pub fn record_tokens(&self, prompt_tokens: usize, completion_tokens: usize) {
        if prompt_tokens > 0 {
            self.prompt_tokens_total
                .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        }
        if completion_tokens > 0 {
            self.completion_tokens_total
                .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        }
    }

    /// Increments the error counter for one OpenAI-style error `code`
    /// (e.g. `"invalid_request"`, `"model_not_found"`, `"internal_error"`).
    pub fn record_error(&self, code: &str) {
        let mut errors = self
            .errors_total
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *errors.entry(code.to_string()).or_insert(0) += 1;
    }

    /// Renders the full registry as Prometheus text-exposition format
    /// (version 0.0.4), labeling every series with `model="<model_id>"` --
    /// `lattice_serve` serves exactly one model per process, so a per-series
    /// model label (rather than a metric-name suffix) keeps the series names
    /// stable across model swaps and matches how most Prometheus exporters
    /// attach instance-identifying labels. `in_flight` is a live snapshot
    /// the caller computes from the shared worker's admission semaphore
    /// (`MetalWorkerClient::available_permits`), not state owned by this
    /// registry.
    pub fn render(&self, model_id: &str, in_flight: usize) -> String {
        let model = escape_label(model_id);
        let mut out = String::new();

        out.push_str(
            "# HELP lattice_http_requests_total Total HTTP requests processed.\n\
             # TYPE lattice_http_requests_total counter\n",
        );
        {
            let requests = self
                .requests_total
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut rows: Vec<_> = requests.iter().collect();
            rows.sort();
            for ((method, route, status), count) in rows {
                out.push_str(&format!(
                    "lattice_http_requests_total{{method=\"{}\",route=\"{}\",status=\"{status}\",model=\"{model}\"}} {count}\n",
                    escape_label(method),
                    escape_label(route),
                ));
            }
        }

        out.push_str(
            "# HELP lattice_http_request_duration_seconds HTTP request latency in seconds.\n\
             # TYPE lattice_http_request_duration_seconds histogram\n",
        );
        {
            let latencies = self
                .route_latency
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut rows: Vec<_> = latencies.iter().collect();
            rows.sort_by(|a, b| a.0.cmp(b.0));
            for (route, hist) in rows {
                let route = escape_label(route);
                let mut cumulative = 0u64;
                for (bound, bucket_count) in LATENCY_BUCKETS_SECONDS.iter().zip(hist.bucket_counts)
                {
                    cumulative += bucket_count;
                    out.push_str(&format!(
                        "lattice_http_request_duration_seconds_bucket{{route=\"{route}\",model=\"{model}\",le=\"{bound}\"}} {cumulative}\n"
                    ));
                }
                out.push_str(&format!(
                    "lattice_http_request_duration_seconds_bucket{{route=\"{route}\",model=\"{model}\",le=\"+Inf\"}} {}\n",
                    hist.count
                ));
                out.push_str(&format!(
                    "lattice_http_request_duration_seconds_sum{{route=\"{route}\",model=\"{model}\"}} {}\n",
                    hist.sum_seconds
                ));
                out.push_str(&format!(
                    "lattice_http_request_duration_seconds_count{{route=\"{route}\",model=\"{model}\"}} {}\n",
                    hist.count
                ));
            }
        }

        out.push_str(&format!(
            "# HELP lattice_prompt_tokens_total Total prompt tokens processed.\n\
             # TYPE lattice_prompt_tokens_total counter\n\
             lattice_prompt_tokens_total{{model=\"{model}\"}} {}\n",
            self.prompt_tokens_total.load(Ordering::Relaxed)
        ));

        out.push_str(&format!(
            "# HELP lattice_completion_tokens_total Total completion tokens generated.\n\
             # TYPE lattice_completion_tokens_total counter\n\
             lattice_completion_tokens_total{{model=\"{model}\"}} {}\n",
            self.completion_tokens_total.load(Ordering::Relaxed)
        ));

        out.push_str(&format!(
            "# HELP lattice_inflight_requests Outstanding (queued + in-flight) requests on the shared worker.\n\
             # TYPE lattice_inflight_requests gauge\n\
             lattice_inflight_requests{{model=\"{model}\"}} {in_flight}\n"
        ));

        out.push_str(
            "# HELP lattice_errors_total Total error responses, labeled by error code.\n\
             # TYPE lattice_errors_total counter\n",
        );
        {
            let errors = self
                .errors_total
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut rows: Vec<_> = errors.iter().collect();
            rows.sort();
            for (code, count) in rows {
                out.push_str(&format!(
                    "lattice_errors_total{{code=\"{}\",model=\"{model}\"}} {count}\n",
                    escape_label(code),
                ));
            }
        }

        out
    }
}

/// Escapes a Prometheus label value per the text-exposition format: a
/// backslash becomes `\\`, a double quote becomes `\"`, a newline becomes
/// `\n`. Order matters -- backslash must be escaped first, or the
/// backslashes this function inserts for `"`/`\n` would themselves be
/// re-escaped.
fn escape_label(raw: &str) -> String {
    raw.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_request_increments_count_and_histogram() {
        let metrics = ServeMetrics::default();
        metrics.record_request("GET", "/health", 200, 0.001);
        metrics.record_request("GET", "/health", 200, 0.002);
        let body = metrics.render("test-model", 0);
        assert!(body.contains(
            "lattice_http_requests_total{method=\"GET\",route=\"/health\",status=\"200\",model=\"test-model\"} 2\n"
        ));
        assert!(body.contains(
            "lattice_http_request_duration_seconds_count{route=\"/health\",model=\"test-model\"} 2\n"
        ));
    }

    #[test]
    fn record_tokens_accumulates_across_calls() {
        let metrics = ServeMetrics::default();
        metrics.record_tokens(10, 5);
        metrics.record_tokens(3, 7);
        let body = metrics.render("m", 0);
        assert!(body.contains("lattice_prompt_tokens_total{model=\"m\"} 13\n"));
        assert!(body.contains("lattice_completion_tokens_total{model=\"m\"} 12\n"));
    }

    #[test]
    fn record_error_counts_by_code() {
        let metrics = ServeMetrics::default();
        metrics.record_error("invalid_request");
        metrics.record_error("invalid_request");
        metrics.record_error("internal_error");
        let body = metrics.render("m", 0);
        assert!(body.contains("lattice_errors_total{code=\"internal_error\",model=\"m\"} 1\n"));
        assert!(body.contains("lattice_errors_total{code=\"invalid_request\",model=\"m\"} 2\n"));
    }

    #[test]
    fn in_flight_gauge_reflects_caller_supplied_snapshot() {
        let metrics = ServeMetrics::default();
        let body = metrics.render("m", 3);
        assert!(body.contains("lattice_inflight_requests{model=\"m\"} 3\n"));
    }

    #[test]
    fn latency_bucket_cumulative_counts_are_monotonic() {
        let metrics = ServeMetrics::default();
        metrics.record_request("POST", "/v1/chat/completions", 200, 0.01);
        metrics.record_request("POST", "/v1/chat/completions", 200, 5.0);
        let body = metrics.render("m", 0);
        // Bucket le="0.025" (>= the 0.01 observation, < the 5.0 one) must
        // read 1; le="+Inf" (== total count) must read 2.
        assert!(body.contains(
            "lattice_http_request_duration_seconds_bucket{route=\"/v1/chat/completions\",model=\"m\",le=\"0.025\"} 1\n"
        ));
        assert!(body.contains(
            "lattice_http_request_duration_seconds_bucket{route=\"/v1/chat/completions\",model=\"m\",le=\"+Inf\"} 2\n"
        ));
    }

    #[test]
    fn label_escaping_handles_special_characters() {
        assert_eq!(escape_label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
