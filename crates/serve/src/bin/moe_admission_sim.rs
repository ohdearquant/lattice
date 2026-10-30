//! Offline MoE expert-cache admission-policy simulator (issue #682 Stage 3).
//!
//! Thin CLI wrapper around [`lattice_inference::moe_admission`] — see that
//! module's doc comment for the full design rationale (why this exists, the
//! trace format, and the fidelity notes for each policy). This binary only
//! parses arguments, reads the trace file, and prints the report; all
//! simulation logic lives in the library module so it is covered by
//! `cargo test -p lattice-inference --lib`.
//!
//! # Usage
//!
//! ```text
//! cargo run --bin moe_admission_sim -- \
//!   --trace routing_trace.jsonl \
//!   --num-slots 32 \
//!   --top-k 8 \
//!   [--window 8] \
//!   [--json-out report.json]
//! ```

use lattice_inference::moe_admission::{
    PolicySpec, SimConfig, format_table, group_by_layer, read_trace, run_simulation,
};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

fn print_usage_and_exit() -> ! {
    eprintln!(
        "moe_admission_sim: replay a JSONL MoE routing trace against the shipped LRU \
         expert-cache policy and challenger admission policies (issue #682 Stage 3)\n\n\
         Usage:\n\
         \x20 moe_admission_sim --trace <path.jsonl> --num-slots <N> --top-k <K> \
         [--window <W>] [--json-out <path>]\n\n\
         Options:\n\
         \x20 --trace <path>      JSONL routing trace: one {{layer_idx, token_idx, \
         selected_ids, gate_weights}} object per line (required)\n\
         \x20 --num-slots <N>     per-layer cache capacity in expert slots (required)\n\
         \x20 --top-k <K>         experts selected per token; N must be >= K (required)\n\
         \x20 --window <W>        freq-admission sliding-window size (default 8)\n\
         \x20 --json-out <path>   also write the full report as JSON to this path"
    );
    std::process::exit(1);
}

struct Args {
    trace: PathBuf,
    num_slots: usize,
    top_k: usize,
    window: usize,
    json_out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut trace: Option<PathBuf> = None;
    let mut num_slots: Option<usize> = None;
    let mut top_k: Option<usize> = None;
    let mut window: usize = 8;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--trace" => {
                i += 1;
                trace = Some(PathBuf::from(raw.get(i).unwrap_or_else(|| {
                    eprintln!("--trace requires an argument");
                    print_usage_and_exit();
                })));
            }
            "--num-slots" => {
                i += 1;
                num_slots = Some(parse_usize_arg("--num-slots", raw.get(i)));
            }
            "--top-k" => {
                i += 1;
                top_k = Some(parse_usize_arg("--top-k", raw.get(i)));
            }
            "--window" => {
                i += 1;
                window = parse_usize_arg("--window", raw.get(i));
            }
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(raw.get(i).unwrap_or_else(|| {
                    eprintln!("--json-out requires an argument");
                    print_usage_and_exit();
                })));
            }
            "-h" | "--help" => print_usage_and_exit(),
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage_and_exit();
            }
        }
        i += 1;
    }

    let trace = trace.unwrap_or_else(|| {
        eprintln!("--trace is required");
        print_usage_and_exit();
    });
    let num_slots = num_slots.unwrap_or_else(|| {
        eprintln!("--num-slots is required");
        print_usage_and_exit();
    });
    let top_k = top_k.unwrap_or_else(|| {
        eprintln!("--top-k is required");
        print_usage_and_exit();
    });

    Args {
        trace,
        num_slots,
        top_k,
        window,
        json_out,
    }
}

fn parse_usize_arg(flag: &str, raw: Option<&String>) -> usize {
    let raw = raw.unwrap_or_else(|| {
        eprintln!("{flag} requires an argument");
        print_usage_and_exit();
    });
    raw.parse::<usize>().unwrap_or_else(|e| {
        eprintln!("{flag}={raw:?} is not a valid non-negative integer: {e}");
        print_usage_and_exit();
    })
}

fn main() {
    let args = parse_args();

    let file = File::open(&args.trace).unwrap_or_else(|e| {
        eprintln!("failed to open trace {}: {e}", args.trace.display());
        std::process::exit(1);
    });
    let records = read_trace(BufReader::new(file)).unwrap_or_else(|e| {
        eprintln!("failed to parse trace {}: {e}", args.trace.display());
        std::process::exit(1);
    });
    if records.is_empty() {
        eprintln!("trace {} contains zero records", args.trace.display());
        std::process::exit(1);
    }
    let num_records = records.len();
    let layers = group_by_layer(records);
    eprintln!(
        "[moe_admission_sim] loaded {num_records} records across {} layers from {}",
        layers.len(),
        args.trace.display()
    );

    let cfg = SimConfig {
        num_slots: args.num_slots,
        top_k: args.top_k,
    };
    let policies = vec![
        PolicySpec::Lru,
        PolicySpec::Arc,
        PolicySpec::FreqAdmission {
            window: args.window,
        },
    ];

    let reports = run_simulation(&layers, &cfg, &policies).unwrap_or_else(|e| {
        eprintln!("simulation failed: {e}");
        std::process::exit(1);
    });

    println!("{}", format_table(&reports));

    if let Some(json_path) = &args.json_out {
        let json = serde_json::to_string_pretty(&reports).unwrap_or_else(|e| {
            eprintln!("failed to serialize report to JSON: {e}");
            std::process::exit(1);
        });
        std::fs::write(json_path, json).unwrap_or_else(|e| {
            eprintln!("failed to write {}: {e}", json_path.display());
            std::process::exit(1);
        });
        eprintln!(
            "[moe_admission_sim] wrote JSON report to {}",
            json_path.display()
        );
    }
}
