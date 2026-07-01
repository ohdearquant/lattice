//! CLI wrapper for [`lattice_inference::weights::w3_mlp_convert::quantize_w3_mlp`].
//!
//! Usage: `quantize_w3_mlp --model-dir <DIR> --output-dir <DIR> [--dry-run]`

use lattice_inference::weights::w3_mlp_convert::quantize_w3_mlp;
use std::path::PathBuf;

fn print_usage_and_exit() -> ! {
    eprintln!(
        "Usage: quantize_w3_mlp --model-dir <DIR> --output-dir <DIR> [--dry-run]\n\n\
         Converts safetensors F16/BF16 MLP weights to packed W3, producing a\n\
         complete mixed .w3/.q4/.f16 output directory."
    );
    std::process::exit(2);
}

fn main() {
    let mut model_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut dry_run = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                    eprintln!("--model-dir requires a value");
                    print_usage_and_exit()
                })));
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                    eprintln!("--output-dir requires a value");
                    print_usage_and_exit()
                })));
            }
            "--dry-run" => dry_run = true,
            "-h" | "--help" => print_usage_and_exit(),
            other => {
                eprintln!("Unrecognized argument: {other}");
                print_usage_and_exit();
            }
        }
    }

    let Some(model_dir) = model_dir else {
        eprintln!("--model-dir is required");
        print_usage_and_exit();
    };
    let Some(output_dir) = output_dir else {
        eprintln!("--output-dir is required");
        print_usage_and_exit();
    };

    match quantize_w3_mlp(&model_dir, &output_dir, dry_run) {
        Ok(report) => {
            println!(
                "quantize_w3_mlp: processed {} tensors (w3={}, q4={}, f16={}), \
                 bytes_in={}, bytes_out={}{}",
                report.tensors_processed,
                report.tensors_w3,
                report.tensors_q4,
                report.tensors_f16,
                report.bytes_in,
                report.bytes_out,
                if dry_run {
                    " [dry-run, nothing written]"
                } else {
                    ""
                }
            );
        }
        Err(e) => {
            eprintln!("quantize_w3_mlp: error: {e}");
            std::process::exit(1);
        }
    }
}
