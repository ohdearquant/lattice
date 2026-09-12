use std::env;
use std::path::Path;
use std::process::ExitCode;

use lattice_inference::CrossEncoderModel;

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let (Some(model_dir), None) = (args.next(), args.next()) else {
        eprintln!("Usage: load_cross_encoder <model-directory>");
        return ExitCode::from(2);
    };

    match CrossEncoderModel::from_directory(Path::new(&model_dir)) {
        Ok(_model) => {
            println!("CROSS_ENCODER_LOADED");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("CROSS_ENCODER_LOAD_ERROR: {error}");
            eprintln!("CROSS_ENCODER_LOAD_ERROR_DEBUG: {error:?}");
            ExitCode::FAILURE
        }
    }
}
