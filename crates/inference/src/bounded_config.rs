use lattice_weights::bounded_config::read_config_json_bounded;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_reader_maps_a_directory_error() {
        let directory = tempfile::tempdir().unwrap();
        let err = read_config_json_bounded(directory.path(), "config.json")
            .expect_err("a directory must be rejected as a config file");
        assert!(
            err.to_string().contains("is not a regular file"),
            "wrong error: {err}"
        );
    }
}
