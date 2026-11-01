use super::*;

#[test]
fn bert_geometry_preserves_projection_direction_and_rejects_invalid_targets() {
    let geometry = BertLoraGeometry::new(2, 7, 19);
    assert_eq!(geometry.num_layers(), 2);
    for layer in 0..geometry.num_layers() {
        for (module, d_in, d_out) in [
            ("query", 7, 7),
            ("key", 7, 7),
            ("value", 7, 7),
            ("attn_output", 7, 7),
            ("ffn_intermediate", 7, 19),
            ("ffn_output", 19, 7),
        ] {
            assert_eq!(
                geometry.projection_shape(layer, module),
                Ok(LoraProjectionShape { d_in, d_out })
            );
        }
    }
    let unknown = geometry.projection_shape(1, "q_proj").unwrap_err();
    assert!(unknown.contains("q_proj") && unknown.contains("layer 1"));
    let out_of_range = geometry.projection_shape(2, "query").unwrap_err();
    assert!(out_of_range.contains("query") && out_of_range.contains("layer 2"));
}
