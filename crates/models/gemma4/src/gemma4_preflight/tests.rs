use super::*;

#[test]
fn duplicate_raw_tensor_inventory_key_is_rejected() {
    // Two raw JSON members naming the same tensor must be rejected before
    // collapsing into the HashMap, not silently resolved to whichever
    // value the raw JSON happened to visit last (mirrors PR #988's
    // `deserialize_weight_map_no_duplicates` test for the analogous
    // `weight_map` shape).
    let json = r#"{"tensors": {
            "model.language_model.norm.weight": {"dtype": "BF16", "shape": [1536]},
            "model.language_model.norm.weight": {"dtype": "F32", "shape": [1]}
        }}"#;
    let result: Result<TensorInventory, _> = serde_json::from_str(json);
    let err = result.expect_err("a duplicate raw tensor-name key must be rejected");
    assert!(err.to_string().contains("duplicate"));
    assert!(err.to_string().contains("model.language_model.norm.weight"));
}
