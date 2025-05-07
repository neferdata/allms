use anyhow::Result;
use regex::Regex;
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde_json::Value;
use tiktoken_rs::{cl100k_base, get_bpe_from_model, CoreBPE};

use crate::llm_models::LLMModel;
#[allow(deprecated)]
use crate::OpenAIModels;

// Get the tokenizer given a model
#[allow(deprecated)]
#[deprecated(
    since = "0.6.1",
    note = "This function is deprecated. Please use the `get_tokenizer` function instead."
)]
pub(crate) fn get_tokenizer_old(model: &OpenAIModels) -> anyhow::Result<CoreBPE> {
    let tokenizer = get_bpe_from_model(model.as_str());
    if let Err(_error) = tokenizer {
        // Fallback to the default chat model
        cl100k_base()
    } else {
        tokenizer
    }
}

// Get the tokenizer given a model
pub(crate) fn get_tokenizer<T: LLMModel>(model: &T) -> anyhow::Result<CoreBPE> {
    let tokenizer = get_bpe_from_model(model.as_str());
    if let Err(_error) = tokenizer {
        // Fallback to the default chat model
        cl100k_base()
    } else {
        tokenizer
    }
}

/// LLMs have a tendency to wrap response Json in ```json{}```. This function sanitizes
pub(crate) fn remove_json_wrapper(json_response: &str) -> String {
    let text_no_json = json_response.replace("json\n", "");
    text_no_json.replace("```", "")
}

/// Reasoning model may include <think></think> portion explaining step-by-step reasoning
pub(crate) fn remove_think_reasoner_wrapper(json_response: &str) -> String {
    // TODO: We may want to make this more model-specific in the future
    let re = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    re.replace_all(json_response, "").to_string()
}

/// Removes the properties wrapper from JSON data if it exists
pub fn remove_properties_wrapper(json_data: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(json_data) {
        Ok(value) => {
            let processed_value = process_value(value);
            processed_value.to_string()
        }
        Err(_) => json_data.to_string(),
    }
}

fn process_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(mut obj) => {
            // If this object has a "properties" field, use its value
            if let Some(properties) = obj.remove("properties") {
                process_value(properties)
            } else {
                // Process all fields recursively
                let processed_obj: serde_json::Map<_, _> = obj
                    .into_iter()
                    .map(|(k, v)| (k, process_value(v)))
                    .collect();
                serde_json::Value::Object(processed_obj)
            }
        }
        serde_json::Value::Array(arr) => {
            // Process array elements recursively
            serde_json::Value::Array(arr.into_iter().map(process_value).collect())
        }
        // For other types (string, number, bool, null), return as is
        other => other,
    }
}

// This function generates a Json schema for the provided type
pub(crate) fn get_type_schema<T: JsonSchema + DeserializeOwned>() -> Result<String> {
    // Instruct the Assistant to answer with the right Json format
    // Output schema is extracted from the type parameter
    let mut schema = schema_for!(T);

    // Modify the schema for `serde_json::Value` fields globally
    fix_value_schema(&mut schema);

    // Convert the schema to a JSON value
    let mut schema_json: Value = serde_json::to_value(&schema)?;

    // Remove '$schema' and 'title' elements that are added by schema_for macro but are not needed
    if let Some(obj) = schema_json.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
    }

    // Convert the modified JSON value back to a pretty-printed JSON string
    Ok(serde_json::to_string_pretty(&schema_json)?)
}

// The Schemars crate uses `Bool(true)` for `Value`, which essentially means "accept anything". We need to replace it with actual `Object` type
fn fix_value_schema(schema: &mut schemars::schema::RootSchema) {
    if let Some(object) = &mut schema.schema.object {
        // Iterate over mutable values in the `properties` BTreeMap
        for subschema in object.properties.values_mut() {
            // Check if the schema is `Bool(true)` (placeholder for `serde_json::Value`)
            if let schemars::schema::Schema::Bool(true) = subschema {
                // Replace `true` with a proper schema for `serde_json::Value`
                *subschema = schemars::schema::Schema::Object(schemars::schema::SchemaObject {
                    instance_type: Some(schemars::schema::InstanceType::Object.into()),
                    ..Default::default()
                });
            }
        }
    }
}

//Used internally to pick a number from range based on its % representation
pub(crate) fn map_to_range(min: u32, max: u32, target: u32) -> f32 {
    // Cap the target to the percentage range [0, 100]
    let capped_target = target.min(100);

    // Calculate the target value in the range [min, max]
    let range = max as f32 - min as f32;
    let percentage = capped_target as f32 / 100.0;
    min as f32 + (range * percentage)
}

//Used internally to pick a number from range based on its % representation
pub(crate) fn map_to_range_f32(min: f32, max: f32, target: u32) -> f32 {
    // Cap the target to the percentage range [0, 100]
    let capped_target = target.min(100);

    // Calculate the target value in the range [min, max]
    let range = max - min;
    let percentage = capped_target as f32 / 100.0;
    min + (range * percentage)
}

#[cfg(test)]
mod tests {
    use schemars::schema::{InstanceType, ObjectValidation, RootSchema, Schema, SchemaObject};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use crate::llm_models::OpenAIModels;
    use crate::utils::{
        fix_value_schema, get_tokenizer, get_type_schema, map_to_range, map_to_range_f32,
        remove_properties_wrapper, remove_think_reasoner_wrapper,
    };

    #[derive(JsonSchema, Serialize, Deserialize)]
    struct SimpleStruct {
        id: i32,
        name: String,
    }

    #[derive(JsonSchema, Serialize, Deserialize)]
    struct StructWithValue {
        data: serde_json::Value,
    }

    #[derive(JsonSchema, Serialize, Deserialize)]
    struct NestedStruct {
        info: SimpleStruct,
        optional_field: Option<String>,
    }

    // Tokenizer tests
    #[test]
    fn it_computes_gpt3_5_tokenization() {
        let bpe = get_tokenizer(&OpenAIModels::Gpt4_32k).unwrap();
        let tokenized: Result<Vec<_>, _> = bpe
            .split_by_token_iter("This is a test         with a lot of spaces", true)
            .collect();
        let tokenized = tokenized.unwrap();
        assert_eq!(
            tokenized,
            vec!["This", " is", " a", " test", "        ", " with", " a", " lot", " of", " spaces"]
        );
    }

    // Generating correct schema for types
    #[test]
    fn test_get_type_schema_simple_struct() {
        let schema_result = get_type_schema::<SimpleStruct>();

        assert!(
            schema_result.is_ok(),
            "Expected schema generation to succeed"
        );

        let schema_json = schema_result.unwrap();
        let schema_value: Value = serde_json::from_str(&schema_json).unwrap();

        // Verify basic structure of the schema
        assert!(
            schema_value.is_object(),
            "Expected schema to be a JSON object"
        );
        let properties = schema_value["properties"].as_object().unwrap();
        assert!(properties.contains_key("id"), "Schema should contain 'id'");
        assert!(
            properties.contains_key("name"),
            "Schema should contain 'name'"
        );
    }

    #[test]
    fn test_get_type_schema_struct_with_value() {
        let schema_result = get_type_schema::<StructWithValue>();

        assert!(
            schema_result.is_ok(),
            "Expected schema generation to succeed"
        );

        let schema_json = schema_result.unwrap();
        let schema_value: Value = serde_json::from_str(&schema_json).unwrap();

        // Verify that the `data` field has been replaced with a proper object schema
        let data_schema = &schema_value["properties"]["data"];
        assert!(
            data_schema.is_object(),
            "Expected 'data' to be a JSON object"
        );
        assert_eq!(
            data_schema["type"].as_str(),
            Some("object"),
            "Expected 'data' field to be of type 'object'"
        );
    }

    #[test]
    fn test_get_type_schema_removes_schema_and_title() {
        let schema_result = get_type_schema::<SimpleStruct>();

        assert!(
            schema_result.is_ok(),
            "Expected schema generation to succeed"
        );

        let schema_json = schema_result.unwrap();
        let schema_value: Value = serde_json::from_str(&schema_json).unwrap();

        // Ensure `$schema` and `title` are removed
        assert!(
            !schema_value.as_object().unwrap().contains_key("$schema"),
            "Schema should not contain '$schema'"
        );
        assert!(
            !schema_value.as_object().unwrap().contains_key("title"),
            "Schema should not contain 'title'"
        );
    }

    #[test]
    fn test_get_type_schema_handles_nested_struct() {
        let schema_result = get_type_schema::<NestedStruct>();

        assert!(
            schema_result.is_ok(),
            "Expected schema generation to succeed"
        );

        let schema_json = schema_result.unwrap();
        let schema_value: Value = serde_json::from_str(&schema_json).unwrap();

        // Verify nested structure
        let info_schema = &schema_value["properties"]["info"];
        assert!(
            info_schema.is_object(),
            "Expected 'info' to be a JSON object"
        );

        // Check that `info` references `SimpleStruct`
        assert!(
            info_schema.get("$ref").is_some(),
            "Expected 'info' to have a $ref to a definition"
        );

        let ref_path = info_schema["$ref"].as_str().unwrap();
        assert_eq!(ref_path, "#/definitions/SimpleStruct");

        // Verify the `SimpleStruct` definition
        let simple_struct_schema = &schema_value["definitions"]["SimpleStruct"];
        let simple_struct_properties = simple_struct_schema["properties"].as_object().unwrap();

        assert!(
            simple_struct_properties.contains_key("id"),
            "SimpleStruct schema should contain 'id'"
        );
        assert!(
            simple_struct_properties.contains_key("name"),
            "SimpleStruct schema should contain 'name'"
        );
    }

    #[test]
    fn test_get_type_schema_pretty_printed_json() {
        let schema_result = get_type_schema::<SimpleStruct>();

        assert!(
            schema_result.is_ok(),
            "Expected schema generation to succeed"
        );

        let schema_json = schema_result.unwrap();

        // Verify pretty-printed formatting by checking indentation
        assert!(
            schema_json.contains('\n'),
            "Expected pretty-printed JSON with line breaks"
        );
        assert!(
            schema_json.contains("  "),
            "Expected pretty-printed JSON with indentation"
        );
    }

    // Fixing how Value is represented in schema
    #[test]
    fn test_fix_value_schema_replaces_bool_true() {
        let mut schema = RootSchema {
            schema: SchemaObject {
                object: Some(Box::new(ObjectValidation {
                    properties: {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert(
                            "test_property".to_string(),
                            Schema::Bool(true), // This should be replaced
                        );
                        map
                    },
                    ..Default::default()
                })),
                ..Default::default()
            },
            ..Default::default()
        };

        fix_value_schema(&mut schema);

        // Assert that the `Bool(true)` was replaced with a `SchemaObject`
        if let Some(object) = &schema.schema.object {
            if let Schema::Object(subschema) = object.properties.get("test_property").unwrap() {
                assert_eq!(subschema.instance_type, Some(InstanceType::Object.into()));
            } else {
                panic!("Expected Schema::Object, but found something else");
            }
        } else {
            panic!("Expected object validation in schema, but none found");
        }
    }

    #[test]
    fn test_fix_value_schema_ignores_other_schemas() {
        let mut schema = RootSchema {
            schema: SchemaObject {
                object: Some(Box::new(ObjectValidation {
                    properties: {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert(
                            "test_property".to_string(),
                            Schema::Object(SchemaObject {
                                instance_type: Some(InstanceType::String.into()),
                                ..Default::default()
                            }), // This should remain unchanged
                        );
                        map
                    },
                    ..Default::default()
                })),
                ..Default::default()
            },
            ..Default::default()
        };

        fix_value_schema(&mut schema);

        // Assert that the schema with `InstanceType::String` remains unchanged
        if let Some(object) = &schema.schema.object {
            if let Schema::Object(subschema) = object.properties.get("test_property").unwrap() {
                assert_eq!(subschema.instance_type, Some(InstanceType::String.into()));
            } else {
                panic!("Expected Schema::Object, but found something else");
            }
        } else {
            panic!("Expected object validation in schema, but none found");
        }
    }

    #[test]
    fn test_fix_value_schema_handles_missing_properties() {
        let mut schema = RootSchema {
            schema: SchemaObject {
                object: Some(Box::new(ObjectValidation {
                    properties: std::collections::BTreeMap::new(), // Empty properties
                    ..Default::default()
                })),
                ..Default::default()
            },
            ..Default::default()
        };

        fix_value_schema(&mut schema);

        // Assert that the properties map is still empty
        if let Some(object) = &schema.schema.object {
            assert!(object.properties.is_empty());
        } else {
            panic!("Expected object validation in schema, but none found");
        }
    }

    #[test]
    fn test_fix_value_schema_handles_missing_object() {
        let mut schema = RootSchema {
            schema: SchemaObject {
                object: None, // No object validation
                ..Default::default()
            },
            ..Default::default()
        };

        fix_value_schema(&mut schema);

        // Assert that the schema's object field is still None
        assert!(schema.schema.object.is_none());
    }

    // Mapping % target to temperature range
    #[test]
    fn test_target_at_min() {
        assert_eq!(map_to_range(0, 100, 0), 0.0);
        assert_eq!(map_to_range(10, 20, 0), 10.0);
    }

    #[test]
    fn test_target_at_max() {
        assert_eq!(map_to_range(0, 100, 100), 100.0);
        assert_eq!(map_to_range(10, 20, 100), 20.0);
    }

    #[test]
    fn test_target_in_middle() {
        assert_eq!(map_to_range(0, 100, 50), 50.0);
        assert_eq!(map_to_range(10, 20, 50), 15.0);
        assert_eq!(map_to_range(0, 1, 50), 0.5);
    }

    #[test]
    fn test_target_out_of_bounds() {
        assert_eq!(map_to_range(0, 100, 3000), 100.0); // Cap to 100
        assert_eq!(map_to_range(0, 100, 200), 100.0); // Cap to 100
        assert_eq!(map_to_range(10, 20, 200), 20.0); // Cap to 100
    }

    #[test]
    fn test_zero_range() {
        assert_eq!(map_to_range(10, 10, 50), 10.0); // Always return min if min == max
        assert_eq!(map_to_range(5, 5, 100), 5.0); // Even at max target
    }

    #[test]
    fn test_negative_behavior_not_applicable() {
        // Not applicable for unsigned inputs but could test edge cases:
        assert_eq!(map_to_range(0, 100, 0), 0.0);
    }

    #[test]
    fn test_target_at_min_f32() {
        assert_eq!(map_to_range_f32(0.0, 100.0, 0), 0.0);
        assert_eq!(map_to_range_f32(10.0, 20.0, 0), 10.0);
    }

    #[test]
    fn test_target_at_max_f32() {
        assert_eq!(map_to_range_f32(0.0, 100.0, 100), 100.0);
        assert_eq!(map_to_range_f32(10.0, 20.0, 100), 20.0);
    }

    #[test]
    fn test_target_in_middle_f32() {
        assert_eq!(map_to_range_f32(0.0, 100.0, 50), 50.0);
        assert_eq!(map_to_range_f32(10.0, 20.0, 50), 15.0);
        assert_eq!(map_to_range_f32(0.0, 1.0, 50), 0.5);
    }

    #[test]
    fn test_target_out_of_bounds_f32() {
        assert_eq!(map_to_range_f32(0.0, 100.0, 3000), 100.0); // Cap to 100
        assert_eq!(map_to_range_f32(0.0, 100.0, 200), 100.0); // Cap to 100
        assert_eq!(map_to_range_f32(10.0, 20.0, 200), 20.0); // Cap to 100
    }

    #[test]
    fn test_zero_range_f32() {
        assert_eq!(map_to_range_f32(10.0, 10.0, 50), 10.0); // Always return min if min == max
        assert_eq!(map_to_range_f32(5.0, 5.0, 100), 5.0); // Even at max target
    }

    #[test]
    fn test_fractional_range_f32() {
        assert_eq!(map_to_range_f32(0.0, 0.5, 50), 0.25);
        assert_eq!(map_to_range_f32(1.5, 3.0, 25), 1.875);
        assert_eq!(map_to_range_f32(-1.0, 1.0, 75), 0.5);
    }

    #[test]
    fn test_large_range_f32() {
        assert_eq!(map_to_range_f32(-1000.0, 1000.0, 50), 0.0); // Midpoint of the range
        assert_eq!(map_to_range_f32(-500.0, 500.0, 25), -250.0); // Quarter point
        assert_eq!(map_to_range_f32(-2000.0, 0.0, 75), -500.0); // Three-quarters
    }

    // Removing <think></think> wrapper
    #[test]
    fn test_remove_think_text() {
        assert_eq!(
            remove_think_reasoner_wrapper("Hello <think>ignore this</think> World"),
            "Hello  World"
        );
        assert_eq!(
            remove_think_reasoner_wrapper("<think>Only this</think>"),
            ""
        );
        assert_eq!(
            remove_think_reasoner_wrapper("No markers here"),
            "No markers here"
        );
        assert_eq!(
            remove_think_reasoner_wrapper(
                "Multiple <think>first</think> parts <think>second</think> remain"
            ),
            "Multiple  parts  remain"
        );
    }

    // Tests for remove_properties_wrapper
    #[test]
    fn test_remove_properties_wrapper_with_wrapper() {
        let input = r#"{"properties": {"name": "John", "age": 30}}"#;
        let expected = r#"{"name":"John","age":30}"#;
        let result = remove_properties_wrapper(input);

        // Parse both strings into Value to compare the actual data structure
        let result_value: Value = serde_json::from_str(&result).unwrap();
        let expected_value: Value = serde_json::from_str(expected).unwrap();
        assert_eq!(
            result_value, expected_value,
            "Should remove properties wrapper"
        );
    }

    #[test]
    fn test_remove_properties_wrapper_without_wrapper() {
        let input = r#"{"name": "John", "age": 30}"#;
        let result = remove_properties_wrapper(input);

        // Parse both strings into Value to compare the actual data structure
        let result_value: Value = serde_json::from_str(&result).unwrap();
        let input_value: Value = serde_json::from_str(input).unwrap();
        assert_eq!(
            result_value, input_value,
            "Should return original string when no properties wrapper exists"
        );
    }

    #[test]
    fn test_remove_properties_wrapper_with_nested_structure() {
        let input = r#"{"properties": {"user": {"properties": {"name": "John", "age": 30}}}}"#;
        let expected = r#"{"user":{"name":"John","age":30}}"#;
        let result = remove_properties_wrapper(input);

        // Parse both strings into Value to compare the actual data structure
        let result_value: Value = serde_json::from_str(&result).unwrap();
        let expected_value: Value = serde_json::from_str(expected).unwrap();
        assert_eq!(
            result_value, expected_value,
            "Should handle nested properties wrappers"
        );
    }

    #[test]
    fn test_remove_properties_wrapper_with_invalid_json() {
        let input = "invalid json";
        let result = remove_properties_wrapper(input);
        assert_eq!(
            result, input,
            "Should return original string for invalid JSON"
        );
    }

    #[test]
    fn test_remove_properties_wrapper_with_array() {
        let input = r#"{"properties": {"items": [1, 2, 3]}}"#;
        let expected = r#"{"items":[1,2,3]}"#;
        let result = remove_properties_wrapper(input);

        // Parse both strings into Value to compare the actual data structure
        let result_value: Value = serde_json::from_str(&result).unwrap();
        let expected_value: Value = serde_json::from_str(expected).unwrap();
        assert_eq!(
            result_value, expected_value,
            "Should handle arrays within properties"
        );
    }

    #[test]
    fn test_remove_properties_wrapper_with_complex_structure() {
        let input = r#"{"properties": {"responses": {"properties": {"items": [{"confidence": 100, "source": "test", "value": {"date": "2024-03-20", "post": "test", "tariffs": false, "url": "https://example.com"}}]}}}}"#;
        let expected = r#"{"responses":{"items":[{"confidence":100,"source":"test","value":{"date":"2024-03-20","post":"test","tariffs":false,"url":"https://example.com"}}]}}"#;
        let result = remove_properties_wrapper(input);

        // Parse both strings into Value to compare the actual data structure
        let result_value: Value = serde_json::from_str(&result).unwrap();
        let expected_value: Value = serde_json::from_str(expected).unwrap();
        assert_eq!(
            result_value, expected_value,
            "Should handle complex nested structures"
        );
    }
}
