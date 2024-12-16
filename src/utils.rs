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

//OpenAI has a tendency to wrap response Json in ```json{}```
//TODO: This function might need to become more sophisticated or handled with better prompt eng
pub(crate) fn sanitize_json_response(json_response: &str) -> String {
    let text_no_json = json_response.replace("json\n", "");
    text_no_json.replace("```", "")
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

#[cfg(test)]
mod tests {
    use crate::llm_models::OpenAIModels;
    use crate::utils::{get_tokenizer, map_to_range};

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
}
