use tiktoken_rs::{cl100k_base, get_bpe_from_model, CoreBPE};

use crate::llm_models::LLMModel;
use crate::models::OpenAIModels;

// Get the tokenizer given a model
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

#[cfg(test)]
mod tests {
    use crate::llm_models::OpenAIModels;
    use crate::utils::get_tokenizer;

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
}
