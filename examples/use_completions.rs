use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use openai_safe::{
    llm_models::{AnthropicModels, OpenAIModels},
    Completions,
};

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct TranslationResponse {
    pub spanish: String,
    pub french: String,
    pub german: String,
    pub polish: String,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // Example context and instructions
    let instructions =
        "Translate this exact English sentence to all the languages in the response type";

    // Get answer using OpenAI
    let openai_api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = OpenAIModels::Gpt4; // Choose the model

    let openai_completion = Completions::new(model, &openai_api_key, None, None);

    match openai_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("OpenAI response: {:?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Anthropic
    let openai_api_key: String =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let model = AnthropicModels::Claude2; // Choose the model

    let anthropic_completion = Completions::new(model, &openai_api_key, None, None);

    match anthropic_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Anthropic response: {:?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
