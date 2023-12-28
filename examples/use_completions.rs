use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use openai_safe::{
    llm_models::{AnthropicModels, MistralModels, OpenAIModels},
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
    let anthropic_api_key: String =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let model = AnthropicModels::ClaudeInstant1_2; // Choose the model

    let anthropic_completion = Completions::new(model, &anthropic_api_key, None, None);

    match anthropic_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Anthropic response: {:?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Mistral
    let mistral_api_key: String =
        std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
    let model = MistralModels::MistralMedium; // Choose the model

    let mistral_completion = Completions::new(model, &mistral_api_key, None, None);

    match mistral_completion
        .debug()
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Mistral response: {:?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
