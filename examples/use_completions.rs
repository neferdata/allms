use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{
    llm::{AnthropicModels, GoogleModels, LLMModel, MistralModels, OpenAIModels},
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
        "Translate the following English sentence to all the languages in the response type: Rust is best for working with LLMs";

    // Get answer using OpenAI
    let openai_api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = OpenAIModels::try_from_str("o1-preview").unwrap_or(OpenAIModels::O1Preview); // Choose the model
    println!("OpenAI model: {:#?}", model.as_str());

    let openai_completion = Completions::new(model, &openai_api_key, None, None);

    match openai_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("OpenAI response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Anthropic
    let anthropic_api_key: String =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let model = AnthropicModels::try_from_str("claude-3-5-sonnet-20240620")
        .unwrap_or(AnthropicModels::Claude3_5Sonnet); // Choose the model
    println!("Anthropic model: {:#?}", model.as_str());

    let anthropic_completion = Completions::new(model, &anthropic_api_key, None, None);

    match anthropic_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Anthropic response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Mistral
    let mistral_api_key: String =
        std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
    let model =
        MistralModels::try_from_str("open-mistral-nemo").unwrap_or(MistralModels::MistralLarge); // Choose the model
    println!("Mistral model: {:#?}", model.as_str());

    let mistral_completion = Completions::new(model, &mistral_api_key, None, None);

    match mistral_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Mistral response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Google GeminiPro
    let model =
        GoogleModels::try_from_str("gemini-1.5-pro").unwrap_or(GoogleModels::Gemini1_5Flash); // Choose the model
    println!("Google Gemini model: {:#?}", model.as_str());

    let google_token_str: String =
        std::env::var("GOOGLE_AI_STUDIO_API_KEY").expect("GOOGLE_AI_STUDIO_API_KEY not set");

    let gemini_completion = Completions::new(model, &google_token_str, None, None);

    match gemini_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Gemini response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
