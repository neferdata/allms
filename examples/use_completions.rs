use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use yup_oauth2::{read_service_account_key, ServiceAccountAuthenticator};

use allms::{
    llm_models::{AnthropicModels, GoogleModels, MistralModels, OpenAIModels},
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
    let model = AnthropicModels::Claude2; // Choose the model

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
    let model = MistralModels::MistralTiny; // Choose the model

    let mistral_completion = Completions::new(model, &mistral_api_key, None, None);

    match mistral_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Mistral response: {:?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Google GeminiPro
    let model = GoogleModels::GeminiProVertex; // Choose the model

    // To authenticate Google we need to use a key associated with a GCP service account with correct grants
    // Load your service account key from a file or an environment variable
    let service_account_key = read_service_account_key("secrets/gcp_sa_key.json")
        .await
        .unwrap();

    // Authenticate with your service account
    let auth = ServiceAccountAuthenticator::builder(service_account_key)
        .build()
        .await
        .unwrap();
    let google_token = auth
        .token(&["https://www.googleapis.com/auth/cloud-platform"])
        .await
        .unwrap();
    let google_token_str = &google_token.token().unwrap();

    let gemini_completion = Completions::new(model, &google_token_str, None, None);

    match gemini_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Gemini response: {:?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
