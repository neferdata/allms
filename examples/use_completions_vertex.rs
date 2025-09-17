use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{llm::GoogleModels, Completions};

mod utils;
use utils::get_vertex_token;

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct TranslationResponse {
    pub spanish: String,
    pub french: String,
    pub german: String,
    pub polish: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // Get Vertex API authentication token
    let google_token_str = get_vertex_token().await?;

    // Example context and instructions
    let instructions =
        "Translate the following English sentence to all the languages in the response type: Rust is best for working with LLMs";

    // Get answer using Google GeminiPro via Vertex AI
    let model = GoogleModels::Gemini2_5Flash;

    // **Pre-requisite**: GeminiPro request through Vertex AI require `GOOGLE_PROJECT_ID` environment variable defined
    let gemini_completion =
        Completions::new(model, &google_token_str, None, None).version("google-vertex");

    match gemini_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Vertex Gemini response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using a fine-tuned model

    // Using a fine-tuned model requires addressing the endpoint directly
    // Replace env variable with the endpoint ID of the fine-tuned model
    let fine_tuned_endpoint_id: String =
        std::env::var("GOOGLE_VERTEX_ENDPOINT_ID").expect("GOOGLE_VERTEX_ENDPOINT_ID not set");
    let model = GoogleModels::endpoint(&fine_tuned_endpoint_id);

    let gemini_completion =
        Completions::new(model, &google_token_str, None, None).version("google-vertex");

    match gemini_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Vertex Gemini response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    Ok(())
}
