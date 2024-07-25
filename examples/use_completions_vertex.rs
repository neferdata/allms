use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use yup_oauth2::{read_service_account_key, ServiceAccountAuthenticator};

use allms::{llm::GoogleModels, Completions};

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

    // Get answer using Google GeminiPro via Vertex AI
    let model = GoogleModels::GeminiProVertex;

    // To authenticate Google Vertex AI we need to use a key associated with a GCP service account with correct permissions
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

    // **Pre-requisite**: GeminiPro request through Vertex AI require `PROJECT_ID` environment variable defined
    let gemini_completion = Completions::new(model, google_token_str, None, None);

    match gemini_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Gemini response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
