use aidapter::OpenAI;
use aidapter::OpenAIModels;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

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
    let api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = OpenAIModels::Gpt4; // Choose the model

    let open_ai = OpenAI::new(&api_key, model, None, None);

    // Example context and instructions
    let instructions =
        "Translate this exact English sentence to all the languages in the response type";

    match open_ai
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Response: {:?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
