use anyhow::{anyhow, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use std::ffi::OsStr;
use std::path::Path;

use allms::{
    assistants::{OpenAIFile, OpenAIVectorStore},
    llm::{
        tools::{
            LLMTools, OpenAICodeInterpreterConfig, OpenAIFileSearchConfig, OpenAIReasoningConfig,
            OpenAIWebSearchConfig,
        },
        OpenAIModels,
    },
    Completions,
};

// Example 1: Basic translation example using reasoning model
#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct TranslationResponse {
    pub spanish: String,
    pub french: String,
    pub german: String,
    pub polish: String,
}

// Example 2: Web search
#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct AINewsArticles {
    pub articles: Vec<AINewsArticle>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct AINewsArticle {
    pub title: String,
    pub url: String,
    pub description: String,
}

// Example 3: File search
#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct ConcertInfo {
    dates: Vec<String>,
    band: String,
    genre: String,
    venue: String,
    city: String,
    country: String,
    ticket_price: String,
}

const BANDS_GENRES: &[(&str, &str)] = &[
    ("Metallica", "Metal"),
    ("The Beatles", "Rock"),
    ("Daft Punk", "Electronic"),
    ("Miles Davis", "Jazz"),
    ("Johnny Cash", "Country"),
];

// Example 4: Code interpreter example
#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct CodeInterpreterResponse {
    pub problem: String,
    pub code: String,
    pub output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // Example 1: Basic translation example using reasoning model
    let instructions =
        "Translate the following English sentence to all the languages in the response type: Rust is best for working with LLMs";

    let openai_api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let reasoning_tool = LLMTools::OpenAIReasoning(OpenAIReasoningConfig::default());

    let openai_responses = Completions::new(OpenAIModels::Gpt4_1Nano, &openai_api_key, None, None)
        .add_tool(reasoning_tool)
        .version("openai_responses");

    match openai_responses
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Translations:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 2: Web search example
    let web_search_tool = LLMTools::OpenAIWebSearch(OpenAIWebSearchConfig::new());
    let openai_responses = Completions::new(OpenAIModels::Gpt4_1Mini, &openai_api_key, None, None)
        .version("openai_responses")
        .add_tool(web_search_tool);

    match openai_responses
        .get_answer::<AINewsArticles>("Find up to 5 most recent news items about Artificial Intelligence, Generative AI, and Large Language Models. 
        For each news item, provide the title, url, and a short description.")
        .await
    {
        Ok(response) => println!("AI news articles:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 3: File search example

    // Read the concert file and upload it to OpenAI
    let path = Path::new("metallica.pdf");
    let bytes = std::fs::read(path)?;
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("Failed to extract file name"))?;
    let openai_file = OpenAIFile::new(None, &openai_api_key)
        .upload(&file_name, bytes)
        .await?;
    let openai_vector_store = OpenAIVectorStore::new(None, "Concerts", &openai_api_key)
        .upload(&[openai_file.id.clone().unwrap_or_default()])
        .await?;

    // Extract concert information using Responses API with file search tool
    let file_search_tool =
        LLMTools::OpenAIFileSearch(OpenAIFileSearchConfig::new(vec![openai_vector_store
            .id
            .clone()
            .unwrap_or_default()]));

    let openai_responses = Completions::new(OpenAIModels::Gpt4_1, &openai_api_key, None, None)
        .version("openai_responses")
        .set_context("bands_genres", &BANDS_GENRES)?
        .add_tool(file_search_tool);

    match openai_responses
        .get_answer::<ConcertInfo>("Extract the information requested in the response type from the attached concert information.
            The response should include the genre of the music the 'band' represents.
            The mapping of bands to genres was provided in 'bands_genres' list.")
        .await
    {
        Ok(response) => println!("Concert Info:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Cleanup
    openai_file.delete().await?;
    openai_vector_store.delete().await?;

    // Example 4: Code interpreter example

    let code_interpreter_tool = LLMTools::OpenAICodeInterpreter(OpenAICodeInterpreterConfig::new());
    let openai_responses = Completions::new(OpenAIModels::Gpt4_1, &openai_api_key, None, None)
        .version("openai_responses")
        .set_context("Code Interpreter", &"You are a personal math tutor. When asked a math question, write and run code to answer the question.".to_string())?
        .add_tool(code_interpreter_tool);

    match openai_responses
        .get_answer::<CodeInterpreterResponse>(
            "I need to solve the equation 3x + 11 = 14. Can you help me?",
        )
        .await
    {
        Ok(response) => println!("Code interpreter response:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    Ok(())
}
