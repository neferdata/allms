use anyhow::{anyhow, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use std::ffi::OsStr;
use std::path::Path;

use allms::{
    files::{AnthropicFile, LLMFiles},
    llm::{
        tools::{LLMTools, MistralCodeInterpreterConfig, MistralWebSearchConfig},
        MistralModels,
    },
    Completions,
};

// Example 1: Web search
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

// Example 2: Code interpreter example
#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct CodeInterpreterResponse {
    pub problem: String,
    pub code: String,
    pub solution: String,
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

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let mistral_api_key: String =
        std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");

    // Example 1: Web search example
    let web_search_tool = LLMTools::MistralWebSearch(MistralWebSearchConfig::new());
    let mistral_responses = Completions::new(
        MistralModels::MistralMedium3_1,
        &mistral_api_key,
        None,
        None,
    )
    .add_tool(web_search_tool);

    match mistral_responses
        .get_answer::<AINewsArticles>("Find up to 5 most recent news items about Artificial Intelligence, Generative AI, and Large Language Models. 
        For each news item, provide the title, url, and a short description.")
        .await
    {
        Ok(response) => println!("AI news articles:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 2: Code interpreter example
    let code_interpreter_tool =
        LLMTools::MistralCodeInterpreter(MistralCodeInterpreterConfig::new());
    let mistral_responses = Completions::new(
        MistralModels::MistralMedium3_1,
        &mistral_api_key,
        None,
        None,
    )
    .add_tool(code_interpreter_tool);

    match mistral_responses
        .get_answer::<CodeInterpreterResponse>(
            "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        )
        .await
    {
        Ok(response) => println!("Code interpreter response:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // // Example 3: File search example

    // // Read the concert file and upload it to Anthropic
    // let path = Path::new("metallica.pdf");
    // let bytes = std::fs::read(path)?;
    // let file_name = path
    //     .file_name()
    //     .and_then(OsStr::to_str)
    //     .map(|s| s.to_string())
    //     .ok_or_else(|| anyhow!("Failed to extract file name"))?;

    // let anthropic_file = AnthropicFile::new(None, &anthropic_api_key)
    //     .upload(&file_name, bytes)
    //     .await?;

    // // Extract concert information using Anthropic API with file search tool
    // let file_search_tool = LLMTools::AnthropicFileSearch(AnthropicFileSearchConfig::new(
    //     anthropic_file.id.clone().unwrap_or_default(),
    // ));

    // let anthropic_responses = Completions::new(
    //     AnthropicModels::Claude4_5Sonnet,
    //     &anthropic_api_key,
    //     None,
    //     None,
    // )
    // .set_context("bands_genres", &BANDS_GENRES)?
    // .add_tool(file_search_tool);

    // match anthropic_responses
    //     .get_answer::<ConcertInfo>("Extract the information requested in the response type from the attached concert information.
    //         The response should include the genre of the music the 'band' represents.
    //         The mapping of bands to genres was provided in 'bands_genres' list.")
    //     .await
    // {
    //     Ok(response) => println!("Concert Info:\n{:#?}", response),
    //     Err(e) => eprintln!("Error: {:?}", e),
    // }

    // // Cleanup
    // anthropic_file.delete().await?;

    Ok(())
}
