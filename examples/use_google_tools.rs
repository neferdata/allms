use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{
    llm::{
        tools::{GeminiCodeInterpreterConfig, GeminiWebSearchConfig, LLMTools},
        GoogleModels,
    },
    Completions,
};

mod utils;
use utils::get_vertex_token;

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

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let google_api_key: String =
        std::env::var("GOOGLE_AI_STUDIO_API_KEY").expect("GOOGLE_AI_STUDIO_API_KEY not set");
    let vertex_token = get_vertex_token().await?;

    // Example 1A: Web search example (with Studio API)
    let web_search_config =
        GeminiWebSearchConfig::new().add_source("https://www.artificialintelligence-news.com/");

    let web_search_tool = LLMTools::GeminiWebSearch(web_search_config);
    let google_responses =
        Completions::new(GoogleModels::Gemini2_5Flash, &google_api_key, None, None)
            .add_tool(web_search_tool.clone());

    match google_responses
        .get_answer::<AINewsArticles>("Find up to 5 most recent news items about Artificial Intelligence, Generative AI, and Large Language Models. 
        For each news item, provide the title, url, and a short description.")
        .await
    {
        Ok(response) => println!("AI news articles:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 1B: Web search example (with Vertex API)
    let google_responses_vertex =
        Completions::new(GoogleModels::Gemini2_5Flash, &vertex_token, None, None)
            .add_tool(web_search_tool)
            .version("google-vertex");

    match google_responses_vertex
        .get_answer::<AINewsArticles>("Find up to 5 most recent news items about Artificial Intelligence, Generative AI, and Large Language Models. 
        For each news item, provide the title, url, and a short description.")
        .await
    {
        Ok(response) => println!("Vertex AI news articles:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 2A: Code interpreter example (with Studio API)
    let code_interpreter_tool = LLMTools::GeminiCodeInterpreter(GeminiCodeInterpreterConfig::new());
    let google_responses =
        Completions::new(GoogleModels::Gemini2_5Pro, &google_api_key, None, None)
            .add_tool(code_interpreter_tool.clone());

    match google_responses
        .get_answer::<CodeInterpreterResponse>(
            "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        )
        .await
    {
        Ok(response) => println!("Code interpreter response:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 2B: Code interpreter example (with Vertex API)
    let google_responses_vertex =
        Completions::new(GoogleModels::Gemini2_5Pro, &vertex_token, None, None)
            .add_tool(code_interpreter_tool)
            .version("google-vertex");

    match google_responses_vertex
        .get_answer::<CodeInterpreterResponse>(
            "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        )
        .await
    {
        Ok(response) => println!("Vertex code interpreter response:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    Ok(())
}
