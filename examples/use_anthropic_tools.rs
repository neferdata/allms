use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{
    llm::{
        tools::{AnthropicCodeExecutionConfig, AnthropicWebSearchConfig, LLMTools},
        AnthropicModels,
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

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let anthropic_api_key: String =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");

    // Example 1: Web search example
    let web_search_tool = LLMTools::AnthropicWebSearch(AnthropicWebSearchConfig::new());
    let anthropic_responses = Completions::new(
        AnthropicModels::Claude4Sonnet,
        &anthropic_api_key,
        None,
        None,
    )
    .add_tool(web_search_tool);

    match anthropic_responses
        .get_answer::<AINewsArticles>("Find up to 5 most recent news items about Artificial Intelligence, Generative AI, and Large Language Models. 
        For each news item, provide the title, url, and a short description.")
        .await
    {
        Ok(response) => println!("AI news articles:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 2: Code interpreter example

    let code_interpreter_tool =
        LLMTools::AnthropicCodeExecution(AnthropicCodeExecutionConfig::new());
    let anthropic_responses = Completions::new(
        AnthropicModels::Claude4Sonnet,
        &anthropic_api_key,
        None,
        None,
    )
    .add_tool(code_interpreter_tool);

    match anthropic_responses
        .get_answer::<CodeInterpreterResponse>(
            "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        )
        .await
    {
        Ok(response) => println!("Code interpreter response:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    Ok(())
}
