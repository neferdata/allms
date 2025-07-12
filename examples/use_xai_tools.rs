use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{
    llm::{
        tools::{LLMTools, XAISearchSource, XAIWebSearchConfig},
        XAIModels,
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

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let xai_api_key: String = std::env::var("XAI_API_KEY").expect("XAI_API_KEY not set");

    // Example 1: Web search example with multiple sources
    let web_search_config = XAIWebSearchConfig::new()
        .add_source(
            XAISearchSource::web()
                .with_allowed_sites(vec!["techcrunch.com".to_string(), "wired.com".to_string()])
                .with_country("US".to_string())
                .with_safe_search(true),
        )
        .add_source(
            XAISearchSource::news()
                .with_excluded_sites(vec!["tabloid.com".to_string()])
                .with_country("US".to_string())
                .with_safe_search(true),
        )
        .add_source(
            XAISearchSource::x()
                .with_included_handles(vec![
                    "openai".to_string(),
                    "anthropic".to_string(),
                    "googleai".to_string(),
                ])
                .with_favorite_count(100)
                .with_view_count(1000),
        )
        .max_search_results(10)
        .return_citations(true);

    let web_search_tool = LLMTools::XAIWebSearch(web_search_config);
    let xai_responses =
        Completions::new(XAIModels::Grok3Mini, &xai_api_key, None, None).add_tool(web_search_tool);

    match xai_responses
        .get_answer::<AINewsArticles>("Find up to 5 most recent news items about Artificial Intelligence, Generative AI, and Large Language Models. 
        For each news item, provide the title, url, and a short description.")
        .await
    {
        Ok(response) => println!("AI news articles:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    Ok(())
}
