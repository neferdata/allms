use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{
    llm::{
        tools::{LLMTools, XAIWebSearchConfig, XAIXSearchConfig},
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

// Example 2: X Search
#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct XPosts {
    pub posts: Vec<XPost>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct XPost {
    pub author: String,
    pub content: String,
    pub url: Option<String>,
    pub date: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let xai_api_key: String = std::env::var("XAI_API_KEY").expect("XAI_API_KEY not set");

    // Example 1: Web search example with domain filters
    let web_search_config = XAIWebSearchConfig::new()
        .add_allowed_domains(&["techcrunch.com".to_string(), "wired.com".to_string()])
        .with_enable_image_understanding(true);

    let web_search_tool = LLMTools::XAIWebSearch(web_search_config);
    let xai_responses =
        Completions::new(XAIModels::Grok4_1FastNonReasoning, &xai_api_key, None, None)
            .add_tool(web_search_tool);

    match xai_responses
        .get_answer::<AINewsArticles>("Find up to 5 most recent news items about Artificial Intelligence, Generative AI, and Large Language Models. 
        For each news item, provide the title, url, and a short description.")
        .await
    {
        Ok(response) => println!("AI news articles:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Example 2: X Search example with date range and handle filters
    let x_search_config = XAIXSearchConfig::new()
        .from_date("2025-01-01".to_string())
        .to_date("2025-12-31".to_string())
        .add_allowed_x_handles(&["@elonmusk".to_string(), "@OpenAI".to_string()])
        .enable_image_understanding(true)
        .enable_video_understanding(true);

    let x_search_tool = LLMTools::XAIXSearch(x_search_config);
    let xai_responses_x =
        Completions::new(XAIModels::Grok4_1FastReasoning, &xai_api_key, None, None)
            .add_tool(x_search_tool);

    match xai_responses_x
        .get_answer::<XPosts>(
            "Find up to 5 recent posts about AI and machine learning from the specified accounts. 
        For each post, provide the author handle, content, and if available, the URL and date.",
        )
        .await
    {
        Ok(response) => println!("X posts:\n{:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    Ok(())
}
