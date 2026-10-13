use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{
    llm::{
        tools::{
            AnthropicWebSearchConfig, GeminiWebSearchConfig, LLMTools, MistralWebSearchConfig,
            OpenAIWebSearchConfig, XAIWebSearchConfig,
        },
        AnthropicModels, GoogleModels, LLMModel, MistralModels, OpenAIModels, PerplexityModels,
        XAIModels,
    },
    Completions,
};

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct EuroWinnerAnswer {
    /// Short factual summary of who won (include year and final score if available).
    pub summary: String,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // Prompt asks for a factual answer without requesting source URLs in the schema,
    // so native provider citations (metadata) are the source of truth for URLs.
    let instructions = "Search the web and answer: which country won the most recent UEFA European Championship (Euro)? \
        Include the final score if available.";

    // Get answer + citations using OpenAI Responses API with web search
    let openai_api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = OpenAIModels::try_from_str("gpt-4.1-mini").unwrap_or(OpenAIModels::Gpt4_1Mini);
    println!("OpenAI model: {:#?}", model.as_str());

    let openai_completion = Completions::new(model, &openai_api_key, None, None)
        .version("openai_responses")
        .add_tool(LLMTools::OpenAIWebSearch(OpenAIWebSearchConfig::new()));

    match openai_completion
        .get_answer_with_metadata::<EuroWinnerAnswer>(instructions, None)
        .await
    {
        Ok(response) => println!("OpenAI Responses API response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer + citations using Anthropic with web search
    let anthropic_api_key: String =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let model = AnthropicModels::try_from_str("claude-sonnet-4-6")
        .unwrap_or(AnthropicModels::ClaudeSonnet4_6);
    println!("Anthropic model: {:#?}", model.as_str());

    let anthropic_completion = Completions::new(model, &anthropic_api_key, None, None)
        .add_tool(LLMTools::AnthropicWebSearch(AnthropicWebSearchConfig::new()));

    match anthropic_completion
        .get_answer_with_metadata::<EuroWinnerAnswer>(instructions, None)
        .await
    {
        Ok(response) => println!("Anthropic response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer + citations using Google Studio with web search
    let model =
        GoogleModels::try_from_str("gemini-2.5-flash").unwrap_or(GoogleModels::Gemini2_5Flash);
    println!("Google Gemini model: {:#?}", model.as_str());

    let google_token_str: String =
        std::env::var("GOOGLE_AI_STUDIO_API_KEY").expect("GOOGLE_AI_STUDIO_API_KEY not set");

    let gemini_completion = Completions::new(model, &google_token_str, None, None)
        .version("google-studio")
        .add_tool(LLMTools::GeminiWebSearch(
            GeminiWebSearchConfig::new().include_web(),
        ));

    match gemini_completion
        .get_answer_with_metadata::<EuroWinnerAnswer>(instructions, None)
        .await
    {
        Ok(response) => println!("Gemini response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer + citations using Mistral with web search
    let mistral_api_key: String =
        std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
    let model = MistralModels::try_from_str("mistral-medium-latest")
        .unwrap_or(MistralModels::MistralMedium3_1);
    println!("Mistral model: {:#?}", model.as_str());

    let mistral_completion = Completions::new(model, &mistral_api_key, None, None)
        .add_tool(LLMTools::MistralWebSearch(MistralWebSearchConfig::new()));

    match mistral_completion
        .get_answer_with_metadata::<EuroWinnerAnswer>(instructions, None)
        .await
    {
        Ok(response) => println!("Mistral response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer + citations using Perplexity (citations are built into online models)
    let model = PerplexityModels::try_from_str("sonar-pro").unwrap_or(PerplexityModels::SonarPro);
    println!("Perplexity model: {:#?}", model.as_str());

    let perplexity_token_str: String =
        std::env::var("PERPLEXITY_API_KEY").expect("PERPLEXITY_API_KEY not set");

    let perplexity_completion = Completions::new(model, &perplexity_token_str, None, None);

    match perplexity_completion
        .get_answer_with_metadata::<EuroWinnerAnswer>(instructions, None)
        .await
    {
        Ok(response) => println!("Perplexity response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer + citations using xAI Grok with web search
    let xai_api_key: String = std::env::var("XAI_API_KEY").expect("XAI_API_KEY not set");
    let model = XAIModels::try_from_str("grok-4.3").unwrap_or(XAIModels::Grok4_3);
    println!("xAI Grok model: {:#?}", model.as_str());

    let xai_completion = Completions::new(model, &xai_api_key, None, None)
        .add_tool(LLMTools::XAIWebSearch(XAIWebSearchConfig::new()));

    match xai_completion
        .get_answer_with_metadata::<EuroWinnerAnswer>(instructions, None)
        .await
    {
        Ok(response) => println!("xAI Grok response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
