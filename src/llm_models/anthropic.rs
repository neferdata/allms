use anyhow::Result;
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::apis::AnthropicApiEndpoints;
use crate::completions::{build_citations, CompletionCitations, ThinkingLevel};
use crate::constants::{ANTHROPIC_API_URL, ANTHROPIC_MESSAGES_API_URL};
use crate::domain::{AnthropicAPICompletionsResponse, AnthropicAPIMessagesResponse};
use crate::llm_models::{
    tools::{
        AnthropicCodeExecutionConfig, AnthropicCodeExecutionToolType, AnthropicComputerUseConfig,
        AnthropicFileSearchConfig, AnthropicImageAnalysisConfig, AnthropicWebSearchConfig,
        AnthropicWebSearchToolType,
    },
    LLMModel, LLMTools,
};

// API Docs: https://docs.anthropic.com/en/docs/about-claude/models/all-models
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum AnthropicModels {
    // Claude 5.x
    ClaudeFable5_1,
    ClaudeMythos5_1,
    ClaudeFable5,
    ClaudeMythos5,
    ClaudeOpus5_5,
    ClaudeSonnet5_5,
    ClaudeHaiku5_5,
    ClaudeOpus5,
    ClaudeSonnet5,
    // Claude 4.x
    ClaudeOpus4_8,
    ClaudeOpus4_7,
    ClaudeSonnet4_6,
    ClaudeOpus4_6,
    Claude4_5Opus,
    Claude4_5Sonnet,
    Claude4_5Haiku,
    Claude4_1Opus,
    Claude4Sonnet,
    Claude4Opus,
    Claude3_7Sonnet,
    Claude3_5Sonnet,
    Claude3_5Haiku,
    Claude3Opus,
    Claude3Sonnet,
    Claude3Haiku,
    // Legacy
    Claude2,
    ClaudeInstant1_2,
}

#[async_trait(?Send)]
impl LLMModel for AnthropicModels {
    fn as_str(&self) -> &str {
        match self {
            AnthropicModels::ClaudeFable5_1 => "claude-fable-5-1",
            AnthropicModels::ClaudeMythos5_1 => "claude-mythos-5-1",
            AnthropicModels::ClaudeFable5 => "claude-fable-5",
            AnthropicModels::ClaudeMythos5 => "claude-mythos-5",
            AnthropicModels::ClaudeOpus5_5 => "claude-opus-5-5",
            AnthropicModels::ClaudeSonnet5_5 => "claude-sonnet-5-5",
            AnthropicModels::ClaudeHaiku5_5 => "claude-haiku-5-5",
            AnthropicModels::ClaudeOpus5 => "claude-opus-5",
            AnthropicModels::ClaudeSonnet5 => "claude-sonnet-5",
            AnthropicModels::ClaudeOpus4_8 => "claude-opus-4-8",
            AnthropicModels::ClaudeOpus4_7 => "claude-opus-4-7",
            AnthropicModels::ClaudeSonnet4_6 => "claude-sonnet-4-6",
            AnthropicModels::ClaudeOpus4_6 => "claude-opus-4-6",
            AnthropicModels::Claude4_5Opus => "claude-opus-4-5",
            AnthropicModels::Claude4_5Sonnet => "claude-sonnet-4-5",
            AnthropicModels::Claude4_5Haiku => "claude-haiku-4-5",
            AnthropicModels::Claude4_1Opus => "claude-opus-4-1-20250805",
            AnthropicModels::Claude4Sonnet => "claude-sonnet-4-20250514",
            AnthropicModels::Claude4Opus => "claude-opus-4-20250514",
            AnthropicModels::Claude3_7Sonnet => "claude-3-7-sonnet-latest",
            AnthropicModels::Claude3_5Sonnet => "claude-3-5-sonnet-latest",
            AnthropicModels::Claude3_5Haiku => "claude-3-5-haiku-latest",
            AnthropicModels::Claude3Opus => "claude-3-opus-latest",
            AnthropicModels::Claude3Sonnet => "claude-3-sonnet-20240229",
            AnthropicModels::Claude3Haiku => "claude-3-haiku-20240307",
            // Legacy
            AnthropicModels::Claude2 => "claude-2.1",
            AnthropicModels::ClaudeInstant1_2 => "claude-instant-1.2",
        }
    }

    // Docs: https://docs.anthropic.com/en/docs/about-claude/models/overview#model-aliases
    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "claude-fable-5-1" => Some(AnthropicModels::ClaudeFable5_1),
            "claude-mythos-5-1" => Some(AnthropicModels::ClaudeMythos5_1),
            "claude-fable-5" => Some(AnthropicModels::ClaudeFable5),
            "claude-mythos-5" => Some(AnthropicModels::ClaudeMythos5),
            "claude-opus-5-5" => Some(AnthropicModels::ClaudeOpus5_5),
            "claude-sonnet-5-5" => Some(AnthropicModels::ClaudeSonnet5_5),
            "claude-haiku-5-5" => Some(AnthropicModels::ClaudeHaiku5_5),
            "claude-opus-5" => Some(AnthropicModels::ClaudeOpus5),
            "claude-sonnet-5" => Some(AnthropicModels::ClaudeSonnet5),
            "claude-opus-4-8" => Some(AnthropicModels::ClaudeOpus4_8),
            "claude-opus-4-7" => Some(AnthropicModels::ClaudeOpus4_7),
            "claude-sonnet-4-6" => Some(AnthropicModels::ClaudeSonnet4_6),
            "claude-opus-4-6" => Some(AnthropicModels::ClaudeOpus4_6),
            "claude-opus-4-5" => Some(AnthropicModels::Claude4_5Opus),
            "claude-opus-4-5-20251101" => Some(AnthropicModels::Claude4_5Opus),
            "claude-sonnet-4-5" => Some(AnthropicModels::Claude4_5Sonnet),
            "claude-sonnet-4-5-20250929" => Some(AnthropicModels::Claude4_5Sonnet),
            "claude-haiku-4-5" => Some(AnthropicModels::Claude4_5Haiku),
            "claude-haiku-4-5-20251001" => Some(AnthropicModels::Claude4_5Haiku),
            "claude-opus-4-1-20250805" => Some(AnthropicModels::Claude4_1Opus),
            "claude-opus-4-1" => Some(AnthropicModels::Claude4_1Opus),
            "claude-sonnet-4-20250514" => Some(AnthropicModels::Claude4Sonnet),
            "claude-sonnet-4-0" => Some(AnthropicModels::Claude4Sonnet),
            "claude-opus-4-20250514" => Some(AnthropicModels::Claude4Opus),
            "claude-opus-4-0" => Some(AnthropicModels::Claude4Opus),
            "claude-3-7-sonnet-latest" => Some(AnthropicModels::Claude3_7Sonnet),
            "claude-3-5-sonnet-20240620" => Some(AnthropicModels::Claude3_5Sonnet),
            "claude-3-5-sonnet-latest" => Some(AnthropicModels::Claude3_5Sonnet),
            "claude-3-5-haiku-latest" => Some(AnthropicModels::Claude3_5Haiku),
            "claude-3-opus-20240229" => Some(AnthropicModels::Claude3Opus),
            "claude-3-opus-latest" => Some(AnthropicModels::Claude3Opus),
            "claude-3-sonnet-20240229" => Some(AnthropicModels::Claude3Sonnet),
            "claude-3-haiku-20240307" => Some(AnthropicModels::Claude3Haiku),
            // Legacy
            "claude-2.1" => Some(AnthropicModels::Claude2),
            "claude-instant-1.2" => Some(AnthropicModels::ClaudeInstant1_2),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        // This is the max tokens allowed for response and not context as per documentation: https://docs.anthropic.com/en/docs/about-claude/models/overview#model-comparison-table
        match self {
            AnthropicModels::ClaudeFable5_1
            | AnthropicModels::ClaudeMythos5_1
            | AnthropicModels::ClaudeFable5
            | AnthropicModels::ClaudeMythos5
            | AnthropicModels::ClaudeOpus5_5
            | AnthropicModels::ClaudeSonnet5_5
            | AnthropicModels::ClaudeHaiku5_5
            | AnthropicModels::ClaudeOpus5
            | AnthropicModels::ClaudeSonnet5
            | AnthropicModels::ClaudeOpus4_8
            | AnthropicModels::ClaudeOpus4_7
            | AnthropicModels::ClaudeOpus4_6 => 128_000,
            AnthropicModels::ClaudeSonnet4_6 => 64_000,
            AnthropicModels::Claude4_5Opus => 64_000,
            AnthropicModels::Claude4_5Sonnet => 64_000,
            AnthropicModels::Claude4_5Haiku => 64_000,
            AnthropicModels::Claude4_1Opus => 32_000,
            AnthropicModels::Claude4Sonnet => 64_000,
            AnthropicModels::Claude4Opus => 32_000,
            AnthropicModels::Claude3_7Sonnet => 64_000,
            AnthropicModels::Claude3_5Sonnet => 8_192,
            AnthropicModels::Claude3_5Haiku => 8_192,
            AnthropicModels::Claude3Opus => 4_096,
            AnthropicModels::Claude3Sonnet => 4_096,
            AnthropicModels::Claude3Haiku => 4_096,
            // Legacy
            AnthropicModels::Claude2 => 4_096,
            AnthropicModels::ClaudeInstant1_2 => 4_096,
        }
    }

    fn get_endpoint(&self) -> String {
        match self {
            AnthropicModels::ClaudeFable5_1
            | AnthropicModels::ClaudeMythos5_1
            | AnthropicModels::ClaudeFable5
            | AnthropicModels::ClaudeMythos5
            | AnthropicModels::ClaudeOpus5_5
            | AnthropicModels::ClaudeSonnet5_5
            | AnthropicModels::ClaudeHaiku5_5
            | AnthropicModels::ClaudeOpus5
            | AnthropicModels::ClaudeSonnet5
            | AnthropicModels::ClaudeOpus4_8
            | AnthropicModels::ClaudeOpus4_7
            | AnthropicModels::ClaudeSonnet4_6
            | AnthropicModels::ClaudeOpus4_6
            | AnthropicModels::Claude4_5Opus
            | AnthropicModels::Claude4_5Sonnet
            | AnthropicModels::Claude4_5Haiku
            | AnthropicModels::Claude4_1Opus
            | AnthropicModels::Claude4Sonnet
            | AnthropicModels::Claude4Opus
            | AnthropicModels::Claude3_7Sonnet
            | AnthropicModels::Claude3_5Sonnet
            | AnthropicModels::Claude3_5Haiku
            | AnthropicModels::Claude3Opus
            | AnthropicModels::Claude3Sonnet
            | AnthropicModels::Claude3Haiku => ANTHROPIC_MESSAGES_API_URL.to_string(),
            // Legacy
            AnthropicModels::Claude2 | AnthropicModels::ClaudeInstant1_2 => {
                ANTHROPIC_API_URL.to_string()
            }
        }
    }

    //This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        tools: Option<&[LLMTools]>,
        _thinking_level: Option<&ThinkingLevel>,
    ) -> serde_json::Value {
        let schema_string = serde_json::to_string(json_schema).unwrap_or_default();
        let base_instructions = self.get_base_instructions(Some(function_call));

        let completions_body = json!({
            "model": self.as_str(),
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "prompt": format!(
                "\n\nHuman:
                {base_instructions}\n\n
                Output Json schema:\n
                {schema_string}\n\n
                {instructions}
                \n\nAssistant:",
            ),
        });

        let base_message = json!({
            "role": "user",
            "content": base_instructions.to_string()
        });

        let user_instructions = format!(
            "<instructions>
            {instructions}
            </instructions>
            <output json schema>
            {schema_string}
            </output json schema>"
        );

        let messages = json!([
            base_message,
            {
                "role": "user",
                "content": self.messages_input_content(&user_instructions, tools),
            }
        ]);

        let mut message_body = json!({
            "model": self.as_str(),
            "max_tokens": max_tokens,
            "messages": messages,
        });

        // Starting with Claude Opus 4.7, temperature is no longer supported
        // https://platform.claude.com/docs/en/about-claude/models/migration-guide
        if self.temperature_supported() {
            message_body["temperature"] = json!(temperature);
        }

        // Add tools if provided
        if let Some(tools_inner) = tools {
            let processed_tools: Vec<Value> = tools_inner
                .iter()
                .filter(|tool| Self::is_messages_hosted_tool(tool))
                .filter(|tool| self.is_supported_tool(tool))
                .map(|tool| self.set_tool_type(tool))
                .filter_map(|tool| LLMTools::get_config_json(&tool))
                .collect::<Vec<Value>>();

            // Only add tools if the processed vector is not empty
            if !processed_tools.is_empty() {
                message_body["tools"] = json!(processed_tools);
            }
        }

        match self {
            AnthropicModels::ClaudeFable5_1
            | AnthropicModels::ClaudeMythos5_1
            | AnthropicModels::ClaudeFable5
            | AnthropicModels::ClaudeMythos5
            | AnthropicModels::ClaudeOpus5_5
            | AnthropicModels::ClaudeSonnet5_5
            | AnthropicModels::ClaudeHaiku5_5
            | AnthropicModels::ClaudeOpus5
            | AnthropicModels::ClaudeSonnet5
            | AnthropicModels::ClaudeOpus4_8
            | AnthropicModels::ClaudeOpus4_7
            | AnthropicModels::ClaudeSonnet4_6
            | AnthropicModels::ClaudeOpus4_6
            | AnthropicModels::Claude4_5Opus
            | AnthropicModels::Claude4_5Sonnet
            | AnthropicModels::Claude4_5Haiku
            | AnthropicModels::Claude4_1Opus
            | AnthropicModels::Claude4Sonnet
            | AnthropicModels::Claude4Opus
            | AnthropicModels::Claude3_7Sonnet
            | AnthropicModels::Claude3_5Sonnet
            | AnthropicModels::Claude3_5Haiku
            | AnthropicModels::Claude3Opus
            | AnthropicModels::Claude3Sonnet
            | AnthropicModels::Claude3Haiku => message_body,
            // Legacy
            AnthropicModels::Claude2 | AnthropicModels::ClaudeInstant1_2 => completions_body,
        }
    }
    /*
     * This function leverages Anthropic API to perform any query as per the provided body.
     *
     * It returns a String the Response object that needs to be parsed based on the self.model.
     */
    async fn call_api(
        &self,
        api_key: &str,
        _version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
        tools: Option<&[LLMTools]>,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_endpoint();

        if debug {
            info!("[debug] Anthropic API URL: {:#?}", model_url);
        }

        //Make the API call
        let client = Client::new();

        // Build request with base headers
        let mut request = client
            .post(model_url)
            .header(header::CONTENT_TYPE, "application/json")
            //Anthropic-specific way of passing API key
            .header("x-api-key", api_key)
            //Required as per documentation
            .header(
                "anthropic-version",
                AnthropicApiEndpoints::messages_default().version(),
            );

        // For Sonnet 4.6 and Opus 4.6 we set the beta header for 1M context window
        if matches!(
            self,
            AnthropicModels::ClaudeSonnet4_6 | AnthropicModels::ClaudeOpus4_6
        ) {
            request = request.header("anthropic-beta", "context-1m-2025-08-07");
        }

        // Add tool-specific headers
        if let Some(tools_list) = tools {
            for tool in tools_list {
                if let Some((header_name, header_value)) = self.get_tool_header(tool) {
                    request = request.header(header_name, header_value);
                }
            }
        }

        //Send request
        let response = request.json(&body).send().await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if debug {
            info!(
                "[debug] Anthropic API response: [{}] {:#?}",
                response_status, response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        //Convert API response to struct representing expected response format
        match self {
            AnthropicModels::ClaudeFable5_1
            | AnthropicModels::ClaudeMythos5_1
            | AnthropicModels::ClaudeFable5
            | AnthropicModels::ClaudeMythos5
            | AnthropicModels::ClaudeOpus5_5
            | AnthropicModels::ClaudeSonnet5_5
            | AnthropicModels::ClaudeHaiku5_5
            | AnthropicModels::ClaudeOpus5
            | AnthropicModels::ClaudeSonnet5
            | AnthropicModels::ClaudeOpus4_8
            | AnthropicModels::ClaudeOpus4_7
            | AnthropicModels::ClaudeSonnet4_6
            | AnthropicModels::ClaudeOpus4_6
            | AnthropicModels::Claude4_5Opus
            | AnthropicModels::Claude4_5Sonnet
            | AnthropicModels::Claude4_5Haiku
            | AnthropicModels::Claude4_1Opus
            | AnthropicModels::Claude4Sonnet
            | AnthropicModels::Claude4Opus
            | AnthropicModels::Claude3_7Sonnet
            | AnthropicModels::Claude3_5Sonnet
            | AnthropicModels::Claude3_5Haiku
            | AnthropicModels::Claude3Opus
            | AnthropicModels::Claude3Sonnet
            | AnthropicModels::Claude3Haiku => {
                let messages_response: AnthropicAPIMessagesResponse =
                    serde_json::from_str(response_text)?;

                let assistant_response = messages_response
                    .content
                    .iter()
                    .filter(|item| item.content_type == "text")
                    .filter_map(|item| item.text.clone())
                    // Sanitize the response to remove the json schema wrapper
                    .map(|text| self.sanitize_json_response(&text))
                    .next_back()
                    .ok_or(anyhow::anyhow!("No assistant response found"))?;

                //Return completions text
                Ok(assistant_response)
            }
            // Legacy
            AnthropicModels::Claude2 | AnthropicModels::ClaudeInstant1_2 => {
                let completions_response: AnthropicAPICompletionsResponse =
                    serde_json::from_str(response_text)?;

                //Return completions text
                Ok(completions_response.completion)
            }
        }
    }

    fn extract_citations(
        &self,
        response_text: &str,
        _version: Option<String>,
    ) -> CompletionCitations {
        let Ok(response) = serde_json::from_str::<AnthropicAPIMessagesResponse>(response_text)
        else {
            return CompletionCitations::default();
        };

        let mut candidates = Vec::new();
        // Prefer citations attached to text blocks
        for block in &response.content {
            if let Some(citations) = &block.citations {
                for citation in citations {
                    if let Some(url) = &citation.url {
                        candidates.push((url.clone(), citation.title.clone()));
                    }
                }
            }
        }
        // Also collect URLs from web_search_tool_result nested content
        for block in &response.content {
            if block.content_type == "web_search_tool_result" {
                if let Some(nested) = &block.content {
                    for item in nested {
                        if let Some(url) = &item.url {
                            candidates.push((url.clone(), item.title.clone()));
                        }
                    }
                }
            }
        }
        build_citations(candidates)
    }
}

impl AnthropicModels {
    // Docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
    pub fn get_supported_tools(&self) -> Vec<LLMTools> {
        match self {
            // Full tool set (computer_20251124-compatible models)
            AnthropicModels::ClaudeFable5_1
            | AnthropicModels::ClaudeMythos5_1
            | AnthropicModels::ClaudeFable5
            | AnthropicModels::ClaudeMythos5
            | AnthropicModels::ClaudeOpus5
            | AnthropicModels::ClaudeSonnet5
            | AnthropicModels::ClaudeOpus4_8
            | AnthropicModels::ClaudeSonnet4_6
            | AnthropicModels::ClaudeOpus4_6
            | AnthropicModels::Claude4_5Opus
            | AnthropicModels::Claude4_5Sonnet
            | AnthropicModels::Claude4_1Opus
            | AnthropicModels::Claude4Sonnet
            | AnthropicModels::Claude4Opus
            | AnthropicModels::Claude3_7Sonnet
            | AnthropicModels::Claude3_5Haiku => {
                vec![
                    LLMTools::AnthropicCodeExecution(AnthropicCodeExecutionConfig::new()),
                    LLMTools::AnthropicComputerUse(AnthropicComputerUseConfig::new(1920, 1080)),
                    LLMTools::AnthropicFileSearch(AnthropicFileSearchConfig::new("".to_string())),
                    LLMTools::AnthropicImageAnalysis(AnthropicImageAnalysisConfig::new(vec![])),
                    LLMTools::AnthropicWebSearch(AnthropicWebSearchConfig::new()),
                ]
            }
            // Claude 5.5 models require computer_toolset_20260801 (not yet modeled); omit computer use
            AnthropicModels::ClaudeOpus5_5
            | AnthropicModels::ClaudeSonnet5_5
            | AnthropicModels::ClaudeHaiku5_5 => {
                vec![
                    LLMTools::AnthropicCodeExecution(AnthropicCodeExecutionConfig::new()),
                    LLMTools::AnthropicFileSearch(AnthropicFileSearchConfig::new("".to_string())),
                    LLMTools::AnthropicImageAnalysis(AnthropicImageAnalysisConfig::new(vec![])),
                    LLMTools::AnthropicWebSearch(AnthropicWebSearchConfig::new()),
                ]
            }
            // As of 2026.04.17 Claude 4.5 Haiku and Claude Opus 4.7 do not seem to support file search
            AnthropicModels::ClaudeOpus4_7 | AnthropicModels::Claude4_5Haiku => {
                vec![
                    LLMTools::AnthropicCodeExecution(AnthropicCodeExecutionConfig::new()),
                    LLMTools::AnthropicComputerUse(AnthropicComputerUseConfig::new(1920, 1080)),
                    LLMTools::AnthropicImageAnalysis(AnthropicImageAnalysisConfig::new(vec![])),
                    LLMTools::AnthropicWebSearch(AnthropicWebSearchConfig::new()),
                ]
            }
            AnthropicModels::Claude3_5Sonnet => {
                vec![
                    LLMTools::AnthropicComputerUse(AnthropicComputerUseConfig::new(1920, 1080)),
                    LLMTools::AnthropicFileSearch(AnthropicFileSearchConfig::new("".to_string())),
                    LLMTools::AnthropicImageAnalysis(AnthropicImageAnalysisConfig::new(vec![])),
                ]
            }
            _ => vec![],
        }
    }

    /// Returns a tuple of (header_name, header_value) for a specific tool, or None if no header is needed
    pub fn get_tool_header(&self, tool: &LLMTools) -> Option<(&'static str, &'static str)> {
        match (self, tool) {
            // Web search per-model headers
            (
                AnthropicModels::ClaudeFable5_1
                | AnthropicModels::ClaudeMythos5_1
                | AnthropicModels::ClaudeFable5
                | AnthropicModels::ClaudeMythos5
                | AnthropicModels::ClaudeOpus5_5
                | AnthropicModels::ClaudeSonnet5_5
                | AnthropicModels::ClaudeHaiku5_5
                | AnthropicModels::ClaudeOpus5
                | AnthropicModels::ClaudeSonnet5
                | AnthropicModels::ClaudeOpus4_8
                | AnthropicModels::ClaudeOpus4_7
                | AnthropicModels::ClaudeSonnet4_6
                | AnthropicModels::ClaudeOpus4_6,
                LLMTools::AnthropicWebSearch(_),
            ) => Some(("anthropic-beta", "code-execution-web-tools-2026-02-09")),
            // Computer use per-model headers
            // https://docs.claude.com/en/docs/agents-and-tools/tool-use/computer-use-tool
            (
                AnthropicModels::ClaudeFable5_1
                | AnthropicModels::ClaudeMythos5_1
                | AnthropicModels::ClaudeFable5
                | AnthropicModels::ClaudeMythos5
                | AnthropicModels::ClaudeOpus5
                | AnthropicModels::ClaudeSonnet5
                | AnthropicModels::ClaudeOpus4_8
                | AnthropicModels::ClaudeOpus4_7
                | AnthropicModels::ClaudeSonnet4_6
                | AnthropicModels::ClaudeOpus4_6
                | AnthropicModels::Claude4_5Opus,
                LLMTools::AnthropicComputerUse(_),
            ) => Some(("anthropic-beta", "computer-use-2025-11-24")),
            (
                AnthropicModels::Claude4_5Sonnet
                | AnthropicModels::Claude4_5Haiku
                | AnthropicModels::Claude4_1Opus
                | AnthropicModels::Claude4Sonnet
                | AnthropicModels::Claude4Opus
                | AnthropicModels::Claude3_7Sonnet,
                LLMTools::AnthropicComputerUse(_),
            ) => Some(("anthropic-beta", "computer-use-2025-01-24")),
            (AnthropicModels::Claude3_5Sonnet, LLMTools::AnthropicComputerUse(_)) => {
                Some(("anthropic-beta", "computer-use-2024-10-22"))
            }
            // File search / image analysis per-model headers
            (
                AnthropicModels::ClaudeFable5_1
                | AnthropicModels::ClaudeMythos5_1
                | AnthropicModels::ClaudeFable5
                | AnthropicModels::ClaudeMythos5
                | AnthropicModels::ClaudeOpus5_5
                | AnthropicModels::ClaudeSonnet5_5
                | AnthropicModels::ClaudeHaiku5_5
                | AnthropicModels::ClaudeOpus5
                | AnthropicModels::ClaudeSonnet5
                | AnthropicModels::ClaudeOpus4_8
                | AnthropicModels::ClaudeOpus4_7
                | AnthropicModels::ClaudeSonnet4_6
                | AnthropicModels::ClaudeOpus4_6
                | AnthropicModels::Claude4_5Opus
                | AnthropicModels::Claude4_5Sonnet
                | AnthropicModels::Claude4_5Haiku
                | AnthropicModels::Claude4_1Opus
                | AnthropicModels::Claude4Sonnet
                | AnthropicModels::Claude4Opus
                | AnthropicModels::Claude3_7Sonnet
                | AnthropicModels::Claude3_5Sonnet
                | AnthropicModels::Claude3_5Haiku,
                LLMTools::AnthropicFileSearch(_) | LLMTools::AnthropicImageAnalysis(_),
            ) => Some((
                "anthropic-beta",
                AnthropicApiEndpoints::files_default().version_static(),
            )),
            // Code execution per-model headers
            _ => {
                // Return None for tools that don't require a header
                None
            }
        }
    }

    fn set_tool_type(&self, tool: &LLMTools) -> LLMTools {
        match (self, tool) {
            // Claude 4.6+ / 5.x use web search tool type 20260209
            (
                AnthropicModels::ClaudeFable5_1
                | AnthropicModels::ClaudeMythos5_1
                | AnthropicModels::ClaudeFable5
                | AnthropicModels::ClaudeMythos5
                | AnthropicModels::ClaudeOpus5_5
                | AnthropicModels::ClaudeSonnet5_5
                | AnthropicModels::ClaudeHaiku5_5
                | AnthropicModels::ClaudeOpus5
                | AnthropicModels::ClaudeSonnet5
                | AnthropicModels::ClaudeOpus4_8
                | AnthropicModels::ClaudeOpus4_7
                | AnthropicModels::ClaudeSonnet4_6
                | AnthropicModels::ClaudeOpus4_6,
                LLMTools::AnthropicWebSearch(config),
            ) => LLMTools::AnthropicWebSearch(
                config
                    .clone()
                    .set_type(AnthropicWebSearchToolType::WebSearch20260209),
            ),
            // Claude 4.5+ / 5.x use code execution tool type 20260120
            (
                AnthropicModels::ClaudeFable5_1
                | AnthropicModels::ClaudeMythos5_1
                | AnthropicModels::ClaudeFable5
                | AnthropicModels::ClaudeMythos5
                | AnthropicModels::ClaudeOpus5_5
                | AnthropicModels::ClaudeSonnet5_5
                | AnthropicModels::ClaudeHaiku5_5
                | AnthropicModels::ClaudeOpus5
                | AnthropicModels::ClaudeSonnet5
                | AnthropicModels::ClaudeOpus4_8
                | AnthropicModels::ClaudeOpus4_7
                | AnthropicModels::ClaudeSonnet4_6
                | AnthropicModels::ClaudeOpus4_6
                | AnthropicModels::Claude4_5Opus
                | AnthropicModels::Claude4_5Sonnet,
                LLMTools::AnthropicCodeExecution(config),
            ) => LLMTools::AnthropicCodeExecution(
                config
                    .clone()
                    .set_type(AnthropicCodeExecutionToolType::CodeExecution20260120),
            ),
            _ => tool.clone(),
        }
    }

    fn is_supported_tool(&self, tool: &LLMTools) -> bool {
        self.get_supported_tools()
            .iter()
            .any(|supported| std::mem::discriminant(tool) == std::mem::discriminant(supported))
    }

    // File search and image analysis are injected into message content, not `tools[]`.
    fn is_messages_hosted_tool(tool: &LLMTools) -> bool {
        !matches!(
            tool,
            LLMTools::AnthropicFileSearch(_) | LLMTools::AnthropicImageAnalysis(_)
        )
    }

    fn messages_input_content(&self, user_instructions: &str, tools: Option<&[LLMTools]>) -> Value {
        let mut content_parts: Vec<Value> = tools
            .unwrap_or(&[])
            .iter()
            .filter(|tool| self.is_supported_tool(tool))
            .flat_map(|tool| match tool {
                LLMTools::AnthropicFileSearch(cfg) => vec![cfg.content()],
                LLMTools::AnthropicImageAnalysis(cfg) => cfg.content(),
                _ => vec![],
            })
            .collect();

        if content_parts.is_empty() {
            json!(user_instructions)
        } else {
            content_parts.push(json!({
                "type": "text",
                "text": user_instructions,
            }));
            json!(content_parts)
        }
    }

    // Sampling params (`temperature` / `top_p` / `top_k`) are rejected on Claude Opus 4.7+
    // and all Claude 5.x models: https://platform.claude.com/docs/en/build-with-claude/thinking
    fn temperature_supported(&self) -> bool {
        !matches!(
            self,
            AnthropicModels::ClaudeFable5_1
                | AnthropicModels::ClaudeMythos5_1
                | AnthropicModels::ClaudeFable5
                | AnthropicModels::ClaudeMythos5
                | AnthropicModels::ClaudeOpus5_5
                | AnthropicModels::ClaudeSonnet5_5
                | AnthropicModels::ClaudeHaiku5_5
                | AnthropicModels::ClaudeOpus5
                | AnthropicModels::ClaudeSonnet5
                | AnthropicModels::ClaudeOpus4_8
                | AnthropicModels::ClaudeOpus4_7
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_models::LLMModel;

    const CITATION_FIXTURE: &str = r#"{
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {
                "type": "web_search_tool_result",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/from-tool",
                        "title": "From Tool"
                    }
                ]
            },
            {
                "type": "text",
                "text": "{\"result\":\"ok\"}",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://example.com/cited",
                        "title": "Cited"
                    },
                    {
                        "type": "web_search_result_location",
                        "url": "https://example.com/from-tool",
                        "title": "From Tool"
                    }
                ]
            }
        ]
    }"#;

    const NO_CITATION_FIXTURE: &str = r#"{
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [{"type": "text", "text": "{\"result\":\"ok\"}"}]
    }"#;

    #[test]
    fn extract_citations_from_text_and_tool_result() {
        let model = AnthropicModels::ClaudeSonnet4_6;
        let citations = model.extract_citations(CITATION_FIXTURE, None);
        assert_eq!(
            citations.urls,
            vec![
                "https://example.com/cited".to_string(),
                "https://example.com/from-tool".to_string(),
            ]
        );
        // text citations are collected first, then tool-result URLs (deduped)
        assert_eq!(citations.items[0].title.as_deref(), Some("Cited"));
        assert!(model
            .get_data(CITATION_FIXTURE, false)
            .unwrap()
            .contains("ok"));
    }

    #[test]
    fn extract_citations_empty_when_absent() {
        let model = AnthropicModels::ClaudeSonnet4_6;
        let citations = model.extract_citations(NO_CITATION_FIXTURE, None);
        assert!(citations.urls.is_empty());
    }
}
