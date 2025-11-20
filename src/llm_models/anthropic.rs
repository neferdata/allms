use anyhow::Result;
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::apis::AnthropicApiEndpoints;
use crate::completions::ThinkingLevel;
use crate::constants::{ANTHROPIC_API_URL, ANTHROPIC_MESSAGES_API_URL};
use crate::domain::{AnthropicAPICompletionsResponse, AnthropicAPIMessagesResponse};
use crate::llm_models::{
    tools::{
        AnthropicCodeExecutionConfig, AnthropicComputerUseConfig, AnthropicFileSearchConfig,
        AnthropicWebSearchConfig,
    },
    LLMModel, LLMTools,
};

// API Docs: https://docs.anthropic.com/en/docs/about-claude/models/all-models
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum AnthropicModels {
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
            AnthropicModels::Claude4_5Sonnet
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
            "content": format!(
                "{base_instructions}"
            )
        });

        let user_instructions = format!(
            "<instructions>
            {instructions}
            </instructions>
            <output json schema>
            {schema_string}
            </output json schema>"
        );

        // The file search tool, if attached, is added to the body of the message
        // We check if the tool is added and if so use it to get the message content to be sent to the model
        let messages = if let Some(file_search_tool_config) = tools.and_then(|tools_inner| {
            tools_inner
                .iter()
                // Check if the tool is supported by the model
                .filter(|tool| {
                    self.get_supported_tools().iter().any(|supported| {
                        std::mem::discriminant(*tool) == std::mem::discriminant(supported)
                    })
                })
                // Find the file search tool
                .find(|tool| matches!(tool, LLMTools::AnthropicFileSearch(_)))
                // Extract the file search tool config
                .and_then(|tool| {
                    tool.get_config_json().and_then(|config_json| {
                        serde_json::from_value::<AnthropicFileSearchConfig>(config_json).ok()
                    })
                })
        }) {
            json!([
                base_message,
                {
                    "role": "user",
                    "content": [
                        // Use the file search tool config to get the content to be sent to the model
                        file_search_tool_config.content(),
                        {
                            "type": "text",
                            "text": user_instructions
                        }
                    ]
                }
            ])
        } else {
            json!([base_message, {
                "role": "user",
                "content": user_instructions
            }])
        };

        let mut message_body = json!({
            "model": self.as_str(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        });

        // Add tools if provided
        if let Some(tools_inner) = tools {
            let processed_tools: Vec<Value> = tools_inner
                .iter()
                // File search is handled separately
                .filter(|tool| !matches!(tool, LLMTools::AnthropicFileSearch(_)))
                .filter(|tool| {
                    self.get_supported_tools().iter().any(|supported| {
                        std::mem::discriminant(*tool) == std::mem::discriminant(supported)
                    })
                })
                .filter_map(LLMTools::get_config_json)
                .collect::<Vec<Value>>();

            // Only add tools if the processed vector is not empty
            if !processed_tools.is_empty() {
                message_body["tools"] = json!(processed_tools);
            }
        }

        match self {
            AnthropicModels::Claude4_5Sonnet
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
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        //Convert API response to struct representing expected response format
        match self {
            AnthropicModels::Claude4_5Sonnet
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
}

impl AnthropicModels {
    pub fn get_supported_tools(&self) -> Vec<LLMTools> {
        match self {
            AnthropicModels::Claude4_5Sonnet
            | AnthropicModels::Claude4_1Opus
            | AnthropicModels::Claude4Sonnet
            | AnthropicModels::Claude4Opus
            | AnthropicModels::Claude3_7Sonnet
            | AnthropicModels::Claude3_5Haiku => {
                vec![
                    LLMTools::AnthropicCodeExecution(AnthropicCodeExecutionConfig::new()),
                    LLMTools::AnthropicComputerUse(AnthropicComputerUseConfig::new(1920, 1080)),
                    LLMTools::AnthropicFileSearch(AnthropicFileSearchConfig::new("".to_string())),
                    LLMTools::AnthropicWebSearch(AnthropicWebSearchConfig::new()),
                ]
            }
            // As of 2025.10.16 Claude 4.5 Haiku does not seem to support file search
            AnthropicModels::Claude4_5Haiku => {
                vec![
                    LLMTools::AnthropicCodeExecution(AnthropicCodeExecutionConfig::new()),
                    LLMTools::AnthropicComputerUse(AnthropicComputerUseConfig::new(1920, 1080)),
                    LLMTools::AnthropicWebSearch(AnthropicWebSearchConfig::new()),
                ]
            }
            AnthropicModels::Claude3_5Sonnet => {
                vec![
                    LLMTools::AnthropicComputerUse(AnthropicComputerUseConfig::new(1920, 1080)),
                    LLMTools::AnthropicFileSearch(AnthropicFileSearchConfig::new("".to_string())),
                ]
            }
            _ => vec![],
        }
    }

    /// Returns a tuple of (header_name, header_value) for a specific tool, or None if no header is needed
    pub fn get_tool_header(&self, tool: &LLMTools) -> Option<(&'static str, &'static str)> {
        match (self, tool) {
            (
                AnthropicModels::Claude4_5Sonnet
                | AnthropicModels::Claude4_5Haiku
                | AnthropicModels::Claude4_1Opus
                | AnthropicModels::Claude4Sonnet
                | AnthropicModels::Claude4Opus
                | AnthropicModels::Claude3_7Sonnet
                | AnthropicModels::Claude3_5Haiku,
                LLMTools::AnthropicCodeExecution(_),
            ) => Some(("anthropic-beta", "code-execution-2025-08-25")),
            (
                // As of 2025.10.16 it is unclear if computer us is supported for 4.5 models
                // https://docs.claude.com/en/docs/agents-and-tools/tool-use/computer-use-tool
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
            (
                AnthropicModels::Claude4_5Sonnet
                | AnthropicModels::Claude4_1Opus
                | AnthropicModels::Claude4Sonnet
                | AnthropicModels::Claude4Opus
                | AnthropicModels::Claude3_7Sonnet
                | AnthropicModels::Claude3_5Sonnet
                | AnthropicModels::Claude3_5Haiku,
                LLMTools::AnthropicFileSearch(_),
            ) => Some((
                "anthropic-beta",
                AnthropicApiEndpoints::files_default().version_static(),
            )),
            _ => {
                // Return None for tools that don't require a header
                None
            }
        }
    }
}
