use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::constants::{MISTRAL_API_URL, MISTRAL_CONVERSATIONS_API_URL};
use crate::domain::{
    MistralAPICompletionsResponse, MistralAPIConversationsChunk,
    MistralAPIConversationsMessageOutputContent, MistralAPIConversationsOutput,
    MistralAPIConversationsResponse, RateLimit,
};
use crate::llm_models::{
    tools::{MistralCodeInterpreterConfig, MistralWebSearchConfig},
    LLMModel, LLMTools,
};
use crate::utils::has_values;

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
//Mistral docs: https://docs.mistral.ai/getting-started/models/models_overview/
pub enum MistralModels {
    // Frontier multimodal models
    MistralLarge2_1,
    MistralMedium3_1,
    MistralMedium3,
    MistralSmall3_2,
    MistralSmall3_1,
    MistralSmall3,
    MistralSmall2,
    // Frontier reasoning models
    MagistralMedium1_2,
    MagistralMedium,
    MagistralSmall1_2,
    // Other frontier models
    Codestral2508,
    Codestral2,
    Ministral3B,
    Ministral8B,
    // Legacy models
    MistralLarge,
    MistralNemo,
    Mistral7B,
    Mixtral8x7B,
    Mixtral8x22B,
    MistralTiny,
    MistralSmall,
    MistralMedium,
}

#[async_trait(?Send)]
impl LLMModel for MistralModels {
    fn as_str(&self) -> &str {
        match self {
            // Frontier multimodal models
            MistralModels::MistralLarge2_1 => "mistral-large-latest",
            MistralModels::MistralMedium3_1 => "mistral-medium-latest",
            MistralModels::MistralMedium3 => "mistral-medium-2505",
            MistralModels::MistralSmall3_2 => "mistral-small-latest",
            MistralModels::MistralSmall3_1 => "mistral-small-2503",
            MistralModels::MistralSmall3 => "mistral-small-2501",
            MistralModels::MistralSmall2 => "mistral-small-2407",
            // Frontier reasoning models
            MistralModels::MagistralMedium1_2 => "magistral-medium-latest",
            MistralModels::MagistralMedium => "magistral-medium-2506",
            MistralModels::MagistralSmall1_2 => "magistral-small-latest",
            // Other frontier models
            MistralModels::Codestral2508 => "codestral-2508",
            MistralModels::Codestral2 => "codestral-2501",
            MistralModels::Ministral3B => "ministral-3b-2410",
            MistralModels::Ministral8B => "ministral-8b-2410",
            // Legacy
            MistralModels::MistralLarge => "mistral-large-latest",
            MistralModels::MistralNemo => "open-mistral-nemo",
            MistralModels::Mistral7B => "open-mistral-7b",
            MistralModels::Mixtral8x7B => "open-mixtral-8x7b",
            MistralModels::Mixtral8x22B => "open-mixtral-8x22b",
            MistralModels::MistralTiny => "mistral-tiny",
            MistralModels::MistralSmall => "mistral-small",
            MistralModels::MistralMedium => "mistral-medium",
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            // Frontier multimodal models
            "mistral-large-latest" => Some(MistralModels::MistralLarge2_1),
            "mistral-large-2411" => Some(MistralModels::MistralLarge2_1),
            "mistral-medium-latest" => Some(MistralModels::MistralMedium3_1),
            "mistral-medium-2508" => Some(MistralModels::MistralMedium3_1),
            "mistral-medium-2505" => Some(MistralModels::MistralMedium3),
            "mistral-small-latest" => Some(MistralModels::MistralSmall3_2),
            "mistral-small-2506" => Some(MistralModels::MistralSmall3_2),
            "mistral-small-2503" => Some(MistralModels::MistralSmall3_1),
            "mistral-small-2501" => Some(MistralModels::MistralSmall3),
            "mistral-small-2407" => Some(MistralModels::MistralSmall2),
            // Frontier reasoning models
            "magistral-medium-latest" => Some(MistralModels::MagistralMedium1_2),
            "magistral-medium-2506" => Some(MistralModels::MagistralMedium),
            "magistral-small-latest" => Some(MistralModels::MagistralSmall1_2),
            // Other frontier models
            "codestral-latest" => Some(MistralModels::Codestral2508),
            "codestral-2508" => Some(MistralModels::Codestral2508),
            "codestral-2501" => Some(MistralModels::Codestral2),
            "ministral-3b-2410" => Some(MistralModels::Ministral3B),
            "ministral-3b-latest" => Some(MistralModels::Ministral3B),
            "ministral-8b-2410" => Some(MistralModels::Ministral8B),
            "ministral-8b-latest" => Some(MistralModels::Ministral8B),
            // Legacy
            "open-mistral-nemo" => Some(MistralModels::MistralNemo),
            "open-mistral-7b" => Some(MistralModels::Mistral7B),
            "open-mixtral-8x7b" => Some(MistralModels::Mixtral8x7B),
            "open-mixtral-8x22b" => Some(MistralModels::Mixtral8x22B),
            "mistral-tiny" => Some(MistralModels::MistralTiny),
            "mistral-small" => Some(MistralModels::MistralSmall),
            "mistral-medium" => Some(MistralModels::MistralMedium),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        match self {
            // Frontier multimodal models
            MistralModels::MistralLarge2_1 => 128_000,
            MistralModels::MistralMedium3_1 => 128_000,
            MistralModels::MistralMedium3 => 128_000,
            MistralModels::MistralSmall3_2 => 128_000,
            MistralModels::MistralSmall3_1 => 128_000,
            MistralModels::MistralSmall3 => 32_000,
            MistralModels::MistralSmall2 => 32_000,
            // Frontier reasoning models
            MistralModels::MagistralMedium1_2 => 128_000,
            MistralModels::MagistralMedium => 128_000,
            MistralModels::MagistralSmall1_2 => 128_000,
            // Other frontier models
            MistralModels::Codestral2508 => 256_000,
            MistralModels::Codestral2 => 256_000,
            MistralModels::Ministral3B => 128_000,
            MistralModels::Ministral8B => 128_000,
            // Legacy
            MistralModels::MistralLarge => 128_000,
            MistralModels::MistralNemo => 128_000,
            MistralModels::Mistral7B => 32_000,
            MistralModels::Mixtral8x7B => 32_000,
            MistralModels::Mixtral8x22B => 64_000,
            MistralModels::MistralTiny => 32_000,
            MistralModels::MistralSmall => 32_000,
            MistralModels::MistralMedium => 32_000,
        }
    }

    fn get_endpoint(&self) -> String {
        MISTRAL_API_URL.to_string()
    }

    // This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        tools: Option<&[LLMTools]>,
    ) -> serde_json::Value {
        if has_values(tools) {
            self.get_conversations_body(
                instructions,
                json_schema,
                function_call,
                max_tokens,
                temperature,
                tools,
            )
        } else {
            self.get_chat_completions_body(
                instructions,
                json_schema,
                function_call,
                max_tokens,
                temperature,
                tools,
            )
        }
    }
    /*
     * This function leverages Mistral API to perform any query as per the provided body.
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
        let model_url = if has_values(tools) {
            // If tools are provided we need to use the conversations API
            MISTRAL_CONVERSATIONS_API_URL.to_string()
        } else {
            self.get_endpoint()
        };

        //Make the API call
        let client = Client::new();

        //Send request
        let response = client
            .post(model_url)
            .header(header::CONTENT_TYPE, "application/json")
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if debug {
            info!(
                "[debug] Mistral API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        if let Ok(data) = self.get_chat_completions_data(response_text, _function_call) {
            Ok(data)
        } else {
            self.get_conversations_data(response_text, _function_call)
        }
    }

    //This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        //Mistral documentation: https://docs.mistral.ai/platform/pricing#rate-limits
        RateLimit {
            tpm: 2_000_000,
            rpm: 360, // 6 requests per second
        }
    }
}

impl MistralModels {
    fn get_supported_tools(&self) -> Vec<LLMTools> {
        vec![
            LLMTools::MistralWebSearch(MistralWebSearchConfig::new()),
            LLMTools::MistralCodeInterpreter(MistralCodeInterpreterConfig::new()),
        ]
    }

    fn get_chat_completions_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        _tools: Option<&[LLMTools]>,
    ) -> serde_json::Value {
        //Prepare the 'messages' part of the body
        let base_instructions = self.get_base_instructions(Some(function_call));
        let system_message = json!({
            "role": "system",
            "content": base_instructions,
        });
        let user_message = json!({
            "role": "user",
            "content": format!(
                "<instructions>
                {instructions}
                </instructions>
                <output json schema>
                {json_schema}
                </output json schema>"
            ),
        });
        json!({
            "model": self.as_str(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": vec![
                system_message,
                user_message,
            ],
        })
    }

    fn get_conversations_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        tools: Option<&[LLMTools]>,
    ) -> serde_json::Value {
        // Prepare the inputs part of the body
        let base_instructions = self.get_base_instructions(Some(function_call));
        let inputs = format!(
            "{base_instructions}
            <instructions>
            {instructions}
            </instructions>
            <output json schema>
            {json_schema}
            </output json schema>"
        );
        // Prepare the completion args part of the body
        let completion_args = json!({
            "max_tokens": max_tokens,
            "temperature": temperature,
        });
        // Prepare the tools part of the body
        let tools = if let Some(tools) = tools {
            // Filter out the tools that are not supported by the model
            let supported_tools = self.get_supported_tools();
            let tools = tools
                .iter()
                .filter(|tool| supported_tools.contains(tool))
                .filter_map(|tool| tool.get_config_json())
                .collect::<Vec<Value>>();

            // If no tools are supported return None
            if tools.is_empty() {
                None
            } else {
                Some(tools)
            }
        } else {
            None
        };

        // Prepare and return the body
        json!({
            "model": self.as_str(),
            "inputs": inputs,
            "completion_args": completion_args,
            "tools": tools,
        })
    }

    fn get_chat_completions_data(
        &self,
        response_text: &str,
        _function_call: bool,
    ) -> Result<String> {
        //Convert API response to struct representing expected response format
        let completions_response: MistralAPICompletionsResponse =
            serde_json::from_str(response_text)?;

        //Parse the response and return the assistant content
        completions_response
            .choices
            .iter()
            .filter_map(|choice| choice.message.as_ref())
            .find(|&message| message.role == Some("assistant".to_string()))
            .and_then(|message| {
                message
                    .content
                    .as_ref()
                    .map(|content| self.sanitize_json_response(content))
            })
            .ok_or_else(|| anyhow!("Assistant role content not found"))
    }

    fn get_conversations_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        //Convert API response to struct representing expected response format
        let conversations_response: MistralAPIConversationsResponse =
            serde_json::from_str(response_text)?;

        //Parse the response and return the assistant content
        let content_text = conversations_response
            .outputs
            .iter()
            .find_map(|output| {
                if let MistralAPIConversationsOutput::MistralAPIConversationsMessageOutput(message_output) = output {
                    message_output.content.as_ref().map(|content| {
                        match content {
                            MistralAPIConversationsMessageOutputContent::MistralAPIConversationsMessageOutputContentString(s) => s.clone(),
                            MistralAPIConversationsMessageOutputContent::MistralAPIConversationsMessageOutputContentChunks(chunks) => {
                                chunks
                                    .iter()
                                    .filter_map(|chunk| {
                                        match chunk {
                                            MistralAPIConversationsChunk::MistralAPIConversationsChunkText(text_chunk) => {
                                                Some(text_chunk.text.clone())
                                            }
                                        }
                                    })
                                    .collect::<Vec<String>>()
                                    .join("")
                            }
                        }
                    })
                } else {
                    None
                }
            })
            .ok_or_else(|| anyhow!("Message output content not found"))?;

        Ok(self.sanitize_json_response(&content_text))
    }
}
