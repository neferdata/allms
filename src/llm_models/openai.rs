#![allow(deprecated)]

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, to_value, Value};

use crate::{
    apis::OpenAiApiEndpoints,
    completions::ThinkingLevel,
    constants::{OPENAI_API_URL, OPENAI_BASE_INSTRUCTIONS, OPENAI_FUNCTION_INSTRUCTIONS},
    domain::{
        OpenAPIChatResponse, OpenAPICompletionsResponse, OpenAPIResponsesContentType,
        OpenAPIResponsesOutputType, OpenAPIResponsesResponse, OpenAPIResponsesRole, RateLimit,
    },
    llm_models::{
        tools::{
            OpenAICodeInterpreterConfig, OpenAIComputerUseConfig, OpenAIFileSearchConfig,
            OpenAIWebSearchConfig,
        },
        LLMModel, LLMTools,
    },
    utils::{map_to_range, remove_json_wrapper, remove_schema_wrappers},
};

// Docs: https://platform.openai.com/docs/models
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIModels {
    // GPT models
    Gpt5_2,
    Gpt5_2Pro,
    Gpt5_1,
    Gpt5,
    Gpt5Mini,
    Gpt5Nano,
    Gpt4_5Preview,
    Gpt4oMini,
    Gpt4_1,
    Gpt4_1Mini,
    Gpt4_1Nano,
    Gpt4o20240806,
    Gpt4o,
    Gpt4TurboPreview,
    Gpt4Turbo,
    Gpt4_32k,
    Gpt4,
    Gpt3_5Turbo16k,
    Gpt3_5Turbo0613,
    Gpt3_5Turbo,
    // Reasoning models
    O4Mini,
    O3,
    O3Mini,
    O1Pro,
    O1,
    O1Preview, // Deprecated
    O1Mini,    // Deprecated
    // Custom models
    Custom { name: String },
    // Legacy models
    TextDavinci003,
}

#[async_trait(?Send)]
impl LLMModel for OpenAIModels {
    fn as_str(&self) -> &str {
        match self {
            OpenAIModels::Gpt5_2 => "gpt-5.2",
            OpenAIModels::Gpt5_2Pro => "gpt-5.2-pro",
            OpenAIModels::Gpt5_1 => "gpt-5.1",
            OpenAIModels::Gpt5 => "gpt-5",
            OpenAIModels::Gpt5Mini => "gpt-5-mini",
            OpenAIModels::Gpt5Nano => "gpt-5-nano",
            OpenAIModels::Gpt3_5Turbo => "gpt-3.5-turbo",
            OpenAIModels::Gpt3_5Turbo0613 => "gpt-3.5-turbo-0613",
            OpenAIModels::Gpt3_5Turbo16k => "gpt-3.5-turbo-16k",
            OpenAIModels::Gpt4 => "gpt-4",
            OpenAIModels::Gpt4_32k => "gpt-4-32k",
            OpenAIModels::TextDavinci003 => "text-davinci-003",
            OpenAIModels::Gpt4Turbo => "gpt-4-turbo",
            OpenAIModels::Gpt4TurboPreview => "gpt-4-turbo-preview",
            OpenAIModels::Gpt4o => "gpt-4o",
            OpenAIModels::Gpt4o20240806 => "gpt-4o-2024-08-06",
            OpenAIModels::Gpt4oMini => "gpt-4o-mini",
            OpenAIModels::Gpt4_1 => "gpt-4.1",
            OpenAIModels::Gpt4_1Mini => "gpt-4.1-mini",
            OpenAIModels::Gpt4_1Nano => "gpt-4.1-nano",
            OpenAIModels::Gpt4_5Preview => "gpt-4.5-preview",
            OpenAIModels::O1Preview => "o1-preview",
            OpenAIModels::O1Mini => "o1-mini",
            OpenAIModels::O1 => "o1",
            OpenAIModels::O1Pro => "o1-pro",
            OpenAIModels::O3 => "o3",
            OpenAIModels::O3Mini => "o3-mini",
            OpenAIModels::O4Mini => "o4-mini",
            OpenAIModels::Custom { name } => name.as_str(),
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "gpt-5.2" => Some(OpenAIModels::Gpt5_2),
            "gpt-5.2-2025-12-11" => Some(OpenAIModels::Gpt5_2),
            "gpt-5.2-pro" => Some(OpenAIModels::Gpt5_2Pro),
            "gpt-5.2-pro-2025-12-11" => Some(OpenAIModels::Gpt5_2Pro),
            "gpt-5.1" => Some(OpenAIModels::Gpt5_1),
            "gpt-5.1-2025-11-13" => Some(OpenAIModels::Gpt5_1),
            "gpt-5" => Some(OpenAIModels::Gpt5),
            "gpt-5-2025-08-07" => Some(OpenAIModels::Gpt5),
            "gpt-5-mini" => Some(OpenAIModels::Gpt5Mini),
            "gpt-5-mini-2025-08-07" => Some(OpenAIModels::Gpt5Mini),
            "gpt-5-nano" => Some(OpenAIModels::Gpt5Nano),
            "gpt-5-nano-2025-08-07" => Some(OpenAIModels::Gpt5Nano),
            "gpt-3.5-turbo" => Some(OpenAIModels::Gpt3_5Turbo),
            "gpt-3.5-turbo-0613" => Some(OpenAIModels::Gpt3_5Turbo0613),
            "gpt-3.5-turbo-16k" => Some(OpenAIModels::Gpt3_5Turbo16k),
            "gpt-4" => Some(OpenAIModels::Gpt4),
            "gpt-4-32k" => Some(OpenAIModels::Gpt4_32k),
            "text-davinci-003" => Some(OpenAIModels::TextDavinci003),
            "gpt-4-turbo" => Some(OpenAIModels::Gpt4Turbo),
            "gpt-4-turbo-preview" => Some(OpenAIModels::Gpt4TurboPreview),
            "gpt-4o" => Some(OpenAIModels::Gpt4o),
            "gpt-4o-2024-08-06" => Some(OpenAIModels::Gpt4o20240806),
            "gpt-4o-mini" => Some(OpenAIModels::Gpt4oMini),
            "gpt-4.1" => Some(OpenAIModels::Gpt4_1),
            "gpt-4.1-mini" => Some(OpenAIModels::Gpt4_1Mini),
            "gpt-4.1-nano" => Some(OpenAIModels::Gpt4_1Nano),
            "gpt-4.5-preview" => Some(OpenAIModels::Gpt4_5Preview),
            "o1-preview" => Some(OpenAIModels::O1Preview),
            "o1-mini" => Some(OpenAIModels::O1Mini),
            "o1" => Some(OpenAIModels::O1),
            "o1-pro" => Some(OpenAIModels::O1Pro),
            "o3" => Some(OpenAIModels::O3),
            "o3-mini" => Some(OpenAIModels::O3Mini),
            "o4-mini" => Some(OpenAIModels::O4Mini),
            _ => Some(OpenAIModels::Custom {
                name: name.to_string(),
            }),
        }
    }

    fn default_max_tokens(&self) -> usize {
        //OpenAI documentation: https://platform.openai.com/docs/models/gpt-3-5
        //This is the max tokens allowed between prompt & response
        match self {
            OpenAIModels::Gpt5_2 => 400_000,
            OpenAIModels::Gpt5_2Pro => 400_000,
            OpenAIModels::Gpt5_1 => 400_000,
            OpenAIModels::Gpt5 => 400_000,
            OpenAIModels::Gpt5Mini => 400_000,
            OpenAIModels::Gpt5Nano => 400_000,
            OpenAIModels::Gpt3_5Turbo => 4096,
            OpenAIModels::Gpt3_5Turbo0613 => 4096,
            OpenAIModels::Gpt3_5Turbo16k => 16384,
            OpenAIModels::Gpt4 => 8192,
            OpenAIModels::Gpt4_32k => 32768,
            OpenAIModels::TextDavinci003 => 4097,
            OpenAIModels::Gpt4Turbo => 128_000,
            OpenAIModels::Gpt4TurboPreview => 128_000,
            OpenAIModels::Gpt4o => 128_000,
            OpenAIModels::Gpt4o20240806 => 128_000,
            OpenAIModels::Gpt4oMini => 128_000,
            OpenAIModels::Gpt4_1 => 1_047_576,
            OpenAIModels::Gpt4_1Mini => 1_047_576,
            OpenAIModels::Gpt4_1Nano => 1_047_576,
            OpenAIModels::Gpt4_5Preview => 128_000,
            OpenAIModels::O1Preview => 128_000,
            OpenAIModels::O1Mini => 128_000,
            OpenAIModels::O1 => 200_000,
            OpenAIModels::O1Pro => 200_000,
            OpenAIModels::O3 => 200_000,
            OpenAIModels::O3Mini => 200_000,
            OpenAIModels::O4Mini => 200_000,
            OpenAIModels::Custom { .. } => 128_000,
        }
    }

    fn get_version_endpoint(&self, version: Option<String>) -> String {
        // If no version provided default to OpenAI Completions API
        let version = version
            .map(|version| OpenAiApiEndpoints::from_str(&version))
            .unwrap_or_default();

        //OpenAI documentation: https://platform.openai.com/docs/models/model-endpoint-compatibility
        match (version, self) {
            #[allow(deprecated)]
            (
                OpenAiApiEndpoints::OpenAI | OpenAiApiEndpoints::OpenAICompletions,
                OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Custom { .. },
            ) => {
                format!(
                    "{OPENAI_API_URL}/v1/chat/completions",
                    OPENAI_API_URL = *OPENAI_API_URL
                )
            }
            (
                OpenAiApiEndpoints::OpenAIResponses,
                OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Gpt5_2Pro
                | OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O1Pro
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini
                | OpenAIModels::Custom { .. },
            )
            // o1 Pro or GPT-5.2 Pro is not supported in Completions API, we redirect to Responses API
            | (
                OpenAiApiEndpoints::OpenAI | OpenAiApiEndpoints::OpenAICompletions,
                OpenAIModels::O1Pro | OpenAIModels::Gpt5_2Pro,
            ) => {
                format!(
                    "{OPENAI_API_URL}/v1/responses",
                    OPENAI_API_URL = *OPENAI_API_URL
                )
            }
            #[allow(deprecated)]
            (
                OpenAiApiEndpoints::OpenAI
                | OpenAiApiEndpoints::OpenAICompletions
                | OpenAiApiEndpoints::OpenAIResponses,
                OpenAIModels::TextDavinci003,
            ) => format!(
                "{OPENAI_API_URL}/v1/completions",
                OPENAI_API_URL = *OPENAI_API_URL
            ),
            #[allow(deprecated)]
            (
                OpenAiApiEndpoints::Azure { version }
                | OpenAiApiEndpoints::AzureCompletions { version },
                OpenAIModels::TextDavinci003
                | OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini
                | OpenAIModels::Custom { .. },
            ) => {
                format!(
                    "{}/openai/deployments/{}/chat/completions?api-version={}",
                    &*OPENAI_API_URL,
                    self.as_str(),
                    version
                )
            }
            (
                OpenAiApiEndpoints::AzureResponses { version },
                OpenAIModels::TextDavinci003
                | OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Gpt5_2Pro
                | OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O1Pro
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini
                | OpenAIModels::Custom { .. },
            )
            // o1 Pro and Gpt-5.2 Pro are not supported in Completions API, we redirect to Responses API
            | (
                OpenAiApiEndpoints::Azure { version }
                | OpenAiApiEndpoints::AzureCompletions { version },
                OpenAIModels::O1Pro | OpenAIModels::Gpt5_2Pro,
            ) => {
                format!(
                    "{}/openai/deployments/{}/responses?api-version={}",
                    &*OPENAI_API_URL,
                    self.as_str(),
                    version
                )
            }
        }
    }

    fn get_base_instructions(&self, function_call: Option<bool>) -> String {
        let function_call = function_call.unwrap_or_else(|| self.function_call_default());
        match function_call {
            true => OPENAI_FUNCTION_INSTRUCTIONS.to_string(),
            false => OPENAI_BASE_INSTRUCTIONS.to_string(),
        }
    }

    fn function_call_default(&self) -> bool {
        //OpenAI documentation: https://platform.openai.com/docs/guides/gpt/function-calling
        match self {
            OpenAIModels::TextDavinci003
            | OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt4_32k
            | OpenAIModels::O1Preview
            | OpenAIModels::O1
            | OpenAIModels::O1Mini
            | OpenAIModels::O1Pro
            | OpenAIModels::O3
            | OpenAIModels::O3Mini
            | OpenAIModels::O4Mini => false,
            OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4TurboPreview
            | OpenAIModels::Gpt4o
            | OpenAIModels::Gpt4o20240806
            | OpenAIModels::Gpt4oMini
            | OpenAIModels::Gpt4_1
            | OpenAIModels::Gpt4_1Mini
            | OpenAIModels::Gpt4_1Nano
            | OpenAIModels::Gpt4_5Preview
            | OpenAIModels::Gpt5
            | OpenAIModels::Gpt5Mini
            | OpenAIModels::Gpt5Nano
            | OpenAIModels::Gpt5_1
            | OpenAIModels::Gpt5_2
            | OpenAIModels::Gpt5_2Pro
            | OpenAIModels::Custom { .. } => true,
        }
    }

    //This method prepares the body of the API call for different models
    #[allow(clippy::too_many_arguments)]
    fn get_version_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        version: Option<String>,
        tools: Option<&[LLMTools]>,
        _thinking_level: Option<&ThinkingLevel>,
    ) -> serde_json::Value {
        // If no version provided default to OpenAI Completions API
        let version = version
            .map(|version| OpenAiApiEndpoints::from_str(&version))
            .unwrap_or_default();

        // Get the base instructions
        let base_instructions = self.get_base_instructions(Some(function_call));

        // Build the main user message
        let user_message_str = format!(
            "<instructions>
            {instructions}
            </instructions>
            <output json schema>
            {json_schema}
            </output json schema>"
        );

        match (version, self) {
            // Chat Completions API Body
            // Docs: https://platform.openai.com/docs/api-reference/completions/create
            #[allow(deprecated)]
            (
                OpenAiApiEndpoints::OpenAI
                | OpenAiApiEndpoints::OpenAICompletions
                | OpenAiApiEndpoints::Azure { .. }
                | OpenAiApiEndpoints::AzureCompletions { .. },
                OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Gpt5_2Pro
                | OpenAIModels::Custom { .. },
            ) => {
                let system_message = json!({
                    "role": "system",
                    "content": base_instructions,
                });

                let mut json_body = match function_call {
                    //If we choose to use function calling
                    //https://platform.openai.com/docs/guides/gpt/function-calling
                    true => {
                        let user_message = json!({
                            "role": "user",
                            "content": instructions,
                        });

                        let function = json!({
                            "name": "analyze_data",
                            "description": "Use this function to compute the answer based on input data, instructions and your language model. Output should be a fully formed JSON object.",
                            "parameters": json_schema,
                        });

                        let function_call = json!({
                            "name": "analyze_data"
                        });

                        //For ChatGPT we ignore max_tokens. It will default to 'inf'
                        json!({
                            "model": self.as_str(),
                            "messages": vec![
                                system_message,
                                user_message,
                            ],
                            "functions": vec![
                                function,
                            ],
                            //This forces ChatGPT to use the function definition
                            "function_call": function_call,
                        })
                    }
                    //https://platform.openai.com/docs/guides/chat/introduction
                    false => {
                        let user_message = json!({
                            "role": "user",
                            "content": user_message_str,
                        });

                        json!({
                            "model": self.as_str(),
                            "messages": vec![
                                system_message,
                                user_message,
                            ],
                        })
                    }
                };
                // For models other than GPT-5, add temperature (GPT-5 models don't support `temperature`)
                if !self.is_gpt5_model() {
                    json_body["temperature"] = json!(temperature);
                }
                json_body
            }
            // Review https://platform.openai.com/docs/guides/reasoning for beta limitations:
            // - Message types: user and assistant messages only, system messages are not supported.
            // - Tools: tools, function calling, and response format parameters are not supported.
            // - Other: temperature, top_p and n are fixed at 1, while presence_penalty and frequency_penalty are fixed at 0.
            // - Assistants and Batch: these models are not supported in the Assistants API or Batch API.
            #[allow(deprecated)]
            (
                OpenAiApiEndpoints::OpenAI
                | OpenAiApiEndpoints::OpenAICompletions
                | OpenAiApiEndpoints::Azure { .. }
                | OpenAiApiEndpoints::AzureCompletions { .. },
                OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini,
            ) => {
                let system_message = json!({
                    "role": "user",
                    "content": base_instructions,
                });

                let user_message = json!({
                    "role": "user",
                    "content": user_message_str,
                });
                json!({
                    "model": self.as_str(),
                    "messages": vec![
                        system_message,
                        user_message,
                    ],
                })
            }
            // Responses API Body
            // Docs: https://platform.openai.com/docs/api-reference/responses/create
            (
                OpenAiApiEndpoints::OpenAIResponses | OpenAiApiEndpoints::AzureResponses { .. },
                OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Gpt5_2Pro
                | OpenAIModels::Custom { .. },
            ) => {
                json!({
                    "model": self.as_str(),
                    "input": user_message_str,
                    "instructions": base_instructions,
                    "max_output_tokens": max_tokens,
                    // GPT-5 models don't support `temperature`
                    "temperature": if self.is_gpt5_model() { json!(null) } else { json!(temperature) },
                    // If tools are provided we add them to the body
                    "tools": tools.map(|tools_inner| tools_inner
                        .iter()
                        .filter(|tool| !matches!(tool, LLMTools::OpenAIReasoning(_)))
                        .filter(|tool| self.get_supported_tools().iter().any(|supported| std::mem::discriminant(*tool) == std::mem::discriminant(supported)))
                        .filter_map(LLMTools::get_config_json)
                        .collect::<Vec<Value>>()
                    ),
                    // TODO: Other fields to be implemented in the future
                    // Structured Outputs Docs: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#how-to-use
                    // "text": {
                    //     "format": {
                    //         "type": "json_schema",
                    //         "name": "output",
                    //         "schema": json_schema,
                    //         "strict": true
                    //     }
                    // } - Structured Outputs is rejecting the json schema auto-generated from T
                    // "previous_response_id" - to implement chained conversations
                })
            }
            // Reasoning models cannot use tools or set temperature in Responses API as well
            (
                OpenAiApiEndpoints::OpenAIResponses | OpenAiApiEndpoints::AzureResponses { .. },
                OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O1Pro
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini,
            )
            // o1 Pro is not supported in Completions API, we use the Responses API body
            | (
                OpenAiApiEndpoints::OpenAI
                | OpenAiApiEndpoints::OpenAICompletions
                | OpenAiApiEndpoints::Azure { .. }
                | OpenAiApiEndpoints::AzureCompletions { .. },
                OpenAIModels::O1Pro,
            ) => {
                // Check if reasoning configuration is provided as a tool
                let reasoning_opt = tools.and_then(|tools_inner| {
                    tools_inner.iter().find_map(|tool| {
                        if let LLMTools::OpenAIReasoning(cfg) = tool {
                            to_value(cfg).ok()
                        } else {
                            None
                        }
                    })
                });
                json!({
                    "model": self.as_str(),
                    "input": user_message_str,
                    "instructions": base_instructions,
                    "max_output_tokens": max_tokens,
                    "reasoning": reasoning_opt,
                    // Reasoning models can use certain tools
                    "tools": tools.map(|tools_inner| tools_inner
                        .iter()
                        .filter(|tool| self.get_supported_tools().iter().any(|supported| std::mem::discriminant(*tool) == std::mem::discriminant(supported)))
                        .filter_map(LLMTools::get_config_json)
                        .collect::<Vec<Value>>()
                    ),
                    // TODO: Other fields to be implemented in the future
                    // Structured Outputs Docs: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#how-to-use
                    // "text": {
                    //     "format": {
                    //         "type": "json_schema",
                    //         "name": "output",
                    //         "schema": json_schema,
                    //         "strict": true
                    //     }
                    // } - Structured Outputs is rejecting the json schema auto-generated from T
                    // "previous_response_id" - to implement chained conversations
                })
            }
            // Legacy Completions API Body
            // For DaVinci model all text goes into the 'prompt' filed of the body
            (_, OpenAIModels::TextDavinci003) => {
                json!({
                    "model": self.as_str(),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "prompt": format!("{base_instructions}{user_message_str}"),
                })
            }
        }
    }
    /*
     * This function leverages OpenAI API to perform any query as per the provided body.
     *
     * It returns a String the Response object that needs to be parsed based on the self.model.
     */
    async fn call_api(
        &self,
        api_key: &str,
        version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
        _tools: Option<&[LLMTools]>,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_version_endpoint(version);

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
                "[debug] OpenAI API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_version_data(
        &self,
        response_text: &str,
        function_call: bool,
        version: Option<String>,
    ) -> Result<String> {
        // If no version provided default to OpenAI Completions API
        let version = version
            .map(|version| OpenAiApiEndpoints::from_str(&version))
            .unwrap_or_default();

        match (version, self) {
            // Chat Completions API Data
            // Docs:https://platform.openai.com/docs/guides/chat/introduction
            #[allow(deprecated)]
            (
                OpenAiApiEndpoints::OpenAI
                | OpenAiApiEndpoints::OpenAICompletions
                | OpenAiApiEndpoints::Azure { .. }
                | OpenAiApiEndpoints::AzureCompletions { .. },
                OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::O1
                | OpenAIModels::O1Mini
                | OpenAIModels::O1Preview
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini
                | OpenAIModels::Custom { .. },
            ) => {
                //Convert API response to struct representing expected response format
                let chat_response: OpenAPIChatResponse = serde_json::from_str(response_text)?;

                //Extract data part
                match chat_response.choices {
                    Some(choices) => Ok(choices
                        .into_iter()
                        .filter_map(|item| {
                            //For function_call the response is in arguments, and for regular call in content
                            match function_call {
                                true => item.message.function_call.map(|function_call| {
                                    self.sanitize_json_response(&function_call.arguments)
                                }),
                                false => item
                                    .message
                                    .content
                                    .map(|content| self.sanitize_json_response(&content)),
                            }
                        })
                        .collect()),
                    None => Err(anyhow!("Unable to retrieve response from OpenAI Chat API")),
                }
            }
            // Responses API Data
            // Docs: https://platform.openai.com/docs/api-reference/responses/create
            (
                OpenAiApiEndpoints::OpenAIResponses | OpenAiApiEndpoints::AzureResponses { .. },
                OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt3_5Turbo0613
                | OpenAIModels::Gpt3_5Turbo16k
                | OpenAIModels::Gpt4
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt4_32k
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Gpt5_2Pro
                | OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O1Pro
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini
                | OpenAIModels::Custom { .. },
            )
            // o1 Pro and GPT-5.2 Pro are not supported in Completions API, we use the Responses API data schema
            | (
                OpenAiApiEndpoints::OpenAI
                | OpenAiApiEndpoints::OpenAICompletions
                | OpenAiApiEndpoints::Azure { .. }
                | OpenAiApiEndpoints::AzureCompletions { .. },
                OpenAIModels::O1Pro | OpenAIModels::Gpt5_2Pro,
            ) => {
                //Convert API response to struct representing expected response format
                let responses_response: OpenAPIResponsesResponse =
                    serde_json::from_str(response_text)?;

                Ok(responses_response
                    .output
                    .into_iter()
                    .filter(|output| {
                        matches!(output.role, Some(OpenAPIResponsesRole::Assistant))
                            && matches!(output.r#type, Some(OpenAPIResponsesOutputType::Message))
                    })
                    .flat_map(|output| output.content.unwrap_or_default())
                    .filter(|content| {
                        matches!(content.r#type, OpenAPIResponsesContentType::OutputText)
                    })
                    .filter_map(|content| content.text)
                    .map(|text| self.sanitize_json_response(&text))
                    .collect())
            }
            (_, OpenAIModels::TextDavinci003) => {
                //Convert API response to struct representing expected response format
                let completions_response: OpenAPICompletionsResponse =
                    serde_json::from_str(response_text)?;

                //Extract data part
                match completions_response.choices {
                    Some(choices) => Ok(choices.into_iter().filter_map(|item| item.text).collect()),
                    None => Err(anyhow!(
                        "Unable to retrieve response from OpenAI Completions API"
                    )),
                }
            }
        }
    }

    /// This function allows to check the rate limits for different models
    /// Rate limit for `Custom` model is assumed based on `GPT-4o` limits
    fn get_rate_limit(&self) -> RateLimit {
        //OpenAI documentation: https://platform.openai.com/account/rate-limits
        //This is the max tokens allowed between prompt & response
        match self {
            OpenAIModels::Gpt5_2 => RateLimit {
                tpm: 40_000_000,
                rpm: 15_000,
            },
            OpenAIModels::Gpt5_2Pro => RateLimit {
                tpm: 30_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt5_1 => RateLimit {
                tpm: 40_000_000,
                rpm: 15_000,
            },
            OpenAIModels::Gpt5 => RateLimit {
                tpm: 40_000_000,
                rpm: 15_000,
            },
            OpenAIModels::Gpt5Mini => RateLimit {
                tpm: 180_000_000,
                rpm: 30_000,
            },
            OpenAIModels::Gpt5Nano => RateLimit {
                tpm: 180_000_000,
                rpm: 30_000,
            },
            OpenAIModels::Gpt3_5Turbo => RateLimit {
                tpm: 50_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt3_5Turbo0613 => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt3_5Turbo16k => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4 => RateLimit {
                tpm: 1_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4Turbo => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4TurboPreview => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4_32k => RateLimit {
                tpm: 300_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4o | OpenAIModels::Custom { .. } => RateLimit {
                tpm: 150_000_000,
                rpm: 50_000,
            },
            OpenAIModels::Gpt4o20240806 => RateLimit {
                tpm: 150_000_000,
                rpm: 50_000,
            },
            OpenAIModels::Gpt4oMini => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::Gpt4_1 => RateLimit {
                tpm: 30_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4_1Mini => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::Gpt4_1Nano => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::Gpt4_5Preview => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::O1Preview => RateLimit {
                tpm: 30_000_000,
                rpm: 10_000,
            },
            OpenAIModels::O1Mini => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::O1 => RateLimit {
                tpm: 30_000_000,
                rpm: 10_000,
            },
            OpenAIModels::O1Pro => RateLimit {
                tpm: 30_000_000,
                rpm: 10_000,
            },
            OpenAIModels::O3 => RateLimit {
                tpm: 30_000_000,
                rpm: 10_000,
            },
            OpenAIModels::O3Mini => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::O4Mini => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::TextDavinci003 => RateLimit {
                tpm: 250_000,
                rpm: 3_000,
            },
        }
    }

    // Accepts a [0-100] percentage range and returns the target temperature based on model ranges
    fn get_normalized_temperature(&self, relative_temp: u32) -> f32 {
        // Temperature range documentation: https://platform.openai.com/docs/api-reference/chat/create
        let min = 0u32;
        let max = 2u32;
        map_to_range(min, max, relative_temp)
    }

    // OpenAI models, especially on Azure, may return JSON with additional properties wrapper. This function removes it.
    fn sanitize_json_response(&self, json_response: &str) -> String {
        let without_wrapper = remove_json_wrapper(json_response);
        remove_schema_wrappers(&without_wrapper)
    }
}

impl OpenAIModels {
    // This function checks if a model supports tool use in Assistants API (e.g. file_search)
    pub fn tools_support(&self) -> bool {
        matches!(
            self,
            OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Custom { .. }
        )
    }

    // This function checks if a model supports Structured Outputs
    // https://openai.com/index/introducing-structured-outputs-in-the-api/
    pub fn structured_output_support(&self) -> bool {
        matches!(
            self,
            OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Gpt4_1
                | OpenAIModels::Gpt4_1Mini
                | OpenAIModels::Gpt4_1Nano
                | OpenAIModels::Gpt4_5Preview
                | OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Custom { .. }
        )
    }

    // This function checks if a model supports use in Assistants API
    // Reasoning models are NOT currently supported
    pub fn assistants_support(&self) -> bool {
        !matches!(
            self,
            OpenAIModels::O1Preview
                | OpenAIModels::O1Mini
                | OpenAIModels::O1
                | OpenAIModels::O1Pro
                | OpenAIModels::O3
                | OpenAIModels::O3Mini
                | OpenAIModels::O4Mini
                // GPT-5.1+ are not supported in Assistants API
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Gpt5_2Pro
        )
    }

    // This function checks if a model is a GPT-5 model
    pub fn is_gpt5_model(&self) -> bool {
        matches!(
            self,
            OpenAIModels::Gpt5
                | OpenAIModels::Gpt5Mini
                | OpenAIModels::Gpt5Nano
                | OpenAIModels::Gpt5_1
                | OpenAIModels::Gpt5_2
                | OpenAIModels::Gpt5_2Pro
        )
    }

    // This function returns a list of supported tools for all models
    pub fn get_supported_tools(&self) -> Vec<LLMTools> {
        match self {
            // Reasoning models only support File Search and Code Interpreter
            OpenAIModels::O1Preview
            | OpenAIModels::O1Mini
            | OpenAIModels::O1
            | OpenAIModels::O1Pro
            | OpenAIModels::O3
            | OpenAIModels::O3Mini
            | OpenAIModels::O4Mini => vec![
                LLMTools::OpenAIFileSearch(OpenAIFileSearchConfig::new(vec![])),
                LLMTools::OpenAICodeInterpreter(OpenAICodeInterpreterConfig::new()),
            ],
            // GPT-5.2 does not support Computer Use as of 2025-12-11
            OpenAIModels::Gpt5_2 => vec![
                LLMTools::OpenAIFileSearch(OpenAIFileSearchConfig::new(vec![])),
                LLMTools::OpenAICodeInterpreter(OpenAICodeInterpreterConfig::new()),
                LLMTools::OpenAIWebSearch(OpenAIWebSearchConfig::new()),
            ],
            // GPT-5.2 Pro does not support Computer Use and Code Interpreter as of 2025-12-11
            OpenAIModels::Gpt5_2Pro => vec![
                LLMTools::OpenAIFileSearch(OpenAIFileSearchConfig::new(vec![])),
                LLMTools::OpenAIWebSearch(OpenAIWebSearchConfig::new()),
            ],
            // All other models support all tools
            _ => vec![
                LLMTools::OpenAIFileSearch(OpenAIFileSearchConfig::new(vec![])),
                LLMTools::OpenAICodeInterpreter(OpenAICodeInterpreterConfig::new()),
                LLMTools::OpenAIWebSearch(OpenAIWebSearchConfig::new()),
                LLMTools::OpenAIComputerUse(OpenAIComputerUseConfig::new(
                    1920,
                    1080,
                    "default".to_string(),
                )),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::llm_models::llm_model::LLMModel;
    use crate::llm_models::OpenAIModels;

    // Tests for calculating max requests per model
    #[test]
    fn test_gpt3_5turbo_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 50_000_000 / ((4096_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo0613_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo0613;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 2_000_000 / ((4096_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo16k_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo16k;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 2_000_000 / ((16384_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt4_max_requests() {
        let model = OpenAIModels::Gpt4;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 1_000_000 / ((8192_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    // Tests of model creation
    #[test]
    fn test_try_from_str_standard_models() {
        assert_eq!(
            OpenAIModels::try_from_str("gpt-3.5-turbo"),
            Some(OpenAIModels::Gpt3_5Turbo)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-3.5-turbo-0613"),
            Some(OpenAIModels::Gpt3_5Turbo0613)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-3.5-turbo-16k"),
            Some(OpenAIModels::Gpt3_5Turbo16k)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4"),
            Some(OpenAIModels::Gpt4)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4-32k"),
            Some(OpenAIModels::Gpt4_32k)
        );
        assert_eq!(
            OpenAIModels::try_from_str("text-davinci-003"),
            Some(OpenAIModels::TextDavinci003)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4-turbo"),
            Some(OpenAIModels::Gpt4Turbo)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4-turbo-preview"),
            Some(OpenAIModels::Gpt4TurboPreview)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4o"),
            Some(OpenAIModels::Gpt4o)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4o-2024-08-06"),
            Some(OpenAIModels::Gpt4o20240806)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4o-mini"),
            Some(OpenAIModels::Gpt4oMini)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4.5-preview"),
            Some(OpenAIModels::Gpt4_5Preview)
        );
    }

    #[test]
    fn test_try_from_str_case_insensitivity() {
        assert_eq!(
            OpenAIModels::try_from_str("GPT-4"),
            Some(OpenAIModels::Gpt4)
        );
        assert_eq!(
            OpenAIModels::try_from_str("GPT-4o-MiNI"),
            Some(OpenAIModels::Gpt4oMini)
        );
        assert_eq!(
            OpenAIModels::try_from_str("GPT-4.5-pREVIEW"),
            Some(OpenAIModels::Gpt4_5Preview)
        );
    }

    #[test]
    fn test_try_from_str_custom_model() {
        assert_eq!(
            OpenAIModels::try_from_str("my-custom-model"),
            Some(OpenAIModels::Custom {
                name: "my-custom-model".to_string()
            })
        );
        assert_eq!(
            OpenAIModels::try_from_str("AnotherModel"),
            Some(OpenAIModels::Custom {
                name: "AnotherModel".to_string()
            })
        );
    }
}
