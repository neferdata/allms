use anyhow::Result;
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::constants::{ANTHROPIC_API_URL, ANTHROPIC_MESSAGES_API_URL};
use crate::domain::{AnthropicAPICompletionsResponse, AnthropicAPIMessagesResponse};
use crate::llm_models::LLMModel;

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum AnthropicModels {
    Claude3_5Sonnet,
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
            AnthropicModels::Claude3_5Sonnet => "claude-3-5-sonnet-20240620",
            AnthropicModels::Claude3Opus => "claude-3-opus-20240229",
            AnthropicModels::Claude3Sonnet => "claude-3-sonnet-20240229",
            AnthropicModels::Claude3Haiku => "claude-3-haiku-20240307",
            // Legacy
            AnthropicModels::Claude2 => "claude-2.1",
            AnthropicModels::ClaudeInstant1_2 => "claude-instant-1.2",
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "claude-3-5-sonnet-20240620" => Some(AnthropicModels::Claude3_5Sonnet),
            "claude-3-opus-20240229" => Some(AnthropicModels::Claude3Opus),
            "claude-3-sonnet-20240229" => Some(AnthropicModels::Claude3Sonnet),
            "claude-3-haiku-20240307" => Some(AnthropicModels::Claude3Haiku),
            // Legacy
            "claude-2.1" => Some(AnthropicModels::Claude2),
            "claude-instant-1.2" => Some(AnthropicModels::ClaudeInstant1_2),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        // This is the max tokens allowed for response and not context as per documentation: https://docs.anthropic.com/claude/reference/input-and-output-sizes
        match self {
            AnthropicModels::Claude3_5Sonnet => 4_096, // 8192 output tokens is in beta and requires the header anthropic-beta: max-tokens-3-5-sonnet-2024-07-15. If the header is not specified, the limit is 4096 tokens. (Source: https://docs.anthropic.com/en/docs/about-claude/models)
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
            AnthropicModels::Claude3_5Sonnet
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
        temperature: &u32,
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

        let message_body = json!({
            "model": self.as_str(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{
                "role": "user",
                "content": format!(
                    "{base_instructions}\n\n
                    Output Json schema:\n
                    {schema_string}\n\n
                    {instructions}"
                )
            }],
        });

        match self {
            AnthropicModels::Claude3_5Sonnet
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
        body: &serde_json::Value,
        debug: bool,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_endpoint();

        //Make the API call
        let client = Client::new();

        //Send request
        let response = client
            .post(model_url)
            .header(header::CONTENT_TYPE, "application/json")
            //Anthropic-specific way of passing API key
            .header("x-api-key", api_key)
            //Required as per documentation
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;

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
            AnthropicModels::Claude3_5Sonnet
            | AnthropicModels::Claude3Opus
            | AnthropicModels::Claude3Sonnet
            | AnthropicModels::Claude3Haiku => {
                let messages_response: AnthropicAPIMessagesResponse =
                    serde_json::from_str(response_text)?;

                let assistant_response = messages_response
                    .content
                    .iter()
                    .map(|item| &item.text)
                    .fold(String::new(), |mut acc, text| {
                        acc.push_str(text);
                        acc
                    });

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
