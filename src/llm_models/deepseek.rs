use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::completions::ThinkingLevel;
use crate::constants::DEEPSEEK_API_URL;
use crate::domain::{DeepSeekAPICompletionsResponse, RateLimit};
use crate::llm_models::{LLMModel, LLMTools};
use crate::utils::map_to_range_f32;

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
//DeepSeek docs: https://api-docs.deepseek.com/quick_start/pricing
pub enum DeepSeekModels {
    DeepSeekChat,
    DeepSeekReasoner,
}

#[async_trait(?Send)]
impl LLMModel for DeepSeekModels {
    fn as_str(&self) -> &str {
        match self {
            DeepSeekModels::DeepSeekChat => "deepseek-chat",
            DeepSeekModels::DeepSeekReasoner => "deepseek-reasoner",
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "deepseek-chat" => Some(DeepSeekModels::DeepSeekChat),
            "deepseek-reasoner" => Some(DeepSeekModels::DeepSeekReasoner),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        match self {
            DeepSeekModels::DeepSeekChat => 8_192,
            DeepSeekModels::DeepSeekReasoner => 8_192,
        }
    }

    fn get_endpoint(&self) -> String {
        DEEPSEEK_API_URL.to_string()
    }

    /// This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        _tools: Option<&[LLMTools]>,
        _thinking_level: Option<&ThinkingLevel>,
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
    ///
    /// This function leverages DeepSeek API to perform any query as per the provided body.
    ///
    /// It returns a String the Response object that needs to be parsed based on the self.model.
    ///
    async fn call_api(
        &self,
        api_key: &str,
        _version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
        _tools: Option<&[LLMTools]>,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_endpoint();

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
                "[debug] DeepSeek API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    ///
    /// This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    ///
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        //Convert API response to struct representing expected response format
        let completions_response: DeepSeekAPICompletionsResponse =
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

    // This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        // DeepSeek documentation: https://api-docs.deepseek.com/quick_start/rate_limit
        // "DeepSeek API does NOT constrain user's rate limit. We will try out best to serve every request."
        RateLimit {
            tpm: 100_000_000, // i.e. very large number
            rpm: 100_000_000,
        }
    }

    // Accepts a [0-100] percentage range and returns the target temperature based on model ranges
    fn get_normalized_temperature(&self, relative_temp: u32) -> f32 {
        // Temperature range documentation: https://api-docs.deepseek.com/quick_start/parameter_settings
        let min = 0.0f32;
        let max = 1.5f32;
        map_to_range_f32(min, max, relative_temp)
    }
}
