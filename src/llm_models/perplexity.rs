#![allow(deprecated)]

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::constants::PERPLEXITY_API_URL;
use crate::domain::{PerplexityAPICompletionsResponse, RateLimit};
use crate::llm_models::LLMModel;
use crate::utils::{map_to_range_f32, remove_json_wrapper, remove_think_reasoner_wrapper};

// Perplexity API Docs: https://docs.perplexity.ai/api-reference/chat-completions
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum PerplexityModels {
    SonarPro,
    Sonar,
    SonarReasoning,
    // Legacy models
    #[deprecated(
        since = "0.12.0",
        note = "`Llama3_1SonarSmall` is deprecated starting February 22, 2025, please use `Sonar` or `SonarPro` instead."
    )]
    Llama3_1SonarSmall,
    #[deprecated(
        since = "0.12.0",
        note = "`Llama3_1SonarLarge` is deprecated starting February 22, 2025, please use `Sonar` or `SonarPro` instead."
    )]
    Llama3_1SonarLarge,
    #[deprecated(
        since = "0.12.0",
        note = "`Llama3_1SonarHuge` is deprecated starting February 22, 2025, please use `Sonar` or `SonarPro` instead."
    )]
    Llama3_1SonarHuge,
}

#[async_trait(?Send)]
impl LLMModel for PerplexityModels {
    fn as_str(&self) -> &str {
        match self {
            PerplexityModels::SonarPro => "sonar-pro",
            PerplexityModels::Sonar => "sonar",
            PerplexityModels::SonarReasoning => "sonar-reasoning",
            // Legacy models
            #[allow(deprecated)]
            PerplexityModels::Llama3_1SonarSmall => "llama-3.1-sonar-small-128k-online",
            #[allow(deprecated)]
            PerplexityModels::Llama3_1SonarLarge => "llama-3.1-sonar-large-128k-online",
            #[allow(deprecated)]
            PerplexityModels::Llama3_1SonarHuge => "llama-3.1-sonar-huge-128k-online",
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "sonar-pro" => Some(PerplexityModels::SonarPro),
            "sonar" => Some(PerplexityModels::Sonar),
            "sonar-reasoning" => Some(PerplexityModels::SonarReasoning),
            // Legacy models
            #[allow(deprecated)]
            "llama-3.1-sonar-small-128k-online" => Some(PerplexityModels::Llama3_1SonarSmall),
            #[allow(deprecated)]
            "llama-3.1-sonar-large-128k-online" => Some(PerplexityModels::Llama3_1SonarLarge),
            #[allow(deprecated)]
            "llama-3.1-sonar-huge-128k-online" => Some(PerplexityModels::Llama3_1SonarHuge),
            _ => None,
        }
    }

    // https://docs.perplexity.ai/guides/model-cards
    fn default_max_tokens(&self) -> usize {
        match self {
            // Docs: https://docs.perplexity.ai/guides/model-cards
            // FYI: sonar-pro has a max output token limit of 8k
            PerplexityModels::SonarPro => 200_000,
            PerplexityModels::Sonar => 127_072,
            PerplexityModels::SonarReasoning => 127_072,
            // Legacy models
            #[allow(deprecated)]
            PerplexityModels::Llama3_1SonarSmall
            | PerplexityModels::Llama3_1SonarLarge
            | PerplexityModels::Llama3_1SonarHuge => 127_072,
        }
    }

    fn get_endpoint(&self) -> String {
        PERPLEXITY_API_URL.to_string()
    }

    //This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        // The total number of tokens requested in max_tokens plus the number of prompt tokens sent in messages must not exceed the context window token limit of model requested.
        // If left unspecified, then the model will generate tokens until either it reaches its stop token or the end of its context window.
        _max_tokens: &usize,
        temperature: &f32,
    ) -> serde_json::Value {
        //Prepare the 'messages' part of the body
        let base_instructions = self.get_base_instructions(Some(function_call));
        let system_message = json!({
            "role": "system",
            "content": base_instructions,
        });
        let schema_string = serde_json::to_string(json_schema).unwrap_or_default();
        let user_message = json!({
            "role": "user",
            "content": format!(
                "Output Json schema:\n
                {schema_string}\n\n
                {instructions}"
            ),
        });
        json!({
            "model": self.as_str(),
            "temperature": temperature,
            "messages": vec![
                system_message,
                user_message,
            ],
        })
    }
    ///
    /// This function leverages Perplexity API to perform any query as per the provided body.
    ///
    /// It returns a String the Response object that needs to be parsed based on the self.model.
    ///
    async fn call_api(
        &self,
        api_key: &str,
        _version: Option<String>,
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
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if debug {
            info!(
                "[debug] Perplexity API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        //Convert API response to struct representing expected response format
        let completions_response: PerplexityAPICompletionsResponse =
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

    /// This function sanitizes the text response from Perplexity models to clean up common formatting issues.
    /// Currently the function checks:
    /// * ```json{}``` wrapper around response
    /// * <think></think> wrapper (for SonarReasoning model only)
    fn sanitize_json_response(&self, json_response: &str) -> String {
        let no_json_text = remove_json_wrapper(json_response);
        if self == &PerplexityModels::SonarReasoning {
            remove_think_reasoner_wrapper(&no_json_text)
        } else {
            no_json_text
        }
    }

    // This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        //Perplexity documentation: https://docs.perplexity.ai/guides/rate-limits
        RateLimit {
            tpm: 50 * 127_072, // 50 requests per minute wit max 127,072 context length
            rpm: 50,           // 50 request per minute
        }
    }

    // Accepts a [0-100] percentage range and returns the target temperature based on model ranges
    fn get_normalized_temperature(&self, relative_temp: u32) -> f32 {
        // Temperature range documentation: https://docs.perplexity.ai/api-reference/chat-completions
        // "The amount of randomness in the response, valued between 0 *inclusive* and 2 *exclusive*."
        let min = 0.0f32;
        let max = 1.99999f32;
        map_to_range_f32(min, max, relative_temp)
    }
}
