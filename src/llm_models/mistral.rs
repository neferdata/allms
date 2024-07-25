use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::constants::MISTRAL_API_URL;
use crate::domain::{MistralAPICompletionsResponse, RateLimit};
use crate::llm_models::LLMModel;
use crate::utils::sanitize_json_response;

#[derive(Deserialize, Serialize, Debug, Clone)]
//Mistral docs: https://docs.mistral.ai/platform/endpoints
pub enum MistralModels {
    MistralLarge,
    MistralNemo,
    Mistral7B,
    Mixtral8x7B,
    Mixtral8x22B,
    // Legacy
    MistralTiny,
    MistralSmall,
    MistralMedium,
}

#[async_trait(?Send)]
impl LLMModel for MistralModels {
    fn as_str(&self) -> &'static str {
        match self {
            MistralModels::MistralLarge => "mistral-large-latest",
            MistralModels::MistralNemo => "open-mistral-nemo",
            MistralModels::Mistral7B => "open-mistral-7b",
            MistralModels::Mixtral8x7B => "open-mixtral-8x7b",
            MistralModels::Mixtral8x22B => "open-mixtral-8x22b",
            // Legacy
            MistralModels::MistralTiny => "mistral-tiny",
            MistralModels::MistralSmall => "mistral-small",
            MistralModels::MistralMedium => "mistral-medium",
        }
    }

    fn default_max_tokens(&self) -> usize {
        match self {
            MistralModels::MistralLarge => 128_000,
            MistralModels::MistralNemo => 128_000,
            MistralModels::Mistral7B => 32_000,
            MistralModels::Mixtral8x7B => 32_000,
            MistralModels::Mixtral8x22B => 64_000,
            // Legacy
            MistralModels::MistralTiny => 32_000,
            MistralModels::MistralSmall => 32_000,
            MistralModels::MistralMedium => 32_000,
        }
    }

    fn get_endpoint(&self) -> String {
        MISTRAL_API_URL.to_string()
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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": vec![
                system_message,
                user_message,
            ],
        })
    }
    /*
     * This function leverages Mistral API to perform any query as per the provided body.
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
        //Convert API response to struct representing expected response format
        let completions_response: MistralAPICompletionsResponse =
            serde_json::from_str(response_text)?;

        //Parse the response and return the assistant content
        completions_response
            .choices
            .iter()
            .filter_map(|choice| choice.message.as_ref())
            .find(|&message| message.role.as_ref() == Some(&"assistant".to_string()))
            .and_then(|message| {
                message
                    .content
                    .as_ref()
                    .map(|content| sanitize_json_response(&content))
            })
            .ok_or_else(|| anyhow!("Assistant role content not found"))
    }

    //This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        //Mistral documentation: https://docs.mistral.ai/platform/pricing#rate-limits
        RateLimit {
            tpm: 2_000_000,
            rpm: 120, // 2 request per second
        }
    }
}
