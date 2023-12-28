use anyhow::Result;
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{domain::AnthropicAPICompletionsResponse, llm_models::LLMModel};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum AnthropicModels {
    Claude2,
}

#[async_trait(?Send)]
impl LLMModel for AnthropicModels {
    fn as_str(&self) -> &'static str {
        match self {
            AnthropicModels::Claude2 => "claude-2.1",
        }
    }

    fn default_max_tokens(&self) -> usize {
        //This is the max tokens allowed for response
        match self {
            AnthropicModels::Claude2 => 4_096,
        }
    }

    fn get_endpoint(&self) -> String {
        match self {
            //TODO: Move to env var
            AnthropicModels::Claude2 => "https://api.anthropic.com/v1/complete".to_string(),
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
        match self {
            AnthropicModels::Claude2 => {
                let schema_string = serde_json::to_string(json_schema).unwrap_or_default();
                let base_instructions = self.get_base_instructions(Some(function_call));
                json!({
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
                })
            }
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
        match self {
            AnthropicModels::Claude2 => {
                //Convert API response to struct representing expected response format
                let completions_response: AnthropicAPICompletionsResponse =
                    serde_json::from_str(response_text)?;

                //Return completions text
                Ok(completions_response.completion)
            }
        }
    }
}
