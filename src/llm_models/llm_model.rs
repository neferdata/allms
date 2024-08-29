use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::constants::OPENAI_BASE_INSTRUCTIONS;
use crate::domain::RateLimit;

///This trait defines functions that need to be implemented for an enum that represents an LLM Model from any of the API providers
#[async_trait(?Send)]
pub trait LLMModel {
    ///Converts each item in the model enum into its string representation
    fn as_str(&self) -> &'static str;
    ///Returns an instance of the enum based on the provided string representation of name
    fn try_from_str(name: &str) -> Option<Self>
    where
        Self: Sized;
    ///Returns max supported number of tokens for each of the variants of the enum
    fn default_max_tokens(&self) -> usize;
    ///Returns the url of the endpoint that should be called for each variant of the LLM Model enum
    fn get_endpoint(&self) -> String;
    ///Provides a list of base instructions that should be added to each prompt when using each of the models
    fn get_base_instructions(&self, _function_call: Option<bool>) -> String {
        OPENAI_BASE_INSTRUCTIONS.to_string()
    }
    ///Returns recommendation if function calling should be used for the specified model
    fn function_call_default(&self) -> bool {
        false
    }
    ///Constructs the body that should be attached to the API call for each of the LLM Models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &u32,
    ) -> serde_json::Value;
    ///Makes the call to the correct API for the selected model
    async fn call_api(
        &self,
        api_key: &str,
        body: &serde_json::Value,
        debug: bool,
    ) -> Result<String>;
    ///Based on the model type extracts the data portion of the API response
    fn get_data(&self, response_text: &str, function_call: bool) -> Result<String>;
    ///Returns the rate limit accepted by the API depending on the used model
    ///If not explicitly defined it will assume 1B tokens or 100k transactions a minute
    fn get_rate_limit(&self) -> RateLimit {
        RateLimit {
            tpm: 100_000_000,
            rpm: 100_000,
        }
    }
    ///Based on the RateLimit for the model calculates how many requests can be send to the API
    fn get_max_requests(&self) -> usize {
        let rate_limit = self.get_rate_limit();

        //Check max requests based on rpm
        let max_requests_from_rpm = rate_limit.rpm;

        //Double check max number of requests based on tpm
        //Assume we will use ~50% of allowed tokens per request (for prompt + response)
        let max_tokens_per_minute = rate_limit.tpm;
        let tpm_per_request = (self.default_max_tokens() as f64 * 0.5).ceil() as usize;
        //Then check how many requests we can process
        let max_requests_from_tpm = max_tokens_per_minute / tpm_per_request;

        //To be safe we go with smaller of the numbers
        std::cmp::min(max_requests_from_rpm, max_requests_from_tpm)
    }
}
