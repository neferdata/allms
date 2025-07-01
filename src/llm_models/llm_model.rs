use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::constants::OPENAI_BASE_INSTRUCTIONS;
use crate::domain::RateLimit;
use crate::llm_models::LLMTools;
use crate::utils::{map_to_range, remove_json_wrapper};

///This trait defines functions that need to be implemented for an enum that represents an LLM Model from any of the API providers
#[async_trait(?Send)]
pub trait LLMModel {
    ///Converts each item in the model enum into its string representation
    fn as_str(&self) -> &str;
    ///Returns an instance of the enum based on the provided string representation of name
    fn try_from_str(name: &str) -> Option<Self>
    where
        Self: Sized;
    ///Returns max supported number of tokens for each of the variants of the enum
    fn default_max_tokens(&self) -> usize;
    ///Returns the url of the endpoint that should be called for each variant of the LLM Model enum
    fn get_endpoint(&self) -> String {
        self.get_version_endpoint(None)
    }
    ///Returns the url of the endpoint that should be called for each variant of the LLM Model enum
    ///It allows to specify which version of the endpoint to use
    fn get_version_endpoint(&self, _version: Option<String>) -> String {
        self.get_endpoint()
    }
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
        temperature: &f32,
        tools: Option<&[LLMTools]>,
    ) -> serde_json::Value {
        self.get_version_body(
            instructions,
            json_schema,
            function_call,
            max_tokens,
            temperature,
            None,
            tools,
        )
    }
    /// An API-version-specific implementation of the body constructor
    #[allow(clippy::too_many_arguments)]
    fn get_version_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        _version: Option<String>,
        tools: Option<&[LLMTools]>,
    ) -> serde_json::Value {
        self.get_body(
            instructions,
            json_schema,
            function_call,
            max_tokens,
            temperature,
            tools,
        )
    }
    ///Makes the call to the correct API for the selected model
    async fn call_api(
        &self,
        api_key: &str,
        version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
        tools: Option<&[LLMTools]>,
    ) -> Result<String>;
    ///Based on the model type extracts the data portion of the API response
    fn get_data(&self, response_text: &str, function_call: bool) -> Result<String> {
        self.get_version_data(response_text, function_call, None)
    }
    /// An API-version-specific implementation of the data extractor
    fn get_version_data(
        &self,
        response_text: &str,
        function_call: bool,
        _version: Option<String>,
    ) -> Result<String> {
        self.get_data(response_text, function_call)
    }
    /// This function sanitizes the text response from LLMs to clean up common formatting issues.
    /// The default implementation of the function removes the common ```json{}``` wrapper returned by most models
    fn sanitize_json_response(&self, json_response: &str) -> String {
        remove_json_wrapper(json_response)
    }
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
    ///Returns the default temperature to be used by the model
    fn get_default_temperature(&self) -> f32 {
        0f32
    }
    ///Returns the normalized temperature for the model
    //Input should be a 0-100 number representing the percentage of max temp for the model
    fn get_normalized_temperature(&self, relative_temp: u32) -> f32 {
        //Assuming 0-1 range for most models. Different ranges require model-specific implementations.
        let min = 0u32;
        let max = 1u32;
        map_to_range(min, max, relative_temp)
    }
}
