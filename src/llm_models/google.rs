#![allow(deprecated)]

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::stream::StreamExt;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::apis::GoogleApiEndpoints;
use crate::constants::{
    GOOGLE_GEMINI_API_URL, GOOGLE_GEMINI_BETA_API_URL, GOOGLE_VERTEX_API_URL,
    GOOGLE_VERTEX_ENDPOINT_API_URL,
};
use crate::domain::{GoogleGeminiProApiResp, RateLimit};
use crate::llm_models::tools::{GeminiCodeInterpreterConfig, GeminiWebSearchConfig};
use crate::llm_models::{LLMModel, LLMTools};

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
// Google Docs: https://ai.google.dev/gemini-api/docs/models/gemini
// Google Vertex Docs: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
pub enum GoogleModels {
    // 2.5
    Gemini2_5Pro,
    Gemini2_5Flash,
    Gemini2_5FlashLite,
    // 2.0
    Gemini2_0Flash,
    Gemini2_0FlashLite,
    // 2.0 - Experimental
    Gemini2_0ProExp,
    Gemini2_0FlashThinkingExp,
    // 1.5
    Gemini1_5Flash,
    Gemini1_5Flash8B,
    Gemini1_5Pro,
    // Fine-tuned models
    FineTunedEndpoint {
        name: String,
    },
    // Legacy approach to Vertex models
    #[deprecated(
        since = "0.19.0",
        note = "Starting 0.19.0 `allms` allows to set the API version to `google-vertex` or `google-studio` instead of using the model name to call the right API."
    )]
    Gemini1_5FlashVertex,
    #[deprecated(
        since = "0.19.0",
        note = "Starting 0.19.0 `allms` allows to set the API version to `google-vertex` or `google-studio` instead of using the model name to call the right API."
    )]
    Gemini1_5Flash8BVertex,
    #[deprecated(
        since = "0.19.0",
        note = "Starting 0.19.0 `allms` allows to set the API version to `google-vertex` or `google-studio` instead of using the model name to call the right API."
    )]
    Gemini1_5ProVertex,
    #[deprecated(
        since = "0.19.0",
        note = "Starting 0.19.0 `allms` allows to set the API version to `google-vertex` or `google-studio` instead of using the model name to call the right API."
    )]
    Gemini2_0FlashVertex,
    #[deprecated(
        since = "0.19.0",
        note = "Starting 0.19.0 `allms` allows to set the API version to `google-vertex` or `google-studio` instead of using the model name to call the right API."
    )]
    Gemini2_0FlashLiteVertex,
    #[deprecated(
        since = "0.19.0",
        note = "Starting 0.19.0 `allms` allows to set the API version to `google-vertex` or `google-studio` instead of using the model name to call the right API."
    )]
    Gemini2_0ProExpVertex,
    #[deprecated(
        since = "0.19.0",
        note = "Starting 0.19.0 `allms` allows to set the API version to `google-vertex` or `google-studio` instead of using the model name to call the right API."
    )]
    Gemini2_0FlashThinkingExpVertex,
}

#[async_trait(?Send)]
impl LLMModel for GoogleModels {
    fn as_str(&self) -> &str {
        match self {
            GoogleModels::Gemini1_5Pro | GoogleModels::Gemini1_5ProVertex => "gemini-1.5-pro",
            GoogleModels::Gemini1_5Flash | GoogleModels::Gemini1_5FlashVertex => "gemini-1.5-flash",
            GoogleModels::Gemini1_5Flash8B | GoogleModels::Gemini1_5Flash8BVertex => {
                "gemini-1.5-flash-8b"
            }
            GoogleModels::Gemini2_0Flash | GoogleModels::Gemini2_0FlashVertex => "gemini-2.0-flash",
            GoogleModels::Gemini2_0FlashLite | GoogleModels::Gemini2_0FlashLiteVertex => {
                "gemini-2.0-flash-lite"
            }
            GoogleModels::Gemini2_0ProExp | GoogleModels::Gemini2_0ProExpVertex => {
                "gemini-2.0-pro-exp-02-05"
            }
            GoogleModels::Gemini2_0FlashThinkingExp
            | GoogleModels::Gemini2_0FlashThinkingExpVertex => {
                "gemini-2.0-flash-thinking-exp-01-21"
            }
            GoogleModels::Gemini2_5Flash => "gemini-2.5-flash",
            GoogleModels::Gemini2_5Pro => "gemini-2.5-pro",
            GoogleModels::Gemini2_5FlashLite => "gemini-2.5-flash-lite",
            GoogleModels::FineTunedEndpoint { name } => name,
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "gemini-1.5-pro" => Some(GoogleModels::Gemini1_5Pro),
            "gemini-1.5-pro-vertex" => Some(GoogleModels::Gemini1_5ProVertex),
            "gemini-1.5-flash" => Some(GoogleModels::Gemini1_5Flash),
            "gemini-1.5-flash-vertex" => Some(GoogleModels::Gemini1_5FlashVertex),
            "gemini-1.5-flash-8b" => Some(GoogleModels::Gemini1_5Flash8B),
            "gemini-1.5-flash-8b-vertex" => Some(GoogleModels::Gemini1_5Flash8BVertex),
            "gemini-2.0-flash" => Some(GoogleModels::Gemini2_0Flash),
            "gemini-2.0-flash-vertex" => Some(GoogleModels::Gemini2_0FlashVertex),
            "gemini-2.0-flash-lite" => Some(GoogleModels::Gemini2_0FlashLite),
            "gemini-2.0-flash-lite-vertex" => Some(GoogleModels::Gemini2_0FlashLiteVertex),
            "gemini-2.0-pro" => Some(GoogleModels::Gemini2_0ProExp),
            "gemini-2.0-pro-exp" => Some(GoogleModels::Gemini2_0ProExp),
            "gemini-2.0-pro-vertex" => Some(GoogleModels::Gemini2_0ProExpVertex),
            "gemini-2.0-flash-thinking" => Some(GoogleModels::Gemini2_0FlashThinkingExp),
            "gemini-2.0-flash-thinking-exp" => Some(GoogleModels::Gemini2_0FlashThinkingExp),
            "gemini-2.0-flash-thinking-vertex" => {
                Some(GoogleModels::Gemini2_0FlashThinkingExpVertex)
            }
            "gemini-2.5-flash" => Some(GoogleModels::Gemini2_5Flash),
            "gemini-2.5-pro" => Some(GoogleModels::Gemini2_5Pro),
            "gemini-2.5-flash-lite" => Some(GoogleModels::Gemini2_5FlashLite),
            // Gemini 1.0 Pro is deprecated starting 2/15/2025. We are re-routing to 1.5 Pro for the model
            "gemini-pro" => Some(GoogleModels::Gemini1_5Pro),
            "gemini-1.0-pro" => Some(GoogleModels::Gemini1_5Pro),
            "gemini-pro-vertex" => Some(GoogleModels::Gemini1_5ProVertex),
            "gemini-1.0-pro-vertex" => Some(GoogleModels::Gemini1_5ProVertex),
            // Fine-tuned models need to be constructed via the endpoint method
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        // Docs: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models
        match self {
            GoogleModels::Gemini1_5Pro | GoogleModels::Gemini1_5ProVertex => 2_097_152,
            GoogleModels::Gemini1_5Flash | GoogleModels::Gemini1_5FlashVertex => 1_048_576,
            GoogleModels::Gemini1_5Flash8B | GoogleModels::Gemini1_5Flash8BVertex => 1_048_576,
            GoogleModels::Gemini2_0Flash | GoogleModels::Gemini2_0FlashVertex => 1_048_576,
            GoogleModels::Gemini2_0FlashLite | GoogleModels::Gemini2_0FlashLiteVertex => 1_048_576,
            GoogleModels::Gemini2_0ProExp | GoogleModels::Gemini2_0ProExpVertex => 2_097_152,
            GoogleModels::Gemini2_0FlashThinkingExp
            | GoogleModels::Gemini2_0FlashThinkingExpVertex => 1_048_576,
            GoogleModels::Gemini2_5Flash => 1_048_576,
            GoogleModels::Gemini2_5Pro => 1_048_576,
            GoogleModels::Gemini2_5FlashLite => 1_048_576,
            // TODO: Is this a good assumption?
            GoogleModels::FineTunedEndpoint { .. } => 1_048_576,
        }
    }

    fn get_version_endpoint(&self, version: Option<String>) -> String {
        // If no version provided default to Google Studio API
        let version = version
            .map(|version| GoogleApiEndpoints::from_str(&version))
            .unwrap_or_default();

        match (self, version) {
            // Google Studio API
            (
                GoogleModels::Gemini1_5Pro
                | GoogleModels::Gemini1_5Flash
                | GoogleModels::Gemini1_5Flash8B
                | GoogleModels::Gemini2_0Flash
                | GoogleModels::Gemini2_0FlashLite
                | GoogleModels::Gemini2_0ProExp
                | GoogleModels::Gemini2_0FlashThinkingExp,
                GoogleApiEndpoints::GoogleStudio,
            ) => format!(
                "{}/{}:generateContent",
                &*GOOGLE_GEMINI_API_URL,
                self.as_str()
            ),
            // 2.5 models are only available in the beta API
            (
                GoogleModels::Gemini2_5Flash
                | GoogleModels::Gemini2_5Pro
                | GoogleModels::Gemini2_5FlashLite,
                GoogleApiEndpoints::GoogleStudio,
            ) => format!(
                "{}/{}:generateContent",
                &*GOOGLE_GEMINI_BETA_API_URL,
                self.as_str()
            ),
            // Fine-tuned models are only available in the Vertex API
            // TODO: Explore fine-tuned models in the Studio API
            (GoogleModels::FineTunedEndpoint { .. }, GoogleApiEndpoints::GoogleStudio) => {
                // Construct Vertex URL when needed
                format!(
                    "{}/{}:generateContent",
                    &*GOOGLE_VERTEX_ENDPOINT_API_URL,
                    self.as_str()
                )
            }
            // Google Vertex API
            (
                GoogleModels::Gemini1_5Pro
                | GoogleModels::Gemini1_5Flash
                | GoogleModels::Gemini1_5Flash8B
                | GoogleModels::Gemini2_0Flash
                | GoogleModels::Gemini2_0FlashLite
                | GoogleModels::Gemini2_0ProExp
                | GoogleModels::Gemini2_0FlashThinkingExp
                | GoogleModels::Gemini2_5Flash
                | GoogleModels::Gemini2_5Pro
                | GoogleModels::Gemini2_5FlashLite,
                GoogleApiEndpoints::GoogleVertex,
            ) => {
                // Construct Vertex URL when needed
                format!(
                    "{}/{}:streamGenerateContent?alt=sse",
                    &*GOOGLE_VERTEX_API_URL,
                    self.as_str()
                )
            }
            // Google Vertex API for fine-tuned models
            (GoogleModels::FineTunedEndpoint { .. }, GoogleApiEndpoints::GoogleVertex) => {
                // Construct Vertex URL when needed
                format!(
                    "{}/{}:generateContent",
                    &*GOOGLE_VERTEX_ENDPOINT_API_URL,
                    self.as_str()
                )
            }
            // Legacy Google Vertex API implementation
            #[allow(deprecated)]
            (
                GoogleModels::Gemini1_5ProVertex
                | GoogleModels::Gemini1_5FlashVertex
                | GoogleModels::Gemini1_5Flash8BVertex
                | GoogleModels::Gemini2_0FlashVertex
                | GoogleModels::Gemini2_0FlashLiteVertex
                | GoogleModels::Gemini2_0ProExpVertex
                | GoogleModels::Gemini2_0FlashThinkingExpVertex,
                _,
            ) => {
                // Construct Vertex URL when needed
                format!(
                    "{}/{}:streamGenerateContent?alt=sse",
                    &*GOOGLE_VERTEX_API_URL,
                    self.as_str()
                )
            }
        }
    }

    //This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        _max_tokens: &usize,
        temperature: &f32,
        tools: Option<&[LLMTools]>,
    ) -> serde_json::Value {
        //Prepare the 'messages' part of the body
        let base_instructions_json = json!({
            "text": self.get_base_instructions(Some(function_call))
        });

        let output_instructions_json = json!({ "text": format!("<output json schema>
                {json_schema}
                </output json schema>") });

        let user_instructions_json = json!({
            "text": format!("<instructions>
                {instructions}
                </instructions>"),
        });

        let mut message_parts = vec![
            base_instructions_json,
            output_instructions_json,
            user_instructions_json,
        ];

        // If the `URL context` tool was configured we include a part with the URLs to be used as context
        if let Some(tools_inner) = tools {
            tools_inner.iter().find_map(|tool| {
                if let LLMTools::GeminiWebSearch(config) = tool {
                    let urls = config.get_context_urls();
                    if !urls.is_empty() {
                        message_parts.push(json!({
                            "text": format!("<url_context>
                                {:?}
                                </url_context>",
                            urls),
                        }));
                    }
                    Some(())
                } else {
                    None
                }
            });
        }

        let contents = json!({
            "role": "user",
            "parts": message_parts,
        });

        let generation_config = json!({
            "temperature": temperature,
        });

        let mut body = json!({
            "contents": contents,
            "generationConfig": generation_config,
        });

        // Include tools if provided
        if let Some(tools_inner) = tools {
            let processed_tools: Vec<Value> = tools_inner
                .iter()
                .filter_map(|tool| {
                    self.get_supported_tools()
                        .iter()
                        .find(|supported| {
                            std::mem::discriminant(tool) == std::mem::discriminant(supported)
                        })
                        .and_then(|_| tool.get_config_json())
                })
                .collect();

            if !processed_tools.is_empty() {
                body["tools"] = json!(processed_tools);
            }
        }

        body
    }

    /*
     * This function leverages Google API to perform any query as per the provided body.
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
        // If no version provided default to Google Studio API
        let api_version = version
            .as_ref()
            .map(|version| GoogleApiEndpoints::from_str(version))
            .unwrap_or_default();

        match (self, api_version) {
            // Google Studio API
            (
                GoogleModels::Gemini1_5Pro
                | GoogleModels::Gemini1_5Flash
                | GoogleModels::Gemini1_5Flash8B
                | GoogleModels::Gemini2_0Flash
                | GoogleModels::Gemini2_0FlashLite
                | GoogleModels::Gemini2_0ProExp
                | GoogleModels::Gemini2_0FlashThinkingExp
                | GoogleModels::Gemini2_5Flash
                | GoogleModels::Gemini2_5Pro
                | GoogleModels::Gemini2_5FlashLite,
                GoogleApiEndpoints::GoogleStudio,
            ) => self.call_api_studio(api_key, version, body, debug).await,
            // Fine-tuned models are only available in the Vertex API
            // TODO: Explore fine-tuned models in the Studio API
            (GoogleModels::FineTunedEndpoint { .. }, GoogleApiEndpoints::GoogleStudio) => {
                self.call_api_vertex(api_key, version, body, debug).await
            }
            // Google Vertex API
            (
                GoogleModels::Gemini1_5Pro
                | GoogleModels::Gemini1_5Flash
                | GoogleModels::Gemini1_5Flash8B
                | GoogleModels::Gemini2_0Flash
                | GoogleModels::Gemini2_0FlashLite
                | GoogleModels::Gemini2_0ProExp
                | GoogleModels::Gemini2_0FlashThinkingExp
                | GoogleModels::Gemini2_5Flash
                | GoogleModels::Gemini2_5Pro
                | GoogleModels::Gemini2_5FlashLite,
                GoogleApiEndpoints::GoogleVertex,
            ) => {
                self.call_api_vertex_stream(api_key, version, body, debug)
                    .await
            }
            // Google Vertex API for fine-tuned models
            (GoogleModels::FineTunedEndpoint { .. }, GoogleApiEndpoints::GoogleVertex) => {
                self.call_api_vertex(api_key, version, body, debug).await
            }
            // Legacy approach to Google Vertex API
            #[allow(deprecated)]
            (
                GoogleModels::Gemini1_5ProVertex
                | GoogleModels::Gemini1_5FlashVertex
                | GoogleModels::Gemini1_5Flash8BVertex
                | GoogleModels::Gemini2_0FlashVertex
                | GoogleModels::Gemini2_0FlashLiteVertex
                | GoogleModels::Gemini2_0ProExpVertex
                | GoogleModels::Gemini2_0FlashThinkingExpVertex,
                _,
            ) => {
                self.call_api_vertex_stream(api_key, version, body, debug)
                    .await
            }
        }
    }

    fn get_version_data(
        &self,
        response_text: &str,
        _function_call: bool,
        version: Option<String>,
    ) -> Result<String> {
        // If no version provided default to Google Studio API
        let version = version
            .map(|version| GoogleApiEndpoints::from_str(&version))
            .unwrap_or_default();

        match (self, version) {
            // Google Studio API
            (
                GoogleModels::Gemini1_5Pro
                | GoogleModels::Gemini1_5Flash
                | GoogleModels::Gemini1_5Flash8B
                | GoogleModels::Gemini2_0Flash
                | GoogleModels::Gemini2_0FlashLite
                | GoogleModels::Gemini2_0ProExp
                | GoogleModels::Gemini2_0FlashThinkingExp
                | GoogleModels::Gemini2_5Flash
                | GoogleModels::Gemini2_5Pro
                | GoogleModels::Gemini2_5FlashLite,
                GoogleApiEndpoints::GoogleStudio,
            ) => self.get_generate_content_data(response_text),
            // Fine-tuned models are only available in the Vertex API
            // TODO: Explore fine-tuned models in the Studio API
            (GoogleModels::FineTunedEndpoint { .. }, GoogleApiEndpoints::GoogleStudio) => {
                self.get_generate_content_data(response_text)
            }
            // Because for Vertex we are using streaming the extraction of data/text is handled in call_api method. Here we only pass the input forward
            (
                GoogleModels::Gemini1_5Pro
                | GoogleModels::Gemini1_5Flash
                | GoogleModels::Gemini1_5Flash8B
                | GoogleModels::Gemini2_0Flash
                | GoogleModels::Gemini2_0FlashLite
                | GoogleModels::Gemini2_0ProExp
                | GoogleModels::Gemini2_0FlashThinkingExp
                | GoogleModels::Gemini2_5Flash
                | GoogleModels::Gemini2_5Pro
                | GoogleModels::Gemini2_5FlashLite,
                GoogleApiEndpoints::GoogleVertex,
            ) => Ok(response_text.to_string()),
            // Google Vertex API for fine-tuned models
            (GoogleModels::FineTunedEndpoint { .. }, GoogleApiEndpoints::GoogleVertex) => {
                self.get_generate_content_data(response_text)
            }
            // Legacy approach to Vertex API implementation
            #[allow(deprecated)]
            (
                GoogleModels::Gemini1_5ProVertex
                | GoogleModels::Gemini1_5FlashVertex
                | GoogleModels::Gemini1_5Flash8BVertex
                | GoogleModels::Gemini2_0FlashVertex
                | GoogleModels::Gemini2_0FlashLiteVertex
                | GoogleModels::Gemini2_0ProExpVertex
                | GoogleModels::Gemini2_0FlashThinkingExpVertex,
                _,
            ) => Ok(response_text.to_string()),
        }
    }

    //This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        //Docs: https://ai.google.dev/gemini-api/docs/rate-limits#tier-3
        match self {
            GoogleModels::Gemini1_5Flash | GoogleModels::Gemini1_5FlashVertex => RateLimit {
                tpm: 4_000_000,
                rpm: 2_000,
            },
            GoogleModels::Gemini1_5Flash8B | GoogleModels::Gemini1_5Flash8BVertex => RateLimit {
                tpm: 4_000_000,
                rpm: 4_000,
            },
            GoogleModels::Gemini1_5Pro | GoogleModels::Gemini1_5ProVertex => RateLimit {
                tpm: 4_000_000,
                rpm: 1_000,
            },
            GoogleModels::Gemini2_0Flash | GoogleModels::Gemini2_0FlashVertex => RateLimit {
                tpm: 30_000_000,
                rpm: 30_000,
            },
            GoogleModels::Gemini2_0FlashLite | GoogleModels::Gemini2_0FlashLiteVertex => {
                RateLimit {
                    tpm: 30_000_000,
                    rpm: 30_000,
                }
            }
            GoogleModels::Gemini2_5Flash => RateLimit {
                tpm: 8_000_000,
                rpm: 10_000,
            },
            GoogleModels::Gemini2_5Pro => RateLimit {
                tpm: 8_000_000,
                rpm: 2_000,
            },
            GoogleModels::Gemini2_5FlashLite => RateLimit {
                tpm: 30_000_000,
                rpm: 30_000,
            },
            // Fine-tuned models use 2.0 Flash and Flash Lite rate limits
            GoogleModels::FineTunedEndpoint { .. } => RateLimit {
                tpm: 30_000_000,
                rpm: 30_000,
            },
            // TODO: No rate limits published for experimental models
            _ => RateLimit {
                tpm: 120_000,
                rpm: 360,
            },
        }
    }
}

impl GoogleModels {
    /// Constructor of the fine-tuned model endpoint
    /// Fine-tuned models are available in the Vertex API via the endpoint ID
    pub fn endpoint(name: &str) -> Self {
        GoogleModels::FineTunedEndpoint {
            name: name.to_string(),
        }
    }

    // Specialized function for calling AI Studio API
    async fn call_api_studio(
        &self,
        api_key: &str,
        version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_version_endpoint(version);

        //Make the API call
        let client = Client::new();

        //Send request
        let url_with_key = format!("{}?key={}", model_url, api_key);
        let response = client
            .post(url_with_key)
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if debug {
            info!(
                "[allms][Google AI Studio] API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    // Specialized function for calling Vertex API with streaming
    async fn call_api_vertex_stream(
        &self,
        api_key: &str,
        version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
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

        //For Vertex we are streaming that data so we need to deserialize each chunk separately
        // Check if the API uses streaming
        if response.status().is_success() {
            let mut stream = response.bytes_stream();
            let mut streamed_response = String::new();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;

                // Convert the chunk (Bytes) to a String
                let mut chunk_str = String::from_utf8(chunk.to_vec()).map_err(|e| anyhow!(e))?;

                // The chunk response starts with "data: " that needs to be remove
                if chunk_str.starts_with("data: ") {
                    // Remove the first 6 characters ("data: ")
                    chunk_str = chunk_str[6..].to_string();
                }

                //Convert response chunk to struct representing expected response format
                let gemini_response: GoogleGeminiProApiResp = serde_json::from_str(&chunk_str)?;

                //Extract the data part from the response
                let part_text = gemini_response
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.content.role.as_deref() == Some("model"))
                    .flat_map(|candidate| &candidate.content.parts)
                    .filter_map(|part| part.text.as_ref())
                    .fold(String::new(), |mut acc, text| {
                        acc.push_str(text);
                        acc
                    });

                //Add the chunk response to output string
                streamed_response.push_str(&part_text);

                // Debug log each chunk if needed
                if debug {
                    info!(
                        "[allms][Google Vertex AI] Received response chunk: {:?}",
                        chunk
                    );
                }
            }
            Ok(self.sanitize_json_response(&streamed_response))
        } else {
            let response_status = response.status();
            let response_txt = response.text().await?;
            Err(anyhow!(
                "[allms][Google][{}] Response body: {:#?}",
                response_status,
                response_txt
            ))
        }
    }

    // Specialized function for calling Vertex API without streaming (used for fine-tuned models)
    async fn call_api_vertex(
        &self,
        api_key: &str,
        version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
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
                "[allms][Google AI Vertex][Fine-tuned] API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    // Specialized function for parsing response of the generateContent API (non-streaming)
    fn get_generate_content_data(&self, response_text: &str) -> Result<String> {
        //Convert response to struct representing expected response format
        let gemini_response: GoogleGeminiProApiResp = serde_json::from_str(response_text)?;

        //Extract the data part from the response
        let data = gemini_response
            .candidates
            .iter()
            .filter(|candidate| candidate.content.role.as_deref() == Some("model"))
            .flat_map(|candidate| &candidate.content.parts)
            .filter_map(|part| part.text.as_ref())
            .fold(String::new(), |mut acc, text| {
                acc.push_str(text);
                acc
            });

        Ok(self.sanitize_json_response(&data))
    }

    fn get_supported_tools(&self) -> Vec<LLMTools> {
        match self {
            GoogleModels::Gemini2_5Pro
            | GoogleModels::Gemini2_5Flash
            | GoogleModels::Gemini2_5FlashLite
            | GoogleModels::Gemini2_0Flash => vec![
                LLMTools::GeminiCodeInterpreter(GeminiCodeInterpreterConfig::new()),
                LLMTools::GeminiWebSearch(GeminiWebSearchConfig::new()),
            ],
            _ => vec![],
        }
    }
}
