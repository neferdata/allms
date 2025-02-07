use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::stream::StreamExt;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::constants::{GOOGLE_GEMINI_API_URL, GOOGLE_VERTEX_API_URL};
use crate::domain::{GoogleGeminiProApiResp, RateLimit};
use crate::llm_models::LLMModel;

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
//Google docs: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
pub enum GoogleModels {
    GeminiPro,
    Gemini1_5Flash,
    Gemini1_5Pro,
    Gemini1_0Pro,
    Gemini2_0Flash,
    Gemini2_0FlashLite,
    Gemini2_0ProExp,
    Gemini2_0FlashThinkingExp,
    // Vertex
    GeminiProVertex,
    Gemini1_5FlashVertex,
    Gemini1_5ProVertex,
    Gemini1_0ProVertex,
}

#[async_trait(?Send)]
impl LLMModel for GoogleModels {
    fn as_str(&self) -> &str {
        match self {
            GoogleModels::GeminiPro | GoogleModels::GeminiProVertex => "gemini-pro",
            GoogleModels::Gemini1_5Pro | GoogleModels::Gemini1_5ProVertex => "gemini-1.5-pro",
            GoogleModels::Gemini1_5Flash | GoogleModels::Gemini1_5FlashVertex => "gemini-1.5-flash",
            GoogleModels::Gemini1_0Pro | GoogleModels::Gemini1_0ProVertex => "gemini-1.0-pro",
            GoogleModels::Gemini2_0Flash => "gemini-2.0-flash-001",
            GoogleModels::Gemini2_0FlashLite => "gemini-2.0-flash-lite-preview-02-05",
            GoogleModels::Gemini2_0ProExp => "gemini-2.0-pro-exp-02-05",
            GoogleModels::Gemini2_0FlashThinkingExp => "gemini-2.0-flash-thinking-exp-01-21",
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "gemini-pro" => Some(GoogleModels::GeminiPro),
            "gemini-pro-vertex" => Some(GoogleModels::GeminiProVertex),
            "gemini-1.5-pro" => Some(GoogleModels::Gemini1_5Pro),
            "gemini-1.5-pro-vertex" => Some(GoogleModels::Gemini1_5ProVertex),
            "gemini-1.5-flash" => Some(GoogleModels::Gemini1_5Flash),
            "gemini-1.5-flash-vertex" => Some(GoogleModels::Gemini1_5FlashVertex),
            "gemini-1.0-pro" => Some(GoogleModels::Gemini1_0Pro),
            "gemini-1.0-pro-vertex" => Some(GoogleModels::Gemini1_0ProVertex),
            "gemini-2.0-flash" => Some(GoogleModels::Gemini2_0Flash),
            "gemini-2.0-flash-lite" => Some(GoogleModels::Gemini2_0FlashLite),
            "gemini-2.0-pro" => Some(GoogleModels::Gemini2_0ProExp),
            "gemini-2.0-pro-exp" => Some(GoogleModels::Gemini2_0ProExp),
            "gemini-2.0-flash-thinking" => Some(GoogleModels::Gemini2_0FlashThinkingExp),
            "gemini-2.0-flash-thinking-exp" => Some(GoogleModels::Gemini2_0FlashThinkingExp),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        //https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models
        match self {
            GoogleModels::GeminiPro | GoogleModels::GeminiProVertex => 32_000,
            GoogleModels::Gemini1_5Pro | GoogleModels::Gemini1_5ProVertex => 1_048_576,
            GoogleModels::Gemini1_5Flash | GoogleModels::Gemini1_5FlashVertex => 1_048_576,
            GoogleModels::Gemini1_0Pro | GoogleModels::Gemini1_0ProVertex => 32_000,
            GoogleModels::Gemini2_0Flash => 1_048_576,
            GoogleModels::Gemini2_0FlashLite => 1_048_576,
            // TODO: Max tokens not yet documented for experimental models. Using defaults from others
            GoogleModels::Gemini2_0ProExp => 1_048_576,
            GoogleModels::Gemini2_0FlashThinkingExp => 1_048_576,
        }
    }

    fn get_endpoint(&self) -> String {
        //The URL requires GOOGLE_REGION and GOOGLE_PROJECT_ID env variables defined to work.
        //If not set GOOGLE_REGION will default to 'us-central1' but GOOGLE_PROJECT_ID needs to be defined.
        match self {
            GoogleModels::GeminiPro
            | GoogleModels::Gemini1_5Pro
            | GoogleModels::Gemini1_5Flash
            | GoogleModels::Gemini1_0Pro 
            | GoogleModels::Gemini2_0Flash
            | GoogleModels::Gemini2_0FlashLite
            | GoogleModels::Gemini2_0ProExp
            | GoogleModels::Gemini2_0FlashThinkingExp => GOOGLE_GEMINI_API_URL.to_string(),
            GoogleModels::GeminiProVertex
            | GoogleModels::Gemini1_5ProVertex
            | GoogleModels::Gemini1_5FlashVertex
            | GoogleModels::Gemini1_0ProVertex => GOOGLE_VERTEX_API_URL.to_string(),
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
    ) -> serde_json::Value {
        //Prepare the 'messages' part of the body
        let base_instructions_json = json!({
            "text": self.get_base_instructions(Some(function_call))
        });

        let schema_string = serde_json::to_string(json_schema).unwrap_or_default();
        let output_instructions_json =
            json!({ "text": format!("'Output Json schema': {schema_string}") });

        let user_instructions_json = json!({
            "text": instructions,
        });

        let contents = json!({
            "role": "user",
            "parts": vec![
                base_instructions_json,
                output_instructions_json,
                user_instructions_json,
            ],
        });

        let generation_config = json!({
            "temperature": temperature,
        });

        json!({
            "contents": contents,
            "generationConfig": generation_config,
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
        match &self {
            GoogleModels::GeminiProVertex
            | GoogleModels::Gemini1_5ProVertex
            | GoogleModels::Gemini1_5FlashVertex
            | GoogleModels::Gemini1_0ProVertex => {
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
                        let mut chunk_str =
                            String::from_utf8(chunk.to_vec()).map_err(|e| anyhow!(e))?;

                        // The chunk response starts with "data: " that needs to be remove
                        if chunk_str.starts_with("data: ") {
                            // Remove the first 6 characters ("data: ")
                            chunk_str = chunk_str[6..].to_string();
                        }

                        //Convert response chunk to struct representing expected response format
                        let gemini_response: GoogleGeminiProApiResp =
                            serde_json::from_str(&chunk_str)?;

                        //Extract the data part from the response
                        let part_text = gemini_response
                            .candidates
                            .iter()
                            .filter(|candidate| candidate.content.role.as_deref() == Some("model"))
                            .flat_map(|candidate| &candidate.content.parts)
                            .map(|part| &part.text)
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
            GoogleModels::GeminiPro
            | GoogleModels::Gemini1_5Pro
            | GoogleModels::Gemini1_5Flash
            | GoogleModels::Gemini1_0Pro
            | GoogleModels::Gemini2_0Flash
            | GoogleModels::Gemini2_0FlashLite
            | GoogleModels::Gemini2_0ProExp
            | GoogleModels::Gemini2_0FlashThinkingExp => {
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
        }
    }

    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        match self {
            //Because for Vertex we are using streaming the extraction of data/text is handled in call_api method. Here we only pass the input forward
            GoogleModels::GeminiProVertex
            | GoogleModels::Gemini1_5ProVertex
            | GoogleModels::Gemini1_5FlashVertex
            | GoogleModels::Gemini1_0ProVertex => Ok(response_text.to_string()),
            GoogleModels::GeminiPro
            | GoogleModels::Gemini1_5Pro
            | GoogleModels::Gemini1_5Flash
            | GoogleModels::Gemini1_0Pro 
            | GoogleModels::Gemini2_0Flash
            | GoogleModels::Gemini2_0FlashLite
            | GoogleModels::Gemini2_0ProExp
            | GoogleModels::Gemini2_0FlashThinkingExp => {
                //Convert response to struct representing expected response format
                let gemini_response: GoogleGeminiProApiResp = serde_json::from_str(response_text)?;

                //Extract the data part from the response
                Ok(gemini_response
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.content.role.as_deref() == Some("model"))
                    .flat_map(|candidate| &candidate.content.parts)
                    .map(|part| &part.text)
                    .fold(String::new(), |mut acc, text| {
                        acc.push_str(text);
                        acc
                    }))
            }
        }
    }

    //This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        //Docs: https://ai.google.dev/gemini-api/docs/models/gemini
        match self {
            GoogleModels::Gemini2_0Flash => RateLimit {
                tpm: 4_000_000, 
                rpm: 2_000,
            },
            GoogleModels::Gemini2_0FlashLite => RateLimit {
                tpm: 4_000_000, 
                rpm: 10,
            },
            // TODO: Update others
            _ => RateLimit {
                tpm: 60 * 32_000, 
                rpm: 60,
            }
        }
        
    }
}
