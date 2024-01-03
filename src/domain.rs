use anyhow::{anyhow, Context, Result};
use log::{error, info};
use reqwest::{header, multipart, Client};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::enums::{OpenAIAssistantRole, OpenAIRunStatus, OpenAIToolTypes};

//OpenAI API response type format for Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAPICompletionsResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<u32>,
    pub model: Option<String>,
    pub choices: Option<Vec<OpenAPICompletionsChoices>>,
    pub usage: Option<OpenAPIUsage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAPICompletionsChoices {
    pub text: Option<String>,
    pub index: Option<u32>,
    pub logprobs: Option<u32>,
    pub finish_reason: Option<String>,
}

//OpenAI API response type format for Chat API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAPIChatResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<u32>,
    pub model: Option<String>,
    pub choices: Option<Vec<OpenAPIChatChoices>>,
    pub usage: Option<OpenAPIUsage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAPIChatChoices {
    pub message: OpenAPIChatMessage,
    pub index: Option<u32>,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAPIChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub function_call: Option<OpenAPIChatFunctionCall>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAPIChatFunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAPIUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct RateLimit {
    pub tpm: usize, // tokens-per-minute
    pub rpm: usize, // requests-per-minute
}

#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct OpenAIDataResponse<T: JsonSchema> {
    pub data: T,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIAssistantResp {
    pub id: String,
    pub object: String,
    pub created_at: u32,
    pub name: Option<String>,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub model: String,
    pub tools: Vec<OpenAITools>,
    pub file_ids: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAITools {
    #[serde(rename(deserialize = "type", serialize = "type"))]
    tool_type: OpenAIToolTypes,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIThreadResp {
    pub id: String,
    pub object: String,
    pub created_at: u32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIMessageListResp {
    pub object: String,
    pub data: Vec<OpenAIMessageResp>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIMessageResp {
    pub id: String,
    pub object: String,
    pub created_at: u32,
    pub thread_id: String,
    pub role: OpenAIAssistantRole,
    pub content: Vec<OpenAIContent>,
    //Other fields omitted as no use for now
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIContent {
    #[serde(rename(deserialize = "type", serialize = "type"))]
    pub content_type: String,
    pub text: Option<OpenAIContentText>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIContentText {
    pub value: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIRunResp {
    pub id: String,
    pub object: String,
    pub created_at: u32,
    pub status: OpenAIRunStatus,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIFile {
    pub id: String,
    debug: bool,
    api_key: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIFileResp {
    id: String,
}

impl OpenAIFile {
    //Constructor
    pub async fn new(
        file_name: &str,
        file_bytes: Vec<u8>,
        open_ai_key: &str,
        debug: bool,
    ) -> Result<Self> {
        let mut new_file = OpenAIFile {
            id: "this-will-be-overwritten".to_string(),
            debug,
            api_key: open_ai_key.to_string(),
        };
        //Upload file and get the ID
        new_file.upload_file(file_name, file_bytes).await?;
        Ok(new_file)
    }

    /*
     * This function uploads a file to OpenAI and assigns it for use with Assistant API
     */
    async fn upload_file(&mut self, file_name: &str, file_bytes: Vec<u8>) -> Result<()> {
        let files_url = "https://api.openai.com/v1/files";

        // Determine MIME type based on file extension
        let mime_type = match Path::new(file_name)
            .extension()
            .and_then(std::ffi::OsStr::to_str)
        {
            Some("pdf") => "application/pdf",
            Some("json") => "application/json",
            Some("txt") => "text/plain",
            _ => anyhow::bail!("Unsupported file type"),
        };

        let form = multipart::Form::new().text("purpose", "assistants").part(
            "file",
            multipart::Part::bytes(file_bytes)
                .file_name(file_name.to_string())
                .mime_str(mime_type)
                .context("Failed to set MIME type")?,
        );

        //Make the API call
        let client = Client::new();

        let response = client
            .post(files_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Files status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let response_deser: OpenAIFileResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[OpenAIAssistant] Files API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        self.id = response_deser.id;

        Ok(())
    }
}

//Anthropic API response type format for Text Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AnthropicAPICompletionsResponse {
    pub id: String,
    #[serde(rename(deserialize = "type", serialize = "type"))]
    pub request_type: String,
    pub completion: String,
    pub stop_reason: String,
    pub model: String,
}

//Mistral API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPICompletionsResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<usize>,
    pub model: Option<String>,
    pub choices: Vec<MistralAPICompletionsChoices>,
    pub usage: Option<MistralAPICompletionsUsage>,
}

//Mistral API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPICompletionsChoices {
    pub index: usize,
    pub message: Option<MistralAPICompletionsMessage>,
    pub finish_reason: String,
}

//Mistral API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPICompletionsMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

//Mistral API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPICompletionsUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}
