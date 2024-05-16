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
    pub file_ids: Option<Vec<String>>,
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
    pub attachments: Option<Vec<OpenAIMessageAttachment>>,
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
pub struct OpenAIMessageAttachment {
    pub file_id: String,
    pub tools: Vec<OpenAIMessageAttachmentTools>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIMessageAttachmentTools {
    #[serde(rename(deserialize = "type", serialize = "type"))]
    pub tool_type: OpenAIToolTypes,
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

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIDFileDeleteResp {
    id: String,
    object: String,
    deleted: bool,
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
        // OpenAI documentation: https://platform.openai.com/docs/assistants/tools/supported-files
        let mime_type = match Path::new(file_name)
            .extension()
            .and_then(std::ffi::OsStr::to_str)
        {
            Some("pdf") => "application/pdf",
            Some("json") => "application/json",
            Some("txt") => "text/plain",
            Some("html") => "text/html",
            Some("c") => "text/x-c",
            Some("cpp") => "text/x-c++",
            Some("docx") => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            Some("java") => "text/x-java",
            Some("md") => "text/markdown",
            Some("php") => "text/x-php",
            Some("pptx") => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
            Some("py") => "text/x-python",
            Some("rb") => "text/x-ruby",
            Some("tex") => "text/x-tex",
            //The below are currently only supported for Code Interpreter but NOT Retrieval
            Some("css") => "text/css",
            Some("jpeg") | Some("jpg") => "image/jpeg",
            Some("js") => "text/javascript",
            Some("gif") => "image/gif",
            Some("png") => "image/png",
            Some("tar") => "application/x-tar",
            Some("ts") => "application/typescript",
            Some("xlsx") => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Some("xml") => "application/xml",
            Some("zip") => "application/zip",
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

    /*
     * This function deletes a file from OpenAI
     */
    pub async fn delete_file(&self) -> Result<()> {
        let files_url = format!("https://api.openai.com/v1/files/{}", self.id);

        //Make the API call
        let client = Client::new();

        let response = client
            .delete(files_url)
            .bearer_auth(&self.api_key)
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

        //Check if the file was successfully deleted
        serde_json::from_str::<OpenAIDFileDeleteResp>(&response_text)
            .map_err(|error| {
                error!(
                    "[OpenAIAssistant] Files Delete API response serialization error: {}",
                    &error
                );
                anyhow!(
                    "[OpenAIAssistant] Files Delete API response serialization error: {}",
                    error
                )
            })
            .and_then(|response| match response.deleted {
                true => Ok(()),
                false => Err(anyhow!("[OpenAIAssistant] Failed to delete the file.")),
            })
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

///Google GeminiPro API response deserialization structs
#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProApiResp {
    pub candidates: Vec<GoogleGeminiProCandidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GoogleGeminiProUsageMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProCandidate {
    pub content: GoogleGeminiProContent,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    #[serde(rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<GoogleGeminiProSafetyRating>>,
    #[serde(rename = "citationMetadata")]
    pub citation_metadata: Option<GoogleGeminiProCitationMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProContent {
    pub parts: Vec<GoogleGeminiProPart>,
    pub role: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProPart {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProSafetyRating {
    pub category: String,
    pub probability: String,
    pub blocked: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProCitationMetadata {
    pub citations: Vec<GoogleGeminiProCitation>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProCitation {
    #[serde(rename = "startIndex")]
    pub start_index: i32,
    #[serde(rename = "endIndex")]
    pub end_index: i32,
    pub uri: String,
    pub title: Option<String>,
    pub license: Option<String>,
    #[serde(rename = "publicationDate")]
    pub publication_date: Option<GoogleGeminiProDate>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProDate {
    pub year: i32,
    pub month: i32,
    pub day: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiProUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: i32,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: i32,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: i32,
}
