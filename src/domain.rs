use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

//Anthropic API response type format for Messages API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AnthropicAPIMessagesResponse {
    pub id: String,
    #[serde(rename(deserialize = "type", serialize = "type"))]
    pub request_type: String,
    pub role: String,
    pub content: Vec<AnthropicAPIMessagesContent>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicAPIMessagesUsage,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AnthropicAPIMessagesContent {
    #[serde(rename(deserialize = "type", serialize = "type"))]
    pub content_type: String,
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AnthropicAPIMessagesUsage {
    pub input_tokens: i32,
    pub output_tokens: i32,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct AllmsError {
    #[serde(rename = "crate")]
    pub crate_name: String,
    pub module: String,
    pub error_message: String,
    pub error_detail: String,
}
