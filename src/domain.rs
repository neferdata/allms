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
    pub prompt_token_count: Option<i32>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<i32>,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AllmsError {
    #[serde(rename = "crate")]
    pub crate_name: String,
    pub module: String,
    pub error_message: String,
    pub error_detail: String,
}

// Perplexity API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PerplexityAPICompletionsResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub object: Option<String>,
    pub created: Option<usize>,
    pub choices: Vec<PerplexityAPICompletionsChoices>,
    pub citations: Option<Vec<String>>,
    pub usage: Option<PerplexityAPICompletionsUsage>,
}

// Perplexity API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PerplexityAPICompletionsChoices {
    pub index: usize,
    pub message: Option<PerplexityAPICompletionsMessage>,
    pub delta: Option<PerplexityAPICompletionsMessage>,
    pub finish_reason: String,
}

// Perplexity API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PerplexityAPICompletionsMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

// Perplexity API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PerplexityAPICompletionsUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// DeepSeek API response type format for Chat Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DeepSeekAPICompletionsResponse {
    pub id: Option<String>,
    pub choices: Vec<DeepSeekAPICompletionsChoices>,
    pub created: Option<usize>,
    pub model: Option<String>,
    pub system_fingerprint: Option<String>,
    pub object: Option<String>,
    pub usage: Option<DeepSeekAPICompletionsUsage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DeepSeekAPICompletionsChoices {
    pub index: usize,
    pub finish_reason: String,
    pub message: Option<DeepSeekAPICompletionsMessage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DeepSeekAPICompletionsMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DeepSeekAPICompletionsUsage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub prompt_cache_hit_tokens: usize,
    pub prompt_cache_miss_tokens: usize,
    pub total_tokens: usize,
    pub completion_tokens_details: Option<DeepSeekAPICompletionsReasoningUsage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DeepSeekAPICompletionsReasoningUsage {
    pub reasoning_tokens: usize,
}

// OpenAI Responses API response type format
#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created_at: Option<i64>,
    pub status: Option<OpenAPIResponsesStatus>,
    pub error: Option<OpenAPIResponsesError>,
    pub incomplete_details: Option<OpenAPIResponsesIncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<i32>,
    pub model: String,
    pub output: Vec<OpenAPIResponsesOutput>,
    // pub previous_response_id: Option<String>,
    // pub reasoning: Option<OpenAPIResponsesReasoning>,
    pub temperature: Option<f32>,
    pub text: OpenAPIResponsesTextFormat,
    // pub tool_choice: OpenAPIResponsesToolChoice,
    // pub tools: Vec<OpenAPIResponsesTool>,
    pub top_p: Option<f32>,
    pub usage: OpenAPIResponsesUsage,
    pub user: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesError {
    pub code: String,
    pub message: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesIncompleteDetails {
    pub reason: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesOutput {
    pub r#type: Option<OpenAPIResponsesOutputType>,
    pub id: Option<String>,
    pub status: Option<OpenAPIResponsesMessageStatus>,
    pub role: Option<OpenAPIResponsesRole>,
    pub content: Option<Vec<OpenAPIResponsesContent>>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesOutputType {
    Message,
    FileSearchCall,
    FunctionCall,
    WebSearchCall,
    ComputerCall,
    Reasoning,
    CodeInterpreterCall,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesMessageStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesRole {
    Assistant,
    User,
    System,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesContent {
    pub r#type: OpenAPIResponsesContentType,
    pub text: Option<String>,
    pub annotations: Option<Vec<OpenAPIResponsesAnnotation>>,
    pub refusal: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesContentType {
    OutputText,
    Refusal,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesAnnotation {
    pub r#type: String,
    pub text: String,
    pub start_index: Option<i32>,
    pub end_index: Option<i32>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesReasoning {
    pub effort: Option<String>,
    pub summary: Option<String>,
    pub service_tier: Option<OpenAPIResponsesServiceTier>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesServiceTier {
    Auto,
    Default,
    Flex,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesTextFormat {
    pub format: OpenAPIResponsesFormat,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesFormat {
    pub r#type: OpenAPIResponsesFormatType,
    pub name: Option<String>,
    pub schema: Option<serde_json::Value>,
    pub description: Option<String>,
    pub strict: Option<bool>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesFormatType {
    Text,
    JsonSchema,
    JsonObject,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(untagged)]
pub enum OpenAPIResponsesToolChoice {
    String(String),
    Object(OpenAPIResponsesToolChoiceObject),
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesToolChoiceObject {
    pub r#type: String,
    pub function: Option<OpenAPIResponsesToolChoiceFunction>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesToolChoiceFunction {
    pub name: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesTool {
    pub r#type: OpenAPIResponsesToolType,
    pub function: Option<OpenAPIResponsesToolFunction>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesToolType {
    Function,
    FileSearch,
    WebSearch,
    Computer,
    CodeInterpreter,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesTruncationStrategy {
    Auto,
    Disabled,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesUsage {
    pub input_tokens: i32,
    pub input_tokens_details: OpenAPIResponsesTokenDetails,
    pub output_tokens: i32,
    pub output_tokens_details: OpenAPIResponsesOutputTokenDetails,
    pub total_tokens: i32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesTokenDetails {
    pub cached_tokens: i32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesOutputTokenDetails {
    pub reasoning_tokens: i32,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAPIResponsesStatus {
    Completed,
    Failed,
    InProgress,
    Incomplete,
}
