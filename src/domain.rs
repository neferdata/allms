use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    pub text: Option<String>,
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

// Mistral Agents API response type format
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPIConversationsResponse {
    pub conversation_id: String,
    pub object: String,
    pub outputs: Vec<MistralAPIConversationsOutput>,
    pub usage: Option<MistralAPIConversationsUsage>,
}

/// Mistral Conversations API output entry types
/// Can be MessageOutputEntry | ToolExecutionEntry | FunctionCallEntry | AgentHandoffEntry
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MistralAPIConversationsOutput {
    #[serde(rename = "message.output")]
    MistralAPIConversationsMessageOutput(MistralAPIConversationsMessageOutput),
    #[serde(rename = "tool.execution")]
    MistralAPIConversationsToolExecution(MistralAPIConversationsToolExecution),
    // TODO: Add other entry types when needed
    // MistralAPIConversationsFunctionCall(MistralAPIConversationsFunctionCall),
    // MistralAPIConversationsAgentHandoff(MistralAPIConversationsAgentHandoff),
}

/// MessageOutputEntry for Mistral Conversations API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPIConversationsMessageOutput {
    pub object: Option<String>,
    #[serde(rename = "type")]
    pub entry_type: Option<String>,
    pub role: Option<String>,
    pub id: Option<String>,
    pub created_at: Option<String>,
    pub agent_id: Option<String>,
    pub completed_at: Option<String>,
    pub model: Option<String>,
    pub content: Option<MistralAPIConversationsMessageOutputContent>,
}

/// Content can be a string or array of chunk types (TextChunk, ImageURLChunk, etc.)
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum MistralAPIConversationsMessageOutputContent {
    MistralAPIConversationsMessageOutputContentString(String),
    MistralAPIConversationsMessageOutputContentChunks(Vec<MistralAPIConversationsChunk>),
}

/// Chunk types for Mistral Conversations API content
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MistralAPIConversationsChunk {
    #[serde(rename = "text")]
    MistralAPIConversationsChunkText(MistralAPIConversationsChunkText),
    // TODO: Add other chunk types when needed
    // ImageURL(ImageURLChunk),
    // ToolFile(ToolFileChunk),
    // DocumentURL(DocumentURLChunk),
    // Think(ThinkChunk),
    // ToolReference(ToolReferenceChunk),
}

/// TextChunk for Mistral Conversations API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPIConversationsChunkText {
    pub text: String,
    #[serde(rename = "type")]
    #[serde(default = "default_text_chunk_type")]
    pub chunk_type: String,
}

fn default_text_chunk_type() -> String {
    "text".to_string()
}

/// ToolExecutionEntry for Mistral Conversations API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPIConversationsToolExecution {
    pub arguments: String,
    pub completed_at: Option<String>,
    pub created_at: String,
    pub id: String,
    pub info: Value,
    pub name: MistralAPIConversationsToolName,
    #[serde(default = "default_entry_object")]
    pub object: String,
    #[serde(rename = "type")]
    #[serde(default = "default_tool_execution_type")]
    pub entry_type: String,
}

/// Tool name enum for Mistral Conversations API
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MistralAPIConversationsToolName {
    #[serde(rename = "web_search")]
    WebSearch,
    #[serde(rename = "web_search_premium")]
    WebSearchPremium,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "image_generation")]
    ImageGeneration,
    #[serde(rename = "document_library")]
    DocumentLibrary,
}

fn default_entry_object() -> String {
    "entry".to_string()
}

fn default_tool_execution_type() -> String {
    "tool.execution".to_string()
}

// Mistral API response type format for Conversations API
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MistralAPIConversationsUsage {
    pub completion_tokens: usize,
    pub connector_tokens: usize,
    pub prompt_tokens: usize,
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
    pub text: Option<String>,
    #[serde(rename = "executableCode")]
    pub executable_code: Option<GoogleGeminiExecutableCode>,
    #[serde(rename = "codeExecutionResult")]
    pub code_execution_result: Option<GoogleGeminiCodeExecutionResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiExecutableCode {
    pub language: String,
    pub code: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGeminiCodeExecutionResult {
    pub outcome: String,
    pub output: Option<String>,
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

/***************************************************************************************************
*
* OpenAI Responses API
*
***************************************************************************************************/
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
    pub text: Option<OpenAPIResponsesTextFormat>,
    // pub tool_choice: OpenAPIResponsesToolChoice,
    // pub tools: Vec<OpenAPIResponsesTool>,
    pub top_p: Option<f32>,
    pub usage: OpenAPIResponsesUsage,
    pub user: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesError {
    pub code: Option<String>,
    pub message: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesIncompleteDetails {
    pub reason: Option<String>,
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
    pub r#type: Option<String>,
    pub text: Option<String>,
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
    pub format: Option<OpenAPIResponsesFormat>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesFormat {
    pub r#type: Option<OpenAPIResponsesFormatType>,
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
    pub r#type: Option<String>,
    pub function: Option<OpenAPIResponsesToolChoiceFunction>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesToolChoiceFunction {
    pub name: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAPIResponsesTool {
    pub r#type: Option<OpenAPIResponsesToolType>,
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
    pub name: Option<String>,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
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

/***************************************************************************************************
*
* xAI
*
***************************************************************************************************/
///
/// xAI API Request
///
#[derive(Serialize, Deserialize)]
pub struct XAIChatRequest {
    pub model: String,
    pub messages: Vec<XAIChatMessage>,
    pub temperature: Option<f32>,
    pub max_completion_tokens: Option<usize>,
    pub response_format: Option<XAIResponseFormat>,
    pub search_parameters: Option<XAIWebSearchConfig>,
    pub tools: Option<Vec<XAITool>>,
    // TODO: Future implementations
    // pub tool_choice: Option<XAIToolChoice>, // Controls which (if any) tool is called by the model. `none` is the default when no tools are present. `auto`` is the default if tools are present.
    // pub parallel_tool_calls: Option<bool>, // If set to false, the model can perform maximum one tool call.
    // pub reasoning_effort: Option<String>, // Can be added later via Reasoning Tool
}

#[derive(Serialize, Deserialize, Default)]
pub struct XAIChatMessage {
    pub role: XAIRole,
    pub content: Option<XAIContentContent>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<XAIToolCall>>,
    pub tool_call_id: Option<String>,
}

impl XAIChatMessage {
    pub fn new(role: XAIRole, content: String) -> Self {
        Self {
            role,
            content: Some(XAIContentContent::String(content)),
            ..Default::default()
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum XAIRole {
    System,
    #[default]
    User,
    Assistant,
    Tool,
    Function,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum XAIContentContent {
    String(String),
    Parts(Vec<XAIContentPart>),
}

#[derive(Serialize, Deserialize)]
pub struct XAIContentPart {
    #[serde(rename = "type")]
    pub content_type: XAIContentType,
    pub text: Option<String>,
    pub image_url: Option<XAIContentImageUrl>,
    pub text_file: Option<String>,
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XAIContentType {
    Text,
    ImageUrl,
    TextFile,
}

#[derive(Serialize, Deserialize)]
pub struct XAIContentImageUrl {
    pub url: String,
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct XAIToolCall {
    pub function: XAIToolFunction,
    pub id: String,
    pub index: Option<u32>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct XAIToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct XAIResponseFormat {
    #[serde(rename = "type")]
    pub r#type: XAIResponseFormatType,
    pub json_schema: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XAIResponseFormatType {
    Text,
    JsonObject,
    JsonSchema,
}

#[derive(Serialize, Deserialize)]
pub struct XAIResponseFormatJsonObject {
    pub r#type: String,
}

#[derive(Serialize, Deserialize)]
pub struct XAIResponseFormatJsonSchema {
    pub r#type: String,
    pub json_schema: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct XAIWebSearchConfig {
    pub from_date: Option<String>,
    pub to_date: Option<String>,
    pub max_search_results: Option<usize>,
    pub mode: Option<XAISearchMode>,
    pub return_citations: Option<bool>,
    pub sources: Option<Vec<XAISearchSource>>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum XAISearchMode {
    On,
    Off,
    #[default]
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum XAISearchSource {
    Web(WebSource),
    X(XSource),
    News(NewsSource),
    Rss(RssSource),
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct WebSource {
    pub allowed_websites: Option<Vec<String>>,
    pub excluded_websites: Option<Vec<String>>,
    pub country: Option<String>,
    pub safe_search: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct XSource {
    pub included_x_handles: Option<Vec<String>>,
    pub excluded_x_handles: Option<Vec<String>>,
    pub post_favorite_count: Option<usize>,
    pub post_view_count: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct NewsSource {
    pub excluded_websites: Option<Vec<String>>,
    pub country: Option<String>,
    pub safe_search: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct RssSource {
    pub links: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct XAITool {
    #[serde(rename = "type")]
    pub tool_type: XAIToolType,
    pub function: Option<XAIToolDefFunction>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XAIToolType {
    Function,
}

#[derive(Serialize, Deserialize)]
pub struct XAIToolDefFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

///
/// xAI API Response
///
#[derive(Debug, Serialize, Deserialize)]
pub struct XAIChatResponse {
    pub id: String,
    pub object: Option<String>, // should be "chat.completion"
    pub created: Option<u64>,
    pub model: Option<String>,
    pub choices: Vec<XAIChatChoice>,
    pub usage: Option<XAIUsage>,
    pub citations: Option<Vec<String>>,
    pub debug_output: Option<Value>,
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XAIChatChoice {
    pub index: u32,
    pub message: XAIAssistantMessage,
    pub finish_reason: Option<String>,
    pub logprobs: Option<Value>, // You can replace with a struct if you want structured access
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XAIAssistantMessage {
    pub role: XAIAssistantMessageRole,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<XAIAssistantMessageToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum XAIAssistantMessageRole {
    #[default]
    Assistant,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XAIAssistantMessageToolCall {
    pub id: String,
    pub index: Option<u32>,
    #[serde(rename = "type")]
    pub call_type: XAIAssistantMessageToolCallType,
    pub function: XAIAssistantMessageToolFunction,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum XAIAssistantMessageToolCallType {
    #[default]
    Function,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XAIAssistantMessageToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XAIUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub completion_tokens_details: Option<XAICompletionTokenDetails>,
    pub prompt_tokens_details: Option<XAIPromptTokenDetails>,
    pub num_sources_used: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XAICompletionTokenDetails {
    pub accepted_prediction_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
    pub rejected_prediction_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XAIPromptTokenDetails {
    pub audio_tokens: Option<u32>,
    pub cached_tokens: Option<u32>,
    pub image_tokens: Option<u32>,
    pub text_tokens: Option<u32>,
}
