use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum OpenAIToolTypes {
    #[serde(rename(deserialize = "code_interpreter", serialize = "code_interpreter"))]
    CodeInterpreter,
    #[serde(rename(deserialize = "retrieval", serialize = "retrieval"))]
    Retrieval,
    #[serde(rename(deserialize = "file_search", serialize = "file_search"))]
    FileSearch,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIAssistantRole {
    #[serde(rename(deserialize = "user", serialize = "user"))]
    User,
    #[serde(rename(deserialize = "assistant", serialize = "assistant"))]
    Assistant,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum OpenAIMessageAttachmentType {
    #[serde(rename(deserialize = "code_interpreter", serialize = "code_interpreter"))]
    CodeInterpreter,
    #[serde(rename(deserialize = "file_search", serialize = "file_search"))]
    FileSearch,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum OpenAIRunStatus {
    #[serde(rename(deserialize = "queued", serialize = "queued"))]
    Queued,
    #[serde(rename(deserialize = "in_progress", serialize = "in_progress"))]
    InProgress,
    #[serde(rename(deserialize = "requires_action", serialize = "requires_action"))]
    RequiresAction,
    #[serde(rename(deserialize = "cancelling", serialize = "cancelling"))]
    Cancelling,
    #[serde(rename(deserialize = "cancelled", serialize = "cancelled"))]
    Cancelled,
    #[serde(rename(deserialize = "failed", serialize = "failed"))]
    Failed,
    #[serde(rename(deserialize = "completed", serialize = "completed"))]
    Completed,
    #[serde(rename(deserialize = "expired", serialize = "expired"))]
    Expired,
}
