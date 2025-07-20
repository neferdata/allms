mod google;
mod openai;

pub use google::GoogleApiEndpoints;
pub use openai::{OpenAiApiEndpoints, OpenAICompletionsAPI,OpenAIAssistantResource, OpenAIAssistantVersion};