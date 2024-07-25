pub mod assistants;
mod completions;
mod constants;
mod domain;
mod enums;
pub mod llm_models;
pub use llm_models as llm;
mod utils;

#[allow(deprecated)]
mod deprecated;

pub use crate::completions::Completions;
#[allow(deprecated)]
pub use crate::deprecated::{
    OpenAI, OpenAIAssistant, OpenAIAssistantVersion, OpenAIFile, OpenAIModels,
};
