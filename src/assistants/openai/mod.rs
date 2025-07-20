pub mod openai_assistant;
pub mod openai_vector_store;

pub use openai_assistant::OpenAIAssistant;
pub use openai_vector_store::{
    OpenAIVectorStore, OpenAIVectorStoreFileCounts, OpenAIVectorStoreStatus,
};

// Re-export structs for backwards compatibility
pub use crate::apis::{OpenAIAssistantResource, OpenAIAssistantVersion};
pub use crate::files::OpenAIFile;
