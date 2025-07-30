pub mod openai_assistant;
pub mod openai_vector_store;

pub use openai_assistant::OpenAIAssistant;
pub use openai_vector_store::{
    OpenAIVectorStore, OpenAIVectorStoreFileCounts, OpenAIVectorStoreStatus,
};

// Re-export structs for backwards compatibility
pub use crate::apis::{OpenAIAssistantResource, OpenAIAssistantVersion};
pub use crate::files::{LLMFiles, OpenAIFile};

// Add inherent methods for OpenAIFile for backwards compatibility
impl OpenAIFile {
    pub fn new(id: Option<String>, api_key: &str) -> Self {
        <Self as LLMFiles>::new(id, api_key)
    }

    pub fn debug(self) -> Self {
        <Self as LLMFiles>::debug(self)
    }

    pub async fn upload(self, file_name: &str, file_bytes: Vec<u8>) -> anyhow::Result<Self> {
        <Self as LLMFiles>::upload(self, file_name, file_bytes).await
    }

    pub async fn delete(&self) -> anyhow::Result<()> {
        <Self as LLMFiles>::delete(self).await
    }

    pub fn get_id(&self) -> Option<&String> {
        <Self as LLMFiles>::get_id(self)
    }

    pub fn is_debug(&self) -> bool {
        <Self as LLMFiles>::is_debug(self)
    }
}
