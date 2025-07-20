use anyhow::Result;
use async_trait::async_trait;

/// Trait for LLM file operations across different providers
///
/// This trait provides a common interface for file operations such as
/// creation, debugging, uploading, and deletion of files for use with
/// LLM assistants and completions.
#[async_trait(?Send)]
pub trait LLMFiles: Send + Sync + Sized {
    /// Create a new file instance
    ///
    /// # Arguments
    /// * `id` - Optional file ID (for existing files)
    /// * `api_key` - API key for the LLM provider
    fn new(id: Option<String>, api_key: &str) -> Self;

    /// Enable debug mode for the file instance
    ///
    /// Returns the modified instance for method chaining
    fn debug(self) -> Self;

    /// Upload a file to the LLM provider
    ///
    /// # Arguments
    /// * `file_name` - Name of the file to upload
    /// * `file_bytes` - File content as bytes
    ///
    /// # Returns
    /// * `Result<Self>` - The file instance with updated ID on success
    async fn upload(self, file_name: &str, file_bytes: Vec<u8>) -> Result<Self>;

    /// Delete a file from the LLM provider
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    async fn delete(&self) -> Result<()>;

    /// Get the file ID if available
    ///
    /// # Returns
    /// * `Option<&String>` - The file ID if it exists
    fn get_id(&self) -> Option<&String>;

    /// Check if debug mode is enabled
    ///
    /// # Returns
    /// * `bool` - True if debug mode is enabled
    fn is_debug(&self) -> bool;
}
