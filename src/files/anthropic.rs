use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use log::{error, info};
use reqwest::{multipart, Client};
use serde::{Deserialize, Serialize};

use crate::{
    apis::AnthropicApiEndpoints, constants::ANTHROPIC_FILES_API_URL, domain::AllmsError,
    files::LLMFiles, utils::get_mime_type,
};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AnthropicFile {
    pub id: Option<String>,
    debug: bool,
    api_key: String,
}

#[async_trait(?Send)]
impl LLMFiles for AnthropicFile {
    /// Create a new file instance
    ///
    /// # Arguments
    /// * `id` - Optional file ID (for existing files)
    /// * `api_key` - API key for the LLM provider
    fn new(id: Option<String>, api_key: &str) -> Self {
        Self {
            id,
            debug: false,
            api_key: api_key.to_string(),
        }
    }

    /// Enable debug mode for the file instance
    ///
    /// Returns the modified instance for method chaining
    fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Upload a file to the LLM provider
    ///
    /// # Arguments
    /// * `file_name` - Name of the file to upload
    /// * `file_bytes` - File content as bytes
    ///
    /// # Returns
    /// * `Result<Self>` - The file instance with updated ID on success
    async fn upload(mut self, file_name: &str, file_bytes: Vec<u8>) -> Result<Self> {
        let files_url = ANTHROPIC_FILES_API_URL.to_string();

        let mime_type = get_mime_type(file_name).ok_or_else(|| anyhow!("Unsupported file type"))?;

        let form = multipart::Form::new().part(
            "file",
            multipart::Part::bytes(file_bytes)
                .file_name(file_name.to_string())
                .mime_str(mime_type)
                .context("Failed to set MIME type")?,
        );

        //Make the API call
        let client = Client::new();

        // Build request with appropriate headers
        let response = client
            .post(files_url)
            // Anthropic-specific way of passing API key
            .header("x-api-key", self.api_key.clone())
            // Required as per documentation
            .header(
                "anthropic-version",
                AnthropicApiEndpoints::messages_default().version(),
            )
            .header(
                "anthropic-beta",
                AnthropicApiEndpoints::files_default().version(),
            )
            // Specify the mime type of the file
            .multipart(form)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] Anthropic Files status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        // Deserialize the string response into the AnthropicFileResp object to confirm if there were any errors
        let response_deser: AnthropicFileResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "files::anthropic".to_string(),
                    error_message: format!(
                        "Anthropic Files API response serialization error: {}",
                        error
                    ),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        self.id = Some(response_deser.id);

        Ok(self)
    }

    /// Delete a file from the LLM provider
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    async fn delete(&self) -> Result<()> {
        let files_url = if let Some(id) = self.id.as_ref() {
            format!("{}/{}", &*ANTHROPIC_FILES_API_URL, id)
        } else {
            return Err(anyhow!("File ID is required to delete a file"));
        };

        //Make the API call
        let client = Client::new();

        // Build request with appropriate headers
        let response = client
            .delete(files_url)
            // Anthropic-specific way of passing API key
            .header("x-api-key", self.api_key.clone())
            // Required as per documentation
            .header(
                "anthropic-version",
                AnthropicApiEndpoints::messages_default().version(),
            )
            .header(
                "anthropic-beta",
                AnthropicApiEndpoints::files_default().version(),
            )
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] Anthropic Files status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Check if the file was successfully deleted
        serde_json::from_str::<AnthropicFileDeleteResp>(&response_text)
            .map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "files::anthropic".to_string(),
                    error_message: format!(
                        "Anthropic Files Delete API response serialization error: {}",
                        error
                    ),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })
            .and_then(|response| match response.result_type {
                AnthropicDeleteResultType::FileDeleted => Ok(()),
            })
    }

    /// Get the file ID if available
    ///
    /// # Returns
    /// * `Option<&String>` - The file ID if it exists
    fn get_id(&self) -> Option<&String> {
        self.id.as_ref()
    }

    /// Check if debug mode is enabled
    ///
    /// # Returns
    /// * `bool` - True if debug mode is enabled
    fn is_debug(&self) -> bool {
        self.debug
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct AnthropicFileResp {
    id: String,
    created_at: String,
    filename: String,
    mime_type: String,
    size_bytes: usize,
    #[serde(rename = "type")]
    file_type: AnthropicFileType,
    downloadable: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
#[serde(rename_all = "snake_case")]
enum AnthropicFileType {
    #[default]
    File,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct AnthropicFileDeleteResp {
    id: String,
    #[serde(rename = "type")]
    result_type: AnthropicDeleteResultType,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
#[serde(rename_all = "snake_case")]
enum AnthropicDeleteResultType {
    #[default]
    FileDeleted,
}
