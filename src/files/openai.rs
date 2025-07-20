use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use log::{error, info};
use reqwest::{header, multipart, Client};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::assistants::{OpenAIAssistantResource, OpenAIAssistantVersion};
use crate::domain::AllmsError;
use crate::files::LLMFiles;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIFile {
    pub id: Option<String>,
    debug: bool,
    api_key: String,
    version: OpenAIAssistantVersion,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIFileResp {
    id: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIDFileDeleteResp {
    id: String,
    object: String,
    deleted: bool,
}

#[async_trait(?Send)]
impl LLMFiles for OpenAIFile {
    /// Constructor
    fn new(id: Option<String>, open_ai_key: &str) -> Self {
        OpenAIFile {
            id,
            debug: false,
            api_key: open_ai_key.to_string(),
            version: OpenAIAssistantVersion::V1, // Default to V1
        }
    }

    ///
    /// This method can be used to turn on debug mode for the OpenAIFile struct
    ///
    fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    ///
    /// This function uploads a file to OpenAI and assigns it for use with Assistant API
    ///
    async fn upload(mut self, file_name: &str, file_bytes: Vec<u8>) -> Result<Self> {
        let files_url = self.version.get_endpoint(&OpenAIAssistantResource::Files);

        // This API sends a form so content type is automatically set by multipart method
        let mut version_headers = self.version.get_headers(&self.api_key);
        version_headers.remove(header::CONTENT_TYPE);

        // Determine MIME type based on file extension
        // OpenAI documentation: https://platform.openai.com/docs/assistants/tools/supported-files
        let mime_type = match Path::new(file_name)
            .extension()
            .and_then(std::ffi::OsStr::to_str)
        {
            Some("pdf") => "application/pdf",
            Some("json") => "application/json",
            Some("txt") => "text/plain",
            Some("html") => "text/html",
            Some("c") => "text/x-c",
            Some("cpp") => "text/x-c++",
            Some("docx") => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            Some("java") => "text/x-java",
            Some("md") => "text/markdown",
            Some("php") => "text/x-php",
            Some("pptx") => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
            Some("py") => "text/x-python",
            Some("rb") => "text/x-ruby",
            Some("tex") => "text/x-tex",
            //The below are currently only supported for Code Interpreter but NOT Retrieval
            Some("css") => "text/css",
            Some("jpeg") | Some("jpg") => "image/jpeg",
            Some("js") => "text/javascript",
            Some("gif") => "image/gif",
            Some("png") => "image/png",
            Some("tar") => "application/x-tar",
            Some("ts") => "application/typescript",
            Some("xlsx") => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Some("xml") => "application/xml",
            Some("zip") => "application/zip",
            _ => anyhow::bail!("Unsupported file type"),
        };

        let form = multipart::Form::new().text("purpose", "assistants").part(
            "file",
            multipart::Part::bytes(file_bytes)
                .file_name(file_name.to_string())
                .mime_str(mime_type)
                .context("Failed to set MIME type")?,
        );

        //Make the API call
        let client = Client::new();

        let response = client
            .post(files_url)
            .headers(version_headers)
            .multipart(form)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Files status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let response_deser: OpenAIFileResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_file".to_string(),
                    error_message: format!("Files API response serialization error: {}", error),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        self.id = Some(response_deser.id);

        Ok(self)
    }

    ///
    /// This function deletes a file from OpenAI
    ///
    async fn delete(&self) -> Result<()> {
        let file_id = if let Some(id) = &self.id {
            id
        } else {
            return Err(anyhow!(
                "[OpenAI][File API] Unable to delete file without an ID."
            ));
        };

        let files_resource = OpenAIAssistantResource::File {
            file_id: file_id.to_string(),
        };
        let files_url = self.version.get_endpoint(&files_resource);
        let version_headers = self.version.get_headers(&self.api_key);

        //Make the API call
        let client = Client::new();

        let response = client
            .delete(files_url)
            .headers(version_headers)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Files status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Check if the file was successfully deleted
        serde_json::from_str::<OpenAIDFileDeleteResp>(&response_text)
            .map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_file".to_string(),
                    error_message: format!(
                        "Files Delete API response serialization error: {}",
                        error
                    ),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })
            .and_then(|response| match response.deleted {
                true => Ok(()),
                false => Err(anyhow!("[OpenAIAssistant] Failed to delete the file.")),
            })
    }

    ///
    /// This function returns the ID of the file if it exists
    ///
    fn get_id(&self) -> Option<&String> {
        self.id.as_ref()
    }

    ///
    /// This function returns the debug mode of the file
    ///
    fn is_debug(&self) -> bool {
        self.debug
    }
}

impl OpenAIFile {
    ///
    /// This method can be used to set the version of Assistants API Beta
    /// Current default is V1
    ///
    pub fn version(mut self, version: OpenAIAssistantVersion) -> Self {
        // Files endpoint currently requires v1 so if v2 is selected we overwrite
        let version = match version {
            OpenAIAssistantVersion::V2 => OpenAIAssistantVersion::V1,
            _ => version,
        };
        self.version = version;
        self
    }
}
