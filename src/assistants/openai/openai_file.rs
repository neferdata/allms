use anyhow::{anyhow, Context, Result};
use log::{error, info};
use reqwest::{header, Client, multipart};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIFile {
    pub id: String,
    debug: bool,
    api_key: String,
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

impl OpenAIFile {
    //Constructor
    pub async fn new(
        file_name: &str,
        file_bytes: Vec<u8>,
        open_ai_key: &str,
        debug: bool,
    ) -> Result<Self> {
        let mut new_file = OpenAIFile {
            id: "this-will-be-overwritten".to_string(),
            debug,
            api_key: open_ai_key.to_string(),
        };
        //Upload file and get the ID
        new_file.upload_file(file_name, file_bytes).await?;
        Ok(new_file)
    }

    /*
     * This function uploads a file to OpenAI and assigns it for use with Assistant API
     */
    async fn upload_file(&mut self, file_name: &str, file_bytes: Vec<u8>) -> Result<()> {
        let files_url = "https://api.openai.com/v1/files";

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
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
            .bearer_auth(&self.api_key)
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
                error!(
                    "[OpenAIAssistant] Files API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        self.id = response_deser.id;

        Ok(())
    }

    /*
     * This function deletes a file from OpenAI
     */
    pub async fn delete_file(&self) -> Result<()> {
        let files_url = format!("https://api.openai.com/v1/files/{}", self.id);

        //Make the API call
        let client = Client::new();

        let response = client
            .delete(files_url)
            .bearer_auth(&self.api_key)
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
                error!(
                    "[OpenAIAssistant] Files Delete API response serialization error: {}",
                    &error
                );
                anyhow!(
                    "[OpenAIAssistant] Files Delete API response serialization error: {}",
                    error
                )
            })
            .and_then(|response| match response.deleted {
                true => Ok(()),
                false => Err(anyhow!("[OpenAIAssistant] Failed to delete the file.")),
            })
    }
}
