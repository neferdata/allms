use anyhow::{anyhow, Result};
use log::{error, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::assistants::OpenAIAssistantVersion;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIVectorStore {
    pub id: Option<String>,
    pub name: String,
    api_key: String,
    status: OpenAIVectorStoreStatus,
    debug: bool,
    version: OpenAIAssistantVersion,
}

const VECTOR_STORE_API: &str = "https://api.openai.com/v1/vector_stores";

impl OpenAIVectorStore {
    /// Constructor
    pub fn new(id: Option<String>, name: &str, api_key: &str) -> Self {
        OpenAIVectorStore {
            id,
            name: name.to_string(),
            api_key: api_key.to_string(),
            status: OpenAIVectorStoreStatus::InProgress,
            debug: false,
            version: OpenAIAssistantVersion::V2,
        }
    }

    ///
    /// This method can be used to set turn on/off the debug mode
    ///
    pub fn debug(mut self) -> Self {
        self.debug = !self.debug;
        self
    }

    /*
     * This function creates a new Vector Store and updates the ID of the struct
     */
    async fn create(&mut self, file_ids: Option<Vec<String>>) -> Result<()> {
        //Make the API call
        let client = Client::new();

        //Get the version-specific header
        let version_headers = self.version.get_headers();

        let mut body = json!({
            "name": self.name.clone(),
        });
        if let Some(ids) = file_ids {
            body["file_ids"] = json!(ids.to_vec());
        }

        let response = client
            .post(VECTOR_STORE_API)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Create API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Assistant object
        let response_deser: OpenAIVectorStoreResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[allms][OpenAI][VectorStore][debug] VectorStore Create API response serialization error: {}",
                    &error
                );
                anyhow!(
                    "[allms][OpenAI][VectorStore][debug] VectorStore Create API response serialization error: {}",
                    error
                )
            })?;

        //Add correct ID & status to self
        self.id = Some(response_deser.id);
        self.status = response_deser.status;

        Ok(())
    }

    ///
    /// This method uploads files to a Vector Store. If no ID was provided the method first creates the Vector Store
    ///
    pub async fn upload(&mut self, file_ids: &[String]) -> Result<Self> {
        // If the Vector Store was not yet created we do that first
        if self.id.is_none() {
            self.create(Some(file_ids.to_vec())).await?;
        } else {
            // If working with existing Vector Store we simply upload files
            self.assign_to_store(file_ids).await?;
        }
        Ok(self.clone())
    }

    /*
     * This function assigns OpenAI Files to an existing Vector Store
     */
    async fn assign_to_store(&self, file_ids: &[String]) -> Result<()> {
        // The function requires an ID of an existing vector store
        let vs_id = if let Some(id) = &self.id {
            id
        } else {
            return Err(anyhow!(
                "[allms][OpenAI][VectorStore][debug] Unable to assign files. No ID provided."
            ));
        };

        // Construct the API url
        let url = format!("{}/{}/file_batches", VECTOR_STORE_API, &vs_id,);

        //Get the version-specific header
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let body = json!({
            "file_ids": file_ids.to_vec(),
        });

        let response = client
            .post(&url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Batch Upload API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize & validate the string response
        serde_json::from_str::<OpenAIVectorStoreFileBatchResp>(&response_text).map_err(|error| {
            error!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Batch Upload API response serialization error: {}",
                &error
            );
            anyhow!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Batch Upload API response serialization error: {}",
                error
            )
        })
        .map(|_| Ok(()))?
    }

    ///
    /// This method checks the status of a Vector Store
    ///
    pub async fn status(&self) -> Result<OpenAIVectorStoreStatus> {
        // Requires an ID of an existing vector store
        let vs_id = if let Some(id) = &self.id {
            id
        } else {
            return Err(anyhow!(
                "[allms][OpenAI][VectorStore][debug] Unable to check status. No ID provided."
            ));
        };

        // Construct the API url
        let url = format!("{}/{}", VECTOR_STORE_API, &vs_id,);

        //Get the version-specific header
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let response = client
            .get(&url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize & validate the string response
        let response_deser: OpenAIVectorStoreResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[allms][OpenAI][VectorStore][debug] VectorStore Status API response serialization error: {}",
                    &error
                );
                anyhow!(
                    "[allms][OpenAI][VectorStore][debug] VectorStore Status API response serialization error: {}",
                    error
                )
            })?;
        Ok(response_deser.status)
    }

    ///
    /// This method checks the counts of files added to a Vector Store and their statuses
    ///
    pub async fn file_count(&self) -> Result<OpenAIVectorStoreFileCounts> {
        // Requires an ID of an existing vector store
        let vs_id = if let Some(id) = &self.id {
            id
        } else {
            return Err(anyhow!(
                "[allms][OpenAI][VectorStore][debug] Unable to check status. No ID provided."
            ));
        };

        // Construct the API url
        let url = format!("{}/{}", VECTOR_STORE_API, &vs_id,);

        //Get the version-specific header
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let response = client
            .get(&url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize & validate the string response
        let response_deser: OpenAIVectorStoreResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[allms][OpenAI][VectorStore][debug] VectorStore Status API response serialization error: {}",
                    &error
                );
                anyhow!(
                    "[allms][OpenAI][VectorStore][debug] VectorStore Status API response serialization error: {}",
                    error
                )
            })?;
        Ok(response_deser.file_counts)
    }

    ///
    /// This method can be used to delete a Vector Store
    ///
    pub async fn delete(&self) -> Result<()> {
        // Requires an ID of an existing vector store
        let vs_id = if let Some(id) = &self.id {
            id
        } else {
            return Err(anyhow!(
                "[allms][OpenAI][VectorStore][debug] Unable to delete. No ID provided."
            ));
        };

        // Construct the API url
        let url = format!("{}/{}", VECTOR_STORE_API, &vs_id,);

        //Get the version-specific header
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let response = client
            .delete(&url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Delete API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize & validate the string response
        serde_json::from_str::<OpenAIVectorStoreDeleteResp>(&response_text).map_err(|error| {
            error!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Delete API response serialization error: {}",
                &error
            );
            anyhow!(
                "[allms][OpenAI][VectorStore][debug] VectorStore Delete API response serialization error: {}",
                error
            )
        })
        .and_then(|response| match response.deleted {
            true => Ok(()),
            false => Err(anyhow!("[OpenAIAssistant] VectorStore Delete API failed to delete the store.")),
        })
    }
}

/******************************************************************************************
*
* API Response serialization / deserialization structs
*
******************************************************************************************/
#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIVectorStoreResp {
    id: String,
    name: String,
    status: OpenAIVectorStoreStatus,
    created_at: i64,
    expires_at: Option<i64>,
    last_active_at: Option<i64>,
    file_counts: OpenAIVectorStoreFileCounts,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIVectorStoreFileCounts {
    in_progress: i32,
    completed: i32,
    failed: i32,
    cancelled: i32,
    total: i32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum OpenAIVectorStoreStatus {
    #[serde(rename(deserialize = "expired", serialize = "expired"))]
    Expired,
    #[serde(rename(deserialize = "in_progress", serialize = "in_progress"))]
    InProgress,
    #[serde(rename(deserialize = "completed", serialize = "completed"))]
    Completed,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIVectorStoreFileBatchResp {
    id: String,
    vector_store_id: String,
    status: OpenAIVectorStoreFileBatchStatus,
    created_at: i64,
    file_counts: OpenAIVectorStoreFileCounts,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
enum OpenAIVectorStoreFileBatchStatus {
    #[serde(rename(deserialize = "in_progress", serialize = "in_progress"))]
    InProgress,
    #[serde(rename(deserialize = "completed", serialize = "completed"))]
    Completed,
    #[serde(rename(deserialize = "cancelled", serialize = "cancelled"))]
    Cancelled,
    #[serde(rename(deserialize = "failed", serialize = "failed"))]
    Failed,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIVectorStoreDeleteResp {
    id: String,
    deleted: bool,
}
