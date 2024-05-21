use anyhow::{anyhow, Result};
use log::error;
use log::info;
use reqwest::{
    header::{self, HeaderMap, HeaderValue},
    Client,
};
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time;
use tokio::time::timeout;

use crate::constants::{OPENAI_API_URL, OPENAI_ASSISTANT_INSTRUCTIONS};
use crate::domain::{
    OpenAIAssistantResp, OpenAIMessageListResp, OpenAIMessageResp, OpenAIRunResp, OpenAIThreadResp,
};
use crate::enums::{OpenAIAssistantRole, OpenAIRunStatus};
use crate::llm_models::{LLMModel, OpenAIModels};
use crate::utils::sanitize_json_response;

/// [OpenAI Docs](https://platform.openai.com/docs/assistants/overview)
///
/// The Assistants API allows you to build AI assistants within your own applications.
/// An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries.
/// The Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling.
/// In the future, we plan to release more OpenAI-built tools, and allow you to provide
/// your own tools on our platform.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIAssistant {
    id: Option<String>,
    thread_id: Option<String>,
    run_id: Option<String>,
    model: OpenAIModels,
    instructions: String,
    debug: bool,
    api_key: String,
    version: OpenAIAssistantVersion,
}

impl OpenAIAssistant {
    //Constructor
    pub fn new(model: OpenAIModels, open_ai_key: &str) -> Self {
        OpenAIAssistant {
            id: None,
            thread_id: None,
            run_id: None,
            model,
            instructions: OPENAI_ASSISTANT_INSTRUCTIONS.to_string(),
            debug: false,
            api_key: open_ai_key.to_string(),
            // Defaulting to V1 for now
            version: OpenAIAssistantVersion::V1,
        }
    }

    ///
    /// This method can be used to turn on debug mode for the Assistant
    ///
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    ///
    /// This method can be used to set the version of Assistants API Beta
    /// Current default is V1
    ///
    pub fn version(mut self, version: OpenAIAssistantVersion) -> Self {
        self.version = version;
        self
    }

    /*
     * This function creates an Assistant and updates the ID of the OpenAIAssistant struct
     */
    async fn create_assistant(&mut self) -> Result<()> {
        //Get the assistant API url
        let assistant_url = format!("{}/assistants", self.version.get_endpoint());

        //Get the version-specific header
        let version_headers = self.version.get_headers();

        //Get the retrieval / file_search part of the payload
        let tools_payload = self.version.get_tools_payload();

        let assistant_body = json!({
            "instructions": self.instructions.clone(),
            "model": self.model.as_str(),
            "tools": tools_payload,
        });

        //Make the API call
        let client = Client::new();

        let response = client
            .post(assistant_url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .json(&assistant_body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Assistant API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Assistant object
        let response_deser: OpenAIAssistantResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[OpenAIAssistant] Assistant API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        //Add correct ID to self
        self.id = Some(response_deser.id);

        Ok(())
    }

    /*
     * This function performs all the orchestration needed to submit a prompt and get and answer
     */
    pub async fn get_answer<T: JsonSchema + DeserializeOwned>(
        mut self,
        message: &str,
        file_ids: &[String],
    ) -> Result<T> {
        // If the assistant and thread are not initialized we do that first
        if self.id.is_none() {
            //Call OpenAI API to get an ID for the assistant
            self.create_assistant().await?;

            //Add first message thus initializing the thread
            self.add_message(OPENAI_ASSISTANT_INSTRUCTIONS, &Vec::new())
                .await?;
        }

        //Step 1: Instruct the Assistant to answer with the right Json format
        //Output schema is extracted from the type parameter
        let schema = schema_for!(T);
        let schema_json: Value = serde_json::to_value(&schema)?;
        let schema_string = serde_json::to_string(&schema_json).unwrap_or_default();

        //We instruct Assistant to answer with that schema
        let schema_message = format!(
            "Response should include only the data portion of a Json formatted as per the following schema: {}. 
            The response should only include well-formatted data, and not the schema itself.
            Do not include any other words or characters, including the word 'json'. Only respond with the data. 
            You need to validate the Json before returning.",
            schema_string
        );
        self.add_message(&schema_message, &Vec::new()).await?;

        //Step 2: Add user message and files to thread
        self.add_message(message, file_ids).await?;

        //Step 3: Kick off processing (aka Run)
        self.start_run().await?;

        //Step 4: Check in on the status of the run
        let operation_timeout = Duration::from_secs(600); // Timeout for the whole operation
        let poll_interval = Duration::from_secs(10);

        let _result = timeout(operation_timeout, async {
            let mut interval = time::interval(poll_interval);
            loop {
                interval.tick().await; // Wait for the next interval tick
                match self.get_run_status().await {
                    Ok(resp) => match resp.status {
                        //Completed successfully. Time to get results.
                        OpenAIRunStatus::Completed => {
                            break Ok(());
                        }
                        //TODO: We will need better handling of requires_action
                        OpenAIRunStatus::RequiresAction
                        | OpenAIRunStatus::Cancelling
                        | OpenAIRunStatus::Cancelled
                        | OpenAIRunStatus::Failed
                        | OpenAIRunStatus::Expired => {
                            return Err(anyhow!("Failed to validate status of the run"));
                        }
                        _ => continue, // Keep polling if in_progress or queued
                    },
                    Err(e) => return Err(e), // Break on error
                }
            }
        })
        .await?;

        //Step 5: Get all messages posted on the thread. This should now include response from the Assistant
        let messages = self.get_message_thread().await?;

        messages
            .into_iter()
            .filter(|message| message.role == OpenAIAssistantRole::Assistant)
            .find_map(|message| {
                message.content.into_iter().find_map(|content| {
                    content.text.and_then(|text| {
                        let sanitized_text = sanitize_json_response(&text.value);
                        serde_json::from_str::<T>(&sanitized_text).ok()
                    })
                })
            })
            .ok_or(anyhow!("No valid response form OpenAI Assistant found."))
    }

    ///
    /// This method can be used to provide data that will be used as context for the prompt.
    /// Using this function you can provide multiple sets of context data by calling it multiple times. New values will be as messages to the thread
    /// It accepts any struct that implements the Serialize trait.
    ///
    pub async fn set_context<T: Serialize>(mut self, dataset_name: &str, data: &T) -> Result<Self> {
        // If the assistant and thread are not initialized we do that first
        if self.id.is_none() {
            //Call OpenAI API to get an ID for the assistant
            self.create_assistant().await?;

            //Add first message thus initializing the thread
            self.add_message(OPENAI_ASSISTANT_INSTRUCTIONS, &Vec::new())
                .await?;
        }

        let serialized_data = if let Ok(json) = serde_json::to_string(&data) {
            json
        } else {
            return Err(anyhow!("Unable serialize provided input data."));
        };
        let message = format!("'{dataset_name}'= {serialized_data}");
        let file_ids = Vec::new();
        self.add_message(&message, &file_ids).await?;
        Ok(self)
    }

    /*
     * This function creates a Thread and updates the thread_id of the OpenAIAssistant struct
     */
    async fn add_message(&mut self, message: &str, file_ids: &[String]) -> Result<()> {
        //Prepare the body that is to be send to OpenAI APIs
        let mut message = json!({
            "role": "user",
            "content": message.to_string(),
        });

        if !file_ids.is_empty() {
            message = self.version.add_message_attachments(&message, file_ids);
        }

        //If there is no thread_id we need to create one
        match self.thread_id {
            None => {
                let body = json!({
                    "messages": vec![message],
                });

                self.create_thread(&body).await
            }
            Some(_) => self.add_message_thread(&message).await,
        }
    }

    /*
     * This function creates a Thread and updates the thread_id of the OpenAIAssistant struct
     */
    async fn create_thread(&mut self, body: &serde_json::Value) -> Result<()> {
        //Get version-specific URL
        let thread_url = format!("{}/threads", self.version.get_endpoint());

        //Get version-specific headers
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let response = client
            .post(thread_url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Threads API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Thread object
        let response_deser: OpenAIThreadResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[OpenAIAssistant] Thread API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        //Add thread_id to self
        self.thread_id = Some(response_deser.id);

        Ok(())
    }

    /*
     * This function adds a message to an existing thread
     */
    async fn add_message_thread(&self, body: &serde_json::Value) -> Result<()> {
        if self.thread_id.is_none() {
            return Err(anyhow!("No active thread detected."));
        }

        //Get version-specific URL
        let message_url = format!(
            "{}/threads/{}/messages",
            self.version.get_endpoint(),
            self.thread_id.clone().unwrap_or_default(),
        );

        //Get version-specific headers
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let response = client
            .post(message_url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Messages API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let _response_deser: OpenAIMessageResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[OpenAIAssistant] Messages API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        Ok(())
    }

    /*
     * This function gets all message posted to an existing thread
     */
    async fn get_message_thread(&self) -> Result<Vec<OpenAIMessageResp>> {
        if self.thread_id.is_none() {
            return Err(anyhow!("No active thread detected."));
        }

        //Get version-specific URL
        let message_url = format!(
            "{}/threads/{}/messages",
            self.version.get_endpoint(),
            self.thread_id.clone().unwrap_or_default(),
        );

        //Get version-specific headers
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let response = client
            .get(message_url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Messages API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into a vector of OpenAIMessageResp objects
        let response_deser: OpenAIMessageListResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[OpenAIAssistant] Messages API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        Ok(response_deser.data)
    }

    /*
     * This function starts an assistant run
     */
    async fn start_run(&mut self) -> Result<()> {
        let assistant_id = if let Some(id) = self.id.clone() {
            id
        } else {
            return Err(anyhow!("No active assistant detected."));
        };

        let thread_id = if let Some(id) = self.thread_id.clone() {
            id
        } else {
            return Err(anyhow!("No active thread detected."));
        };

        //Get version-specific URL
        let run_url = format!("{}/threads/{}/runs", self.version.get_endpoint(), thread_id,);

        //Get version-specific headers
        let version_headers = self.version.get_headers();

        let body = json!({
            "assistant_id": assistant_id,
        });

        //Make the API call
        let client = Client::new();

        let response = client
            .post(run_url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Messages API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let response_deser: OpenAIRunResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[OpenAIAssistant] Run API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        //Update run_id
        self.run_id = Some(response_deser.id);

        Ok(())
    }

    /*
     * This function checks the status of an assistant run
     */
    async fn get_run_status(&self) -> Result<OpenAIRunResp> {
        let thread_id = if let Some(id) = self.thread_id.clone() {
            id
        } else {
            return Err(anyhow!("No active thread detected."));
        };

        let run_id = if let Some(id) = self.run_id.clone() {
            id
        } else {
            return Err(anyhow!("No active run detected."));
        };

        //Get version-specific URL
        let run_url = format!(
            "{}/threads/{}/runs/{}",
            self.version.get_endpoint(),
            thread_id,
            run_id,
        );

        //Get version-specific headers
        let version_headers = self.version.get_headers();

        //Make the API call
        let client = Client::new();

        let response = client
            .get(run_url)
            .headers(version_headers)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Run status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let response_deser: OpenAIRunResp =
            serde_json::from_str(&response_text).map_err(|error| {
                error!(
                    "[OpenAIAssistant] Run API response serialization error: {}",
                    &error
                );
                anyhow!("Error: {}", error)
            })?;

        Ok(response_deser)
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum OpenAIAssistantVersion {
    V1,
    V2,
}

impl OpenAIAssistantVersion {
    pub(crate) fn get_endpoint(&self) -> String {
        //OpenAI documentation: https://platform.openai.com/docs/models/model-endpoint-compatibility
        match self {
            OpenAIAssistantVersion::V1 | OpenAIAssistantVersion::V2 => {
                format!("{OPENAI_API_URL}/v1", OPENAI_API_URL = *OPENAI_API_URL)
            }
        }
    }

    pub(crate) fn get_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        match self {
            OpenAIAssistantVersion::V1 => {
                headers.insert("OpenAI-Beta", HeaderValue::from_static("assistants=v1"))
            }
            OpenAIAssistantVersion::V2 => {
                headers.insert("OpenAI-Beta", HeaderValue::from_static("assistants=v2"))
            }
        };
        headers
    }

    pub(crate) fn get_tools_payload(&self) -> Vec<Value> {
        match self {
            OpenAIAssistantVersion::V1 => vec![json!({
                "type": "retrieval"
            })],
            OpenAIAssistantVersion::V2 => vec![json!({
                "type": "file_search"
            })],
        }
    }

    pub(crate) fn add_message_attachments(
        &self,
        message_payload: &Value,
        file_ids: &[String],
    ) -> Value {
        let mut message_payload = message_payload.clone();
        match self {
            OpenAIAssistantVersion::V1 => {
                message_payload["file_ids"] = json!(file_ids);
            }
            OpenAIAssistantVersion::V2 => {
                let file_search_json = json!({
                    "type": "file_search"
                });
                let attachments_vec: Vec<Value> = file_ids
                    .iter()
                    .map(|file_id| {
                        json!({
                            "file_id": file_id.to_string(),
                            "tools": [file_search_json.clone()]
                        })
                    })
                    .collect();
                message_payload["attachments"] = json!(attachments_vec);
            }
        }
        message_payload
    }
}
