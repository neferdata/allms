use std::time::Duration;

use crate::domain::{
    OpenAIAssistantResp, OpenAIMessageListResp, OpenAIMessageResp, OpenAIRunResp, OpenAIThreadResp,
};
use crate::enums::{OpenAIAssistantRole, OpenAIRunStatus};
use crate::utils::sanitize_json_response;
use crate::{constants::OPENAI_ASSISTANT_INSTRUCTIONS, models::OpenAIModels};
use anyhow::{anyhow, Result};
use log::error;
use log::info;
use reqwest::{header, Client};
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time;
use tokio::time::timeout;

/// [OpenAI Docs](https://platform.openai.com/docs/assistants/overview)
///
/// The Assistants API allows you to build AI assistants within your own applications.
/// An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries.
/// The Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling.
/// In the future, we plan to release more OpenAI-built tools, and allow you to provide
/// your own tools on our platform.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIAssistant {
    id: String,
    thread_id: Option<String>,
    run_id: Option<String>,
    model: OpenAIModels,
    instructions: String,
    debug: bool,
    api_key: String,
}

impl OpenAIAssistant {
    //Constructor
    pub async fn new(model: OpenAIModels, open_ai_key: &str, debug: bool) -> Result<Self> {
        let mut new_assistant = OpenAIAssistant {
            id: "this_will_change".to_string(),
            thread_id: None,
            run_id: None,
            model,
            instructions: OPENAI_ASSISTANT_INSTRUCTIONS.to_string(),
            debug,
            api_key: open_ai_key.to_string(),
        };
        //Call OpenAI API to get an ID for the assistant
        new_assistant.create_assistant().await?;

        Ok(new_assistant)
    }

    /*
     * This function creates an Assistant and updates the ID of the OpenAIAssistant struct
     */
    async fn create_assistant(&mut self) -> Result<()> {
        //Get the API url
        let assistant_url = "https://api.openai.com/v1/assistants";

        let code_interpreter = json!({
            "type": "retrieval",
        });
        let assistant_body = json!({
            "instructions": self.instructions.clone(),
            "model": self.model.as_str(),
            "tools": vec![code_interpreter],
        });

        //Make the API call
        let client = Client::new();

        let response = client
            .post(assistant_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
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
        self.id = response_deser.id;

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
        let message = match file_ids.is_empty() {
            false => json!({
                "role": "user",
                "content": message.to_string(),
                "file_ids": file_ids.to_vec(),
            }),
            true => json!({
                "role": "user",
                "content": message.to_string(),
            }),
        };

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
        let thread_url = "https://api.openai.com/v1/threads";

        //Make the API call
        let client = Client::new();

        let response = client
            .post(thread_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
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

        let message_url = format!(
            "https://api.openai.com/v1/threads/{}/messages",
            self.thread_id.clone().unwrap_or_default()
        );

        //Make the API call
        let client = Client::new();

        let response = client
            .post(message_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
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

        let message_url = format!(
            "https://api.openai.com/v1/threads/{}/messages",
            self.thread_id.clone().unwrap_or_default()
        );

        //Make the API call
        let client = Client::new();

        let response = client
            .get(message_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
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
        if self.thread_id.is_none() {
            return Err(anyhow!("No active thread detected."));
        }

        let run_url = format!(
            "https://api.openai.com/v1/threads/{}/runs",
            self.thread_id.clone().unwrap_or_default()
        );

        let body = json!({
            "assistant_id": self.id.clone(),
        });

        //Make the API call
        let client = Client::new();

        let response = client
            .post(run_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
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
        if self.thread_id.is_none() {
            return Err(anyhow!("No active thread detected."));
        }

        if self.run_id.is_none() {
            return Err(anyhow!("No active run detected."));
        }

        let run_url = format!(
            "https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
            thread_id = self.thread_id.clone().unwrap_or_default(),
            run_id = self.run_id.clone().unwrap_or_default(),
        );

        //Make the API call
        let client = Client::new();

        let response = client
            .get(run_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header("OpenAI-Beta", "assistants=v1")
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
