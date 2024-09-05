use reqwest::header::{self, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::constants::OPENAI_API_URL;

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIAssistantVersion {
    V1,
    V2,
    Azure,
}

impl OpenAIAssistantVersion {
    pub(crate) fn get_endpoint(&self, resource: &OpenAIAssistantResource) -> String {
        //OpenAI documentation: https://platform.openai.com/docs/models/model-endpoint-compatibility
        let base_url = match self {
            OpenAIAssistantVersion::V1 | OpenAIAssistantVersion::V2 => {
                format!("{OPENAI_API_URL}/v1", OPENAI_API_URL = *OPENAI_API_URL)
            }
            OpenAIAssistantVersion::Azure => {
                format!("{OPENAI_API_URL}/openai", OPENAI_API_URL = *OPENAI_API_URL)
            }
        };

        let path = match resource {
            OpenAIAssistantResource::Assistants => format!("{base_url}/assistants"),
            OpenAIAssistantResource::Assistant { assistant_id } => {
                format!("{base_url}/assistants/{assistant_id}")
            }
            OpenAIAssistantResource::Threads => format!("{base_url}/threads"),
            OpenAIAssistantResource::Messages { thread_id } => {
                format!("{base_url}/threads/{thread_id}/messages")
            }
            OpenAIAssistantResource::Runs { thread_id } => {
                format!("{base_url}/threads/{thread_id}/runs")
            }
            OpenAIAssistantResource::Run { thread_id, run_id } => {
                format!("{base_url}/threads/{thread_id}/runs/{run_id}")
            }
            OpenAIAssistantResource::Files => format!("{base_url}/files"),
            OpenAIAssistantResource::File { file_id } => format!("{base_url}/files/{file_id}"),
            OpenAIAssistantResource::VectorStores => format!("{base_url}/vector_stores"),
            OpenAIAssistantResource::VectorStore { vector_store_id } => {
                format!("{base_url}/vector_stores/{vector_store_id}")
            }
            OpenAIAssistantResource::VectorStoreFileBatches { vector_store_id } => {
                format!("{base_url}/vector_stores/{vector_store_id}/file_batches")
            }
        };

        // Add suffix if needed
        match self {
            OpenAIAssistantVersion::Azure => format!("{path}?api-version=2024-05-01-preview"),
            _ => path,
        }
    }

    pub(crate) fn get_headers(&self, api_key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        match self {
            OpenAIAssistantVersion::V1 => {
                // Try to create the header value from the bearer token
                if let Ok(bearer_header) = HeaderValue::from_str(&format!("Bearer {api_key}")) {
                    headers.insert("Authorization", bearer_header);
                } else {
                    headers.insert(
                        "Error",
                        HeaderValue::from_static("Invalid Authorization Header"),
                    );
                };
                headers.insert("OpenAI-Beta", HeaderValue::from_static("assistants=v1"));
            }
            OpenAIAssistantVersion::V2 => {
                // Try to create the header value from the bearer token
                if let Ok(bearer_header) = HeaderValue::from_str(&format!("Bearer {api_key}")) {
                    headers.insert("Authorization", bearer_header);
                } else {
                    headers.insert(
                        "Error",
                        HeaderValue::from_static("Invalid Authorization Header"),
                    );
                };
                headers.insert("OpenAI-Beta", HeaderValue::from_static("assistants=v2"));
            }
            OpenAIAssistantVersion::Azure => {
                // Try to create the header value from the bearer token
                if let Ok(api_key_header) = HeaderValue::from_str(api_key) {
                    headers.insert("api-key", api_key_header);
                } else {
                    headers.insert(
                        "Error",
                        HeaderValue::from_static("Invalid Authorization Header"),
                    );
                };
            }
        };
        headers
    }

    pub(crate) fn get_tools_payload(&self) -> Value {
        match self {
            OpenAIAssistantVersion::V1 => json!([{
                "type": "retrieval"
            }]),
            OpenAIAssistantVersion::V2 | OpenAIAssistantVersion::Azure => json!([{
                "type": "file_search"
            }]),
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
            OpenAIAssistantVersion::V2 | OpenAIAssistantVersion::Azure => {
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

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIAssistantResource {
    Assistants,
    Assistant { assistant_id: String },
    Threads,
    Messages { thread_id: String },
    Runs { thread_id: String },
    Run { thread_id: String, run_id: String },
    Files,
    File { file_id: String },
    VectorStores,
    VectorStore { vector_store_id: String },
    VectorStoreFileBatches { vector_store_id: String },
}
