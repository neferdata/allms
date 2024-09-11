use anyhow::{anyhow, Result};
use reqwest::header::{self, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;

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

impl FromStr for OpenAIAssistantVersion {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "v1" => Ok(OpenAIAssistantVersion::V1),
            "v2" => Ok(OpenAIAssistantVersion::V2),
            "azure" => Ok(OpenAIAssistantVersion::Azure),
            _ => Err(anyhow!("Invalid version: {}", s)),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    const OPENAI_API_URL: &str = "https://api.openai.com";

    #[test]
    fn test_v1_assistants_endpoint() {
        let version = OpenAIAssistantVersion::V1;
        let resource = OpenAIAssistantResource::Assistants;
        let expected_url = format!("{}/v1/assistants", OPENAI_API_URL);
        assert_eq!(version.get_endpoint(&resource), expected_url);
    }

    #[test]
    fn test_azure_assistant_endpoint() {
        let version = OpenAIAssistantVersion::Azure;
        let resource = OpenAIAssistantResource::Assistant {
            assistant_id: "123".to_string(),
        };
        let expected_url = format!(
            "{}/openai/assistants/123?api-version=2024-05-01-preview",
            OPENAI_API_URL
        );
        assert_eq!(version.get_endpoint(&resource), expected_url);
    }

    #[test]
    fn test_v2_threads_endpoint() {
        let version = OpenAIAssistantVersion::V2;
        let resource = OpenAIAssistantResource::Threads;
        let expected_url = format!("{}/v1/threads", OPENAI_API_URL);
        assert_eq!(version.get_endpoint(&resource), expected_url);
    }

    #[test]
    fn test_azure_file_batches_endpoint() {
        let version = OpenAIAssistantVersion::Azure;
        let resource = OpenAIAssistantResource::VectorStoreFileBatches {
            vector_store_id: "abc".to_string(),
        };
        let expected_url = format!(
            "{}/openai/vector_stores/abc/file_batches?api-version=2024-05-01-preview",
            OPENAI_API_URL
        );
        assert_eq!(version.get_endpoint(&resource), expected_url);
    }

    #[test]
    fn test_v1_run_endpoint() {
        let version = OpenAIAssistantVersion::V1;
        let resource = OpenAIAssistantResource::Run {
            thread_id: "xyz".to_string(),
            run_id: "456".to_string(),
        };
        let expected_url = format!("{}/v1/threads/xyz/runs/456", OPENAI_API_URL);
        assert_eq!(version.get_endpoint(&resource), expected_url);
    }

    #[test]
    fn test_v1_tools_payload() {
        let version = OpenAIAssistantVersion::V1;
        let expected_payload: Value = json!([{
            "type": "retrieval"
        }]);
        assert_eq!(version.get_tools_payload(), expected_payload);
    }

    #[test]
    fn test_v2_tools_payload() {
        let version = OpenAIAssistantVersion::V2;
        let expected_payload: Value = json!([{
            "type": "file_search"
        }]);
        assert_eq!(version.get_tools_payload(), expected_payload);
    }

    #[test]
    fn test_azure_tools_payload() {
        let version = OpenAIAssistantVersion::Azure;
        let expected_payload: Value = json!([{
            "type": "file_search"
        }]);
        assert_eq!(version.get_tools_payload(), expected_payload);
    }
}
