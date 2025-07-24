#![allow(deprecated)]

use anyhow::{anyhow, Result};
use reqwest::header::{self, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;

use crate::constants::{DEFAULT_AZURE_VERSION, OPENAI_API_URL};

///
/// Enum of supported Completions and Responses APIs (non-Assistant APIs)
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum OpenAiApiEndpoints {
    #[deprecated(note = "Use OpenAICompletions instead")]
    OpenAI,
    #[default]
    OpenAICompletions,
    OpenAIResponses,
    #[deprecated(note = "Use AzureCompletions instead")]
    Azure {
        version: String,
    },
    AzureCompletions {
        version: String,
    },
    AzureResponses {
        version: String,
    },
}

/// Type alias for backward compatibility
pub type OpenAICompletionsAPI = OpenAiApiEndpoints;

impl OpenAiApiEndpoints {
    /// Default version of Azure set to `2025-01-01-preview` as of 5/9/2025
    pub fn default_azure_version() -> String {
        "2025-01-01-preview".to_string()
    }

    /// Parses a string into `OpenAiApiEndpoints`.
    ///
    /// Supported formats (case-insensitive):
    /// - `"OpenAI"` or `"openai_completions"` -> `OpenAiApiEndpoints::OpenAICompletions`
    /// - `"openai_responses"` -> `OpenAiApiEndpoints::OpenAIResponses`
    /// - `"azure:<version>"` or `"azure_completions:<version>"` -> `OpenAiApiEndpoints::AzureCompletions { version }`
    /// - `"azure_responses:<version>"` -> `OpenAiApiEndpoints::AzureResponses { version }`
    ///
    /// Returns default for others.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        let s_lower = s.to_lowercase();
        match s_lower.as_str() {
            "openai" | "openai_completions" => OpenAiApiEndpoints::OpenAICompletions,
            "openai_responses" => OpenAiApiEndpoints::OpenAIResponses,
            _ if s_lower.starts_with("azure") || s_lower.starts_with("azure_completions") => {
                let version = s_lower
                    .strip_prefix("azure:")
                    .or_else(|| s_lower.strip_prefix("azure_completions:"))
                    .map(|v| v.trim().to_string())
                    .unwrap_or_else(OpenAICompletionsAPI::default_azure_version);

                OpenAICompletionsAPI::AzureCompletions { version }
            }
            _ if s_lower.starts_with("azure_responses") => {
                let version = s_lower
                    .strip_prefix("azure_responses:")
                    .map(|v| v.trim().to_string())
                    .unwrap_or_else(OpenAICompletionsAPI::default_azure_version);

                OpenAICompletionsAPI::AzureResponses { version }
            }
            _ => OpenAiApiEndpoints::default(),
        }
    }
}

///
/// OpenAI Assistant Version
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIAssistantVersion {
    V1,
    V2,
    Azure,
    AzureVersion { version: String },
}

impl OpenAIAssistantVersion {
    pub(crate) fn get_endpoint(&self, resource: &OpenAIAssistantResource) -> String {
        //OpenAI documentation: https://platform.openai.com/docs/models/model-endpoint-compatibility
        let trimmed_api_url = (*OPENAI_API_URL).trim_end_matches('/');
        let base_url = match self {
            OpenAIAssistantVersion::V1 | OpenAIAssistantVersion::V2 => {
                format!("{trimmed_api_url}/v1")
            }
            OpenAIAssistantVersion::Azure | OpenAIAssistantVersion::AzureVersion { .. } => {
                format!("{trimmed_api_url}/openai")
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

        // Add Azure version suffix if needed
        match self {
            OpenAIAssistantVersion::Azure => {
                format!("{path}?api-version={}", DEFAULT_AZURE_VERSION)
            }
            OpenAIAssistantVersion::AzureVersion { version } => {
                format!("{path}?api-version={version}")
            }
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
            OpenAIAssistantVersion::Azure | OpenAIAssistantVersion::AzureVersion { .. } => {
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
            OpenAIAssistantVersion::V2
            | OpenAIAssistantVersion::Azure
            | OpenAIAssistantVersion::AzureVersion { .. } => json!([{
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
            OpenAIAssistantVersion::V2
            | OpenAIAssistantVersion::Azure
            | OpenAIAssistantVersion::AzureVersion { .. } => {
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

    /// Parses a string into `OpenAIAssistantVersion`.
    ///
    /// Supported formats (case-insensitive):
    /// - `"v1"` -> `OpenAIAssistantVersion::V1`
    /// - `"v2"` -> `OpenAIAssistantVersion::V2`
    /// - `"azure"` -> `OpenAIAssistantVersion::Azure`
    /// - `"azure:<version>"` -> `OpenAIAssistantVersion::AzureVersion { version }`
    ///
    /// Returns an error for unrecognized formats.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s_lower = s.to_lowercase();
        match s_lower.as_str() {
            "v1" => Ok(OpenAIAssistantVersion::V1),
            "v2" => Ok(OpenAIAssistantVersion::V2),
            _ if s_lower.starts_with("azure") => {
                // Check if the string contains a version after "azure:"
                if let Some(version) = s_lower.strip_prefix("azure:") {
                    Ok(OpenAIAssistantVersion::AzureVersion {
                        version: version.trim().to_string(),
                    })
                } else {
                    // Backward compatibility: if it's just "azure", use a default version
                    Ok(OpenAIAssistantVersion::Azure)
                }
            }
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
    const DEFAULT_AZURE_VERSION: &str = "2024-06-01";

    #[test]
    fn test_v1_assistants_endpoint() {
        let version = OpenAIAssistantVersion::V1;
        let resource = OpenAIAssistantResource::Assistants;
        let expected_url = format!("{}/v1/assistants", OPENAI_API_URL);
        assert_eq!(version.get_endpoint(&resource), expected_url);
    }

    #[test]
    fn test_azure_assistant_endpoint() {
        let version = OpenAIAssistantVersion::AzureVersion {
            version: "2024-05-01-preview".to_string(),
        };
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
    fn test_default_azure_assistant_endpoint() {
        let version = OpenAIAssistantVersion::from_str("azure").unwrap();
        let resource = OpenAIAssistantResource::Assistant {
            assistant_id: "123".to_string(),
        };
        let expected_url = format!(
            "{}/openai/assistants/123?api-version={}",
            OPENAI_API_URL, DEFAULT_AZURE_VERSION
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
        let version = OpenAIAssistantVersion::AzureVersion {
            version: "2024-05-01-preview".to_string(),
        };
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
        let version = OpenAIAssistantVersion::AzureVersion {
            version: "2024-05-01-preview".to_string(),
        };
        let expected_payload: Value = json!([{
            "type": "file_search"
        }]);
        assert_eq!(version.get_tools_payload(), expected_payload);
    }

    // Deserializing from string
    #[test]
    fn test_v1_version() {
        let result = OpenAIAssistantVersion::from_str("v1");
        assert_eq!(result.unwrap(), OpenAIAssistantVersion::V1);
    }

    #[test]
    fn test_v2_version() {
        let result = OpenAIAssistantVersion::from_str("v2");
        assert_eq!(result.unwrap(), OpenAIAssistantVersion::V2);
    }

    #[test]
    fn test_azure_with_version() {
        let result = OpenAIAssistantVersion::from_str("azure:2024-09-01");
        assert_eq!(
            result.unwrap(),
            OpenAIAssistantVersion::AzureVersion {
                version: "2024-09-01".to_string(),
            }
        );
    }

    #[test]
    fn test_azure_with_spaces_in_version() {
        let result = OpenAIAssistantVersion::from_str("azure: 2024-09-01 ");
        assert_eq!(
            result.unwrap(),
            OpenAIAssistantVersion::AzureVersion {
                version: "2024-09-01".to_string(), // Spaces trimmed
            }
        );
    }

    #[test]
    fn test_azure_default_version() {
        let result = OpenAIAssistantVersion::from_str("azure");
        assert_eq!(result.unwrap(), OpenAIAssistantVersion::Azure);
    }

    #[test]
    fn test_invalid_version() {
        let result = OpenAIAssistantVersion::from_str("invalid_version");
        assert!(result.is_err());
        assert_eq!(
            format!("{}", result.unwrap_err()),
            "Invalid version: invalid_version"
        );
    }
}
