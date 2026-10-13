use anyhow::Result;
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::completions::{build_citations, CompletionCitations, ThinkingLevel};
use crate::constants::XAI_API_URL;
use crate::domain::{RateLimit, XAIChatMessage, XAIChatResponse, XAIResponseOutput, XAIRole};
use crate::llm_models::{LLMModel, LLMTools};

// API Docs: https://docs.x.ai/docs/models
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum XAIModels {
    Grok4_3,
    Grok4_1FastReasoning,
    Grok4_1FastNonReasoning,
    Grok4FastReasoning,
    Grok4FastNonReasoning,
    Grok4,
    GrokCodeFast1,
    Grok3,
    Grok3Mini,
    Grok3Fast,
    Grok3MiniFast,
}

#[async_trait(?Send)]
impl LLMModel for XAIModels {
    fn as_str(&self) -> &str {
        match self {
            XAIModels::Grok4_3 => "grok-4.3",
            XAIModels::Grok4_1FastReasoning => "grok-4-1-fast-reasoning",
            XAIModels::Grok4_1FastNonReasoning => "grok-4-1-fast-non-reasoning",
            XAIModels::Grok4FastReasoning => "grok-4-fast-reasoning",
            XAIModels::Grok4FastNonReasoning => "grok-4-fast-non-reasoning",
            XAIModels::Grok4 => "grok-4",
            XAIModels::GrokCodeFast1 => "grok-code-fast-1",
            XAIModels::Grok3 => "grok-3",
            XAIModels::Grok3Mini => "grok-3-mini",
            XAIModels::Grok3Fast => "grok-3-fast",
            XAIModels::Grok3MiniFast => "grok-3-mini-fast",
        }
    }

    // Docs: https://docs.x.ai/docs/models
    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "grok-4.3" => Some(XAIModels::Grok4_3),
            "grok-4-1-fast" => Some(XAIModels::Grok4_1FastReasoning),
            "grok-4-1-fast-reasoning-latest" => Some(XAIModels::Grok4_1FastReasoning),
            "grok-4-1-fast-reasoning" => Some(XAIModels::Grok4_1FastReasoning),
            "grok-4-1-fast-non-reasoning" => Some(XAIModels::Grok4_1FastNonReasoning),
            "grok-4-1-fast-non-reasoning-latest" => Some(XAIModels::Grok4_1FastNonReasoning),
            "grok-4-fast" => Some(XAIModels::Grok4FastReasoning),
            "grok-4-fast-reasoning" => Some(XAIModels::Grok4FastReasoning),
            "grok-4-fast-reasoning-latest" => Some(XAIModels::Grok4FastReasoning),
            "grok-4-fast-non-reasoning" => Some(XAIModels::Grok4FastNonReasoning),
            "grok-4-fast-non-reasoning-latest" => Some(XAIModels::Grok4FastNonReasoning),
            "grok-4" => Some(XAIModels::Grok4),
            "grok-4-latest" => Some(XAIModels::Grok4),
            "grok-4-0709" => Some(XAIModels::Grok4),
            "grok-code-fast-1" => Some(XAIModels::GrokCodeFast1),
            "grok-code-fast" => Some(XAIModels::GrokCodeFast1),
            "grok-code-fast-1-0825" => Some(XAIModels::GrokCodeFast1),
            "grok-3" => Some(XAIModels::Grok3),
            "grok-3-latest" => Some(XAIModels::Grok3),
            "grok-3-beta" => Some(XAIModels::Grok3),
            "grok-3-mini" => Some(XAIModels::Grok3Mini),
            "grok-3-mini-latest" => Some(XAIModels::Grok3Mini),
            "grok-3-mini-beta" => Some(XAIModels::Grok3Mini),
            "grok-3-fast" => Some(XAIModels::Grok3Fast),
            "grok-3-fast-latest" => Some(XAIModels::Grok3Fast),
            "grok-3-fast-beta" => Some(XAIModels::Grok3Fast),
            "grok-3-mini-fast" => Some(XAIModels::Grok3MiniFast),
            "grok-3-mini-fast-latest" => Some(XAIModels::Grok3MiniFast),
            "grok-3-mini-fast-beta" => Some(XAIModels::Grok3MiniFast),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        // Docs: https://docs.x.ai/docs/models
        match self {
            XAIModels::Grok4_3 => 1_000_000,
            XAIModels::Grok4_1FastReasoning => 2_097_152,
            XAIModels::Grok4_1FastNonReasoning => 2_097_152,
            XAIModels::Grok4FastReasoning => 2_097_152,
            XAIModels::Grok4FastNonReasoning => 2_097_152,
            XAIModels::Grok4 => 256_000,
            XAIModels::GrokCodeFast1 => 256_000,
            XAIModels::Grok3 => 131_072,
            XAIModels::Grok3Mini => 131_072,
            XAIModels::Grok3Fast => 131_072,
            XAIModels::Grok3MiniFast => 131_072,
        }
    }

    fn get_endpoint(&self) -> String {
        XAI_API_URL.to_string()
    }

    //This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        tools: Option<&[LLMTools]>,
        _thinking_level: Option<&ThinkingLevel>,
    ) -> serde_json::Value {
        // Get system instructions
        let base_instructions = self.get_base_instructions(Some(function_call));

        // Set the structured output schema

        // TODO: Using structured output with JSON Schema is not working. Attaching schema to instructions until fixed.
        // let response_format = Some(XAIResponseFormat{
        //     r#type: XAIResponseFormatType::JsonSchema,
        //     json_schema: Some(json_schema.clone()),
        // });

        let instructions = format!(
            "<instructions>{}</instructions>
            <output_json_schema>{:?}</output_json_schema>",
            instructions, json_schema,
        );

        // Build tools array if provided
        let tools = if let Some(tools_inner) = tools {
            let processed_tools: Vec<Value> = tools_inner
                .iter()
                .filter_map(LLMTools::get_config_json)
                .collect::<Vec<Value>>();

            if processed_tools.is_empty() {
                None
            } else {
                Some(processed_tools)
            }
        } else {
            None
        };

        // Build JSON body directly like other providers
        json!({
            "model": self.as_str(),
            "instructions": base_instructions,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "input": vec![
                XAIChatMessage::new(XAIRole::User, instructions.to_string()),
            ],
            "tools": tools,
        })
    }

    /*
     * This function leverages xAI API to perform any query as per the provided body.
     *
     * It returns a String the Response object that needs to be parsed based on the self.model.
     */
    async fn call_api(
        &self,
        api_key: &str,
        _version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
        _tools: Option<&[LLMTools]>,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_endpoint();

        if debug {
            info!("[debug] xAI API URL: {:#?}", model_url);
        }

        //Make the API call
        let client = Client::new();

        //Send request
        let response = client
            .post(model_url)
            .header(header::CONTENT_TYPE, "application/json")
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if debug {
            info!(
                "[debug] xAI API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        //Convert API response to struct representing expected response format
        let messages_response: XAIChatResponse = serde_json::from_str(response_text)?;

        // Extract text from message outputs
        let assistant_response = messages_response
            .output
            .iter()
            .filter_map(|output| {
                if let XAIResponseOutput::Message(message) = output {
                    if message.role == "assistant" {
                        message.content.iter().find_map(|content| {
                            if content.content_type == "output_text" {
                                content.text.as_ref()
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .fold(String::new(), |mut acc, text| {
                acc.push_str(&self.sanitize_json_response(text));
                acc
            });

        //Return completions text
        Ok(assistant_response)
    }

    fn extract_citations(
        &self,
        response_text: &str,
        _version: Option<String>,
    ) -> CompletionCitations {
        let Ok(response) = serde_json::from_str::<XAIChatResponse>(response_text) else {
            return CompletionCitations::default();
        };

        let mut candidates = Vec::new();
        for output in &response.output {
            match output {
                XAIResponseOutput::Message(message) => {
                    for content in &message.content {
                        if let Some(annotations) = &content.annotations {
                            for annotation in annotations {
                                if let Some(url) = &annotation.url {
                                    candidates.push((url.clone(), annotation.title.clone()));
                                }
                            }
                        }
                    }
                }
                XAIResponseOutput::WebSearchCall(call) => {
                    if let Some(url) = &call.action.url {
                        candidates.push((url.clone(), None));
                    }
                    if let Some(sources) = &call.action.sources {
                        for source in sources {
                            if let Some(url) = source.get("url").and_then(|v| v.as_str()) {
                                let title = source
                                    .get("title")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                candidates.push((url.to_string(), title));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        build_citations(candidates)
    }

    /// This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        //xAI documentation: https://docs.x.ai/developers/models
        match self {
            XAIModels::Grok4_3 => RateLimit {
                tpm: 10_000_000,
                rpm: 1_800,
            },
            XAIModels::Grok4_1FastReasoning => RateLimit {
                tpm: 4_000_000,
                rpm: 480,
            },
            XAIModels::Grok4_1FastNonReasoning => RateLimit {
                tpm: 4_000_000,
                rpm: 480,
            },
            XAIModels::Grok4FastReasoning => RateLimit {
                tpm: 4_000_000,
                rpm: 480,
            },
            XAIModels::Grok4FastNonReasoning => RateLimit {
                tpm: 4_000_000,
                rpm: 480,
            },
            XAIModels::Grok4 => RateLimit {
                tpm: 2_000_000,
                rpm: 480,
            },
            XAIModels::GrokCodeFast1 => RateLimit {
                tpm: 2_000_000,
                rpm: 480,
            },
            XAIModels::Grok3 => RateLimit {
                tpm: 2_000_000, // Not documented. Assuming same as Grok4.
                rpm: 600,
            },
            XAIModels::Grok3Mini => RateLimit {
                tpm: 2_000_000, // Not documented. Assuming same as Grok4.
                rpm: 480,
            },
            _ => RateLimit {
                // Not documented. Assuming same as Grok4.
                tpm: 2_000_000,
                rpm: 480,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_models::LLMModel;

    const CITATION_FIXTURE: &str = r#"{
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "shoes",
                    "url": "https://search.example/q",
                    "sources": [
                        {"url": "https://shop.example/a", "title": "Shop A"},
                        {"url": "https://shop.example/b"}
                    ]
                }
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "{\"result\":\"ok\"}",
                    "annotations": [
                        {"type": "url_citation", "url": "https://shop.example/a", "title": "Shop A"},
                        {"type": "url_citation", "url": "https://news.example/c", "title": "News"}
                    ]
                }]
            }
        ]
    }"#;

    const NO_CITATION_FIXTURE: &str = r#"{
        "output": [{
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "{\"result\":\"ok\"}"}]
        }]
    }"#;

    #[test]
    fn extract_citations_from_annotations_and_web_search() {
        let model = XAIModels::Grok3Mini;
        let citations = model.extract_citations(CITATION_FIXTURE, None);
        assert_eq!(
            citations.urls,
            vec![
                "https://search.example/q".to_string(),
                "https://shop.example/a".to_string(),
                "https://shop.example/b".to_string(),
                "https://news.example/c".to_string(),
            ]
        );
        assert_eq!(citations.items[1].title.as_deref(), Some("Shop A"));
        assert!(model
            .get_data(CITATION_FIXTURE, false)
            .unwrap()
            .contains("ok"));
    }

    #[test]
    fn extract_citations_empty_when_absent() {
        let model = XAIModels::Grok3Mini;
        let citations = model.extract_citations(NO_CITATION_FIXTURE, None);
        assert!(citations.urls.is_empty());
    }
}
