use anyhow::Result;
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::constants::XAI_API_URL;
use crate::domain::{
    XAIAssistantMessageRole, XAIChatMessage, XAIChatRequest, XAIChatResponse, XAIRole,
};
use crate::llm_models::{LLMModel, LLMTools};

// API Docs: https://docs.x.ai/docs/models
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum XAIModels {
    Grok4,
    Grok3,
    Grok3Mini,
    Grok3Fast,
    Grok3MiniFast,
}

#[async_trait(?Send)]
impl LLMModel for XAIModels {
    fn as_str(&self) -> &str {
        match self {
            XAIModels::Grok4 => "grok-4",
            XAIModels::Grok3 => "grok-3",
            XAIModels::Grok3Mini => "grok-3-mini",
            XAIModels::Grok3Fast => "grok-3-fast",
            XAIModels::Grok3MiniFast => "grok-3-mini-fast",
        }
    }

    // Docs: https://docs.anthropic.com/en/docs/about-claude/models/overview#model-aliases
    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "grok-4" => Some(XAIModels::Grok4),
            "grok-4-latest" => Some(XAIModels::Grok4),
            "grok-4-0709" => Some(XAIModels::Grok4),
            "grok-3" => Some(XAIModels::Grok3),
            "grok-3-mini" => Some(XAIModels::Grok3Mini),
            "grok-3-fast" => Some(XAIModels::Grok3Fast),
            "grok-3-mini-fast" => Some(XAIModels::Grok3MiniFast),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        // This is the max tokens allowed for response and not context as per documentation: https://docs.anthropic.com/en/docs/about-claude/models/overview#model-comparison-table
        match self {
            XAIModels::Grok4 => 256_000,
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
        _tools: Option<&[LLMTools]>,
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

        // TODOs:
        // Search Tool
        // TextFile tool

        let chat_request = XAIChatRequest {
            model: self.as_str().to_string(),
            max_completion_tokens: Some(*max_tokens),
            temperature: Some(*temperature),
            messages: vec![
                XAIChatMessage::new(XAIRole::System, base_instructions),
                XAIChatMessage::new(XAIRole::User, instructions.to_string()),
            ],
            response_format: None,
            tools: None,
            search_parameters: None,
        };

        serde_json::to_value(chat_request).unwrap_or_default()
    }

    /*
     * This function leverages Anthropic API to perform any query as per the provided body.
     *
     * It returns a String the Response object that needs to be parsed based on the self.model.
     */
    async fn call_api(
        &self,
        api_key: &str,
        _version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_endpoint();

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

        let assistant_response = messages_response
            .choices
            .iter()
            .map(|item| &item.message)
            .filter(|message| message.role == XAIAssistantMessageRole::Assistant)
            .filter_map(|message| {
                // Use content or reasoning_content if present
                message
                    .content
                    .as_ref()
                    .or(message.reasoning_content.as_ref())
            })
            .fold(String::new(), |mut acc, content| {
                acc.push_str(content);
                acc
            });

        //Return completions text
        Ok(assistant_response)
    }
}
