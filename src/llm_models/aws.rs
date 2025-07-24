use anyhow::{anyhow, Result};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    types::{ContentBlock, ConversationRole, InferenceConfiguration, Message, SystemContentBlock},
    Client,
};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::constants::{AWS_BEDROCK_API_URL, AWS_REGION};
use crate::domain::RateLimit;
use crate::llm_models::{LLMModel, LLMTools};

#[derive(Serialize, Deserialize)]
struct AwsBedrockRequestBody {
    instructions: String,
    json_schema: Value,
    max_tokens: i32,
    temperature: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
// AWS Bedrock docs: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
pub enum AwsBedrockModels {
    NovaPro,
    NovaLite,
    NovaMicro,
}

#[async_trait(?Send)]
impl LLMModel for AwsBedrockModels {
    fn as_str(&self) -> &str {
        match self {
            AwsBedrockModels::NovaPro => "amazon.nova-pro-v1:0",
            AwsBedrockModels::NovaLite => "amazon.nova-lite-v1:0",
            AwsBedrockModels::NovaMicro => "amazon.nova-micro-v1:0",
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "amazon.nova-pro-v1:0" => Some(AwsBedrockModels::NovaPro),
            "amazon.nova-lite-v1:0" => Some(AwsBedrockModels::NovaLite),
            "amazon.nova-micro-v1:0" => Some(AwsBedrockModels::NovaMicro),
            _ => None,
        }
    }

    fn default_max_tokens(&self) -> usize {
        match self {
            AwsBedrockModels::NovaPro => 5_120,
            AwsBedrockModels::NovaLite => 5_120,
            AwsBedrockModels::NovaMicro => 5_120,
        }
    }

    fn get_endpoint(&self) -> String {
        format!("{}/model/{}/converse", &*AWS_BEDROCK_API_URL, self.as_str())
    }

    /// AWS Bedrock implementation leverages AWS Bedrock SKD, therefore data is only passed by this method to `call_api` method where the actual logic is implemented
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        _function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
        _tools: Option<&[LLMTools]>,
    ) -> serde_json::Value {
        let body = AwsBedrockRequestBody {
            instructions: instructions.to_string(),
            json_schema: json_schema.clone(),
            max_tokens: *max_tokens as i32,
            temperature: *temperature,
        };

        // Return the body serialized as a JSON value
        serde_json::to_value(body).unwrap()
    }

    /// This function leverages AWS Bedrock SDK to perform any query as per the provided body.
    async fn call_api(
        &self,
        // AWS Bedrock SDK utilizes `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables for request authentication
        // Docs: https://docs.aws.amazon.com/sdk-for-rust/latest/dg/credproviders.html
        _api_key: &str,
        _version: Option<String>,
        body: &serde_json::Value,
        debug: bool,
        _tools: Option<&[LLMTools]>,
    ) -> Result<String> {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(&**AWS_REGION)
            .load()
            .await;
        let client = Client::new(&sdk_config);

        // Get request info from body
        let request_body_opt: Option<AwsBedrockRequestBody> =
            serde_json::from_value(body.clone()).ok();
        let (instructions_opt, json_schema_opt, max_tokens_opt, temperature_opt) = request_body_opt
            .map_or_else(
                || (None, None, None, None),
                |request_body| {
                    (
                        Some(request_body.instructions),
                        Some(request_body.json_schema),
                        Some(request_body.max_tokens),
                        Some(request_body.temperature),
                    )
                },
            );

        // Get base instructions
        let base_instructions = self.get_base_instructions(None);

        let converse_builder = client
            .converse()
            .model_id(self.as_str())
            .system(SystemContentBlock::Text(base_instructions));

        // Add user instructions including the expected output schema if specifed
        let instructions = instructions_opt.unwrap_or_default();
        let user_instructions = json_schema_opt
            .map(|schema| {
                format!(
                    "<instructions>
                    {instructions}
                    </instructions>
                    <output json schema>
                    {schema}
                    </output json schema>"
                )
            })
            .unwrap_or(instructions);
        let converse_builder = converse_builder.messages(
            Message::builder()
                .role(ConversationRole::User)
                .content(ContentBlock::Text(user_instructions))
                .build()
                .map_err(|_| anyhow!("failed to build message"))?,
        );

        // If specified add inference config
        let converse_builder = if max_tokens_opt.is_some() || temperature_opt.is_some() {
            let inference_config = InferenceConfiguration::builder()
                .set_max_tokens(max_tokens_opt)
                .set_temperature(temperature_opt)
                .build();
            converse_builder.set_inference_config(Some(inference_config))
        } else {
            converse_builder
        };

        // Send request
        let converse_response = converse_builder.send().await?;

        if debug {
            info!(
                "[debug] AWS Bedrock API response: {:#?}",
                &converse_response
            );
        }

        //Parse the response and return the assistant content
        let text = converse_response
            .output()
            .ok_or(anyhow!("no output"))?
            .as_message()
            .map_err(|_| anyhow!("output not a message"))?
            .content()
            .first()
            .ok_or(anyhow!("no content in message"))?
            .as_text()
            .map_err(|_| anyhow!("content is not text"))?
            .to_string();
        Ok(self.sanitize_json_response(&text))
    }

    /// AWS Bedrock implementation leverages AWS Bedrock SDK, therefore data extraction is implemented directly in `call_api` method and this method only passes the data on
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        Ok(response_text.to_string())
    }

    //This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        // Docs: https://docs.aws.amazon.com/general/latest/gr/bedrock.html
        match self {
            AwsBedrockModels::NovaPro => RateLimit {
                tpm: 400_000,
                rpm: 100,
            },
            AwsBedrockModels::NovaLite | AwsBedrockModels::NovaMicro => RateLimit {
                tpm: 2_000_000,
                rpm: 1_000,
            },
        }
    }
}
