use anyhow::{anyhow, Result as Anysult};
use lazy_static::lazy_static;
use log::error;
use log::{info, warn};
use reqwest::{header, Client};
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tiktoken_rs::{cl100k_base, get_bpe_from_model, CoreBPE};
use tokio::time::{self, timeout};

lazy_static! {
    pub static ref OPENAI_API_URL: String = std::env::var("OPENAI_API_URL").unwrap();
}

//Generic OpenAI instructions
const OPENAI_BASE_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
Step 1: Review and understand the 'instructions' from the *Instructions* section.
Step 2: Based on the 'instructions' process the data provided in the *Input data* section using your language model.
Step 3: Prepare a response by processing the 'input data' as per the 'instructions'. 
Step 4: Convert the response to a Json object. The Json object must match the schema provided in the *Output Json schema* section.
Step 5: Validate that the Json object matches the 'output Json schema' and correct if needed. If you are not able to generate a valid Json based on the 'input data' and 'instructions' please respond with "Error calculating the answer."
Step 6: Respond ONLY with properly formatted Json object. No other words or text, only valid Json in the answer.
"#;

const OPENAI_FUNCTION_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
Step 1: Review and understand the 'instructions' from the *Instructions* section.
Step 2: Based on the 'instructions' process the data provided in the *Input data* section using your language model.
Step 3: Prepare a response by processing the 'input data' as per the 'instructions'. 
Step 4: Convert the response to a Json object. The Json object must match the schema provided in the function definition.
Step 5: Validate that the Json object matches the function properties and correct if needed. If you are not able to generate a valid Json based on the 'input data' and 'instructions' please respond with "Error calculating the answer."
Step 6: Respond ONLY with properly formatted Json object. No other words or text, only valid Json in the answer.
"#;

const OPENAI_ASSISTANT_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
1: Review and understand the content of user messages passed to you in the thread.
2: Review and consider any files the user provided attached to the messages.
3: Prepare response using your language model based on the user messages and provided files.
4: Respond ONLY with properly formatted data portion of a Json. No other words or text, only valid Json in your answers. 
"#;

//OpenAI API response type format for Completions API
#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAPICompletionsResponse {
    id: Option<String>,
    object: Option<String>,
    created: Option<u32>,
    model: Option<String>,
    choices: Option<Vec<OpenAPICompletionsChoices>>,
    usage: Option<OpenAPIUsage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAPICompletionsChoices {
    text: Option<String>,
    index: Option<u32>,
    logprobs: Option<u32>,
    finish_reason: Option<String>,
}

//OpenAI API response type format for Chat API
#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAPIChatResponse {
    id: Option<String>,
    object: Option<String>,
    created: Option<u32>,
    model: Option<String>,
    choices: Option<Vec<OpenAPIChatChoices>>,
    usage: Option<OpenAPIUsage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAPIChatChoices {
    message: OpenAPIChatMessage,
    index: Option<u32>,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAPIChatMessage {
    role: String,
    content: Option<String>,
    function_call: Option<OpenAPIChatFunctionCall>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAPIChatFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAPIUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIRateLimit {
    tpm: usize, // tokens-per-minute
    rpm: usize, // requests-per-minute
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum OpenAIModels {
    Gpt3_5Turbo,
    Gpt3_5Turbo0613,
    Gpt3_5Turbo16k,
    Gpt4,
    Gpt4_32k,
    TextDavinci003,
    Gpt4Turbo,
}

#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct OpenAIDataResponse<T: JsonSchema> {
    pub data: T,
}

impl OpenAIModels {
    pub fn as_str(&self) -> &'static str {
        match self {
            //In an API call, you can describe functions to gpt-3.5-turbo-0613 and gpt-4-0613
            //On June 27, 2023 the stable gpt-3.5-turbo will be automatically upgraded to gpt-3.5-turbo-0613
            OpenAIModels::Gpt3_5Turbo => "gpt-3.5-turbo",
            OpenAIModels::Gpt3_5Turbo0613 => "gpt-3.5-turbo-0613",
            OpenAIModels::Gpt3_5Turbo16k => "gpt-3.5-turbo-16k",
            OpenAIModels::Gpt4 => "gpt-4-0613",
            OpenAIModels::Gpt4_32k => "gpt-4-32k",
            OpenAIModels::TextDavinci003 => "text-davinci-003",
            OpenAIModels::Gpt4Turbo => "gpt-4-1106-preview",
        }
    }

    pub fn default_max_tokens(&self) -> usize {
        //OpenAI documentation: https://platform.openai.com/docs/models/gpt-3-5
        //This is the max tokens allowed between prompt & response
        match self {
            OpenAIModels::Gpt3_5Turbo => 4096,
            OpenAIModels::Gpt3_5Turbo0613 => 4096,
            OpenAIModels::Gpt3_5Turbo16k => 16384,
            OpenAIModels::Gpt4 => 8192,
            OpenAIModels::Gpt4_32k => 32768,
            OpenAIModels::TextDavinci003 => 4097,
            OpenAIModels::Gpt4Turbo => 128_000,
        }
    }

    fn get_endpoint(&self) -> String {
        //OpenAI documentation: https://platform.openai.com/docs/models/model-endpoint-compatibility
        match self {
            OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4_32k => {
                format!(
                    "{OPENAI_API_URL}/v1/chat/completions",
                    OPENAI_API_URL = *OPENAI_API_URL
                )
            }
            OpenAIModels::TextDavinci003 => format!(
                "{OPENAI_API_URL}/v1/completions",
                OPENAI_API_URL = *OPENAI_API_URL
            ),
        }
    }

    fn get_base_instructions(&self, function_call: Option<bool>) -> String {
        let function_call = function_call.unwrap_or_else(|| self.function_call_default());
        match function_call {
            true => OPENAI_FUNCTION_INSTRUCTIONS.to_string(),
            false => OPENAI_BASE_INSTRUCTIONS.to_string(),
        }
    }

    fn function_call_default(&self) -> bool {
        //OpenAI documentation: https://platform.openai.com/docs/guides/gpt/function-calling
        match self {
            OpenAIModels::TextDavinci003 | OpenAIModels::Gpt3_5Turbo | OpenAIModels::Gpt4_32k => {
                false
            }
            OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo => true,
        }
    }

    //This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &u32,
    ) -> serde_json::Value {
        match self {
            //https://platform.openai.com/docs/api-reference/completions/create
            //For DaVinci model all text goes into the 'prompt' filed of the body
            OpenAIModels::TextDavinci003 => {
                let schema_string = serde_json::to_string(json_schema).unwrap_or_default();
                let base_instructions = self.get_base_instructions(Some(function_call));
                json!({
                    "model": self.as_str(),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "prompt": format!(
                        "{base_instructions}\n\n
                        Output Json schema:\n
                        {schema_string}\n\n
                        {instructions}",
                    ),
                })
            }
            OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4_32k => {
                let base_instructions = self.get_base_instructions(Some(function_call));
                let system_message = json!({
                    "role": "system",
                    "content": base_instructions,
                });

                match function_call {
                    //If we choose to use function calling
                    //https://platform.openai.com/docs/guides/gpt/function-calling
                    true => {
                        let user_message = json!({
                            "role": "user",
                            "content": instructions,
                        });

                        let function = json!({
                            "name": "analyze_data",
                            "description": "Use this function to compute the answer based on input data, instructions and your language model. Output should be a fully formed JSON object.",
                            "parameters": json_schema,
                        });

                        let function_call = json!({
                            "name": "analyze_data"
                        });

                        //For ChatGPT we ignore max_tokens. It will default to 'inf'
                        json!({
                            "model": self.as_str(),
                            "temperature": temperature,
                            "messages": vec![
                                system_message,
                                user_message,
                            ],
                            "functions": vec![
                                function,
                            ],
                            //This forces ChatGPT to use the function definition
                            "function_call": function_call,
                        })
                    }
                    //https://platform.openai.com/docs/guides/chat/introduction
                    false => {
                        let schema_string = serde_json::to_string(json_schema).unwrap_or_default();

                        let user_message = json!({
                            "role": "user",
                            "content": format!(
                                "Output Json schema:\n
                                {schema_string}\n\n
                                {instructions}"
                            ),
                        });
                        //For ChatGPT we ignore max_tokens. It will default to 'inf'
                        json!({
                            "model": self.as_str(),
                            "temperature": temperature,
                            "messages": vec![
                                system_message,
                                user_message,
                            ],
                        })
                    }
                }
            }
        }
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_data(&self, response_text: &str, function_call: bool) -> Anysult<String> {
        match self {
            //https://platform.openai.com/docs/api-reference/completions/create
            OpenAIModels::TextDavinci003 => {
                //Convert API response to struct representing expected response format
                let completions_response: OpenAPICompletionsResponse =
                    serde_json::from_str(response_text)?;

                //Extract data part
                match completions_response.choices {
                    Some(choices) => Ok(choices.into_iter().filter_map(|item| item.text).collect()),
                    None => Err(anyhow!(
                        "Unable to retrieve response from OpenAI Completions API"
                    )),
                }
            }
            //https://platform.openai.com/docs/guides/chat/introduction
            OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4_32k => {
                //Convert API response to struct representing expected response format
                let chat_response: OpenAPIChatResponse = serde_json::from_str(response_text)?;

                //Extract data part
                match chat_response.choices {
                    Some(choices) => Ok(choices
                        .into_iter()
                        .filter_map(|item| {
                            //For function_call the response is in arguments, and for regular call in content
                            match function_call {
                                true => item
                                    .message
                                    .function_call
                                    .map(|function_call| function_call.arguments),
                                false => item.message.content,
                            }
                        })
                        .collect()),
                    None => Err(anyhow!("Unable to retrieve response from OpenAI Chat API")),
                }
            }
        }
    }

    //This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> OpenAIRateLimit {
        //OpenAI documentation: https://platform.openai.com/account/rate-limits
        //This is the max tokens allowed between prompt & response
        match self {
            OpenAIModels::Gpt3_5Turbo => OpenAIRateLimit {
                tpm: 90_000,
                rpm: 3_500,
            },
            OpenAIModels::Gpt3_5Turbo0613 => OpenAIRateLimit {
                tpm: 90_000,
                rpm: 3_500,
            },
            OpenAIModels::Gpt3_5Turbo16k => OpenAIRateLimit {
                tpm: 180_000,
                rpm: 3_500,
            },
            OpenAIModels::Gpt4 => OpenAIRateLimit {
                tpm: 10_000,
                rpm: 200,
            },
            OpenAIModels::Gpt4Turbo => OpenAIRateLimit {
                tpm: 10_000,
                rpm: 200,
            },
            OpenAIModels::Gpt4_32k => OpenAIRateLimit {
                tpm: 10_000,
                rpm: 200,
            },
            OpenAIModels::TextDavinci003 => OpenAIRateLimit {
                tpm: 250_000,
                rpm: 3_000,
            },
        }
    }

    //This function checks how many requests can be sent to an OpenAI model within a minute
    pub fn get_max_requests(&self) -> usize {
        let rate_limit = self.get_rate_limit();

        //Check max requests based on rpm
        let max_requests_from_rpm = rate_limit.rpm;

        //Double check max number of requests based on tpm
        //Assume we will use ~50% of allowed tokens per request (for prompt + response)
        let max_tokens_per_minute = rate_limit.tpm;
        let tpm_per_request = (self.default_max_tokens() as f64 * 0.5).ceil() as usize;
        //Then check how many requests we can process
        let max_requests_from_tpm = max_tokens_per_minute / tpm_per_request;

        //To be safe we go with smaller of the numbers
        std::cmp::min(max_requests_from_rpm, max_requests_from_tpm)
    }
}

pub struct OpenAI {
    model: OpenAIModels,
    //For prompt & response
    max_tokens: usize,
    temperature: u32,
    input_json: Option<String>,
    debug: bool,
    function_call: bool,
    api_key: String,
}

impl OpenAI {
    //Constructor
    pub fn new(
        open_ai_key: &str,
        model: OpenAIModels,
        max_tokens: Option<usize>,
        temperature: Option<u32>,
    ) -> Self {
        OpenAI {
            //If no max tokens limit is provided we default to max allowed for the model
            max_tokens: max_tokens.unwrap_or_else(|| model.default_max_tokens()),
            function_call: model.function_call_default(),
            model,
            temperature: temperature.unwrap_or(0u32), //Low number makes the output less random and more deterministic
            input_json: None,
            debug: false,
            api_key: open_ai_key.to_string(),
        }
    }

    /*
     * This function turns on debug mode which will info! the prompt to log when executing it.
     */
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /*
     * This function turns on/off function calling mode when interacting with OpenAI API.
     */
    pub fn function_calling(mut self, function_call: bool) -> Self {
        self.function_call = function_call;
        self
    }

    /*
     * This method can be used to provide values that will be used as context for the prompt.
     * Using this function you can provide multiple input values by calling it multiple times. New values will be appended with the category name
     * It accepts any instance that implements the Serialize trait.
     */
    pub fn set_context<T: Serialize>(mut self, input_name: &str, input_data: &T) -> Anysult<Self> {
        let input_json = if let Ok(json) = serde_json::to_string(&input_data) {
            json
        } else {
            return Err(anyhow!("Unable serialize provided input data."));
        };
        let line_break = match self.input_json {
            Some(_) => "\n\n".to_string(),
            None => "".to_string(),
        };
        let new_json = format!(
            "{}{}{}: {}",
            self.input_json.unwrap_or_default(),
            line_break,
            input_name,
            input_json,
        );
        self.input_json = Some(new_json);
        Ok(self)
    }

    /*
     * This method is used to check how many tokens would most likely remain for the response
     * This is accomplished by estimating number of tokens needed for system/base instructions, user prompt, and function components including schema definition.
     */
    pub fn check_prompt_tokens<T: JsonSchema + DeserializeOwned>(
        &self,
        instructions: &str,
    ) -> Anysult<usize> {
        //Output schema is extracted from the type parameter
        let schema = schema_for!(T);
        let json_value: Value = serde_json::to_value(&schema)?;

        let prompt = format!(
            "Instructions:
            {instructions}

            Input data:
            {input_json}
            
            Respond ONLY with the data portion of a valid Json object. No schema definition required. No other words.", 
            instructions = instructions,
            input_json = self.input_json.clone().unwrap_or_default(),
        );

        let full_prompt = format!(
            "{}{}{}",
            //Base (system) instructions
            self.model.get_base_instructions(Some(self.function_call)),
            //Instructions & context data
            prompt,
            //Output schema
            serde_json::to_string(&json_value).unwrap_or_default()
        );

        //Check how many tokens are required for prompt
        let bpe = get_tokenizer(&self.model)?;
        let prompt_tokens = bpe.encode_with_special_tokens(&full_prompt).len();

        //Assuming another 5% overhead for json formatting
        Ok((prompt_tokens as f64 * 1.05) as usize)
    }

    /*
     * This function leverages OpenAI API to perform any query as per the provided body.
     *
     * It returns a String the Response object that needs to be parsed based on the self.model.
     */
    async fn call_openai_api(&self, body: &serde_json::Value) -> Anysult<String> {
        //Get the API url
        let model_url = self.model.get_endpoint();

        //Make the API call
        let client = Client::new();

        let response = client
            .post(model_url)
            .header(header::CONTENT_TYPE, "application/json")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    /*
     * This method is used to submit a prompt to OpenAI and process the response.
     * When calling the function you need to specify the type parameter as the response will match the schema of that type.
     * The prompt in this function is written in a way to instruct OpenAI to behave like a computer function that calculates an output based on provided input and its language model.
     */
    pub async fn get_answer<T: JsonSchema + DeserializeOwned>(
        self,
        instructions: &str,
    ) -> Anysult<T> {
        //Output schema is extracted from the type parameter
        let schema = schema_for!(T);
        let json_value: Value = serde_json::to_value(&schema)?;

        let prompt = format!(
            "Instructions:
            {instructions}

            Input data:
            {input_json}
            
            Respond ONLY with the data portion of a valid Json object. No schema definition required. No other words.", 
            instructions = instructions,
            input_json = self.input_json.clone().unwrap_or_default(),
        );

        //Validate how many tokens remain for the response (and how many are used for prompt)
        let prompt_tokens = self
            .check_prompt_tokens::<T>(instructions)
            .unwrap_or_default();

        if prompt_tokens >= self.max_tokens {
            return Err(anyhow!(
                "The provided prompt requires more tokens than allocated."
            ));
        }
        let response_tokens = self.max_tokens - prompt_tokens;

        //Throw a warning if after processing the prompt there might be not enough tokens for response
        //This assumes response will be similar size as input. Because this is not always correct this is a warning and not an error
        if prompt_tokens * 2 >= self.max_tokens {
            warn!(
                "{} tokens remaining for response: {} allocated, {} used for prompt",
                response_tokens.to_string(),
                self.max_tokens.to_string(),
                prompt_tokens.to_string(),
            );
        };

        //Build the API body depending on the used model
        let model_body = self.model.get_body(
            &prompt,
            &json_value,
            self.function_call,
            &response_tokens,
            &self.temperature,
        );

        //Display debug info if requested
        if self.debug {
            info!("[debug] Model body: {:#?}", model_body);
            info!(
                "[debug] Prompt accounts for approx {} tokens, leaving {} tokens for answer.",
                prompt_tokens.to_string(),
                response_tokens.to_string(),
            );
        }

        let response_text = self.call_openai_api(&model_body).await?;

        //Extract data from the returned response text based on the used model
        let response_string = self.model.get_data(&response_text, self.function_call)?;

        if self.debug {
            info!("[debug] OpenAI response data: {}", response_string);
        }
        //Deserialize the string response into the expected output type
        let response_deser: anyhow::Result<T, anyhow::Error> =
            serde_json::from_str(&response_string).map_err(|error| {
                error!("[OpenAI] Response serialization error: {}", &error);
                anyhow!("Error: {}", error)
            });
        // Sometimes openai responds with a json object that has a data property. If that's the case, we need to extract the data property and deserialize that.
        if let Err(_e) = response_deser {
            let response_deser: OpenAIDataResponse<T> = serde_json::from_str(&response_text)
                .map_err(|error| {
                    error!("[OpenAI] Response serialization error: {}", &error);
                    anyhow!("Error: {}", error)
                })?;
            Ok(response_deser.data)
        } else {
            Ok(response_deser.unwrap())
        }
    }
}

// Get the tokenizer given a model
fn get_tokenizer(model: &OpenAIModels) -> anyhow::Result<CoreBPE> {
    let tokenizer = get_bpe_from_model(model.as_str());
    if let Err(_error) = tokenizer {
        // Fallback to the default chat model
        cl100k_base()
    } else {
        tokenizer
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIAssistantResp {
    id: String,
    object: String,
    created_at: u32,
    name: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    model: String,
    tools: Vec<OpenAITools>,
    file_ids: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAITools {
    #[serde(rename(deserialize = "type", serialize = "type"))]
    tool_type: OpenAIToolTypes,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
enum OpenAIToolTypes {
    #[serde(rename(deserialize = "code_interpreter", serialize = "code_interpreter"))]
    CodeInterpreter,
    #[serde(rename(deserialize = "retrieval", serialize = "retrieval"))]
    Retrieval,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIThreadResp {
    id: String,
    object: String,
    created_at: u32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIMessageListResp {
    object: String,
    data: Vec<OpenAIMessageResp>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIMessageResp {
    id: String,
    object: String,
    created_at: u32,
    thread_id: String,
    role: OpenAIAssistantRole,
    content: Vec<OpenAIContent>,
    //Other fields omitted as no use for now
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
enum OpenAIAssistantRole {
    #[serde(rename(deserialize = "user", serialize = "user"))]
    User,
    #[serde(rename(deserialize = "assistant", serialize = "assistant"))]
    Assistant,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIContent {
    #[serde(rename(deserialize = "type", serialize = "type"))]
    content_type: String,
    text: Option<OpenAIContentText>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIContentText {
    value: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIRunResp {
    id: String,
    object: String,
    created_at: u32,
    status: OpenAIRunStatus,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
enum OpenAIRunStatus {
    #[serde(rename(deserialize = "queued", serialize = "queued"))]
    Queued,
    #[serde(rename(deserialize = "in_progress", serialize = "in_progress"))]
    InProgress,
    #[serde(rename(deserialize = "requires_action", serialize = "requires_action"))]
    RequiresAction,
    #[serde(rename(deserialize = "cancelling", serialize = "cancelling"))]
    Cancelling,
    #[serde(rename(deserialize = "cancelled", serialize = "cancelled"))]
    Cancelled,
    #[serde(rename(deserialize = "failed", serialize = "failed"))]
    Failed,
    #[serde(rename(deserialize = "completed", serialize = "completed"))]
    Completed,
    #[serde(rename(deserialize = "expired", serialize = "expired"))]
    Expired,
}

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
    pub async fn new(model: OpenAIModels, open_ai_key: &str, debug: bool) -> Anysult<Self> {
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
    async fn create_assistant(&mut self) -> Anysult<()> {
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
    ) -> Anysult<T> {
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
        self.add_message(&message, &file_ids).await?;

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
                    content.text.map(|text| {
                        let sanitized_text = sanitize_json_response(&text.value);
                        serde_json::from_str::<T>(&sanitized_text).ok()
                    })
                    .flatten()
                })
            })
            .ok_or(anyhow!("No valid response form OpenAI Assistant found."))
    }

    /*
     * This function creates a Thread and updates the thread_id of the OpenAIAssistant struct
     */
    async fn add_message(&mut self, message: &str, file_ids: &[String]) -> Anysult<()> {
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
    async fn create_thread(&mut self, body: &serde_json::Value) -> Anysult<()> {
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
    async fn add_message_thread(&self, body: &serde_json::Value) -> Anysult<()> {
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
    async fn get_message_thread(&self) -> Anysult<Vec<OpenAIMessageResp>> {
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
    async fn start_run(&mut self) -> Anysult<()> {
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
    async fn get_run_status(&self) -> Anysult<OpenAIRunResp> {
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

//OpenAI has a tendency to wrap response Json in ```json{}```
//TODO: This function might need to become more sophisticated or handled with better prompt eng
fn sanitize_json_response(json_response: &str) -> String {
    let text_no_json = json_response.replace("json\n", "");
    text_no_json.replace("```", "")
}

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

impl OpenAIFile {
    //Constructor
    pub async fn new(file_bytes: Vec<u8>, open_ai_key: &str, debug: bool) -> Anysult<Self> {
        let mut new_file = OpenAIFile {
            id: "this-will-be-overwritten".to_string(),
            debug,
            api_key: open_ai_key.to_string(),
        };
        //Upload file and get the ID
        new_file.upload_file(file_bytes).await?;
        Ok(new_file)
    }

    /*
     * This function uploads a file to OpenAI and assigns it for use with Assistant API
     */
    async fn upload_file(&mut self, file_bytes: Vec<u8>) -> Anysult<()> {
        let files_url = "https://api.openai.com/v1/files";

        let form = reqwest::multipart::Form::new()
            .text("purpose", "assistants")
            .part(
                "file",
                reqwest::multipart::Part::bytes(file_bytes).file_name("file.pdf"),
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
}

#[cfg(test)]
mod tests {
    use crate::get_tokenizer;
    use crate::OpenAIModels;

    #[test]
    fn it_computes_gpt3_5_tokenization() {
        let bpe = get_tokenizer(&crate::OpenAIModels::Gpt4_32k).unwrap();
        let tokenized: Result<Vec<_>, _> = bpe
            .split_by_token_iter("This is a test         with a lot of spaces", true)
            .collect();
        let tokenized = tokenized.unwrap();
        assert_eq!(
            tokenized,
            vec!["This", " is", " a", " test", "        ", " with", " a", " lot", " of", " spaces"]
        );
    }

    // Tests for calculating max requests per model
    #[test]
    fn test_gpt3_5turbo_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(3500, 90000 / ((4096 as f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo0613_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo0613;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(3500, 90000 / ((4096 as f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo16k_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo16k;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(3500, 180000 / ((16384 as f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt4_max_requests() {
        let model = OpenAIModels::Gpt4;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(200, 10000 / ((8192 as f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }
}
