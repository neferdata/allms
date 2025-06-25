use anyhow::{anyhow, Result};
use log::{error, info, warn};
use reqwest::{header, Client};
use schemars::{schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    constants::{OPENAI_API_URL, OPENAI_BASE_INSTRUCTIONS, OPENAI_FUNCTION_INSTRUCTIONS},
    domain::{OpenAIDataResponse, OpenAPIChatResponse, OpenAPICompletionsResponse, RateLimit},
    utils::get_tokenizer_old,
};

/// This is a LEGACY implementation of OpenAI LLM Models that will not be maintained going forward (after May 2024).
/// For current implementation refer to `llm_models` module.
#[deprecated(
    since = "0.6.1",
    note = "This struct is deprecated. Please use the `llm_models::OpenAIModels` struct for latest functionality."
)]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum OpenAIModels {
    Gpt3_5Turbo,
    Gpt3_5Turbo0613,
    Gpt3_5Turbo16k,
    Gpt4,
    Gpt4_32k,
    TextDavinci003,
    Gpt4Turbo,
    Gpt4o,
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
            OpenAIModels::Gpt4o => "gpt-4o",
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
            OpenAIModels::Gpt4o => 128_000,
        }
    }

    pub(crate) fn get_endpoint(&self) -> String {
        //OpenAI documentation: https://platform.openai.com/docs/models/model-endpoint-compatibility
        match self {
            OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4o
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

    pub(crate) fn get_base_instructions(&self, function_call: Option<bool>) -> String {
        let function_call = function_call.unwrap_or_else(|| self.function_call_default());
        match function_call {
            true => OPENAI_FUNCTION_INSTRUCTIONS.to_string(),
            false => OPENAI_BASE_INSTRUCTIONS.to_string(),
        }
    }

    pub(crate) fn function_call_default(&self) -> bool {
        //OpenAI documentation: https://platform.openai.com/docs/guides/gpt/function-calling
        match self {
            OpenAIModels::TextDavinci003 | OpenAIModels::Gpt3_5Turbo | OpenAIModels::Gpt4_32k => {
                false
            }
            OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4o => true,
        }
    }

    //This method prepares the body of the API call for different models
    pub(crate) fn get_body(
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
            | OpenAIModels::Gpt4o
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
    /*
     * This function leverages OpenAI API to perform any query as per the provided body.
     *
     * It returns a String the Response object that needs to be parsed based on the self.model.
     */
    pub async fn call_api(
        &self,
        api_key: &str,
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
                "[debug] OpenAI API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    pub(crate) fn get_data(&self, response_text: &str, function_call: bool) -> Result<String> {
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
            | OpenAIModels::Gpt4o
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
    fn get_rate_limit(&self) -> RateLimit {
        //OpenAI documentation: https://platform.openai.com/account/rate-limits
        //This is the max tokens allowed between prompt & response
        match self {
            OpenAIModels::Gpt3_5Turbo => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt3_5Turbo0613 => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt3_5Turbo16k => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4 => RateLimit {
                tpm: 300_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4Turbo => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4_32k => RateLimit {
                tpm: 300_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4o => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::TextDavinci003 => RateLimit {
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

/// This is a LEGACY implementation of Chat Completions with OpenAI API.
/// This implementation will not be maintained going forward (after May 2024).
/// For current implementation refer to `llm_models` and `completions` modules.
///
/// [Chat Completions API](https://platform.openai.com/docs/guides/text-generation/chat-completions-api)
///
/// Chat models take a list of messages as input and return a model-generated message as output.
/// Although the chat format is designed to make multi-turn conversations easy,
/// it’s just as useful for single-turn tasks without any conversation.
#[deprecated(
    since = "0.6.1",
    note = "This struct is deprecated. Please use the `Completions` struct for latest functionality including GPT-4o."
)]
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
    pub fn set_context<T: Serialize>(mut self, input_name: &str, input_data: &T) -> Result<Self> {
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
    ) -> Result<usize> {
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
        let bpe = get_tokenizer_old(&self.model)?;
        let prompt_tokens = bpe.encode_with_special_tokens(&full_prompt).len();

        //Assuming another 5% overhead for json formatting
        Ok((prompt_tokens as f64 * 1.05) as usize)
    }

    /*
     * This method is used to submit a prompt to OpenAI and process the response.
     * When calling the function you need to specify the type parameter as the response will match the schema of that type.
     * The prompt in this function is written in a way to instruct OpenAI to behave like a computer function that calculates an output based on provided input and its language model.
     */
    pub async fn get_answer<T: JsonSchema + DeserializeOwned>(
        self,
        instructions: &str,
    ) -> Result<T> {
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
                response_tokens, self.max_tokens, prompt_tokens,
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
                prompt_tokens, response_tokens,
            );
        }

        let response_text = self
            .model
            .call_api(&self.api_key, &model_body, self.debug)
            .await?;

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

#[cfg(test)]
mod tests {
    use crate::utils::get_tokenizer_old;
    use crate::OpenAIModels;

    #[test]
    fn it_computes_gpt3_5_tokenization() {
        let bpe = get_tokenizer_old(&OpenAIModels::Gpt4_32k).unwrap();
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
        let expected_max = std::cmp::min(10_000, 2_000_000 / ((4096_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo0613_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo0613;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 2_000_000 / ((4096_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo16k_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo16k;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 2_000_000 / ((16384_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt4_max_requests() {
        let model = OpenAIModels::Gpt4;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 300_000 / ((8192_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }
}
