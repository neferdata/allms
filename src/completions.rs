use anyhow::{anyhow, Result};
use log::{error, info, warn};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};

use crate::domain::{AllmsError, OpenAIDataResponse};
use crate::llm_models::{LLMModel, LLMTools};
use crate::utils::{get_tokenizer, get_type_schema};

/// Completions APIs take a list of messages as input and return a model-generated message as output.
/// Although the Completions format is designed to make multi-turn conversations easy,
/// it’s just as useful for single-turn tasks without any conversation.
pub struct Completions<T: LLMModel> {
    model: T,
    //For prompt & response
    max_tokens: usize,
    temperature: f32,
    input_json: Option<String>,
    debug: bool,
    function_call: bool,
    api_key: String,
    version: Option<String>,
    tools: Option<Vec<LLMTools>>,
}

impl<T: LLMModel> Completions<T> {
    /// Constructor for the Completions API
    pub fn new(
        model: T,
        api_key: &str,
        max_tokens: Option<usize>,
        temperature: Option<u32>,
    ) -> Self {
        let temperature = temperature
            .map(|temp| model.get_normalized_temperature(temp))
            .unwrap_or(model.get_default_temperature());
        Completions {
            //If no max tokens limit is provided we default to max allowed for the model
            max_tokens: max_tokens.unwrap_or_else(|| model.default_max_tokens()),
            function_call: model.function_call_default(),
            model,
            temperature,
            input_json: None,
            debug: false,
            api_key: api_key.to_string(),
            version: None,
            tools: None,
        }
    }

    ///
    /// This function turns on debug mode which will info! the prompt to log when executing it.
    ///
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    ///
    /// This function turns on/off function calling mode when interacting with OpenAI API.
    ///
    pub fn function_calling(mut self, function_call: bool) -> Self {
        self.function_call = function_call;
        self
    }

    ///
    /// This method can be used to define the model temperature used by the Assistant
    /// This method accepts % target of the acceptable range for the model
    ///
    pub fn temperature(mut self, temp_target: u32) -> Self {
        self.temperature = self.model.get_normalized_temperature(temp_target);
        self
    }

    ///
    /// This method can be used to define the model temperature used by the Assistant
    /// Using this method the temperature can be set directly without any validation of the range accepted by the model
    /// For a range-safe implementation please consider using `OpenAIAssistant::temperature` method
    ///
    pub fn temperature_unchecked(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    ///
    /// This method can be used to set the version of Completions API to be used
    /// This is currently used for OpenAI models which can be run on OpenAI API or Azure API
    ///
    pub fn version(mut self, version: &str) -> Self {
        // TODO: We should use the model trait to check which versions are allowed
        self.version = Some(version.to_string());
        self
    }

    ///
    /// This method can be used to inform the model to use a tool.
    /// Different models support different tool implementations.
    ///
    pub fn add_tool(mut self, tool: LLMTools) -> Self {
        self.tools = Some(match self.tools {
            Some(mut tools) => {
                tools.push(tool);
                tools
            }
            None => vec![tool],
        });
        self
    }

    ///
    /// This method can be used to provide values that will be used as context for the prompt.
    /// Using this function you can provide multiple input values by calling it multiple times. New values will be appended with the category name
    /// It accepts any instance that implements the Serialize trait.
    ///
    pub fn set_context<U: Serialize>(mut self, input_name: &str, input_data: &U) -> Result<Self> {
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
            "{}{}<{}>{}</{}>",
            self.input_json.unwrap_or_default(),
            line_break,
            input_name,
            input_json,
            input_name,
        );
        self.input_json = Some(new_json);
        Ok(self)
    }

    ///
    /// This method is used to check how many tokens would most likely remain for the response
    /// This is accomplished by estimating number of tokens needed for system/base instructions, user prompt, and function components including schema definition.
    ///
    pub fn check_prompt_tokens<U: JsonSchema + DeserializeOwned>(
        &self,
        instructions: &str,
    ) -> Result<usize> {
        //Output schema is extracted from the type parameter
        let schema = get_type_schema::<U>()?;

        let context_text = self
            .input_json
            .as_ref()
            .map(|context| format!("\n\n{}", &context))
            .unwrap_or_default();

        let prompt = format!(
            "Instructions:
            {instructions}{context_text}
            
            Respond ONLY with the data portion of a valid Json object. No schema definition required. No other words.", 
        );

        let full_prompt = format!(
            "{}{}{}",
            //Base (system) instructions
            self.model.get_base_instructions(Some(self.function_call)),
            //Instructions & context data
            prompt,
            //Output schema
            schema
        );

        //Check how many tokens are required for prompt
        let bpe = get_tokenizer(&self.model)?;
        let prompt_tokens = bpe.encode_with_special_tokens(&full_prompt).len();

        //Assuming another 5% overhead for json formatting
        Ok((prompt_tokens as f64 * 1.05) as usize)
    }

    ///
    /// This method is used to submit a prompt to OpenAI and process the response.
    /// When calling the function you need to specify the type parameter as the response will match the schema of that type.
    /// The prompt in this function is written in a way to instruct OpenAI to behave like a computer function that calculates an output based on provided input and its language model.
    ///
    pub async fn get_answer<U: JsonSchema + DeserializeOwned>(
        self,
        instructions: &str,
    ) -> Result<U> {
        //Output schema is extracted from the type parameter
        let schema = get_type_schema::<U>()?;
        let json_schema = serde_json::from_str(&schema)?;

        let context_text = self
            .input_json
            .as_ref()
            .map(|context| format!("\n\n{}", &context))
            .unwrap_or_default();

        let prompt = format!("{instructions}{context_text}");

        //Validate how many tokens remain for the response (and how many are used for prompt)
        let prompt_tokens = self
            .check_prompt_tokens::<U>(instructions)
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
        let model_body = self.model.get_version_body(
            &prompt,
            &json_schema,
            self.function_call,
            &response_tokens,
            &self.temperature,
            self.version.clone(),
            self.tools.as_deref(),
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
            .call_api(
                &self.api_key,
                self.version.clone(),
                &model_body,
                self.debug,
                self.tools.as_deref(),
            )
            .await?;

        //Extract data from the returned response text based on the used model
        let response_string = self
            .model
            .get_version_data(&response_text, self.function_call, self.version)
            .map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: format!("assistants::completions::{}", self.model.as_str()),
                    error_message: format!(
                        "Completions API response serialization error: {}",
                        error
                    ),
                    error_detail: response_text.to_string(),
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        if self.debug {
            info!("[debug] Completions response data: {}", response_string);
        }
        //Deserialize the string response into the expected output type
        let response_deser: anyhow::Result<U, anyhow::Error> =
            serde_json::from_str(&response_string).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: format!("assistants::completions::{}", self.model.as_str()),
                    error_message: format!(
                        "Completions API response serialization error: {}",
                        error
                    ),
                    error_detail: response_string,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            });
        // Sometimes openai responds with a json object that has a data property. If that's the case, we need to extract the data property and deserialize that.
        // TODO: This is OpenAI specific and should be implemented within the model.
        if let Err(_e) = response_deser {
            let response_deser: OpenAIDataResponse<U> = serde_json::from_str(&response_text)
                .map_err(|error| {
                    let error = AllmsError {
                        crate_name: "allms".to_string(),
                        module: format!("assistants::completions::{}", self.model.as_str()),
                        error_message: format!(
                            "Completions API response serialization error: {}",
                            error
                        ),
                        error_detail: response_text,
                    };
                    error!("{:?}", error);
                    anyhow!("{:?}", error)
                })?;
            Ok(response_deser.data)
        } else {
            response_deser
        }
    }
}
