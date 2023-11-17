use anyhow::{anyhow, Result};
use log::{error, info, warn};
use reqwest::{header, Client};
use schemars::{schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

use crate::{domain::OpenAIDataResponse, models::OpenAIModels, utils::get_tokenizer};

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
    async fn call_openai_api(&self, body: &serde_json::Value) -> Result<String> {
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
