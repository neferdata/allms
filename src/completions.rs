use anyhow::{anyhow, Result};
use log::{error, info, warn};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::HashSet;

use crate::domain::AllmsError;
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
    thinking_level: Option<ThinkingLevel>,
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
            thinking_level: None,
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
    /// This method can be used to set the thinking level for the model
    /// This is currently used for Gemini 3 models
    ///
    pub fn thinking_level(mut self, thinking_level: ThinkingLevel) -> Self {
        self.thinking_level = Some(thinking_level);
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
        let (response_string, _response_text) = self.complete_raw::<U>(instructions).await?;
        Self::deserialize_answer(&response_string, self.model.as_str())
    }

    ///
    /// Same prompt/schema path as [`get_answer`], but also returns requested side-channel metadata
    /// (e.g. native web-search citation URLs) extracted from the provider response.
    ///
    /// `kinds`:
    /// - `None` — extract all defined metadata kinds ([`CompletionMetadataKind::ALL`])
    /// - `Some(&[])` — extract nothing (empty metadata)
    /// - `Some(&[...])` — extract only the listed kinds
    ///
    pub async fn get_answer_with_metadata<U: JsonSchema + DeserializeOwned>(
        self,
        instructions: &str,
        kinds: Option<&[CompletionMetadataKind]>,
    ) -> Result<CompletionAnswer<U>> {
        let resolved = kinds.unwrap_or(CompletionMetadataKind::ALL);
        let (response_string, response_text) = self.complete_raw::<U>(instructions).await?;
        let metadata = self
            .model
            .extract_metadata(&response_text, self.version.clone(), resolved);
        let answer = Self::deserialize_answer(&response_string, self.model.as_str())?;
        Ok(CompletionAnswer { answer, metadata })
    }

    /// Shared request path: build body → call API → extract answer JSON string.
    /// Returns `(answer_json, raw_response_text)`.
    async fn complete_raw<U: JsonSchema + DeserializeOwned>(
        &self,
        instructions: &str,
    ) -> Result<(String, String)> {
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
            self.thinking_level.as_ref(),
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
            .get_version_data(&response_text, self.function_call, self.version.clone())
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

        Ok((response_string, response_text))
    }

    fn deserialize_answer<U: DeserializeOwned>(
        response_string: &str,
        model_name: &str,
    ) -> Result<U> {
        serde_json::from_str(response_string).map_err(|error| {
            let error = AllmsError {
                crate_name: "allms".to_string(),
                module: format!("assistants::completions::{model_name}"),
                error_message: format!("Completions API response serialization error: {}", error),
                error_detail: response_string.to_string(),
            };
            error!("{:?}", error);
            anyhow!("{:?}", error)
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    Low,
    #[default]
    High,
}

impl ThinkingLevel {
    pub fn as_str(&self) -> &str {
        match self {
            ThinkingLevel::Low => "low",
            ThinkingLevel::High => "high",
        }
    }

    pub fn try_from_str(s: &str) -> Option<Self> {
        match s {
            "low" => Some(ThinkingLevel::Low),
            "high" => Some(ThinkingLevel::High),
            _ => None,
        }
    }
}

/// Kinds of side-channel metadata that can be extracted from a provider response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompletionMetadataKind {
    Citations,
}

impl CompletionMetadataKind {
    /// All kinds currently defined (kept in sync when variants are added).
    pub const ALL: &'static [CompletionMetadataKind] = &[CompletionMetadataKind::Citations];
}

/// A single citation/source URL from a provider response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionCitation {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Deduped citation/source URLs from the provider response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionCitations {
    /// Deduped citation/source URLs (order preserved).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub urls: Vec<String>,
    /// Citation items with optional titles (first-seen title per URL).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<CompletionCitation>,
}

/// Typed bag of optional side-channel data from a provider response.
/// Fields are `None` when not requested (or not extracted); skipped when serializing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<CompletionCitations>,
}

/// Structured answer plus requested metadata from a Completions call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionAnswer<T> {
    pub answer: T,
    pub metadata: CompletionMetadata,
}

/// Build [`CompletionCitations`] from candidate `(url, title)` pairs.
/// Skips empty / non-http(s) URLs; deduplicates by full URL (case-sensitive), preserving order.
pub(crate) fn build_citations(
    candidates: impl IntoIterator<Item = (String, Option<String>)>,
) -> CompletionCitations {
    let mut seen = HashSet::new();
    let mut urls = Vec::new();
    let mut items = Vec::new();

    for (url, title) in candidates {
        let url = url.trim().to_string();
        if url.is_empty() || !is_http_url(&url) || !seen.insert(url.clone()) {
            continue;
        }
        urls.push(url.clone());
        items.push(CompletionCitation { url, title });
    }

    CompletionCitations { urls, items }
}

fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_citations_dedupes_and_skips_invalid() {
        let citations = build_citations([
            ("https://a.example".to_string(), Some("A".to_string())),
            ("https://a.example".to_string(), Some("A2".to_string())),
            ("".to_string(), None),
            ("ftp://nope".to_string(), None),
            ("  https://b.example  ".to_string(), None),
        ]);
        assert_eq!(
            citations.urls,
            vec![
                "https://a.example".to_string(),
                "https://b.example".to_string()
            ]
        );
        assert_eq!(citations.items[0].title.as_deref(), Some("A"));
        assert_eq!(citations.items[1].title, None);
    }
}
