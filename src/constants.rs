use lazy_static::lazy_static;

lazy_static! {
    pub(crate) static ref OPENAI_API_URL: String =
        std::env::var("OPENAI_API_URL").unwrap_or("https://api.openai.com".to_string());
}

lazy_static! {
    pub(crate) static ref ANTHROPIC_API_URL: String = std::env::var("ANTHROPIC_API_URL")
        .unwrap_or("https://api.anthropic.com/v1/complete".to_string());
}

lazy_static! {
    pub(crate) static ref MISTRAL_API_URL: String = std::env::var("MISTRAL_API_URL")
        .unwrap_or("https://api.mistral.ai/v1/chat/completions".to_string());
}

//Generic OpenAI instructions
pub(crate) const OPENAI_BASE_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
Step 1: Review and understand the 'instructions' from the *Instructions* section.
Step 2: Based on the 'instructions' process the data provided in the *Input data* section using your language model.
Step 3: Prepare a response by processing the 'input data' as per the 'instructions'. 
Step 4: Convert the response to a Json object. The Json object must match the schema provided in the *Output Json schema* section.
Step 5: Validate that the Json object matches the 'output Json schema' and correct if needed. If you are not able to generate a valid Json based on the 'input data' and 'instructions' please respond with "Error calculating the answer."
Step 6: Respond ONLY with properly formatted Json object. No other words or text, only valid Json in the answer.
"#;

pub(crate) const OPENAI_FUNCTION_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
Step 1: Review and understand the 'instructions' from the *Instructions* section.
Step 2: Based on the 'instructions' process the data provided in the *Input data* section using your language model.
Step 3: Prepare a response by processing the 'input data' as per the 'instructions'. 
Step 4: Convert the response to a Json object. The Json object must match the schema provided in the function definition.
Step 5: Validate that the Json object matches the function properties and correct if needed. If you are not able to generate a valid Json based on the 'input data' and 'instructions' please respond with "Error calculating the answer."
Step 6: Respond ONLY with properly formatted Json object. No other words or text, only valid Json in the answer.
"#;

pub(crate) const OPENAI_ASSISTANT_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
1: Review and understand the content of user messages passed to you in the thread.
2: Review and consider any files the user provided attached to the messages.
3: Prepare response using your language model based on the user messages and provided files.
4: Respond ONLY with properly formatted data portion of a Json. No other words or text, only valid Json in your answers. 
"#;
