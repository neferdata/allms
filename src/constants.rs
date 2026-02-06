use lazy_static::lazy_static;

lazy_static! {
    pub(crate) static ref OPENAI_API_URL: String =
        std::env::var("OPENAI_API_URL").unwrap_or("https://api.openai.com".to_string());
}

lazy_static! {
    pub(crate) static ref ANTHROPIC_API_URL: String = std::env::var("ANTHROPIC_API_URL")
        .unwrap_or("https://api.anthropic.com/v1/complete".to_string());
    pub(crate) static ref ANTHROPIC_MESSAGES_API_URL: String =
        std::env::var("ANTHROPIC_MESSAGES_API_URL")
            .unwrap_or("https://api.anthropic.com/v1/messages".to_string());
    pub(crate) static ref ANTHROPIC_MESSAGES_VERSION: String =
        std::env::var("ANTHROPIC_MESSAGES_VERSION").unwrap_or("2023-06-01".to_string());
    pub(crate) static ref ANTHROPIC_FILES_VERSION: String =
        std::env::var("ANTHROPIC_FILES_VERSION").unwrap_or("files-api-2025-04-14".to_string());
    pub(crate) static ref ANTHROPIC_FILES_API_URL: String =
        std::env::var("ANTHROPIC_FILES_API_URL")
            .unwrap_or("https://api.anthropic.com/v1/files".to_string());
}

lazy_static! {
    pub(crate) static ref MISTRAL_API_URL: String = std::env::var("MISTRAL_API_URL")
        .unwrap_or("https://api.mistral.ai/v1/chat/completions".to_string());
    pub(crate) static ref MISTRAL_CONVERSATIONS_API_URL: String =
        std::env::var("MISTRAL_CONVERSATIONS_API_URL")
            .unwrap_or("https://api.mistral.ai/v1/conversations".to_string());
}

lazy_static! {
    pub(crate) static ref GOOGLE_VERTEX_API_URL: String = {
        let region = std::env::var("GOOGLE_REGION").unwrap_or("us-central1".to_string());
        let project_id = std::env::var("GOOGLE_PROJECT_ID").expect("PROJECT_ID not set");

        format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models",
                region, project_id, region)
    };
    pub(crate) static ref GOOGLE_VERTEX_ENDPOINT_API_URL: String = {
        let region = std::env::var("GOOGLE_REGION").unwrap_or("us-central1".to_string());
        let project_id = std::env::var("GOOGLE_PROJECT_ID").expect("PROJECT_ID not set");

        format!("https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints")
    };
}

lazy_static! {
    pub(crate) static ref GOOGLE_GEMINI_API_URL: String = std::env::var("GOOGLE_GEMINI_API_URL")
        .unwrap_or("https://generativelanguage.googleapis.com/v1/models".to_string());
    pub(crate) static ref GOOGLE_GEMINI_BETA_API_URL: String =
        std::env::var("GOOGLE_GEMINI_BETA_API_URL")
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta/models".to_string());
}

lazy_static! {
    pub(crate) static ref PERPLEXITY_API_URL: String = std::env::var("PERPLEXITY_API_URL")
        .unwrap_or("https://api.perplexity.ai/chat/completions".to_string());
}

lazy_static! {
    /// Docs: https://docs.aws.amazon.com/general/latest/gr/bedrock.html
    pub(crate) static ref AWS_REGION: String = std::env::var("AWS_REGION").unwrap_or("us-east-1".to_string());
    pub(crate) static ref AWS_BEDROCK_API_URL: String = {
        format!("https://bedrock.{}.amazonaws.com", &*AWS_REGION)
    };
}

lazy_static! {
    pub(crate) static ref AWS_ACCESS_KEY_ID: String =
        std::env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID not set");
    pub(crate) static ref AWS_SECRET_ACCESS_KEY: String =
        std::env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY not set");
}

lazy_static! {
    pub(crate) static ref DEEPSEEK_API_URL: String = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or("https://api.deepseek.com/chat/completions".to_string());
}

lazy_static! {
    pub(crate) static ref XAI_API_URL: String =
        std::env::var("XAI_API_URL").unwrap_or("https://api.x.ai/v1/responses".to_string());
}

//Generic OpenAI instructions
pub(crate) const OPENAI_BASE_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
Step 1: Review and understand the 'instructions'.
Step 2: Prepare a response by processing the provided data as per the 'instructions'. 
Step 3: Convert the response to a Json object. The Json object must match the schema provided as the `output json schema`.
Step 4: Validate that the Json object matches the 'output json schema' and correct if needed. If you are not able to generate a valid Json, respond with "Error calculating the answer."
Step 5: Respond ONLY with properly formatted Json object. No other words or text, only valid Json in the answer.
"#;

pub(crate) const OPENAI_FUNCTION_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
Step 1: Review and understand the 'instructions'.
Step 2: Prepare a response by processing the provided data as per the 'instructions'. 
Step 3: Convert the response to a Json object. The Json object must match the schema provided in the function definition.
Step 4: Validate that the Json object matches the function properties and correct if needed. If you are not able to generate a valid Json, respond with "Error calculating the answer."
Step 5: Respond ONLY with properly formatted Json object. No other words or text, only valid Json in the answer.
"#;

pub(crate) const OPENAI_ASSISTANT_INSTRUCTIONS: &str = r#"You are a computer function. You are expected to perform the following tasks:
1: Review and understand the content of user messages passed to you in the thread.
2: Review and consider any files the user provided attached to the messages.
3: Prepare response using your language model based on the user messages and provided files.
4: Respond ONLY with properly formatted data portion of a Json. No other words or text, only valid Json in your answers. 
"#;

pub(crate) const OPENAI_ASSISTANT_POLL_FREQ: usize = 10;

pub(crate) const DEFAULT_AZURE_VERSION: &str = "2024-06-01";
