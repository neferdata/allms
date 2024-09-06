# allms: One Library to rule them aLLMs
[![crates.io](https://img.shields.io/crates/v/allms.svg)](https://crates.io/crates/allms)
[![docs.rs](https://docs.rs/allms/badge.svg)](https://docs.rs/allms)

This Rust library is specialized in providing type-safe interactions with APIs of the following LLM providers: OpenAI, Anthropic, Mistral, Google Gemini. (More providers to be added in the future.) It's designed to simplify the process of experimenting with different models. It de-risks the process of migrating between providers reducing vendor lock-in issues. It also standardizes serialization of sending requests to LLM APIs and interpreting the responses, ensuring that the JSON data is handled in a type-safe manner. With allms you can focus on creating effective prompts and providing LLM with the right context, instead of worrying about differences in API implementations.

## Features

- Support for various LLM models including OpenAI (GPT-3.5, GPT-4), Anthropic (Claude, Claude Instant), Mistral, or Google GeminiPro.
- Easy-to-use functions for chat/text completions and assistants. Use the same struct and methods regardless of which model you choose.
- Automated response deserialization to custom types.
- Standardized approach to providing context with support of function calling, tools, and file uploads.
- Enhanced developer productivity with automated token calculations, rate limits and debug mode.
- Extensibility enabling easy adoption of other models with standardized trait.
- Asynchronous support using Tokio.

### Foundational Models
OpenAI:
- APIs: Chat Completions, Function Calling, Assistants (v1 & v2), Files, Vector Stores, Tools (file_search)
- Models: GPT-4o, GPT-4, GPT-4 32k, GPT-4 Turbo, GPT-3.5 Turbo, GPT-3.5 Turbo 16k

Azure OpenAI
- APIs: Assistants, Files, Vector Stores, Tools
- Models: as per model deployments in Azure OpenAI Studio

Anthropic:
- APIs: Messages, Text Completions
- Models: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku, Claude 2.0, Claude Instant 1.2

Mistral:
- APIs: Chat Completions
- Models: Mistral Large, Mistral Nemo, Mistral 7B, Mixtral 8x7B, Mixtral 8x22B, Mistral Medium, Mistral Small, Mistral Tiny

Google Vertex AI / AI Studio:
- APIs: Chat Completions (including streaming)
- Models: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro

### Prerequisites
- OpenAI: API key (passed in model constructor)
- Azure OpenAI: environment variable `OPENAI_API_URL` set to your Azure OpenAI resource endpoint. Model deployment names in Azure OpenAI Stuido need to match `OpenAIModels::as_str()`
- Anthropic: API key (passed in model constructor)
- Mistral: API key (passed in model constructor)
- Google AI Studio: API key (passed in model constructor)
- Google Vertex AI: GCP service account key (used to obtain access token) + GCP project ID (set as environment variable)

### Examples
Explore the `examples` directory to see more use cases and how to use different LLM providers and endpoint types.

Using `Completions` API with different foundational models:
```
let openai_answer = Completions::new(OpenAIModels::Gpt4o, &API_KEY, None, None)
    .get_answer::<T>(instructions)
    .await?

let anthropic_answer = Completions::new(AnthropicModels::Claude2, &API_KEY, None, None)
    .get_answer::<T>(instructions)
    .await?

let mistral_answer = Completions::new(MistralModels::MistralSmall, &API_KEY, None, None)
    .get_answer::<T>(instructions)
    .await?

let google_answer = Completions::new(GoogleModels::GeminiPro, &API_KEY, None, None)
    .get_answer::<T>(instructions)
    .await?
```

Example:
```
RUST_LOG=info RUST_BACKTRACE=1 cargo run --example use_completions
```

Using `Assistant` API to analyze your files with `File` and `VectorStore` capabilities:
```
// Create a File
let openai_file = OpenAIFile::new(None, &API_KEY)
    .upload(&file_name, bytes)
    .await?;

// Create a Vector Store
let openai_vector_store = OpenAIVectorStore::new(None, "Name", &API_KEY)
    .upload(&[openai_file.id.clone().unwrap_or_default()])
    .await?;

// Extract data using Assistant 
let openai_answer = OpenAIAssistant::new(OpenAIModels::Gpt4o, &API_KEY)
    .version(OpenAIAssistantVersion::V2)
    .vector_store(openai_vector_store.clone())
    .await?
    .get_answer::<T>(instructions, &[])
    .await?;
```

Example:
```
RUST_LOG=info RUST_BACKTRACE=1 cargo run --example use_openai_assistant
```

## License
This project is licensed under dual MIT/Apache-2.0 license. See the [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) files for details.
