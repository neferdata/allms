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

### Prerequisites
- OpenAI: API key (passed in model constructor)
- Anthropic: API key (passed in model constructor)
- Mistral: API key (passed in model constructor)
- Google AI Studio: API key (passed in model constructor)
- Google Vertex AI: GCP service account key (used to obtain access token) + GCP project ID (set as environment variable)

### Examples
Explore the `examples` directory to see more use cases and how to use different LLM providers and endpoint types.

This is the output of calling the assistant api with metallica.pdf

```
RUST_LOG=info RUST_BACKTRACE=1 cargo run --example use_openai_assistant
```

This program will send this press release to OpenAI Assistant API and get the data requested in the response type back:

```
pub struct ConcertInfo {
    dates: Vec<String>,
    band: String,
    venue: String,
    city: String,
    country: String,
    ticket_price: String,
}
```

<img width="600" src="/examples/metallica.png">

Output:
```
Running `target/debug/examples/use_openai_assistant`

ConcertInfo { dates: ["Friday September 6, 2019"], band: "Metallica and the San Francisco Symphony", venue: "Chase Center", city: "San Francisco", country: "USA", ticket_price: "Information not available" }
```

## License
This project is licensed under dual MIT/Apache-2.0 license. See the [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) files for details.
