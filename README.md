# AIdapter: One Library to rule them aLLMs
[![crates.io](https://img.shields.io/crates/v/aidapter.svg)](https://crates.io/crates/aidapter)
[![docs.rs](https://docs.rs/aidapter/badge.svg)](https://docs.rs/aidapter)

This Rust library is specialized in providing type-safe interactions with the OpenAI API. It's designed to simplify the process of sending requests to OpenAI and interpreting the responses, ensuring that the JSON data is handled in a type-safe manner. This guarantees that the data conforms to predefined structures, reducing runtime errors and increasing the reliability of applications using OpenAI's powerful AI models like GPT-3.5 and GPT-4.

## Features

- Support for various OpenAI models including GPT-3.5, GPT-4, etc.
- Easy-to-use functions for completions, chat responses, and other OpenAI features.
- Structured response handling.
- Rate limit handling.
- Asynchronous support using Tokio.

### Prerequisites
- An OpenAI API key.

### Examples
Explore the `examples` directory to see more use cases and how to handle different types of responses from the
OpenAI API.

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
