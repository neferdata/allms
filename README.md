# OpenAI Typesafe Rust Library
[![crates.io](https://img.shields.io/crates/v/openai-typesafe-rs.svg)](https://crates.io/crates/openai-typesafe-rs)
[![docs.rs](https://docs.rs/openai-typesafe-rs/badge.svg)](https://docs.rs/openai-typesafe-rs)

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

This is the output of calling the assistant api with sample-bill.pdf

```
RUST_LOG=info RUST_BACKTRACE=1 cargo run --example use_openai_assistant
```

This program will send this invoice pdf to OpenAI Assistant API and get the invoice data back.

<img width="800" src="/examples/bill-image.png">


Output:
```
Running `target/debug/examples/use_openai_assistant`

Invoice: Invoice { invoice_number: "12345678190", vendor_name: "Peoples Gas", payment_amount: 147.82, payment_date: "2021-03-04" }
```

## License
This project is licensed under dual MIT/Apache-2.0 license. See the [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) files for details.