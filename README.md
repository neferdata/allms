## OpenAI-rs

Hand picked Rust bindings for OpenAI's API.

## Usage

```rust
// Upload file to OpenAI
let openai_file = OpenAIFile::new(bytes, &CLIENT_OPENAI_API_KEY, true)
    .await
    .map_err(|e| {
        log::error!("Request to OpenAI failed: {:?}", e);
        actix_web::error::ErrorInternalServerError("Internal server error")
    })?;

// Extract invoice detail using Assistant API
OpenAIAssistant::new(OpenAIModels::Gpt4Turbo, &CLIENT_OPENAI_API_KEY, true)
    .await
    .map_err(error::ErrorInternalServerError)?
    .get_answer::<Invoice>(
        "Extract the following information from the attached invoice: invoice number, vendor name, payment amount, payment date.",
        &[openai_file.id],
    )
    .await
    .map(Json)
    .map_err(error::ErrorInternalServerError)
```




