use std::ffi::OsStr;
use std::path::Path;

use allms::assistants::{OpenAIAssistant, OpenAIAssistantVersion, OpenAIFile, OpenAIVectorStore};
use allms::llm_models::OpenAIModels;

use anyhow::{anyhow, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct ConcertInfo {
    dates: Vec<String>,
    band: String,
    genre: String,
    venue: String,
    city: String,
    country: String,
    ticket_price: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    // Read concert file
    let path = Path::new("metallica.pdf");
    let bytes = std::fs::read(path)?;
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("Failed to extract file name"))?;

    // Upload concert file to OpenAI
    let openai_file = OpenAIFile::new(&file_name, bytes, &api_key, false).await?;

    let bands_genres = vec![
        ("Metallica", "Metal"),
        ("The Beatles", "Rock"),
        ("Daft Punk", "Electronic"),
        ("Miles Davis", "Jazz"),
        ("Johnny Cash", "Country"),
    ];

    // Create a Vector Store and assign the file to it
    let openai_vector_store = OpenAIVectorStore::new(None, "Concerts", &api_key)
        .debug()
        .upload(&[openai_file.id.clone()]).await?;

    // Extract concert information using Assistant API
    let concert_info = OpenAIAssistant::new(OpenAIModels::Gpt4o, &api_key, false)
        .await?
        // Constructor defaults to V1
        .version(OpenAIAssistantVersion::V2)
        .set_context(
            "bands_genres",
            &bands_genres
        )
        .await?
        .get_answer::<ConcertInfo>(
            "Extract the information requested in the response type from the attached concert information.
            The response should include the genre of the music the 'band' represents.
            The mapping of bands to genres was provided in 'bands_genres' list in a previous message.",
            &[openai_file.id.clone()],
        )
        .await?;

    println!("Concert Info: {:#?}", concert_info);

    //Remove the file from OpenAI
    openai_file.delete_file().await?;

    Ok(())
}
