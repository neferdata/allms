use std::ffi::OsStr;
use std::path::Path;

use allms::assistants::{OpenAIAssistant, OpenAIAssistantVersion, OpenAIFile, OpenAIVectorStore};
use allms::files::LLMFiles;
use allms::llm::OpenAIModels;

use anyhow::{anyhow, Result};

const CONCERT_INFO_SCHEMA: &str = r#"
{
    "type": "object",
    "properties": {
        "dates": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "band": {
            "type": "string"
        },
        "genre": {
            "type": "string"
        },
        "venue": {
            "type": "string"
        },
        "city": {
            "type": "string"
        },
        "country": {
            "type": "string"
        },
        "ticket_price": {
            "type": "string"
        }
    }
}
"#;

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

    let openai_file = OpenAIFile::new(None, &api_key)
        .debug()
        .upload(&file_name, bytes)
        .await?;

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
        .upload(&[openai_file.id.clone().unwrap_or_default()])
        .await?;

    let status = openai_vector_store.status().await?;
    println!(
        "Vector Store: {:?}; Status: {:?}",
        &openai_vector_store.id, &status
    );

    let file_count = openai_vector_store.file_count().await?;
    println!(
        "Vector Store: {:?}; File count: {:?}",
        &openai_vector_store.id, &file_count
    );

    // Extract concert information using Assistant API
    let concert_info = OpenAIAssistant::new(OpenAIModels::Gpt4_1Mini, &api_key)
        .debug()
        // Constructor defaults to V1
        .version(OpenAIAssistantVersion::V2)
        .vector_store(openai_vector_store.clone())
        .await?
        .set_context(
            "bands_genres",
            &bands_genres
        )
        .await?
        .get_json_answer(
            "Extract the information requested in the response type from the attached concert information.
            The response should include the genre of the music the 'band' represents.
            The mapping of bands to genres was provided in 'bands_genres' list in a previous message.",
            CONCERT_INFO_SCHEMA,
            &[],
        )
        .await?;

    println!("Concert Info: {:#?}", concert_info);

    //Remove the file from OpenAI
    openai_file.delete().await?;

    // Delete the Vector Store
    openai_vector_store.delete().await?;

    Ok(())
}
