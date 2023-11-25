use std::path::Path;

use openai_safe::OpenAIAssistant;
use openai_safe::OpenAIFile;
use openai_safe::OpenAIModels;

use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct ConcertInfo {
    dates: Vec<String>,
    band: String,
    venue: String,
    city: String,
    country: String,
    ticket_price: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    // Read invoice file
    let path = Path::new("examples/metallica.pdf");
    let bytes = std::fs::read(path)?;

    let openai_file = OpenAIFile::new(bytes, &api_key, true).await?;

    let exciting_bands = vec![
        "Metallica Inc.",
        "Guns N' Roses",
        "Slayer",
        "Iron Maiden",
        "Black Sabbath",
    ];

    // Extract concert information using Assistant API
    let concert_info = OpenAIAssistant::new(OpenAIModels::Gpt4Turbo, &api_key, true)
        .await?
        .set_context(
            "exciting_bands",
            &exciting_bands
        )
        .await?
        .get_answer::<ConcertInfo>(
            "Extract the information requested in the response type from the attached concert information.
            Before returning a response try to match the band name you discovered to the names provided in the 'exciting_bands' list.
            Change the name of the band to the one in 'exciting_bands' if the names are similar.",
            &[openai_file.id],
        )
        .await?;

    println!("Concert Info: {:?}", concert_info);
    Ok(())
}
