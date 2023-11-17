use std::path::Path;

use openai_rs::OpenAIAssistant;
use openai_rs::OpenAIFile;
use openai_rs::OpenAIModels;


use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct Invoice {
    invoice_number: String,
    vendor_name: String,
    payment_amount: f32,
    payment_date: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    // Read invoice file
    let path = Path::new("examples/sample-bill.pdf");
    let bytes = std::fs::read(path)?;

    let openai_file = OpenAIFile::new(bytes, &api_key, true)
        .await?;

    // Extract invoice detail using Assistant API
    let invoice = 
        OpenAIAssistant::new(OpenAIModels::Gpt4Turbo, &api_key, true)
        .await?
        .get_answer::<Invoice>(
            "Extract the following information from the attached invoice: invoice number, vendor name, payment amount, payment date.",
            &[openai_file.id],
        )
        .await?;

    println!("Invoice: {:?}", invoice);
    Ok(())
}
