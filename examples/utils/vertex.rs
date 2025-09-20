use anyhow::Result;
use yup_oauth2::{read_service_account_key, ServiceAccountAuthenticator};

// Reusable function for Vertex API authentication
pub async fn get_vertex_token() -> Result<String> {
    let service_account_key = read_service_account_key("secrets/gcp_sa_key.json")
        .await
        .unwrap();

    let auth = ServiceAccountAuthenticator::builder(service_account_key)
        .build()
        .await
        .unwrap();

    let google_token = auth
        .token(&["https://www.googleapis.com/auth/cloud-platform"])
        .await
        .unwrap();

    Ok(google_token.token().unwrap().to_string())
}
