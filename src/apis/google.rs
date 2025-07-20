use serde::{Deserialize, Serialize};

// Enum of supported Completions APIs
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum GoogleApiEndpoints {
    GoogleStudio,
    GoogleVertex,
}

impl GoogleApiEndpoints {
    /// Defaulting to OpenAICompletions
    pub fn default() -> Self {
        GoogleApiEndpoints::GoogleStudio
    }

    /// Parses a string into `GoogleApiEndpoints`.
    ///
    /// Supported formats (case-insensitive):
    /// - `"google-studio"` -> `GoogleApiEndpoints::GoogleStudio`
    /// - `"google-vertex"` -> `GoogleApiEndpoints::GoogleVertex`
    ///
    /// Returns default for others.
    pub fn from_str(s: &str) -> Self {
        let s_lower = s.to_lowercase();
        match s_lower.as_str() {
            "google-studio" => GoogleApiEndpoints::GoogleStudio,
            "google-vertex" => GoogleApiEndpoints::GoogleVertex,
            _ => GoogleApiEndpoints::default(),
        }
    }
}
