use serde::{Deserialize, Serialize};

use crate::constants::{ANTHROPIC_FILES_VERSION, ANTHROPIC_MESSAGES_VERSION};

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum AnthropicApiEndpoints {
    Messages { version: String },
    Files { version: String },
}

impl AnthropicApiEndpoints {
    pub fn messages_default() -> Self {
        AnthropicApiEndpoints::Messages {
            version: ANTHROPIC_MESSAGES_VERSION.to_string(),
        }
    }

    pub fn messages(version: String) -> Self {
        AnthropicApiEndpoints::Messages { version }
    }

    pub fn files_default() -> Self {
        AnthropicApiEndpoints::Files {
            version: ANTHROPIC_FILES_VERSION.to_string(),
        }
    }

    pub fn files(version: String) -> Self {
        AnthropicApiEndpoints::Files { version }
    }

    pub fn version(&self) -> String {
        match self {
            AnthropicApiEndpoints::Messages { version } => version.to_string(),
            AnthropicApiEndpoints::Files { version } => version.to_string(),
        }
    }

    pub fn version_static(&self) -> &'static str {
        match self {
            AnthropicApiEndpoints::Messages { .. } => ANTHROPIC_MESSAGES_VERSION.as_str(),
            AnthropicApiEndpoints::Files { .. } => ANTHROPIC_FILES_VERSION.as_str(),
        }
    }
}
