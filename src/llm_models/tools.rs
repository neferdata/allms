use serde::{Deserialize, Serialize};
use serde_json::{to_value, Value};

///
/// Enum of all the tools that can be used with different LLM providers
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum LLMTools {
    OpenAIFileSearch(OpenAIFileSearchConfig),
    OpenAIWebSearch(OpenAIWebSearchConfig),
    OpenAIComputerUse(OpenAIComputerUseConfig),
    OpenAIReasoning(OpenAIReasoningConfig),
    OpenAICodeInterpreter(OpenAICodeInterpreterConfig),
}

impl LLMTools {
    pub fn get_config_json(&self) -> Option<Value> {
        match self {
            LLMTools::OpenAIFileSearch(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAIWebSearch(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAIComputerUse(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAIReasoning(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAICodeInterpreter(cfg) => to_value(cfg).ok(),
        }
    }
}

///
/// OpenAI File Search tool config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub struct OpenAIFileSearchConfig {
    #[serde(rename = "type")]
    pub tool_type: OpenAIFileSearchToolType,
    pub vector_store_ids: Vec<String>,
    pub max_num_results: Option<usize>,
}

impl OpenAIFileSearchConfig {
    pub fn new(vector_store_ids: Vec<String>) -> Self {
        Self {
            tool_type: OpenAIFileSearchToolType::FileSearch,
            vector_store_ids,
            max_num_results: None,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIFileSearchToolType {
    #[serde(rename = "file_search")]
    FileSearch,
}

///
/// OpenAI Web Search tool config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub struct OpenAIWebSearchConfig {
    #[serde(rename = "type")]
    pub tool_type: OpenAIWebSearchToolType,
    pub search_context_size: Option<OpenAIWebSearchContextSize>,
}

impl Default for OpenAIWebSearchConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAIWebSearchConfig {
    pub fn new() -> Self {
        Self {
            tool_type: OpenAIWebSearchToolType::default(),
            search_context_size: None,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum OpenAIWebSearchToolType {
    #[serde(rename = "web_search_preview")]
    #[default]
    WebSearchPreview,
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview20250311,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIWebSearchContextSize {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

///
/// OpenAI Computer Use tool config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub struct OpenAIComputerUseConfig {
    #[serde(rename = "type")]
    pub tool_type: OpenAIComputerUseToolType,
    pub display_height: usize,
    pub display_width: usize,
    pub environment: String,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIComputerUseToolType {
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview,
}

///
/// OpenAI Reasoning config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub struct OpenAIReasoningConfig {
    pub effort: Option<OpenAIReasoningEffort>,
    pub summary: Option<OpenAIReasoningSummary>,
}

impl OpenAIReasoningConfig {
    pub fn new(
        effort: Option<OpenAIReasoningEffort>,
        summary: Option<OpenAIReasoningSummary>,
    ) -> Self {
        Self { effort, summary }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIReasoningEffort {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIReasoningSummary {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "none")]
    Concise,
    #[serde(rename = "detailed")]
    Detailed,
}

///
/// OpenAI Code Interpreter tool config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub struct OpenAICodeInterpreterConfig {
    #[serde(rename = "type")]
    pub tool_type: OpenAICodeInterpreterToolType,
    pub container: OpenAICodeInterpreterContainerConfig,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAICodeInterpreterToolType {
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
}

// Can be a container ID or an object that specifies uploaded file IDs to make available to your code.
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
#[serde(untagged)]
pub enum OpenAICodeInterpreterContainerConfig {
    ContainerId(String),
    CodeInterpreterContainerAuto(CodeInterpreterContainerAutoConfig),
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub struct CodeInterpreterContainerAutoConfig {
    #[serde(rename = "type")]
    container_type: OpenAICodeInterpreterContainerType,
    file_ids: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum OpenAICodeInterpreterContainerType {
    #[serde(rename = "auto")]
    #[default]
    Auto,
}

impl Default for OpenAICodeInterpreterConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAICodeInterpreterConfig {
    pub fn new() -> Self {
        Self {
            tool_type: OpenAICodeInterpreterToolType::CodeInterpreter,
            container: OpenAICodeInterpreterContainerConfig::CodeInterpreterContainerAuto(
                CodeInterpreterContainerAutoConfig::default(),
            ),
        }
    }
}
