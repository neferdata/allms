use log::warn;
use serde::{Deserialize, Serialize};
use serde_json::{to_value, Value};

use crate::domain::XAISearchMode;

// Re-export XAIWebSearchConfig and XAISearchSource with their implemented methods
pub use crate::domain::XAISearchSource;
pub use crate::domain::XAIWebSearchConfig;

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
    AnthropicCodeExecution(AnthropicCodeExecutionConfig),
    AnthropicComputerUse(AnthropicComputerUseConfig),
    AnthropicWebSearch(AnthropicWebSearchConfig),
    XAIWebSearch(XAIWebSearchConfig),
}

impl LLMTools {
    pub fn get_config_json(&self) -> Option<Value> {
        match self {
            LLMTools::OpenAIFileSearch(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAIWebSearch(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAIComputerUse(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAIReasoning(cfg) => to_value(cfg).ok(),
            LLMTools::OpenAICodeInterpreter(cfg) => to_value(cfg).ok(),
            LLMTools::AnthropicCodeExecution(cfg) => to_value(cfg).ok(),
            LLMTools::AnthropicComputerUse(cfg) => to_value(cfg).ok(),
            LLMTools::AnthropicWebSearch(cfg) => to_value(cfg).ok(),
            LLMTools::XAIWebSearch(cfg) => to_value(cfg).ok(),
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

///
/// Anthropic Code Execution tool config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub struct AnthropicCodeExecutionConfig {
    pub name: AnthropicCodeExecutionName,
    #[serde(rename = "type")]
    pub tool_type: AnthropicCodeExecutionToolType,
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicCodeExecutionName {
    #[serde(rename = "code_execution")]
    #[default]
    CodeExecution,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicCodeExecutionToolType {
    #[serde(rename = "code_execution_20250522")]
    #[default]
    CodeExecution20250522,
}

impl AnthropicCodeExecutionConfig {
    pub fn new() -> Self {
        Self {
            name: AnthropicCodeExecutionName::default(),
            tool_type: AnthropicCodeExecutionToolType::default(),
            cache_control: None,
        }
    }

    pub fn cache_control(mut self, ttl: AnthropicCacheControllTTL) -> Self {
        self.cache_control = Some(AnthropicCacheControl {
            cache_type: AnthropicCacheControlType::default(),
            ttl,
        });
        self
    }
}

///
/// Anthropic Computer Use tool config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub struct AnthropicComputerUseConfig {
    pub name: AnthropicComputerUseName,
    #[serde(rename = "type")]
    pub tool_type: AnthropicComputerUseToolType,
    pub display_height_px: usize,
    pub display_width_px: usize,
    pub cache_control: Option<AnthropicCacheControl>,
    pub display_number: Option<usize>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicComputerUseName {
    #[serde(rename = "computer")]
    #[default]
    Computer,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicComputerUseToolType {
    #[serde(rename = "computer_20241022")]
    #[default]
    Computer20241022,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub struct AnthropicCacheControl {
    #[serde(rename = "type")]
    pub cache_type: AnthropicCacheControlType,
    pub ttl: AnthropicCacheControllTTL,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicCacheControlType {
    #[serde(rename = "ephemeral")]
    #[default]
    Ephemeral,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicCacheControllTTL {
    #[serde(rename = "5m")]
    #[default]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

impl AnthropicComputerUseConfig {
    pub fn new(display_height_px: usize, display_width_px: usize) -> Self {
        Self {
            name: AnthropicComputerUseName::default(),
            tool_type: AnthropicComputerUseToolType::default(),
            display_height_px: display_height_px.max(1),
            display_width_px: display_width_px.max(1),
            cache_control: None,
            display_number: None,
        }
    }

    pub fn cache_control(mut self, ttl: AnthropicCacheControllTTL) -> Self {
        self.cache_control = Some(AnthropicCacheControl {
            cache_type: AnthropicCacheControlType::default(),
            ttl,
        });
        self
    }

    pub fn display_number(mut self, display_number: usize) -> Self {
        self.display_number = Some(display_number);
        self
    }
}

/// xAI Web Search tool config
///
impl Default for XAIWebSearchConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl XAIWebSearchConfig {
    pub fn new() -> Self {
        Self {
            from_date: None,
            to_date: None,
            max_search_results: None,
            mode: None,
            return_citations: None,
            sources: None,
        }
    }

    pub fn from_date(mut self, from_date: String) -> Self {
        self.from_date = Some(from_date);
        self
    }

    pub fn to_date(mut self, to_date: String) -> Self {
        self.to_date = Some(to_date);
        self
    }

    pub fn max_search_results(mut self, max_search_results: usize) -> Self {
        self.max_search_results = Some(max_search_results);
        self
    }

    pub fn mode(mut self, mode: XAISearchMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn return_citations(mut self, return_citations: bool) -> Self {
        self.return_citations = Some(return_citations);
        self
    }

    pub fn add_source(mut self, source: XAISearchSource) -> Self {
        if let Some(sources) = self.sources {
            let mut new_sources = sources.clone();
            new_sources.push(source);
            self.sources = Some(new_sources);
        } else {
            self.sources = Some(vec![source]);
        }
        self
    }
}

///
/// Anthropic Web Search tool config
///
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub struct AnthropicWebSearchConfig {
    pub name: AnthropicWebSearchName,
    #[serde(rename = "type")]
    pub tool_type: AnthropicWebSearchToolType,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub cache_control: Option<AnthropicCacheControl>,
    pub max_uses: Option<usize>,
    pub user_location: Option<AnthropicWebSearchUserLocation>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicWebSearchName {
    #[serde(rename = "web_search")]
    #[default]
    WebSearch,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicWebSearchToolType {
    #[serde(rename = "web_search_20250305")]
    #[default]
    WebSearch20250305,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub struct AnthropicWebSearchUserLocation {
    #[serde(rename = "type")]
    pub location_type: AnthropicWebSearchUserLocationType,
    pub city: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, Default)]
pub enum AnthropicWebSearchUserLocationType {
    #[serde(rename = "approximate")]
    #[default]
    Approximate,
}

impl AnthropicWebSearchConfig {
    pub fn new() -> Self {
        Self {
            name: AnthropicWebSearchName::default(),
            tool_type: AnthropicWebSearchToolType::default(),
            allowed_domains: None,
            blocked_domains: None,
            cache_control: None,
            max_uses: None,
            user_location: None,
        }
    }

    pub fn cache_control(mut self, ttl: AnthropicCacheControllTTL) -> Self {
        self.cache_control = Some(AnthropicCacheControl {
            cache_type: AnthropicCacheControlType::default(),
            ttl,
        });
        self
    }

    pub fn allowed_domains(mut self, allowed_domains: Vec<String>) -> Self {
        self.allowed_domains = Some(allowed_domains);
        self.blocked_domains = None;
        self
    }

    pub fn blocked_domains(mut self, blocked_domains: Vec<String>) -> Self {
        self.blocked_domains = Some(blocked_domains);
        self.allowed_domains = None;
        self
    }

    pub fn max_uses(mut self, max_uses: usize) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    pub fn user_location(mut self, user_location: AnthropicWebSearchUserLocation) -> Self {
        self.user_location = Some(user_location);
        self
    }
}

impl AnthropicWebSearchUserLocation {
    pub fn new(location_type: AnthropicWebSearchUserLocationType) -> Self {
        Self {
            location_type,
            city: None,
            country: None,
            region: None,
            timezone: None,
        }
    }

    pub fn city(mut self, city: String) -> Self {
        self.city = Some(city);
        self
    }

    pub fn country(mut self, country: String) -> Self {
        self.country = Some(country);
        self
    }

    pub fn region(mut self, region: String) -> Self {
        self.region = Some(region);
        self
    }

    pub fn timezone(mut self, timezone: String) -> Self {
        self.timezone = Some(timezone);
        self
    }
}

/// XAISearchSource constructors and helper methods
///
impl XAISearchSource {
    /// Create a new Web search source
    pub fn web() -> Self {
        XAISearchSource::Web(crate::domain::WebSource {
            allowed_websites: None,
            excluded_websites: None,
            country: None,
            safe_search: None,
        })
    }

    /// Create a new X (Twitter) search source
    pub fn x() -> Self {
        XAISearchSource::X(crate::domain::XSource {
            included_x_handles: None,
            excluded_x_handles: None,
            post_favorite_count: None,
            post_view_count: None,
        })
    }

    /// Create a new News search source
    pub fn news() -> Self {
        XAISearchSource::News(crate::domain::NewsSource {
            excluded_websites: None,
            country: None,
            safe_search: None,
        })
    }

    /// Create a new RSS search source with the given links
    pub fn rss(links: Vec<String>) -> Self {
        XAISearchSource::Rss(crate::domain::RssSource { links })
    }

    /// Add allowed websites to a Web search source
    /// This will clear any excluded_websites to avoid conflicts
    pub fn with_allowed_sites(mut self, allowed_websites: Vec<String>) -> Self {
        if let XAISearchSource::Web(ref mut web_source) = self {
            web_source.allowed_websites = Some(allowed_websites);
            if web_source.excluded_websites.is_some() {
                warn!("[allms][xAI][Tools] Allowed websites will clear any excluded websites");
                web_source.excluded_websites = None; // Clear conflicting parameter
            }
        }
        self
    }

    /// Add excluded websites to a Web or News search source
    /// For Web sources, this will clear any allowed_websites to avoid conflicts
    pub fn with_excluded_sites(mut self, excluded_websites: Vec<String>) -> Self {
        match &mut self {
            XAISearchSource::Web(web_source) => {
                web_source.excluded_websites = Some(excluded_websites);
                if web_source.allowed_websites.is_some() {
                    warn!("[allms][xAI][Tools] Excluded websites will clear any allowed websites");
                    web_source.allowed_websites = None; // Clear conflicting parameter
                }
            }
            XAISearchSource::News(news_source) => {
                news_source.excluded_websites = Some(excluded_websites);
            }
            _ => {} // Ignore for other source types
        }
        self
    }

    /// Add country filter to a Web or News search source
    pub fn with_country(mut self, country: String) -> Self {
        match &mut self {
            XAISearchSource::Web(web_source) => {
                web_source.country = Some(country);
            }
            XAISearchSource::News(news_source) => {
                news_source.country = Some(country);
            }
            _ => {} // Ignore for other source types
        }
        self
    }

    /// Add safe search setting to a Web or News search source
    pub fn with_safe_search(mut self, safe_search: bool) -> Self {
        match &mut self {
            XAISearchSource::Web(web_source) => {
                web_source.safe_search = Some(safe_search);
            }
            XAISearchSource::News(news_source) => {
                news_source.safe_search = Some(safe_search);
            }
            _ => {} // Ignore for other source types
        }
        self
    }

    /// Add included X handles to an X search source
    /// This will clear any excluded_x_handles to avoid conflicts
    pub fn with_included_handles(mut self, included_x_handles: Vec<String>) -> Self {
        if let XAISearchSource::X(ref mut x_source) = self {
            x_source.included_x_handles = Some(included_x_handles);
            if x_source.excluded_x_handles.is_some() {
                warn!("[allms][xAI][Tools] Included X handles will clear any excluded X handles");
                x_source.excluded_x_handles = None; // Clear conflicting parameter
            }
        }
        self
    }

    /// Add excluded X handles to an X search source
    /// This will clear any included_x_handles to avoid conflicts
    pub fn with_excluded_handles(mut self, excluded_x_handles: Vec<String>) -> Self {
        if let XAISearchSource::X(ref mut x_source) = self {
            x_source.excluded_x_handles = Some(excluded_x_handles);
            if x_source.included_x_handles.is_some() {
                warn!("[allms][xAI][Tools] Excluded X handles will clear any included X handles");
                x_source.included_x_handles = None; // Clear conflicting parameter
            }
        }
        self
    }

    /// Add minimum favorite count filter to an X search source
    pub fn with_favorite_count(mut self, post_favorite_count: usize) -> Self {
        if let XAISearchSource::X(ref mut x_source) = self {
            x_source.post_favorite_count = Some(post_favorite_count);
        }
        self
    }

    /// Add minimum view count filter to an X search source
    pub fn with_view_count(mut self, post_view_count: usize) -> Self {
        if let XAISearchSource::X(ref mut x_source) = self {
            x_source.post_view_count = Some(post_view_count);
        }
        self
    }
}
