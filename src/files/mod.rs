mod anthropic;
mod llm_files;
mod openai;

/// Main trait for LLM files
pub use llm_files::LLMFiles;

/// OpenAI file implementation
pub use openai::OpenAIFile;

/// Anthropic file implementation
pub use anthropic::AnthropicFile;
