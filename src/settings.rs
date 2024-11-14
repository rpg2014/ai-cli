use std::path::PathBuf;

use config::Config;
use serde;

use crate::constants::DEFAULT_CONFIG_CONTENT;

#[derive(Debug, serde::Deserialize)]
pub struct Settings {
    pub verbosity: Option<String>,
    pub ai_backend: String,
    pub model_config: ModelConfig,
    pub aws_settings: AwsSettings,
}
#[derive(Debug, serde::Deserialize)]
pub struct AwsSettings {
    pub profile: Option<String>,
    pub region: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    pub verbose_prompt: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub sample_len: usize,
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub weight_file: Option<String>,
    pub tokenizer: Option<String>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub dtype: Option<String>,
}

impl Settings {
    pub fn new() -> Result<Self, config::ConfigError> {
        let config_path = dirs::config_dir()  // Gets the config directory cross-platform
        .map(|mut path| {
            path.push("your_program_name");  // Replace with your actual program name
            path.push("config");
            path
        })
        .unwrap_or_else(|| PathBuf::from("config"));  // Fallback to local config

    // Create the directory if it doesn't exist
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).ok();  // Ignore error if dir already exists
    }
    // Check if config file exists, if not create it with defaults
    if !config_path.with_extension("toml").exists() {
        std::fs::write(
            config_path.with_extension("toml"),
            DEFAULT_CONFIG_CONTENT
        ).ok();  // Using ok() to ignore write errors
    }

    let settings = Config::builder()
            .add_source(config::File::with_name(config_path.to_str().unwrap()).required(false))
            .add_source(config::Environment::with_prefix("AI_CLI"))
            .set_default("model_config.verbose_prompt", false)?
            .set_default("model_config.temperature", 0.8_f64)?
            .set_default("model_config.top_p", 0.9_f64)?
            .set_default("model_config.seed", rand::random::<u64>())?
            .set_default("model_config.sample_len", 100)?
            .set_default("model_config.repeat_penalty", 1.1)?
            .set_default("model_config.repeat_last_n", 64)?
            .set_default("model_config.dtype", "f32")?
            .set_default("aws_settings.region", "us-east-1")?
            .set_default("ai_backend", "local")?
            .build()?;

        settings.try_deserialize()
    }
}
