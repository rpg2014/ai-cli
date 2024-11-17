use std::path::PathBuf;

use clap_verbosity_flag::LogLevel;
use config::Config;

use crate::{ai_backend::local::WhichModel, constants::DEFAULT_CONFIG_CONTENT};

/// Top Level settings object
#[derive(Debug, serde::Deserialize)]
pub struct Settings {
    /// Verbosity setting, CLI arg takes precident 
    pub verbosity: Option<String>,
    // Which AI backend to use by default, bedrock or local
    pub ai_backend: String,
    /// The local model configuration
    pub local_model_config: LocalModelConfig,
    /// Various AWS setting such as profile (not respected yet) and region
    pub aws_settings: AwsSettings,
}

/// AWS related settings
#[derive(Debug, serde::Deserialize)]
pub struct AwsSettings {
    pub profile: Option<String>,
    pub region: String,
}

/// Config options for the local LLM setting
#[derive(Debug, serde::Deserialize)]
pub struct LocalModelConfig {
    /// Run on CPU rather than on GPU.
    pub cpu: bool,
    /// Which local model to pull (2, 3)
    pub model: WhichModel,
    /// whether to use the quantized version of the model, 2 only supported
    pub quantized: bool,
    /// log the split up tokens in the prompt
    pub verbose_prompt: bool,
    /// Model temperature - controls randomness of outputs (0.0-1.0)
    pub temperature: Option<f64>,
    /// Top-p sampling threshold (0.0-1.0) - controls diversity of outputs
    pub top_p: Option<f64>,
    /// Random seed for reproducible outputs
    pub seed: u64,
    /// Maximum number of tokens to generate
    pub sample_len: usize,
    /// Identifier for the model to use - HF model repo
    pub model_id: Option<String>,
    /// Model revision/version - HF git branch
    pub revision: Option<String>,
    /// Path to model weights file
    pub weight_file: Option<String>,
    /// Path to tokenizer file
    pub tokenizer: Option<String>,
    /// Penalty factor for repeated tokens (>1.0 reduces repetition)
    pub repeat_penalty: f32,
    /// Number of previous tokens to consider for repeat penalty
    pub repeat_last_n: usize,
    /// Data type for model weights (e.g. "f32", "f16")
    pub dtype: Option<String>,
}

impl Settings {
    pub fn new() -> Result<Self, config::ConfigError> {
        // I personally like my config files in .config on mac
        let config_path = dirs::home_dir() // Gets the config directory cross-platform
            .map(|mut path| {
                path.push(".config");
                path.push("ai");
                path.push("config");
                path
            })
            .unwrap_or_else(|| PathBuf::from("config")); // Fallback to local config

        // create ~/.config/ai if it doesn't exist
        let config_parent_dir = config_path.parent().unwrap();
        if !config_parent_dir.exists() {
            // info! doesnn't work here as this get's run before we set up the log subscriber
            println!("Creating config directory: {:?}", &config_parent_dir);
            std::fs::create_dir_all(config_parent_dir).unwrap();
        }

        // Check if config file exists, if not create it with defaults
        let config_file = config_path.with_extension("toml");
        if !config_file.exists() {
            println!("Creating config file: {:?}", &config_file);
            std::fs::write(&config_file, DEFAULT_CONFIG_CONTENT)
                .expect("Failed to write config file");
        }

        let settings = Config::builder()
            .add_source(config::File::with_name(config_path.to_str().unwrap()).required(false))
            .add_source(config::File::with_name("config").required(false))
            .set_default("local_model_config.cpu", false)?
            .set_default("local_model_config.model", "V2")?
            .set_default("local_model_config.quantized", true)?
            .set_default("local_model_config.verbose_prompt", false)?
            .set_default("local_model_config.temperature", 0.8_f64)?
            .set_default("local_model_config.top_p", 0.9_f64)?
            .set_default("local_model_config.seed", rand::random::<u64>())?
            .set_default("local_model_config.sample_len", 100)?
            .set_default("local_model_config.repeat_penalty", 1.1)?
            .set_default("local_model_config.repeat_last_n", 64)?
            .set_default("local_model_config.dtype", "f32")?
            .set_default("aws_settings.region", "us-east-1")?
            .set_default("ai_backend", "local")?
            .build()?;

        settings.try_deserialize()
    }
}

#[derive(Debug)]
pub struct ConfigLogLevel {}

impl LogLevel for ConfigLogLevel {
    fn default() -> Option<clap_verbosity_flag::Level> {
        // read from settings options
        let settings = Settings::new().unwrap();
        let log_level = settings.verbosity.unwrap_or_else(|| "error".to_string());
        let level = match log_level.as_str() {
            "error" => Some(clap_verbosity_flag::Level::Error),
            "warn" => Some(clap_verbosity_flag::Level::Warn),
            "info" => Some(clap_verbosity_flag::Level::Info),
            "debug" => Some(clap_verbosity_flag::Level::Debug),
            "trace" => Some(clap_verbosity_flag::Level::Trace),
            _ => Some(clap_verbosity_flag::Level::Error),
        };
        level
    }
}
