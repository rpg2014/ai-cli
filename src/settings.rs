use config::Config;
use serde;

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
        let settings = Config::builder()
            .add_source(config::File::with_name("config").required(false))
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
