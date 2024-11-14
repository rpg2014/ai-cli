use std::time::{Duration, Instant};

use crate::ai_backend::AiBackend;
use crate::ai_backend::{BedrockAiBackend, LocalAiBackend};
use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use clap_verbosity_flag::LogLevel;
use indicatif::{ProgressBar, ProgressStyle};


use crate::settings::Settings;
use tracing::info;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum WhichModel {
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, name = "ai")]
pub struct AiCliArgs {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,

    #[arg(long)]
    pub prompt: String,

    #[arg(long, default_value = "2")]
    pub model: WhichModel,

    #[arg(long)]
    pub quantized: bool,

    #[arg(long)]
    pub ai_backend: Option<String>,

    #[command(flatten)]
    pub verbose: clap_verbosity_flag::Verbosity<ConfigLogLevel>,
}

pub struct AiCli {
    settings: Settings,
    args: AiCliArgs,
    start: Instant,
}

impl AiCli {
    pub fn new(settings: Settings, args: AiCliArgs, start: Option<Instant>) -> Self {
        Self {
            settings,
            args,
            start: start.unwrap_or(Instant::now()),
        }
    }
    pub fn exec(self) -> Result<()> {
        info!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            self.settings.model_config.temperature.unwrap_or(0.),
            self.settings.model_config.repeat_penalty,
            self.settings.model_config.repeat_last_n
        );
        // get from args, fallback to settings obj
        let backend = match self.args.ai_backend {
            Some(ref backend) => backend,
            None => &self.settings.ai_backend,
        };
        
        
        let local_model: Box<dyn AiBackend> = match backend.as_str() {
            "bedrock" => {
                info!("Using Bedrock AI backend");
                Box::new(BedrockAiBackend::new(self.settings, self.args, self.start))
            }
            "local" => {
                info!("Using Local AI backend");
                Box::new(LocalAiBackend::new(self.settings, self.args, self.start))
            }
            _ => {
                return Err(E::msg(format!(
                    "Unknown backend: {}",
                    backend
                )));
            }
        };
        info!("Beginning inference");
        
        let bar = ProgressBar::new_spinner();
        bar.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap().tick_strings(&[
            "⣾",
			"⣽",
			"⣻",
			"⢿",
			"⡿",
			"⣟",
			"⣯",
			"⣷",
            // full block
            "⣿"
// "▹▹▹▹▹",
//                 "▸▹▹▹▹",
//                 "▹▸▹▹▹",
//                 "▹▹▸▹▹",
//                 "▹▹▹▸▹",
//                 "▹▹▹▹▸",
//                 "▪▪▪▪▪",
            ]));
        bar.tick();
        bar.enable_steady_tick(Duration::from_millis(100));
        bar.set_message("Thinking...");
        let result = local_model.invoke()?; //print result
        bar.finish_with_message("Done");
        info!("response time: {:?}", self.start.elapsed());
        info!("{:?}", result);
        println!("{}", result);
        Ok(())
    }
}
