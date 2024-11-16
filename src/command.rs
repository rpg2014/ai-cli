use std::time::{Duration, Instant};

use crate::ai_backend::AiBackend;
use crate::ai_backend::{BedrockAiBackend, LocalAiBackend};
use anyhow::{Error as E, Result};
use clap::{Parser, Subcommand, ValueEnum};
use clap_verbosity_flag::{Level, LogLevel};
use indicatif::{ProgressBar, ProgressStyle};

use crate::settings::Settings;
use tracing::info;

#[derive(Clone, ValueEnum, Debug, Subcommand)]
pub enum AiCliCommands {
    Config,
    Chat
    // Image,
    // Code,
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
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long, short)]
    pub tracing: bool,

    #[arg(long, short = 'b')]
    pub ai_backend: Option<String>,

    #[command(flatten)]
    pub verbose: clap_verbosity_flag::Verbosity<ConfigLogLevel>,

    #[command(subcommand)]
    pub command: Option<AiCliCommands>,
    
    #[arg(trailing_var_arg = true)]
    pub other_args: Vec<String>,
}

pub struct AiCli {
    pub settings: Settings,
    pub args: AiCliArgs,
    start: Instant,
    log_level: Level,
    pub prompt: String,
}

impl AiCli {
    pub fn new(
        settings: Settings,
        args: AiCliArgs,
        start: Option<Instant>,
        log_level: Level,
        prompt: String,
    ) -> Self {
        Self {
            settings,
            args,
            start: start.unwrap_or(Instant::now()),
            log_level,
            prompt,
        }
    }
    pub fn exec(self) -> Result<()> {
        match self.args.command {
            Some(AiCliCommands::Config) => {
                // pretty println settings, args and log level
                println!("Settings: {:#?}", self.settings);
                println!("Args: {:#?}", self.args);
                println!("Log level: {:#?}", self.log_level);
                Ok(())
            }
            Some(_) | None => {
                // check prompt is not empty
                if self.prompt.is_empty() {
                    return Err(anyhow::anyhow!("Prompt is empty"))
                }
                info!(
                    "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
                    self.settings.local_model_config.temperature.unwrap_or(0.),
                    self.settings.local_model_config.repeat_penalty,
                    self.settings.local_model_config.repeat_last_n
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
                        return Err(E::msg(format!("Unknown backend: {}", backend)));
                    }
                };
                info!("Beginning inference");
                let mut bar: Option<ProgressBar> = None;
                // if match verbosity is info or below
                if self.log_level < Level::Info {
                    let temp_bar = ProgressBar::new_spinner();
                    temp_bar.set_style(
                        ProgressStyle::with_template("{spinner:.green} {msg}")
                            .unwrap()
                            .tick_strings(&[
                                "⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽", "⣾", // full block
                                "⣿", // "▹▹▹▹▹",
                                    //                 "▸▹▹▹▹",
                                    //                 "▹▸▹▹▹",
                                    //                 "▹▹▸▹▹",
                                    //                 "▹▹▹▸▹",
                                    //                 "▹▹▹▹▸",
                                    //                 "▪▪▪▪▪",
                            ]),
                    );
                    temp_bar.tick();
                    temp_bar.enable_steady_tick(Duration::from_millis(100));
                    temp_bar.set_message("Thinking...");
                    bar = Some(temp_bar);
                }
                let result = local_model.invoke(self.prompt)?; //print result
                if let Some(bar) = bar {
                    bar.finish_with_message("Done");
                }

                info!("response time: {:?}", self.start.elapsed());
                info!("{:?}", result);
                println!("{}", result);
                Ok(())
            }
        }
    }
}
