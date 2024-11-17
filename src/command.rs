use std::time::{Duration, Instant};

use crate::ai_backend::AiBackend;
use crate::ai_backend::{BedrockAiBackend, LocalAiBackend};
use anyhow::{Error as E, Result};
use clap::{Parser, Subcommand};
use clap_verbosity_flag::Level;
use indicatif::{ProgressBar, ProgressStyle};

use crate::settings::{ConfigLogLevel, Settings};
use tracing::info;

#[derive(Clone, Debug, Subcommand)]
pub enum AiCliCommands {
    /// Prints the Settings, arguments, and the log verbosity
    Config,
    /// Generate a bash one liner based off of the prompt
    Generate,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, name = "ai")]
pub struct AiCliArgs {
    /// Enable tracing functionality which will generate a trace-timestamp.json file
    /// containing detailed execution information for debugging and profiling. Load into Chrome to view
    #[arg(long, short)]
    pub tracing: bool,

    /// Specify which AI backend to use for processing requests:
    /// - "bedrock": Use Amazon Bedrock managed AI service
    /// - "local": Use local LLM model (Phi 2 or 3) pulled from Hugging face
    /// 
    /// If not specified, the backend will be read from config file, defaulting to "local"
    #[arg(long, short = 'b')]
    pub ai_backend: Option<String>,

    /// Control log output verbosity level:
    /// - v: warnings
    /// - vv: info
    /// - vvv: debug
    /// - vvvv: trace
    /// 
    /// Default level is error if not specified, overrides the config setting
    #[command(flatten)]
    pub verbose: clap_verbosity_flag::Verbosity<ConfigLogLevel>,

    /// Specify a command to execute. Currently supported commands:
    /// - config: Display current configuration settings
    /// - generate: Generate a bash script based off of the prompt (default)
    #[command(subcommand)]
    pub command: Option<AiCliCommands>,

    /// The input prompt/query to send to the AI model when using generate mode.
    /// Multiple words can be provided and will be joined into a single prompt.
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
                    return Err(anyhow::anyhow!("Prompt is empty"));
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
                        Box::new(BedrockAiBackend::new(self.settings))
                    }
                    "local" => {
                        info!("Using Local AI backend");
                        Box::new(LocalAiBackend::new(self.settings, self.start))
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
                #[cfg(feature = "clipboard")]{
                    let mut clipboard = arboard::Clipboard::new()?;
                    clipboard.set_text(result)?;
                }
                Ok(())
            }
        }
    }
}
