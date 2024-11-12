use std::time::Instant;

use crate::{device, hub_load_safetensors, settings};
use crate::text_generation::{Model, TextGeneration};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::mixformer::Config;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use clap::{Parser, ValueEnum};
use clap_verbosity_flag::{LevelFilter, LogLevel};
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_nn::VarBuilder;
use tracing::info;
use crate::settings::Settings;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
}
#[derive(Debug)]
pub struct ConfigLogLevel {

}

 impl LogLevel for ConfigLogLevel {
    fn default() -> Option<clap_verbosity_flag::Level> {
        // read from settings options 
        let settings = Settings::new().unwrap();
        let log_level = settings.verbosity.unwrap_or_else( || {"info".to_string()});
        let level = match log_level.as_str() {
            "error" => Some(clap_verbosity_flag::Level::Error),
            "warn" => Some(clap_verbosity_flag::Level::Warn),
            "info" => Some(clap_verbosity_flag::Level::Info),
            "debug" => Some(clap_verbosity_flag::Level::Debug),
            "trace" => Some(clap_verbosity_flag::Level::Trace),
            _ => Some(clap_verbosity_flag::Level::Info),
        };
        level
        
    }
    
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct AiCliArgs {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value = "2")]
    model: WhichModel,

    #[arg(long)]
    quantized: bool,

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
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        info!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            self.settings.model_config.temperature.unwrap_or(0.),
            self.settings.model_config.repeat_penalty,
            self.settings.model_config.repeat_last_n
        );
        let (model, tokenizer, device) = self.load_local_model()?;
        info!("loaded the model in {:?}", self.start.elapsed());
        match self.args.prompt {
            Some(prompt) => {
                let mut pipeline = TextGeneration::new(
                    model,
                    tokenizer,
                    self.settings.model_config.seed,
                    self.settings.model_config.temperature,
                    self.settings.model_config.top_p,
                    self.settings.model_config.repeat_penalty,
                    self.settings.model_config.repeat_last_n,
                    self.settings.model_config.verbose_prompt,
                    &device,
                );
                
                // Use tokio runtime to run the async method
                tokio::runtime::Runtime::new()?.block_on(async {
                    let mut stdout = tokio::io::stdout();
                    pipeline.run(&prompt, self.settings.model_config.sample_len, &mut stdout).await
                })?;
                info!("generated the output in {:?}", self.start.elapsed());
                Ok(())
            }
            
            None => anyhow::bail!("Prompt not provided"),
        }
    }

    fn load_local_model(&self) -> Result<(Model, Tokenizer, Device)> {
        
        let repo = self.get_repo_for_local_model()?;
        let tokenizer_filename = match &self.settings.model_config.tokenizer {
            Some(file) => std::path::PathBuf::from(file),
            None => match self.args.model {
                WhichModel::V2 | WhichModel::V3 => repo.get("tokenizer.json")?,
            },
        };
        let filenames = match &self.settings.model_config.weight_file {
            Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
            None => {
                if self.args.quantized {
                    match self.args.model {
                        WhichModel::V2 => vec![repo.get("model-v2-q4k.gguf")?],
                        WhichModel::V3 => anyhow::bail!(
                            "use the quantized or quantized-phi examples for quantized phi-v3"
                        ),
                    }
                } else {
                    match self.args.model {
                        WhichModel::V2 => hub_load_safetensors(
                            &repo,
                            "model.safetensors.index.json",
                        )?,
                        WhichModel::V3 => hub_load_safetensors(
                            &repo,
                            "model.safetensors.index.json",
                        )?,
                    }
                }
            }
        };
        // println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    
        // let start = std::time::Instant::now();
        let config = || match self.args.model {
            WhichModel::V2 => Config::v2(),
            WhichModel::V3 => {
                panic!("use the quantized or quantized-phi examples for quantized phi-v3")
            }
        };
        let device = device(self.args.cpu)?;
        let model = if self.args.quantized {
            let config = config();
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &filenames[0],
                &device,
            )?;
            let model = match self.args.model {
                WhichModel::V2 => QMixFormer::new_v2(&config, vb)?,
                WhichModel::V3 => {
                    anyhow::bail!("Quantized Phi-3 not supported")
                }
            };
            Model::Quantized(model)
        } else {
            let dtype = match &self.settings.model_config.dtype {
                Some(dtype) => dtype.parse()?,
                None => {
                    if self.args.model == WhichModel::V3 {
                        device.bf16_default_to_f32()
                    } else {
                        DType::F32
                    }
                }
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            match self.args.model {
                WhichModel::V2 => {
                    let config_filename = repo.get("config.json")?;
                    let config = std::fs::read_to_string(config_filename)?;
                    let config: PhiConfig = serde_json::from_str(&config)?;
                    let phi = Phi::new(&config, vb)?;
                    Model::Phi(phi)
                }
                WhichModel::V3 => {
                    let config_filename = repo.get("config.json")?;
                    let config = std::fs::read_to_string(config_filename)?;
                    let config: Phi3Config = serde_json::from_str(&config)?;
                    let phi3 = Phi3::new(&config, vb)?;
                    Model::Phi3(phi3)
                }
            }
        };

        info!("loaded the model, device: {:?}", device);
        //return
        Ok((model, tokenizer, device))
    }

    fn get_repo_for_local_model(&self) -> Result<ApiRepo> {
        info!("Loading the model, parsing model from args and settings");
        let api = Api::new()?;
        let model_id = match &self.settings.model_config.model_id {
            Some(model_id) => model_id.to_string(),
            None => {
                if self.args.quantized {
                    "lmz/candle-quantized-phi".to_string()
                } else {
                    match self.args.model {
                        WhichModel::V2 => "microsoft/phi-2".to_string(),
                        WhichModel::V3 => "microsoft/Phi-3-mini-4k-instruct".to_string(),
                    }
                }
            }
        };
        let revision = match &self.settings.model_config.revision {
            Some(rev) => rev.to_string(),
            None => {
                if self.args.quantized {
                    "main".to_string()
                } else {
                    match self.args.model {
                        WhichModel::V2 => "main".to_string(),
                        WhichModel::V3 => "main".to_string(),
                    }
                }
            }
        };
        info!("Loading model {model_id} revision {revision}");
        Ok(api.repo(Repo::with_revision(model_id, RepoType::Model, revision)))
    }

}
