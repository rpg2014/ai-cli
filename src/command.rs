use std::time::Instant;

use crate::{device, hub_load_safetensors};
use crate::text_generation::{Model, TextGeneration};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::mixformer::Config;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_nn::VarBuilder;
use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tokio::io::AsyncWriteExt;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct AiCli {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long)]
    prompt: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    /// Lets you add a specific model to use from hf
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "2")]
    model: WhichModel,

    /// the hf git revision (main usually) to pull the repo for
    #[arg(long)]
    revision: Option<String>,


    /// Lets you specify a specific weigt file to use within the hf repo
    #[arg(long)]
    weight_file: Option<String>,

    /// lets you specify a tokenizer json string to use isntead of from hf
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The dtype to be used for running the model, e.g. f32, bf16, or f16.
    #[arg(long)]
    dtype: Option<String>,

    #[command(flatten)]
    pub verbose: clap_verbosity_flag::Verbosity,
}

impl AiCli {
    
    pub fn exec(self, start: &Instant) -> Result<()> {
        
        let (model, tokenizer, device) = self.load_local_model()?;
        info!("loaded the model in {:?}", start.elapsed());
        match self.prompt {
            Some(prompt) => {
                let mut pipeline = TextGeneration::new(
                    model,
                    tokenizer,
                    self.seed,
                    self.temperature,
                    self.top_p,
                    self.repeat_penalty,
                    self.repeat_last_n,
                    self.verbose_prompt,
                    &device,
                );
                
                // Use tokio runtime to run the async method
                tokio::runtime::Runtime::new()?.block_on(async {
                    let mut stdout = tokio::io::stdout();
                    pipeline.run(&prompt, self.sample_len, &mut stdout).await
                })?;
                info!("generated the output in {:?}", start.elapsed());
                Ok(())
            }
            
            None => anyhow::bail!("Prompt not provided"),
        }
    }

    fn load_local_model(&self) -> Result<(Model, Tokenizer, Device)> {
        info!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        info!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            self.temperature.unwrap_or(0.),
            self.repeat_penalty,
            self.repeat_last_n
        );
    
        
        let repo = self.get_repo_for_local_model()?;
        let tokenizer_filename = match &self.tokenizer {
            Some(file) => std::path::PathBuf::from(file),
            None => match self.model {
                WhichModel::V2 | WhichModel::V3 => repo.get("tokenizer.json")?,
            },
        };
        let filenames = match &self.weight_file {
            Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
            None => {
                if self.quantized {
                    match self.model {
                        WhichModel::V2 => vec![repo.get("model-v2-q4k.gguf")?],
                        WhichModel::V3 => anyhow::bail!(
                            "use the quantized or quantized-phi examples for quantized phi-v3"
                        ),
                    }
                } else {
                    match self.model {
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
        let config = || match self.model {
            WhichModel::V2 => Config::v2(),
            WhichModel::V3 => {
                panic!("use the quantized or quantized-phi examples for quantized phi-v3")
            }
        };
        let device = device(self.cpu)?;
        let model = if self.quantized {
            let config = config();
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &filenames[0],
                &device,
            )?;
            let model = match self.model {
                WhichModel::V2 => QMixFormer::new_v2(&config, vb)?,
                WhichModel::V3 => {
                    anyhow::bail!("Quantized Phi-3 not supported")
                }
            };
            Model::Quantized(model)
        } else {
            let dtype = match &self.dtype {
                Some(dtype) => std::str::FromStr::from_str(&dtype)?,
                None => {
                    if self.model == WhichModel::V3 {
                        device.bf16_default_to_f32()
                    } else {
                        DType::F32
                    }
                }
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            match self.model {
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
        let api = Api::new()?;
        let model_id = match &self.model_id {
            Some(model_id) => model_id.to_string(),
            None => {
                if self.quantized {
                    "lmz/candle-quantized-phi".to_string()
                } else {
                    match self.model {
                        WhichModel::V2 => "microsoft/phi-2".to_string(),
                        WhichModel::V3 => "microsoft/Phi-3-mini-4k-instruct".to_string(),
                    }
                }
            }
        };
        let revision = match &self.revision {
            Some(rev) => rev.to_string(),
            None => {
                if self.quantized {
                    "main".to_string()
                } else {
                    match self.model {
                        WhichModel::V2 => "main".to_string(),
                        WhichModel::V3 => "main".to_string(),
                    }
                }
            }
        };
        Ok(api.repo(Repo::with_revision(model_id, RepoType::Model, revision)))
    }

}
