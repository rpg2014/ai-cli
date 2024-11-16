use std::time::Instant;

use anyhow::{Error as E, Result};
use clap::ValueEnum;
use serde::Deserialize;
use tracing::info;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::mixformer::Config;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use super::common::AiBackend;
use crate::text_generation::{Model, TextGeneration};
use crate::AiCliArgs;
use crate::Settings;
use crate::{device, hub_load_safetensors};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Deserialize)]
pub enum WhichModel {
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
}

pub struct LocalAiBackend {
    settings: Settings,
    args: AiCliArgs,
    start: std::time::Instant,
}

impl LocalAiBackend {
    pub fn new(settings: Settings, args: AiCliArgs, start: Instant) -> Self {
        Self {
            settings,
            args,
            start,
        }
    }

    pub fn load_local_model(&self) -> Result<(Model, Tokenizer, Device)> {
        let repo = self.get_repo_for_local_model()?;
        let tokenizer_filename = match &self.settings.local_model_config.tokenizer {
            Some(file) => std::path::PathBuf::from(file),
            None => match self.settings.local_model_config.model {
                WhichModel::V2 | WhichModel::V3 => repo.get("tokenizer.json")?,
            },
        };
        let filenames = match &self.settings.local_model_config.weight_file {
            Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
            None => {
                if self.settings.local_model_config.quantized {
                    match self.settings.local_model_config.model {
                        WhichModel::V2 => vec![repo.get("model-v2-q4k.gguf")?],
                        WhichModel::V3 => anyhow::bail!(
                            "use the quantized or quantized-phi examples for quantized phi-v3"
                        ),
                    }
                } else {
                    match self.settings.local_model_config.model {
                        WhichModel::V2 => {
                            hub_load_safetensors(&repo, "model.safetensors.index.json")?
                        }
                        WhichModel::V3 => {
                            hub_load_safetensors(&repo, "model.safetensors.index.json")?
                        }
                    }
                }
            }
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let config = || match self.settings.local_model_config.model {
            WhichModel::V2 => Config::v2(),
            WhichModel::V3 => {
                panic!("use the quantized or quantized-phi examples for quantized phi-v3")
            }
        };
        let device = device(self.settings.local_model_config.cpu)?;
        let model = if self.settings.local_model_config.quantized {
            let config = config();
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &filenames[0],
                &device,
            )?;
            let model = match self.settings.local_model_config.model {
                WhichModel::V2 => QMixFormer::new_v2(&config, vb)?,
                WhichModel::V3 => {
                    anyhow::bail!("Quantized Phi-3 not supported")
                }
            };
            Model::Quantized(model)
        } else {
            let dtype = match &self.settings.local_model_config.dtype {
                Some(dtype) => dtype.parse()?,
                None => {
                    if self.settings.local_model_config.model == WhichModel::V3 {
                        device.bf16_default_to_f32()
                    } else {
                        DType::F32
                    }
                }
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            match self.settings.local_model_config.model {
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
        Ok((model, tokenizer, device))
    }

    fn get_repo_for_local_model(&self) -> Result<ApiRepo> {
        info!("Loading the model, parsing model from args and settings");
        let api = Api::new()?;
        let model_id = match &self.settings.local_model_config.model_id {
            Some(model_id) => model_id.to_string(),
            None => {
                if self.settings.local_model_config.quantized {
                    "lmz/candle-quantized-phi".to_string()
                } else {
                    match self.settings.local_model_config.model {
                        WhichModel::V2 => "microsoft/phi-2".to_string(),
                        WhichModel::V3 => "microsoft/Phi-3-mini-4k-instruct".to_string(),
                    }
                }
            }
        };
        let revision = match &self.settings.local_model_config.revision {
            Some(rev) => rev.to_string(),
            None => {
                if self.settings.local_model_config.quantized {
                    "main".to_string()
                } else {
                    match self.settings.local_model_config.model {
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

impl AiBackend for LocalAiBackend {
    fn invoke(&self, prompt: String) -> Result<String> {
        info!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        let (model, tokenizer, device) = self.load_local_model()?;
        info!("loaded the model in {:?}", self.start.elapsed());

        let mut pipeline = TextGeneration::new(
            model,
            tokenizer,
            self.settings.local_model_config.seed,
            self.settings.local_model_config.temperature,
            self.settings.local_model_config.top_p,
            self.settings.local_model_config.repeat_penalty,
            self.settings.local_model_config.repeat_last_n,
            self.settings.local_model_config.verbose_prompt,
            &device,
        );
        let mut string_buffer = std::io::Cursor::new(Vec::new());
        // Use tokio runtime to run the async method
        tokio::runtime::Runtime::new()?.block_on(async {
            // pass in string buffer stream into run function
            pipeline
                .run(
                    &prompt,
                    self.settings.local_model_config.sample_len,
                    &mut string_buffer,
                )
                .await
        })?;
        info!("generated the output in {:?}", self.start.elapsed());
        Ok(String::from_utf8(string_buffer.into_inner())?)
    }
}
