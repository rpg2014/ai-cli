use std::time::Instant;

use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseStreamOutput, Message, SystemContentBlock,
};
use aws_sdk_bedrockruntime::Client;

use tracing::{debug, info};

use crate::command::WhichModel;
use crate::constants::SYSTEM_PROMPT;
use crate::text_generation::{Model, TextGeneration};
use crate::AiCliArgs;
use crate::Settings;
use crate::{device, hub_load_safetensors};
use anyhow::{Error as E, Result};

use aws_config::{BehaviorVersion, Region};
use candle_core::{DType, Device};
use candle_transformers::models::mixformer::Config;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use clap::Parser;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use candle_nn::VarBuilder;
pub trait AiBackend {
    fn invoke(&self) -> Result<String>;
}

pub struct BedrockAiBackend {
    settings: Settings,
    args: AiCliArgs,
    start: std::time::Instant,
}
impl BedrockAiBackend {
    pub fn new(settings: Settings, args: AiCliArgs, start: Instant) -> Self {
        Self {
            settings,
            args,
            start,
        }
    }
    fn get_converse_output_text(
        output: ConverseStreamOutput,
    ) -> Result<String, ConverseStreamOutputError> {
        Ok(match output {
            ConverseStreamOutput::ContentBlockDelta(event) => match event.delta() {
                Some(delta) => {
                    debug!("{:?}",delta);
                    delta.as_text().cloned().unwrap_or_else(|_| "".into())
                },
                None => "".into(),
            },
            // rest log and return empty string
            ConverseStreamOutput::MessageStart(e) => {
                debug!("MessageStart: {:?}", e);
                "".into()
            },
            ConverseStreamOutput::MessageStop(e) => {
                debug!("MessageStop: {:?}", e);
                "".into()
            },
            ConverseStreamOutput::Metadata(e) =>{
                debug!("Metadata: {:?}", e);
                "".into()
            },
            ConverseStreamOutput::ContentBlockStart(e) => {
                debug!("ContentBlockStart: {:?}", e);
                "".into()
            },
            ConverseStreamOutput::ContentBlockStop(e) => {
                debug!("ContentBlockStop: {:?}", e);
                "".into()
            },
            _ => {
                debug!("Received non-content block delta");
                "".into()
            },
        })
    }
}
impl AiBackend for BedrockAiBackend {
    fn invoke(&self) -> Result<String> {
        // Clone the necessary fields to move into the async block
        let prompt = self.args.prompt.clone();
        let region = String::from(self.settings.aws_settings.region.as_str());
        info!("Prompt input is: {}", prompt);
        info!("Using region: {}", region);

        let result = tokio::runtime::Runtime::new()?.block_on(async {
            let sdk_config = aws_config::defaults(BehaviorVersion::latest())
                .region(Region::new(region))
                .load()
                .await;
            info!("Creating bedrock client");
            let client = Client::new(&sdk_config);
            info!("Client created");
            let response = client
                .converse_stream()
                .model_id("anthropic.claude-3-haiku-20240307-v1:0")
                .messages(
                    Message::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(prompt))
                        .build()
                        .map_err(|_| anyhow::anyhow!("failed to build message"))?,
                )
                .set_system(Some(vec![SystemContentBlock::Text(SYSTEM_PROMPT.to_string())]))
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send message: {:?}", e))?;
            info!("Response received");
            let mut stream = response.stream;

            let mut response_text = String::new();
            info!("Starting response stream");
            loop {
                let token = stream.recv().await;
                match token {
                    Ok(Some(text)) => {
                        debug!("Received token");
                        let next = BedrockAiBackend::get_converse_output_text(text);
                        match next {
                            Ok(text) => {
                                debug!("{}", text);
                                response_text.push_str(&text);
                            }
                            Err(e) => {
                                let string_clone = e
                                    .meta()
                                    .message()
                                    .unwrap_or_else(|| "Unable to see stream error message")
                                    .to_string();
                                return Err(anyhow::anyhow!(string_clone));
                            }
                        }
                    }
                    // means the stream is complete
                    Ok(None) => break,
                    Err(e) => {
                        if let Some(error) = e.as_service_error() {
                            return Err(anyhow::anyhow!(error
                                .meta()
                                .message()
                                .unwrap_or("Unable to open stream error message")
                                .to_string()));
                        }
                        anyhow::bail!("Unable to see stream error message");
                    }
                }
            }
            // Since this is a stream, you might want to collect the response
            // This is a placeholder - you'll need to implement actual response handling
            Ok(response_text)
        })?;
        

        Ok(result)
    }
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

impl AiBackend for LocalAiBackend {
    fn invoke(&self) -> Result<String> {
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
            self.settings.model_config.seed,
            self.settings.model_config.temperature,
            self.settings.model_config.top_p,
            self.settings.model_config.repeat_penalty,
            self.settings.model_config.repeat_last_n,
            self.settings.model_config.verbose_prompt,
            &device,
        );
        let mut string_buffer = std::io::Cursor::new(Vec::new());
        // Use tokio runtime to run the async method
        tokio::runtime::Runtime::new()?.block_on(async {
            // pass in string buffer stream into run function

            pipeline
                .run(
                    &self.args.prompt,
                    self.settings.model_config.sample_len,
                    &mut string_buffer,
                )
                .await
        })?;
        info!("generated the output in {:?}", self.start.elapsed());
        Ok(String::from_utf8(string_buffer.into_inner())?)
    }
}
