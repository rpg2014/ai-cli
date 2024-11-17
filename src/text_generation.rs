use crate::token_output_stream;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
// use candle_transformers::models::mixformer::MixFormerSequentialForCausalLM as MixFormer;
use candle_transformers::models::phi::Model as Phi;
use candle_transformers::models::phi3::Model as Phi3;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

pub enum Model {
    // MixFormer(MixFormer),
    Phi(Phi),
    Phi3(Phi3),
    Quantized(QMixFormer),
}

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    /// Async runs the text generation model on the given prompt for a specified number of tokens
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt to generate from
    /// * `sample_len` - Maximum number of tokens to generate
    /// * `stream` - An async channel or stream to send generated tokens
    pub async fn run<S>(&mut self, prompt: &str, sample_len: usize, stream: &mut S) -> Result<()>
    where
        S: tokio::io::AsyncWrite + Unpin,
    {
        // Encode the prompt text into tokens
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?;
        debug!("Encoded tokens: {tokens:?}");
        // Check for empty prompts which are not supported
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }

        // Print verbose token information if enabled
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }

        // Initialize token tracking
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;

        // Get the end of text token
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };

        // Write initial prompt to stream
        stream.write_all(prompt.as_bytes()).await?;

        // Track generation time and position
        let start_gen = std::time::Instant::now();
        let mut pos = 0;

        // Main generation loop
        for index in 0..sample_len {
            // Get context size - full context for first iteration, single token after
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

            // Prepare input tensor
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            // Get logits from the appropriate model
            let logits = match &mut self.model {
                // Model::MixFormer(m) => m.forward(&input)?,
                Model::Phi(m) => m.forward(&input)?,
                Model::Quantized(m) => m.forward(&input)?,
                Model::Phi3(m) => m.forward(&input, pos)?.i((.., 0, ..))?,
            };

            // Process logits
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            // Apply repeat penalty if configured
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            // Sample next token
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            // Check for end of text
            if next_token == eos_token {
                if let Some(t) = self.tokenizer.decode_rest()? {
                    stream.write_all(t.as_bytes()).await?;
                }
                break;
            }

            // Write generated token to stream
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                stream.write_all(t.as_bytes()).await?;
            }
            pos += context_size;
        }

        // Flush the stream to ensure all data is written
        stream.flush().await?;

        // Print generation statistics
        let dt = start_gen.elapsed();
        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}
