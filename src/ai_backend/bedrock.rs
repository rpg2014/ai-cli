use std::time::Instant;

use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseStreamOutput, Message, SystemContentBlock,
};
use aws_sdk_bedrockruntime::Client;
use aws_config::{BehaviorVersion, Region};

use tracing::{debug, info};
use anyhow::Result;

use crate::constants::SYSTEM_PROMPT;
use crate::AiCliArgs;
use crate::Settings;
use super::common::AiBackend;

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
            Ok(response_text)
        })?;

        Ok(result)
    }
}
