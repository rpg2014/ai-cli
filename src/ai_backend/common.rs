use anyhow::Result;

pub trait AiBackend {
    fn invoke(&self, prompt: String) -> Result<String>;
}
