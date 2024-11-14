use anyhow::Result;

pub trait AiBackend {
    fn invoke(&self) -> Result<String>;
}
