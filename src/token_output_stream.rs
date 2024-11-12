use candle_core::Result;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
/// The struct maintains state about the current tokenization process including:
/// - The tokenizer instance
/// - Vector of processed tokens
/// - Previous and current indices for tracking progress
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    /// Creates a new TokenOutputStream with the given tokenizer
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    /// Consumes self and returns the underlying tokenizer
    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    /// Helper function to decode a slice of tokens into a String
    /// Returns an error if decoding fails
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    /// Processes the next token in the stream
    /// Returns Some(String) if a complete word is formed, None otherwise
    /// Implementation based on Hugging Face's text-generation-inference https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        // Get previously decoded text 
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        // Add new token and decode
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        // Return new complete word if one is formed
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Decodes any remaining tokens that haven't formed complete words yet
    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Decodes all tokens processed so far into a single String
    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    /// Looks up the token ID for a given string in the vocabulary
    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    /// Returns a reference to the underlying tokenizer
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Resets the stream state by clearing tokens and indices
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
