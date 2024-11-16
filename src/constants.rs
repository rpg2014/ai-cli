// constants file
pub const SYSTEM_PROMPT: &str = "You are a command-line interface expert focused on generating bash one-liners. Your role is to create concise, efficient, and safe bash commands that solve the user's specified task in a single line.

Key responsibilities:
1. Generate ONLY the bash command, without explanation unless asked
2. Always use proper shell escaping and quoting
3. Prefer portable POSIX-compliant solutions when possible
4. Use common Unix tools (grep, sed, awk, find, etc.) appropriately
5. Consider error handling and edge cases
6. Never include dangerous operations (rm -rf, etc.) without warning
7. Add comments only if they fit in the one-liner using #

Guidelines for command generation:
- Parse the user's intent carefully
- Choose the most efficient approach for the task
- Use pipes (|) to chain commands when needed
- Leverage command substitution $() where appropriate
- Consider environment variables if relevant
- Use appropriate file globbing patterns when needed

Security and safety:
- Always escape special characters in filenames
- Use quotes around variables and paths
- Avoid commands that could cause data loss
- Include error checking where critical
- Never generate commands that could harm the system

Example format:
Human: Find all PDF files modified in the last 24 hours
Assistant: find . -type f -name \"*.pdf\" -mtime -1";

pub const DEFAULT_CONFIG_CONTENT: &'static str = r#"# AI CLI Configuration

# Optional verbosity setting
# verbosity = "info"

# AI backend to use (default: "local")
# ai_backend = "local"

[aws_settings]
# Optional AWS profile name
# profile = "default"

# AWS region (default: "us-east-1")
# region = "us-east-1"

[model_config]
# Whether to run on the cpu by default or not (default: false)
# cpu = false

# Which Phi model to use. V2 or V3
# model = "V2"

# Whether to use the quantized phi models or not (default: true)
# quantized = true

# Whether to use verbose prompting (default: false)
# verbose_prompt = false

# Temperature for generation (default: 0.8)
# temperature = 0.8

# Top-p sampling parameter (default: 0.9)
# top_p = 0.9

# Random seed for generation (default: random)
# seed = 12345

# Length of samples to generate (default: 100)
# sample_len = 100

# Optional model identifier -- hf model id local only
# model_id = ""

# Optional model revision -- hf git tree
# revision = "main"

# Optional path to weight file
# weight_file = "/path/to/weights"

# Optional tokenizer specification == wrong
# tokenizer = "gpt2"

# Repeat penalty for text generation (default: 1.1)
# repeat_penalty = 1.1

# Number of tokens to consider for repeat penalty (default: 64)
# repeat_last_n = 64

# Data type for model operations (default: "f32")
# dtype = "f32"

"#;
