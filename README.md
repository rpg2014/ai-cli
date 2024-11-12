# AI CLI Text Generation Tool

A Rust-based command-line interface for text generation using Microsoft's Phi language models.

## Features

- Support for Phi-2 and Phi-3 language models
- Flexible deployment options:
  - Run on CPU or GPU
  - Quantized and non-quantized model support
- Configurable model parameters
  - Temperature
  - Repeat penalty
  - Sampling length
- Customizable logging levels
- Easy prompt-based text generation

## Prerequisites

- Rust programming language
- Cargo package manager

## Installation

```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli
cargo build --release
```

## Usage

### Basic Usage

```bash
# Generate text with a prompt
cargo run -- --prompt "Your text prompt here"

# Specify model version
cargo run -- --model 2 --prompt "Phi-2 prompt"
cargo run -- --model 3 --prompt "Phi-3 prompt"

# Run on CPU
cargo run -- --cpu --prompt "CPU-based generation"

# Use quantized model
cargo run -- --quantized --prompt "Quantized model generation"
```

### Configuration

Configuration can be set in `config.toml`:
- Model selection
- Model parameters
- Logging verbosity

## Command-line Options

- `--cpu`: Force CPU usage instead of GPU
- `--tracing`: Enable performance tracing
- `--prompt`: Input text prompt for generation
- `--model`: Model version (2 or 3)
- `--quantized`: Use quantized model
- `--verbose`: Set logging verbosity

## Model Support

- Phi-2: Non-quantized and quantized support
- Phi-3: Non-quantized support (experimental)

## License

[Add your project's license here]

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the project repository.
