# AI CLI: Natural Language to Bash One-Liner Generator

A flexible command-line tool that transforms natural language prompts into executable bash one-liners using Large Language Models (LLMs).

## Features

- Generate bash one-liners from natural language prompts
- Support for multiple AI backends
  - Local models
  - AWS Bedrock
- Logging and tracing
- Configurable model parameters
- Cross-platform compatibility (CPU/GPU)

## Prerequisites

- Rust programming language
- Cargo package manager
- (Optional) AWS Bedrock credentials

## Installation

```bash
git clone https://github.com/rpg2014/ai-cli.git
cd ai-cli
cargo install --path .
```
### Optional features

- `accelerate`: Enable GPU acceleration using the Accelerate library for improved performance - Mac only
- `mkl`: Use Intel Math Kernel Library (MKL) for optimized computational performance
- `metal`: Enable GPU acceleration on Apple devices using Metal - Mac only
- `clipboard`: Automatically copy the generated bash one-liner to your system clipboard

#### Using optional features
Install the cli with the following command with the features you want:

```bash
cargo install --path . --features metal,clipboard
```

## Usage

### Basic Usage

```bash
# Generate a bash one-liner from a natural language prompt
ai create a directory and list its contents

# Set verbose logging (Defaults to Error, each v drops it down a level (Warn, info, debug, trace))
ai -vv your prompt

# Enable performance tracing.  Generates a trace-timestamp.json file that can be loaded into Chrome
ai --tracing "your prompt"
```

## Available Commands

### Generate
Generate a bash one-liner based on a natural language prompt.

```bash
# Basic generation
ai create a backup of all txt files in the current directory

# With additional options
ai --backend local list all running docker containers

# Explictly specify the generate command
ai generate list all files in the directory from largest to smallest
```

### Config
Print the current settings, arguments, and log verbosity.

```bash
# Display current configuration
ai config
```

## Configuration

Configuration can be customized in `~/.config/ai/config.toml`:
- AI backend selection
- Model parameters
- Logging settings

A default config file is written when first launched.  The configuration can also be overridden on a per project bases by putting a `config.toml` file in the current directory.

## Command-line Options

- `--verbose`: Set logging verbosity
- `--tracing`: Enable performance tracing
- `--backend`: Select AI backend (local/bedrock)

## Supported Backends

- Local AI Models
- AWS Bedrock

## Performance

- Supports CPU and GPU execution
- Chrome tracing for performance analysis

## License

[Add your project's license here]

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the project repository.
