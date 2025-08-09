---
# this_file: src_docs/md/getting-started/installation.md
title: "Installation Guide"
description: "Complete installation guide for Claif Knollm with all dependencies and optional features."
---

# Installation Guide

Get Claif Knollm installed and running on your system in just a few minutes.

## System Requirements

Claif Knollm works on all major platforms and requires:

- **Python 3.11 or later** (3.12+ recommended for best performance)
- **pip** or **uv** package manager
- **Internet connection** for provider API access

### Supported Platforms

- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+, Alpine 3.14+)
- ✅ **macOS** (10.15 Catalina+)
- ✅ **Windows** (Windows 10+, Windows Server 2019+)

## Quick Installation

### Standard Installation

The fastest way to get started:

```bash
pip install claif-knollm
```

Or using the modern `uv` package manager (recommended):

```bash
uv pip install claif-knollm
```

### Installation with CLI Tools

For full CLI functionality with rich formatting:

```bash
pip install claif-knollm[cli]
```

### Development Installation

If you want to contribute or need the latest features:

```bash
pip install claif-knollm[dev]
```

### Full Installation

For all features including documentation and testing tools:

```bash
pip install claif-knollm[all]
```

## Verify Installation

Test that Claif Knollm is properly installed:

```bash
python -c "from claif_knollm import ModelRegistry; print('✅ Knollm installed successfully!')"
```

Or test the CLI:

```bash
knollm --version
```

You should see output like:

```
claif-knollm 1.0.2
```

## Package Extras

Claif Knollm offers several optional feature sets:

| Extra | Description | Install Command |
|-------|-------------|-----------------|
| `cli` | Rich CLI interface with formatting and colors | `pip install claif-knollm[cli]` |
| `dev` | Development tools (testing, linting, formatting) | `pip install claif-knollm[dev]` |
| `docs` | Documentation building tools | `pip install claif-knollm[docs]` |
| `all` | All optional features combined | `pip install claif-knollm[all]` |

### Core Dependencies

The base installation includes:

- **pydantic** ≥ 2.5.0 - Data validation and type safety
- **httpx** ≥ 0.25.0 - Async HTTP client
- **pyyaml** ≥ 6.0.0 - Configuration file support
- **typing-extensions** ≥ 4.8.0 - Enhanced type hints

### CLI Dependencies (`[cli]`)

Additional dependencies for the CLI:

- **rich** ≥ 13.7.0 - Rich text and beautiful formatting
- **fire** ≥ 0.5.0 - Automatic CLI generation
- **click** ≥ 8.1.0 - CLI framework
- **tabulate** ≥ 0.9.0 - Table formatting

## Environment Setup

### API Keys Configuration

Claif Knollm needs API keys for the providers you want to use. Set them as environment variables:

```bash
# Core providers
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Fast providers
export GROQ_API_KEY="your-groq-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export CEREBRAS_API_KEY="your-cerebras-key"

# Other providers (optional)
export MISTRAL_API_KEY="your-mistral-key"
export COHERE_API_KEY="your-cohere-key"
export AI21_API_KEY="your-ai21-key"
```

### Shell Configuration

Add API keys to your shell profile for persistence:

=== "Bash/Zsh"

    Add to `~/.bashrc` or `~/.zshrc`:
    
    ```bash
    # Claif Knollm API Keys
    export OPENAI_API_KEY="your-openai-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
    export GOOGLE_API_KEY="your-google-key"
    # ... add more as needed
    ```

=== "Fish"

    Add to `~/.config/fish/config.fish`:
    
    ```fish
    # Claif Knollm API Keys
    set -gx OPENAI_API_KEY "your-openai-key"
    set -gx ANTHROPIC_API_KEY "your-anthropic-key" 
    set -gx GOOGLE_API_KEY "your-google-key"
    # ... add more as needed
    ```

=== "PowerShell"

    Add to your PowerShell profile:
    
    ```powershell
    # Claif Knollm API Keys
    $env:OPENAI_API_KEY = "your-openai-key"
    $env:ANTHROPIC_API_KEY = "your-anthropic-key"
    $env:GOOGLE_API_KEY = "your-google-key"
    # ... add more as needed
    ```

### Configuration File

Alternatively, create a configuration file at `~/.config/knollm/config.toml`:

```toml
[providers.openai]
api_key = "your-openai-key"
enabled = true

[providers.anthropic]
api_key = "your-anthropic-key"
enabled = true

[providers.google]
api_key = "your-google-key"
enabled = true

[routing]
strategy = "balanced"
fallback_providers = ["openai", "anthropic", "groq"]
```

## Installation Troubleshooting

### Common Issues

#### Python Version Too Old

**Error:** `Python 3.11+ is required`

**Solution:** Update Python to 3.11 or later:

```bash
# Using pyenv (recommended)
pyenv install 3.12.0
pyenv global 3.12.0

# Using conda
conda install python=3.12

# Using system package manager (Ubuntu)
sudo apt update
sudo apt install python3.12
```

#### Package Installation Fails

**Error:** `Failed building wheel for claif-knollm`

**Solution:** Install build dependencies:

```bash
# Ubuntu/Debian
sudo apt install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc

# macOS (with Homebrew)
brew install python-dev

# Then retry installation
pip install --upgrade pip setuptools wheel
pip install claif-knollm
```

#### Import Errors

**Error:** `ModuleNotFoundError: No module named 'claif_knollm'`

**Solution:** Ensure you're using the right Python environment:

```bash
# Check which Python you're using
which python
python --version

# Check installed packages
pip list | grep claif-knollm

# If needed, reinstall in the correct environment
pip uninstall claif-knollm
pip install claif-knollm
```

#### CLI Not Found

**Error:** `knollm: command not found`

**Solution:** Ensure the CLI is installed and in your PATH:

```bash
# Install with CLI support
pip install claif-knollm[cli]

# Check if it's in PATH
which knollm

# If not found, add pip's bin directory to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs** - Run with `--verbose` flag for detailed output
2. **Search issues** - Look through [GitHub Issues](https://github.com/twardoch/claif_knollm/issues)
3. **Create an issue** - Report bugs or request help
4. **Join discussions** - Participate in [GitHub Discussions](https://github.com/twardoch/claif_knollm/discussions)

## Upgrade Instructions

### Upgrading from Previous Versions

To upgrade to the latest version:

```bash
pip install --upgrade claif-knollm
```

### Version-Specific Upgrade Notes

#### Upgrading to 1.0.0+

Breaking changes:
- Configuration format changed from YAML to TOML
- Provider names are now lowercase (e.g., `OpenAI` → `openai`)
- Some CLI commands have new syntax

Migration steps:
1. Update configuration files to new TOML format
2. Update provider names in your code
3. Review CLI scripts for syntax changes

#### Upgrading to 0.9.0+

- New dependency: `pydantic` v2.0+
- Python 3.11+ now required
- CLI interface redesigned with new commands

## Development Setup

For contributors or advanced users who want to install from source:

```bash
# Clone the repository
git clone https://github.com/twardoch/claif_knollm.git
cd claif_knollm

# Install in development mode
pip install -e .[dev]

# Or using uv (recommended)
uv pip install -e .[dev]

# Run tests to verify installation
python -m pytest tests/
```

## Next Steps

Now that Claif Knollm is installed:

1. **[Quick Start →](quickstart.md)** - Build your first application
2. **[Configuration →](configuration.md)** - Set up providers and preferences
3. **[Provider Guide →](../providers/)** - Learn about available providers
4. **[API Reference →](../api/)** - Dive into the technical details

---

<div class="admonition success">
<p class="admonition-title">🎉 Installation Complete!</p>
<p>Claif Knollm is now ready to use. Continue to the <a href="quickstart/">Quick Start guide</a> to build your first application.</p>
</div>