---
# this_file: src_docs/md/getting-started/configuration.md
title: "Configuration Guide"
description: "Complete guide to configuring Claif Knollm providers, routing strategies, and system preferences."
---

# Configuration Guide

Claif Knollm offers flexible configuration options to match your specific needs. This guide covers everything from basic setup to advanced customization.

## Configuration Methods

Knollm supports multiple configuration approaches:

1. **Environment Variables** - Quick setup for API keys
2. **Configuration Files** - Comprehensive settings management
3. **Runtime Configuration** - Dynamic configuration in code
4. **CLI Arguments** - Command-line overrides

## Environment Variables

### Basic Provider Setup

Set API keys as environment variables:

```bash
# Essential providers
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export GOOGLE_API_KEY="AIza..."

# Fast inference providers
export GROQ_API_KEY="gsk_..."
export DEEPSEEK_API_KEY="sk-..."
export CEREBRAS_API_KEY="csk-..."

# Additional providers
export MISTRAL_API_KEY="..."
export COHERE_API_KEY="..."
export AI21_API_KEY="..."
export HUGGINGFACE_API_KEY="..."
```

### Global Settings

Configure global behavior:

```bash
# Routing preferences
export KNOLLM_DEFAULT_STRATEGY="balanced"
export KNOLLM_FALLBACK_PROVIDERS="openai,anthropic,groq"

# Performance settings
export KNOLLM_CACHE_TTL="3600"
export KNOLLM_REQUEST_TIMEOUT="30"
export KNOLLM_MAX_RETRIES="3"

# Cost controls
export KNOLLM_MAX_COST_PER_REQUEST="0.10"
export KNOLLM_DAILY_BUDGET="50.00"
export KNOLLM_ALERT_THRESHOLD="40.00"
```

## Configuration Files

### Main Configuration File

Create `~/.config/knollm/config.toml`:

```toml
[general]
log_level = "INFO"
cache_directory = "~/.cache/knollm"
data_directory = "~/.local/share/knollm"

[routing]
strategy = "balanced"  # cost_optimized, speed_optimized, quality_optimized, balanced
fallback_providers = ["openai", "anthropic", "groq"]
enable_caching = true
cache_ttl = 3600
max_retries = 3
retry_backoff = 2.0

[costs]
max_cost_per_request = 0.10
daily_budget = 50.00
monthly_budget = 1500.00
alert_threshold = 0.80
currency = "USD"

[performance]
request_timeout = 30
concurrent_requests = 10
rate_limit_buffer = 0.1
enable_monitoring = true

# Provider-specific configurations
[providers.openai]
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
enabled = true
tier = "premium"
rate_limit = 60  # requests per minute
preferred_models = ["gpt-4o-mini", "gpt-4o"]
cost_multiplier = 1.0

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
base_url = "https://api.anthropic.com"
enabled = true
tier = "premium"
rate_limit = 40
preferred_models = ["claude-3-haiku", "claude-3-sonnet"]
cost_multiplier = 1.0

[providers.groq]
api_key_env = "GROQ_API_KEY"
base_url = "https://api.groq.com/openai/v1"
enabled = true
tier = "budget"
rate_limit = 100
preferred_models = ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]
cost_multiplier = 0.1  # Very cheap

[providers.google]
api_key_env = "GOOGLE_API_KEY"
base_url = "https://generativelanguage.googleapis.com/v1"
enabled = true
tier = "premium"
rate_limit = 60
preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
cost_multiplier = 0.8

# Disable providers you don't want to use
[providers.together]
enabled = false

[providers.huggingface]
enabled = false
```

### Project-Specific Configuration

Create `knollm.toml` in your project directory:

```toml
[routing]
strategy = "cost_optimized"
fallback_providers = ["groq", "deepseek", "openai"]

[costs]
max_cost_per_request = 0.01
daily_budget = 5.00

[providers.openai]
enabled = false  # Disable expensive provider

[providers.groq]
preferred_models = ["llama-3.1-8b-instant"]
```

## Runtime Configuration

### Programmatic Configuration

Configure Knollm directly in your Python code:

```python
from claif_knollm import KnollmClient, RoutingStrategy, KnollmConfig

# Create custom configuration
config = KnollmConfig(
    routing_strategy=RoutingStrategy.COST_OPTIMIZED,
    fallback_providers=["groq", "deepseek", "openai"],
    max_cost_per_request=0.05,
    enable_caching=True,
    cache_ttl=1800
)

# Initialize client with custom config
client = KnollmClient(config=config)
```

### Dynamic Provider Configuration

Add or modify providers at runtime:

```python
from claif_knollm import ProviderConfig, AuthType

# Configure a custom provider
custom_provider = ProviderConfig(
    name="custom_provider",
    base_url="https://api.custom.com/v1",
    auth_type=AuthType.API_KEY,
    api_key="your-api-key",
    rate_limit=100,
    enabled=True
)

# Add to client
client.add_provider(custom_provider)
```

## Routing Strategies

Configure how Knollm selects providers:

### Strategy Types

| Strategy | Description | Best For |
|----------|-------------|----------|
| `cost_optimized` | Always chooses cheapest option | Development, high-volume |
| `speed_optimized` | Prioritizes fastest response | Real-time applications |
| `quality_optimized` | Selects highest-quality models | Production, important tasks |
| `balanced` | Balances cost, speed, quality | General purpose |
| `round_robin` | Distributes load evenly | Load testing, fairness |
| `adaptive` | Learns from usage patterns | Long-running applications |

### Strategy Configuration

```toml
[routing]
strategy = "adaptive"

[routing.cost_optimized]
max_acceptable_cost = 0.02
quality_threshold = 0.6

[routing.speed_optimized]
max_acceptable_latency = 500  # milliseconds
quality_threshold = 0.7

[routing.quality_optimized]
min_quality_score = 0.9
max_acceptable_cost = 0.20

[routing.balanced]
cost_weight = 0.4
speed_weight = 0.3
quality_weight = 0.3

[routing.adaptive]
learning_rate = 0.1
memory_window = 1000  # requests
exploration_rate = 0.05
```

## Cost Management

### Budget Controls

Set spending limits and alerts:

```toml
[costs]
# Absolute limits
max_cost_per_request = 0.50
hourly_budget = 10.00
daily_budget = 100.00
monthly_budget = 2000.00

# Alert thresholds (as percentage of budget)
warning_threshold = 0.75
alert_threshold = 0.90
critical_threshold = 0.95

# Cost tracking
enable_cost_tracking = true
cost_log_file = "~/.local/share/knollm/costs.json"
cost_report_interval = "daily"  # daily, weekly, monthly

# Currency and formatting
currency = "USD"
decimal_places = 4
```

### Provider Cost Modifiers

Adjust relative costs for decision-making:

```toml
[providers.openai]
cost_multiplier = 1.0  # Baseline

[providers.anthropic] 
cost_multiplier = 1.1  # Slightly more expensive

[providers.groq]
cost_multiplier = 0.1  # Very cheap

[providers.together]
cost_multiplier = 0.3  # Budget option
```

## Security Configuration

### API Key Management

Secure API key handling:

```toml
[security]
# API key storage
encrypt_config = true
config_password_env = "KNOLLM_CONFIG_PASSWORD"
keyring_service = "knollm"

# Request security
verify_ssl = true
ca_bundle_path = "/etc/ssl/certs/ca-certificates.crt"
timeout = 30
max_redirects = 5

# Privacy
log_request_bodies = false
log_response_bodies = false
mask_api_keys_in_logs = true
```

### Rate Limiting

Configure rate limiting to respect provider limits:

```toml
[rate_limiting]
enable_global_rate_limiting = true
global_rate_limit = 1000  # requests per minute
burst_limit = 100
backoff_strategy = "exponential"
max_backoff = 300  # seconds

[providers.openai]
rate_limit = 60
burst_allowance = 10

[providers.anthropic]
rate_limit = 40
burst_allowance = 5
```

## Logging and Monitoring

### Logging Configuration

```toml
[logging]
level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
format = "detailed"  # simple, detailed, json
file_path = "~/.local/share/knollm/knollm.log"
max_file_size = "10MB"
backup_count = 5

# Component-specific logging
[logging.components]
routing = "DEBUG"
providers = "INFO"
costs = "INFO"
cache = "WARNING"
```

### Monitoring

```toml
[monitoring]
enable_metrics = true
metrics_port = 8080
metrics_path = "/metrics"

# Performance tracking
track_latency = true
track_costs = true
track_error_rates = true
track_cache_hits = true

# Health checks
health_check_interval = 60  # seconds
provider_health_checks = true
```

## CLI Configuration

### Global CLI Settings

```toml
[cli]
default_output_format = "table"  # table, json, yaml, csv
color_output = true
pager = "auto"  # auto, never, always
editor = "vim"

[cli.table]
max_width = 120
show_headers = true
grid_style = "rounded"

[cli.json]
indent = 2
sort_keys = true
```

### Command Aliases

```toml
[cli.aliases]
ls = "models search"
find = "models search"
cheap = "models cheapest"
compare = "models compare"
providers = "providers list"
```

## Advanced Configuration

### Custom Model Filters

Define reusable model filters:

```toml
[filters.coding]
required_capabilities = ["code_generation", "function_calling"]
max_cost_per_1k_tokens = 0.02
min_context_window = 32000

[filters.vision]
required_capabilities = ["vision", "multimodal"]
min_quality_score = 0.8

[filters.budget]
max_cost_per_1k_tokens = 0.005
exclude_providers = ["openai"]
```

### Plugin Configuration

Configure third-party plugins:

```toml
[plugins]
enabled = ["cost_tracker", "performance_monitor"]

[plugins.cost_tracker]
export_format = "csv"
export_interval = "daily"
export_path = "~/.local/share/knollm/costs/"

[plugins.performance_monitor]
track_all_requests = true
alert_on_failures = true
```

## Configuration Validation

Validate your configuration:

```bash
# Check configuration syntax
knollm config validate

# Test provider connections
knollm config test-providers

# Show effective configuration
knollm config show
```

## Troubleshooting

### Common Configuration Issues

**Problem:** "Provider not found" errors
**Solution:** Check provider name spelling and ensure it's enabled:

```toml
[providers.openai]
enabled = true
```

**Problem:** API key not found
**Solution:** Verify environment variable names and values:

```bash
echo $OPENAI_API_KEY
```

**Problem:** Budget exceeded errors
**Solution:** Check and adjust budget settings:

```toml
[costs]
daily_budget = 100.00  # Increase as needed
```

### Configuration Precedence

Settings are applied in this order (later overrides earlier):

1. Default values
2. Global config file (`~/.config/knollm/config.toml`)
3. Project config file (`./knollm.toml`)
4. Environment variables
5. Runtime configuration
6. CLI arguments

## Next Steps

With Knollm configured:

1. **[Quick Start →](quickstart.md)** - Build your first application
2. **[Provider Guide →](../providers/)** - Learn about available providers
3. **[Cost Optimization →](../guides/optimization.md)** - Minimize expenses
4. **[Best Practices →](../guides/best-practices.md)** - Production tips

---

<div class="admonition tip">
<p class="admonition-title">💡 Configuration Tips</p>
<ul>
<li>Start with the <code>balanced</code> strategy for general use</li>
<li>Use <code>cost_optimized</code> for development and testing</li>
<li>Always set budget limits to avoid unexpected charges</li>
<li>Enable monitoring in production environments</li>
</ul>
</div>