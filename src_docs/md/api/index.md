---
# this_file: src_docs/md/api/index.md
title: "API Reference"
description: "Complete API documentation for Claif Knollm with examples, type hints, and detailed method descriptions."
---

# API Reference

Complete technical documentation for the Claif Knollm Python API. This reference covers all classes, methods, and data structures with detailed examples and type information.

## Core API Components

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } **Client API**

    ---

    Main `KnollmClient` class for making requests with intelligent routing and failover.

    [:octicons-arrow-right-24: Client Reference](client.md)

-   :material-database:{ .lg .middle } **Registry API**

    ---

    Model and provider registries for searching, filtering, and managing LLM resources.

    [:octicons-arrow-right-24: Registry Reference](registry.md)

-   :material-route:{ .lg .middle } **Routing API**

    ---

    Intelligent routing engine with strategies for cost, quality, and speed optimization.

    [:octicons-arrow-right-24: Routing Reference](routing.md)

-   :material-wrench:{ .lg .middle } **Utilities**

    ---

    Helper functions, data models, and utility classes for common operations.

    [:octicons-arrow-right-24: Utilities Reference](utilities.md)

</div>

## Quick API Overview

### Main Classes

| Class | Purpose | Import Path |
|-------|---------|-------------|
| [`KnollmClient`](client.md#knollmclient) | Main client for API requests | `claif_knollm.KnollmClient` |
| [`ModelRegistry`](registry.md#modelregistry) | Model database and search | `claif_knollm.ModelRegistry` |
| [`ProviderRegistry`](registry.md#providerregistry) | Provider management | `claif_knollm.ProviderRegistry` |
| [`RoutingEngine`](routing.md#routingengine) | Request routing logic | `claif_knollm.RoutingEngine` |

### Key Data Models

| Model | Purpose | Import Path |
|-------|---------|-------------|
| [`Model`](utilities.md#model) | LLM model representation | `claif_knollm.models.Model` |
| [`Provider`](utilities.md#provider) | Provider configuration | `claif_knollm.models.Provider` |
| [`SearchFilter`](utilities.md#searchfilter) | Search parameters | `claif_knollm.models.SearchFilter` |
| [`CompletionRequest`](utilities.md#completionrequest) | Request parameters | `claif_knollm.models.CompletionRequest` |

### Enumerations

| Enum | Purpose | Values |
|------|---------|--------|
| [`RoutingStrategy`](routing.md#routingstrategy) | Routing algorithms | `COST_OPTIMIZED`, `QUALITY_OPTIMIZED`, etc. |
| [`ModelCapability`](utilities.md#modelcapability) | Model capabilities | `TEXT_GENERATION`, `VISION`, etc. |
| [`ProviderTier`](utilities.md#providertier) | Provider categories | `FREE`, `BUDGET`, `PREMIUM`, etc. |

## Basic Usage Example

Here's a complete example showing the main API components:

```python
from claif_knollm import (
    KnollmClient,
    ModelRegistry, 
    RoutingStrategy,
    SearchFilter,
    ModelCapability
)
from decimal import Decimal

# Initialize the model registry
registry = ModelRegistry()

# Search for suitable models
search_filter = SearchFilter(
    required_capabilities=[ModelCapability.CHAT_COMPLETION],
    max_cost_per_1k_tokens=Decimal("0.01"),
    min_quality_score=0.8,
    active_only=True,
    limit=5
)

models = registry.search_models(search_filter)
print(f"Found {len(models.models)} suitable models")

# Initialize client with routing strategy
client = KnollmClient(
    routing_strategy=RoutingStrategy.BALANCED,
    fallback_providers=["openai", "anthropic", "groq"],
    enable_caching=True,
    cache_ttl=3600
)

# Make a request
response = await client.create_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing briefly."}
    ],
    max_tokens=150,
    temperature=0.7
)

print(f"Response from {response.provider}/{response.model}:")
print(response.content)
print(f"Cost: ${response.cost:.6f}")
```

## API Design Principles

### Type Safety

All API methods include comprehensive type hints:

```python
from typing import List, Optional, Union, Dict, Any
from decimal import Decimal

async def create_completion(
    self,
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs: Any
) -> CompletionResponse:
    """Create a chat completion with intelligent routing."""
```

### Async/Await Support

All network operations are async by default:

```python
# Async operations
response = await client.create_completion(messages)
models = await registry.search_models_async(filter)
status = await client.check_provider_health("openai")

# Sync alternatives available
response = client.create_completion_sync(messages)
models = registry.search_models(filter)
```

### Error Handling

Comprehensive error hierarchy with specific exception types:

```python
from claif_knollm.exceptions import (
    KnollmError,           # Base exception
    ProviderError,         # Provider-specific errors  
    ModelNotFoundError,    # Model not available
    RateLimitError,        # Rate limiting
    CostExceededError,     # Budget limits
    RoutingError          # Routing failures
)

try:
    response = await client.create_completion(messages)
except RateLimitError as e:
    print(f"Rate limited by {e.provider}, retrying in {e.retry_after}s")
except CostExceededError as e:
    print(f"Request would cost ${e.cost:.4f}, exceeds limit of ${e.limit:.4f}")
except ProviderError as e:
    print(f"Provider {e.provider} error: {e.message}")
```

### Configuration Options

Flexible configuration through multiple methods:

```python
# 1. Constructor parameters
client = KnollmClient(
    routing_strategy=RoutingStrategy.COST_OPTIMIZED,
    max_cost_per_request=0.10,
    fallback_providers=["groq", "deepseek"]
)

# 2. Configuration objects
from claif_knollm import KnollmConfig

config = KnollmConfig(
    routing_strategy="balanced",
    enable_caching=True,
    cache_ttl=3600
)
client = KnollmClient(config=config)

# 3. Environment variables
# KNOLLM_ROUTING_STRATEGY=cost_optimized
# KNOLLM_MAX_COST_PER_REQUEST=0.05
client = KnollmClient()  # Reads from environment

# 4. Configuration files
# ~/.config/knollm/config.toml
client = KnollmClient.from_config_file()
```

## Response Objects

### CompletionResponse

All completion requests return a structured response:

```python
@dataclass
class CompletionResponse:
    """Response from a completion request."""
    
    content: str                    # Generated text
    model: str                      # Actual model used  
    provider: str                   # Provider that handled request
    usage: TokenUsage              # Token consumption details
    cost: Optional[Decimal]        # Request cost in USD
    latency: float                 # Response time in seconds
    metadata: Dict[str, Any]       # Additional provider data
    cached: bool                   # Whether response was cached
    
    # Quality metrics
    confidence_score: Optional[float]
    safety_score: Optional[float]
```

### SearchResult

Model search operations return paginated results:

```python
@dataclass  
class SearchResult:
    """Result from a model search operation."""
    
    models: List[Model]            # Matching models
    providers: List[Provider]      # Associated providers
    total_count: int              # Total matches (before pagination)
    page_size: int                # Results per page
    page_offset: int              # Current page offset
    search_time_ms: float         # Search duration
    filters_applied: SearchFilter # Original search criteria
```

## Authentication and Security

### API Key Management

Secure handling of provider API keys:

```python
from claif_knollm.auth import APIKeyManager

# Load from environment variables
key_manager = APIKeyManager.from_environment()

# Load from secure keyring
key_manager = APIKeyManager.from_keyring("knollm")

# Manual configuration
key_manager = APIKeyManager({
    "openai": "sk-...",
    "anthropic": "sk-ant-...",
    "google": "AIza..."
})

client = KnollmClient(api_key_manager=key_manager)
```

### Request Security

All requests include security features:

- **SSL Certificate Verification** - Enabled by default
- **Request Timeouts** - Configurable per request
- **Rate Limiting** - Automatic backoff and retry
- **API Key Masking** - Sensitive data hidden in logs

## Performance Features

### Caching

Intelligent response caching to reduce costs and latency:

```python
# Enable caching with custom TTL
client = KnollmClient(
    enable_caching=True,
    cache_ttl=3600,  # 1 hour
    cache_size=1000  # Max cached responses
)

# Cache key includes request parameters
response = await client.create_completion(
    messages=[{"role": "user", "content": "Hello"}],
    cache_key="greeting"  # Optional explicit key
)

# Manual cache operations
client.clear_cache()
client.get_cache_stats()
```

### Concurrent Requests

Handle multiple requests efficiently:

```python
import asyncio

# Batch requests
requests = [
    {"messages": [{"role": "user", "content": f"Question {i}"}]}
    for i in range(10)
]

responses = await client.create_completions_batch(requests)

# Concurrent with different models
tasks = [
    client.create_completion(messages, model="gpt-4o-mini"),
    client.create_completion(messages, model="claude-3-haiku"),
    client.create_completion(messages, model="gemini-1.5-flash")
]

responses = await asyncio.gather(*tasks)
```

### Monitoring and Observability

Track performance and costs:

```python
# Get provider statistics
stats = client.get_provider_stats()
for provider, data in stats.items():
    print(f"{provider}: {data.request_count} requests, ${data.total_cost:.4f}")

# Performance metrics
metrics = client.get_performance_metrics()
print(f"Average latency: {metrics.avg_latency_ms}ms")
print(f"Success rate: {metrics.success_rate:.2%}")

# Cost tracking
costs = client.get_cost_breakdown(period="today")
for provider, cost in costs.items():
    print(f"{provider}: ${cost:.4f}")
```

## What's Next?

Dive into specific API components:

1. **[Client API →](client.md)** - Complete KnollmClient reference
2. **[Registry API →](registry.md)** - Model and provider registries  
3. **[Routing API →](routing.md)** - Intelligent routing system
4. **[Utilities →](utilities.md)** - Data models and helpers

---

<div class="admonition example">
<p class="admonition-title">💡 API Design Philosophy</p>
<p>Claif Knollm's API is designed to be:</p>
<ul>
<li><strong>Type-safe</strong> - Complete type hints and validation</li>
<li><strong>Async-first</strong> - Non-blocking operations by default</li>
<li><strong>Error-aware</strong> - Comprehensive exception handling</li>
<li><strong>Provider-agnostic</strong> - Unified interface across all providers</li>
<li><strong>Performance-oriented</strong> - Caching, batching, and monitoring</li>
</ul>
</div>