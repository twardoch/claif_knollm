---
# this_file: src_docs/md/getting-started/quickstart.md
title: "Quick Start Guide"
description: "Get up and running with Claif Knollm in under 5 minutes. Learn the basics of provider routing, model selection, and cost optimization."
---

# Quick Start Guide

Get up and running with Claif Knollm in under 5 minutes! This guide will walk you through the essential features and show you how to build your first multi-provider LLM application.

## Installation

First, install Claif Knollm with pip:

```bash
pip install claif-knollm
```

For development or CLI usage, install with extra dependencies:

```bash
pip install claif-knollm[cli]
```

## Your First Knollm Application

### 1. Basic Model Search

Start by exploring the model catalog:

```python
from claif_knollm import ModelRegistry, SearchFilter, ModelCapability

# Initialize the registry
registry = ModelRegistry()

# Search for vision-capable models under $0.01 per 1k tokens
search_filter = SearchFilter(
    required_capabilities=[ModelCapability.VISION],
    max_cost_per_1k_tokens=0.01,
    min_context_window=32000,
    limit=5
)

models = registry.search_models(search_filter)

print(f"Found {len(models.models)} vision models:")
for model in models.models:
    provider = model.provider.title()
    context = f"{model.context_window:,}" if model.context_window else "Unknown"
    print(f"  • {model.id} ({provider}) - {context} tokens")
```

### 2. Smart Provider Routing

Use intelligent routing to automatically select the best provider:

```python
from claif_knollm import KnollmClient, RoutingStrategy

# Initialize client with cost-optimized routing
client = KnollmClient(
    routing_strategy=RoutingStrategy.COST_OPTIMIZED,
    fallback_providers=["openai", "anthropic", "groq"]
)

# Make a request - Knollm will choose the optimal provider
response = await client.create_completion(
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    max_tokens=150,
    temperature=0.7
)

print(f"Response from {response.provider}/{response.model}:")
print(response.content)
print(f"Cost: ${response.cost:.4f}" if response.cost else "Cost: Free")
```

### 3. Find the Cheapest Models

Optimize costs by finding the most affordable options:

```python
from claif_knollm import ModelRegistry, ModelCapability

registry = ModelRegistry()

# Find the 5 cheapest models with function calling
cheap_models = registry.get_cheapest_models(
    limit=5,
    capabilities=[ModelCapability.FUNCTION_CALLING]
)

print("Cheapest function-calling models:")
for model in cheap_models:
    if model.is_free:
        cost = "FREE"
    elif model.metrics and model.metrics.cost_per_1k_input_tokens:
        cost = f"${model.metrics.cost_per_1k_input_tokens}"
    else:
        cost = "Unknown"
    
    print(f"  • {model.id} ({model.provider}) - {cost}")
```

### 4. Compare Multiple Models

Make informed decisions by comparing models across different criteria:

```python
from claif_knollm import ModelRegistry

registry = ModelRegistry()

# Compare top models across cost, speed, and quality
comparison = registry.compare_models(
    models=["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash"],
    criteria=["cost", "quality", "context_window"],
    weights={"cost": 0.4, "quality": 0.4, "context_window": 0.2}
)

print("Model Comparison Results:")
for rank, (model, score) in enumerate(comparison, 1):
    print(f"  {rank}. {model.id} (Score: {score:.3f})")
    print(f"     Provider: {model.provider.title()}")
    print(f"     Context: {model.context_window:,} tokens" if model.context_window else "     Context: Unknown")
```

## CLI Quick Tour

Knollm includes a powerful CLI for exploring providers and models:

### List Providers

```bash
# List all active providers
knollm providers list

# Filter by tier
knollm providers list --tier premium

# Filter by capability
knollm providers list --capability vision
```

### Search Models

```bash
# Basic text search
knollm models search --query "gpt-4"

# Advanced filtering
knollm models search \
  --capability function_calling \
  --min-context 32000 \
  --max-cost 0.02 \
  --limit 10

# Find cheapest models
knollm models cheapest --capability vision --limit 5
```

### Get Model Information

```bash
# Detailed model info
knollm models info "gpt-4o-mini"

# Compare models
knollm models compare "gpt-4o-mini" "claude-3-haiku" "gemini-1.5-flash"
```

### Library Recommendations

```bash
# Get library recommendations for specific use cases
knollm libraries recommend "async"
knollm libraries recommend "structured_output"
knollm libraries recommend "multi_provider"

# List all libraries
knollm libraries list --min-rating 5.0

# Get detailed library info
knollm libraries info "httpx"
```

## Common Use Cases

### Use Case 1: Cost-Optimized Chatbot

Build a chatbot that automatically uses the cheapest available provider:

```python
from claif_knollm import KnollmClient, RoutingStrategy

class CostOptimizedChatbot:
    def __init__(self):
        self.client = KnollmClient(
            routing_strategy=RoutingStrategy.COST_OPTIMIZED,
            fallback_providers=["groq", "deepseek", "openai"]
        )
    
    async def chat(self, message: str, history: list = None) -> str:
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        response = await self.client.create_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        
        print(f"Used: {response.provider}/{response.model}")
        if response.cost:
            print(f"Cost: ${response.cost:.6f}")
        
        return response.content

# Usage
bot = CostOptimizedChatbot()
reply = await bot.chat("What's the capital of France?")
print(reply)
```

### Use Case 2: High-Quality Analysis

For tasks requiring the highest quality, use quality-optimized routing:

```python
from claif_knollm import KnollmClient, RoutingStrategy, CompletionRequest

async def analyze_document(document_text: str) -> str:
    client = KnollmClient(routing_strategy=RoutingStrategy.QUALITY_OPTIMIZED)
    
    response = await client.create_completion(
        messages=[
            {
                "role": "system", 
                "content": "You are an expert analyst. Provide detailed, accurate analysis."
            },
            {
                "role": "user", 
                "content": f"Analyze this document:\n\n{document_text}"
            }
        ],
        max_tokens=1000,
        temperature=0.1  # Low temperature for consistent, accurate results
    )
    
    return response.content

# Usage
analysis = await analyze_document("Your document text here...")
print(analysis)
```

### Use Case 3: Multi-Modal Vision Analysis

Use the model registry to find and use vision-capable models:

```python
from claif_knollm import ModelRegistry, KnollmClient, ModelCapability

async def analyze_image(image_url: str, question: str) -> str:
    # Find the best vision model
    registry = ModelRegistry()
    vision_model = registry.find_optimal_model(
        required_capabilities=[ModelCapability.VISION],
        quality_threshold=0.8
    )
    
    if not vision_model:
        raise ValueError("No suitable vision model found")
    
    client = KnollmClient()
    
    response = await client.create_completion(
        model=vision_model.id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    )
    
    return response.content

# Usage
result = await analyze_image(
    "https://example.com/image.jpg",
    "What do you see in this image?"
)
print(result)
```

## Configuration

### Environment Variables

Set up your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google
export GOOGLE_API_KEY="your-google-key"

# Groq
export GROQ_API_KEY="your-groq-key"
```

### Configuration File

Create a `knollm_config.toml` file for advanced configuration:

```toml
[routing]
strategy = "balanced"
fallback_providers = ["openai", "anthropic", "groq"]
cache_ttl = 3600

[costs]
max_daily_spend = 10.00
alert_threshold = 8.00

[providers.openai]
api_key_env = "OPENAI_API_KEY"
rate_limit = 60
preferred = true

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"  
rate_limit = 40

[providers.groq]
api_key_env = "GROQ_API_KEY"
rate_limit = 100
```

## Next Steps

Now that you've got the basics down, explore these advanced topics:

### 🔍 **Deep Dive into Providers**
Learn about the [complete provider ecosystem](../providers/) and how to choose the right providers for your needs.

### 🤖 **Master Model Selection**  
Explore the [model database](../models/) and learn advanced [search techniques](../models/search.md).

### 🐍 **Python Library Guide**
Discover the [best Python libraries](../libraries/) for LLM integration and get specific [recommendations](../libraries/recommendations.md).

### 📊 **Cost Optimization**
Master [cost optimization strategies](../guides/optimization.md) and learn to minimize your LLM expenses.

### 🚀 **Production Deployment**
Learn [best practices](../guides/best-practices.md) for deploying Knollm in production environments.

---

<div class="admonition example">
<p class="admonition-title">💡 Try It Yourself</p>
<p>Copy any of the code examples above and run them in your Python environment. All examples are tested and ready to use!</p>
</div>

<div class="admonition tip">
<p class="admonition-title">🚀 Pro Tips</p>
<ul>
<li>Use <code>RoutingStrategy.COST_OPTIMIZED</code> for development and testing</li>
<li>Use <code>RoutingStrategy.QUALITY_OPTIMIZED</code> for production workloads</li>
<li>Always set up fallback providers for reliability</li>
<li>Monitor your usage with <code>client.get_provider_stats()</code></li>
</ul>
</div>