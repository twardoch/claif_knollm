#!/usr/bin/env python3
# this_file: src/claif_knollm/data_loader.py

"""
Data loading and processing for the Knollm provider catalog.

This module loads and processes model data from JSON files in the work/model_catalog/
directory, normalizing different provider formats and extracting model capabilities
from naming patterns and specifications.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from decimal import Decimal
import logging

from .models import (
    Provider, Model, APILibrary, ModelCapability, ProviderTier, 
    AuthType, ModelModality, LibraryCategory, LibraryFeature,
    ModelMetrics
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and processes provider and model data from JSON files."""
    
    def __init__(self, work_dir: Optional[Path] = None):
        """Initialize the data loader.
        
        Args:
            work_dir: Path to the work directory containing model_catalog and api_inference
        """
        if work_dir is None:
            # Default to work directory relative to this file
            work_dir = Path(__file__).parent.parent.parent / "work"
        
        self.work_dir = Path(work_dir)
        self.model_catalog_dir = self.work_dir / "model_catalog"
        self.api_inference_dir = self.work_dir / "api_inference"
        
        # Cache for loaded data
        self._providers: Dict[str, Provider] = {}
        self._models: Dict[str, List[Model]] = {}
        self._libraries: Dict[str, APILibrary] = {}
        
        # Provider configuration mappings
        self._provider_configs = self._load_provider_configs()
        
        # Capability extraction patterns
        self._capability_patterns = self._init_capability_patterns()
        
        # Pricing data cache
        self._pricing_cache: Dict[str, Dict[str, Any]] = {}
    
    def _load_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load provider configurations from the dump_models.py script."""
        provider_configs = {}
        
        # Read the dump_models.py file to extract provider configurations
        dump_models_path = self.model_catalog_dir / "dump_models.py"
        
        if not dump_models_path.exists():
            logger.warning(f"dump_models.py not found at {dump_models_path}")
            return provider_configs
        
        try:
            with open(dump_models_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract PROVIDER_CONFIG section
            config_match = re.search(
                r'PROVIDER_CONFIG = """(.*?)"""\s*\.strip\(\)', 
                content, 
                re.DOTALL
            )
            
            if config_match:
                config_text = config_match.group(1)
                for line in config_text.strip().split('\n'):
                    if not line.strip():
                        continue
                    
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 4:
                        name, kind, api_key_env, url_env = parts[:4]
                        provider_configs[name] = {
                            'kind': kind,
                            'api_key_env': api_key_env,
                            'url_env': url_env
                        }
            
            # Extract URL configurations
            url_config_match = re.search(
                r'PROVIDER_URL_CONFIG = """(.*?)"""\s*\.strip\(\)',
                content,
                re.DOTALL
            )
            
            if url_config_match:
                url_text = url_config_match.group(1)
                for line in url_text.strip().split('\n'):
                    if not line.strip() or ',' not in line:
                        continue
                    
                    env_var, url = [part.strip() for part in line.split(',', 1)]
                    
                    # Find which provider uses this URL env var
                    for provider_name, config in provider_configs.items():
                        if config.get('url_env') == env_var:
                            config['base_url'] = url
                            break
        
        except Exception as e:
            logger.error(f"Error loading provider configs: {e}")
        
        return provider_configs
    
    def _init_capability_patterns(self) -> Dict[ModelCapability, List[str]]:
        """Initialize regex patterns for extracting capabilities from model names."""
        return {
            ModelCapability.VISION: [
                r'vision', r'visual', r'image', r'multimodal', r'omni'
            ],
            ModelCapability.FUNCTION_CALLING: [
                r'function', r'tool', r'agent'
            ],
            ModelCapability.CODE_GENERATION: [
                r'code', r'codex', r'coder', r'programming', r'dev'
            ],
            ModelCapability.REASONING: [
                r'reason', r'think', r'o1', r'reasoning'
            ],
            ModelCapability.MATH: [
                r'math', r'mathematical', r'calculation'
            ],
            ModelCapability.EMBEDDINGS: [
                r'embed', r'vector', r'similarity'
            ],
            ModelCapability.FINE_TUNING: [
                r'fine', r'custom', r'tuned'
            ],
            ModelCapability.IMAGE_GENERATION: [
                r'dall-e', r'image-gen', r'stable-diffusion', r'midjourney'
            ],
            ModelCapability.AUDIO_GENERATION: [
                r'audio', r'music', r'sound', r'tts', r'speech'
            ],
            ModelCapability.SPEECH_TO_TEXT: [
                r'whisper', r'stt', r'speech-to-text', r'transcri'
            ],
            ModelCapability.TEXT_TO_SPEECH: [
                r'tts', r'text-to-speech', r'voice'
            ]
        }
    
    def extract_capabilities_from_name(self, model_name: str) -> Set[ModelCapability]:
        """Extract capabilities from model name using pattern matching."""
        capabilities = set()
        model_lower = model_name.lower()
        
        for capability, patterns in self._capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, model_lower):
                    capabilities.add(capability)
                    break
        
        # All models have basic text generation
        capabilities.add(ModelCapability.TEXT_GENERATION)
        
        # Most modern models support chat completion
        if not any(x in model_lower for x in ['embed', 'whisper', 'dall-e']):
            capabilities.add(ModelCapability.CHAT_COMPLETION)
        
        # Most models support streaming
        capabilities.add(ModelCapability.STREAMING)
        
        return capabilities
    
    def infer_context_window(self, model_name: str) -> Optional[int]:
        """Infer context window from model name."""
        model_lower = model_name.lower()
        
        # Common context window patterns
        context_patterns = [
            (r'(\d+)k', lambda x: int(x) * 1000),
            (r'(\d+)k-context', lambda x: int(x) * 1000),
            (r'context-(\d+)k', lambda x: int(x) * 1000),
            (r'(\d+)-turbo', lambda x: int(x) * 1000 if int(x) > 10 else None),
        ]
        
        for pattern, converter in context_patterns:
            match = re.search(pattern, model_lower)
            if match:
                try:
                    result = converter(match.group(1))
                    if result and result > 1000:  # Sanity check
                        return result
                except (ValueError, TypeError):
                    continue
        
        # Default context windows for known model families
        if 'gpt-4' in model_lower:
            if 'turbo' in model_lower or '1106' in model_lower or '0125' in model_lower:
                return 128000
            elif 'vision' in model_lower:
                return 128000
            else:
                return 8192
        elif 'gpt-3.5' in model_lower:
            if 'turbo' in model_lower and ('1106' in model_lower or '0125' in model_lower):
                return 16385
            elif 'turbo' in model_lower:
                return 4096
            else:
                return 4097
        elif 'claude-3' in model_lower:
            return 200000
        elif 'claude-2' in model_lower:
            return 100000
        elif 'gemini' in model_lower:
            if 'pro' in model_lower:
                return 2000000  # Gemini Pro has very large context
            else:
                return 32000
        elif 'llama' in model_lower:
            if '2-70b' in model_lower or '2-13b' in model_lower:
                return 4096
            elif '3' in model_lower:
                return 8192
            else:
                return 2048
        
        return None
    
    def infer_pricing_tier(self, provider_name: str, model_name: str) -> ProviderTier:
        """Infer pricing tier from provider and model name."""
        model_lower = model_name.lower()
        
        # Free tier providers
        if provider_name in ['aihorde', 'huggingface']:
            return ProviderTier.FREE
        
        # Enterprise providers
        if provider_name in ['azure', 'aws', 'google-cloud']:
            return ProviderTier.ENTERPRISE
        
        # Premium models
        if any(x in model_lower for x in ['gpt-4', 'claude-3-opus', 'o1-preview']):
            return ProviderTier.PREMIUM
        
        # Budget models  
        if any(x in model_lower for x in ['gpt-3.5', 'claude-3-haiku', 'gemma', 'llama-2']):
            return ProviderTier.BUDGET
        
        # Standard by default
        return ProviderTier.STANDARD
    
    def normalize_model_data(self, data: Any, provider_name: str) -> List[Dict[str, Any]]:
        """Normalize model data from different provider formats."""
        models = []
        
        try:
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    # OpenAI format: {"object": "list", "data": [...]}
                    models = data["data"]
                else:
                    # Object format: {"model-name": {...}, "sample_spec": {...}}
                    for key, value in data.items():
                        if key != "sample_spec" and isinstance(value, dict):
                            model_data = {"id": key}
                            model_data.update(value)
                            models.append(model_data)
                        elif key != "sample_spec":
                            # Simple key-value where key is model name
                            models.append({"id": key})
            
            elif isinstance(data, list):
                # Direct array format: [{"id": "model-name", ...}]
                models = data
            
            else:
                logger.warning(f"Unknown data format for provider {provider_name}: {type(data)}")
                return []
        
        except Exception as e:
            logger.error(f"Error normalizing data for provider {provider_name}: {e}")
            return []
        
        # Ensure all models have at least an 'id' field
        normalized = []
        for model in models:
            if isinstance(model, dict) and model.get("id"):
                normalized.append(model)
            elif isinstance(model, str):
                normalized.append({"id": model})
        
        return normalized
    
    def load_models_from_provider(self, provider_name: str) -> List[Model]:
        """Load models for a specific provider from JSON file."""
        json_file = self.model_catalog_dir / f"models_{provider_name}.json"
        
        if not json_file.exists():
            logger.warning(f"Model data file not found: {json_file}")
            return []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_models = self.normalize_model_data(data, provider_name)
            models = []
            
            for model_data in raw_models:
                try:
                    model_id = model_data.get("id", "").strip()
                    if not model_id:
                        continue
                    
                    # Extract capabilities from name
                    capabilities = self.extract_capabilities_from_name(model_id)
                    
                    # Infer context window
                    context_window = self.infer_context_window(model_id)
                    
                    # Determine modalities
                    input_modalities = {ModelModality.TEXT}
                    output_modalities = {ModelModality.TEXT}
                    
                    if ModelCapability.VISION in capabilities:
                        input_modalities.add(ModelModality.IMAGE)
                    if ModelCapability.IMAGE_GENERATION in capabilities:
                        output_modalities.add(ModelModality.IMAGE)
                    if ModelCapability.AUDIO_GENERATION in capabilities:
                        output_modalities.add(ModelModality.AUDIO)
                    
                    # Create model object
                    model = Model(
                        id=model_id,
                        display_name=model_data.get("name", model_id),
                        provider=provider_name,
                        context_window=context_window,
                        capabilities=capabilities,
                        input_modalities=input_modalities,
                        output_modalities=output_modalities,
                        supports_streaming=ModelCapability.STREAMING in capabilities,
                        supports_function_calling=ModelCapability.FUNCTION_CALLING in capabilities,
                        supports_system_messages=True,  # Most models support this
                        pricing_tier=self.infer_pricing_tier(provider_name, model_id),
                        is_available=True,
                        is_deprecated=False,
                        tags=self._extract_model_tags(model_id),
                        description=model_data.get("description"),
                        last_updated=datetime.utcnow()
                    )
                    
                    models.append(model)
                
                except Exception as e:
                    logger.error(f"Error creating model {model_data.get('id', 'unknown')} from {provider_name}: {e}")
                    continue
            
            logger.info(f"Loaded {len(models)} models from {provider_name}")
            return models
        
        except Exception as e:
            logger.error(f"Error loading models from {json_file}: {e}")
            return []
    
    def _extract_model_tags(self, model_id: str) -> Set[str]:
        """Extract tags from model ID for categorization."""
        tags = set()
        model_lower = model_id.lower()
        
        # Model family tags
        families = ['gpt', 'claude', 'gemini', 'llama', 'mistral', 'qwen', 'yi']
        for family in families:
            if family in model_lower:
                tags.add(family)
                break
        
        # Size tags
        if any(x in model_lower for x in ['large', 'xl', '70b', '405b']):
            tags.add('large')
        elif any(x in model_lower for x in ['small', 'mini', '7b', '8b']):
            tags.add('small')
        elif any(x in model_lower for x in ['medium', '13b', '14b']):
            tags.add('medium')
        
        # Capability tags
        if 'vision' in model_lower or 'visual' in model_lower:
            tags.add('multimodal')
        if 'turbo' in model_lower:
            tags.add('fast')
        if 'instruct' in model_lower:
            tags.add('instruct')
        if 'chat' in model_lower:
            tags.add('chat')
        if 'code' in model_lower:
            tags.add('coding')
        
        return tags
    
    def create_provider_from_config(self, provider_name: str) -> Optional[Provider]:
        """Create a Provider object from configuration data."""
        config = self._provider_configs.get(provider_name)
        if not config:
            logger.warning(f"No configuration found for provider: {provider_name}")
            return None
        
        base_url = config.get('base_url', '')
        if not base_url:
            logger.warning(f"No base URL found for provider: {provider_name}")
            return None
        
        # Determine auth type
        auth_type = AuthType.API_KEY
        if config.get('kind') == 'ant':
            auth_type = AuthType.CUSTOM_HEADER
        elif config.get('kind') == 'url':
            auth_type = AuthType.NO_AUTH
        
        # Load models to determine capabilities
        models = self.load_models_from_provider(provider_name)
        
        # Aggregate capabilities from models
        all_capabilities = set()
        max_context = 0
        supports_streaming = False
        supports_function_calling = False
        supports_vision = False
        supports_multimodal = False
        
        for model in models:
            all_capabilities.update(model.capabilities)
            if model.context_window:
                max_context = max(max_context, model.context_window)
            if model.supports_streaming:
                supports_streaming = True
            if model.supports_function_calling:
                supports_function_calling = True
            if ModelCapability.VISION in model.capabilities:
                supports_vision = True
            if len(model.input_modalities) > 1 or len(model.output_modalities) > 1:
                supports_multimodal = True
        
        # Determine provider tier
        tier = ProviderTier.STANDARD
        if provider_name in ['aihorde', 'huggingface']:
            tier = ProviderTier.FREE
        elif provider_name in ['openai', 'anthropic', 'google']:
            tier = ProviderTier.PREMIUM
        elif provider_name in ['groq', 'deepseek', 'cerebras']:
            tier = ProviderTier.BUDGET
        
        try:
            provider = Provider(
                name=provider_name,
                display_name=provider_name.replace('_', ' ').title(),
                description=f"LLM API provider offering {len(models)} models",
                base_url=base_url,
                auth_type=auth_type,
                auth_header=config.get('api_key_env') if auth_type == AuthType.CUSTOM_HEADER else None,
                supported_capabilities=all_capabilities,
                max_context_window=max_context if max_context > 0 else None,
                supports_streaming=supports_streaming,
                supports_function_calling=supports_function_calling,
                supports_vision=supports_vision,
                supports_multimodal=supports_multimodal,
                tier=tier,
                free_tier_available=(tier == ProviderTier.FREE),
                is_active=True,
                model_count=len(models),
                required_env_vars=[config.get('api_key_env')] if config.get('api_key_env') else [],
                last_updated=datetime.utcnow()
            )
            
            return provider
            
        except Exception as e:
            logger.error(f"Error creating provider {provider_name}: {e}")
            return None
    
    def load_all_providers(self) -> Dict[str, Provider]:
        """Load all providers from configuration and model data."""
        providers = {}
        
        # Get all provider names from JSON files
        if not self.model_catalog_dir.exists():
            logger.error(f"Model catalog directory not found: {self.model_catalog_dir}")
            return providers
        
        json_files = list(self.model_catalog_dir.glob("models_*.json"))
        provider_names = []
        
        for json_file in json_files:
            # Extract provider name from filename (models_<provider>.json)
            match = re.match(r'models_(.+)\.json$', json_file.name)
            if match:
                provider_names.append(match.group(1))
        
        logger.info(f"Found {len(provider_names)} providers: {provider_names}")
        
        # Create provider objects
        for provider_name in provider_names:
            try:
                provider = self.create_provider_from_config(provider_name)
                if provider:
                    providers[provider_name] = provider
                    logger.debug(f"Created provider: {provider_name}")
                else:
                    logger.warning(f"Failed to create provider: {provider_name}")
            except Exception as e:
                logger.error(f"Error loading provider {provider_name}: {e}")
                continue
        
        self._providers = providers
        logger.info(f"Successfully loaded {len(providers)} providers")
        return providers
    
    def load_all_models(self) -> Dict[str, List[Model]]:
        """Load all models from all providers."""
        all_models = {}
        
        for provider_name in self._providers.keys():
            models = self.load_models_from_provider(provider_name)
            if models:
                all_models[provider_name] = models
        
        self._models = all_models
        total_models = sum(len(models) for models in all_models.values())
        logger.info(f"Successfully loaded {total_models} models from {len(all_models)} providers")
        
        return all_models
    
    def load_api_libraries(self) -> Dict[str, APILibrary]:
        """Load API library information from the api_inference directory."""
        libraries = {}
        
        # Library data based on the README.md analysis
        library_data = {
            "requests": {
                "display_name": "Requests",
                "description": "The most popular Python HTTP library with simple, synchronous interface",
                "category": LibraryCategory.HTTP_CLIENT,
                "rating": 5.0,
                "features": {LibraryFeature.TYPE_SAFETY},
                "pros": ["Unbeatable simplicity", "Largest community support", "Perfect for scripts"],
                "cons": ["No async support", "No native streaming"],
                "use_cases": ["Simple scripts", "Synchronous applications", "Learning/prototyping"]
            },
            "httpx": {
                "display_name": "HTTPX", 
                "description": "Modern HTTP client with sync/async interfaces and native streaming support",
                "category": LibraryCategory.HTTP_CLIENT,
                "rating": 7.0,
                "features": {LibraryFeature.ASYNC_SUPPORT, LibraryFeature.STREAMING},
                "pros": ["Best general-purpose HTTP library", "Excellent performance", "Modern async/sync design"],
                "cons": ["Slightly more complex than requests"],
                "use_cases": ["Modern applications", "Async/await usage", "Streaming responses"]
            },
            "aiohttp": {
                "display_name": "aiohttp",
                "description": "Async-first HTTP client/server framework for high-performance scenarios", 
                "category": LibraryCategory.HTTP_CLIENT,
                "rating": 6.0,
                "features": {LibraryFeature.ASYNC_SUPPORT, LibraryFeature.STREAMING},
                "pros": ["Peak performance for high-concurrency", "Mature async ecosystem"],
                "cons": ["More complex API", "Async-only design"],
                "use_cases": ["High-concurrency workloads", "Production async services"]
            },
            "openai": {
                "display_name": "OpenAI Python",
                "description": "Official OpenAI Python library, the gold standard for OpenAI-compatible APIs",
                "category": LibraryCategory.OPENAI_CLIENT,
                "rating": 7.0,
                "features": {LibraryFeature.ASYNC_SUPPORT, LibraryFeature.STREAMING, LibraryFeature.FUNCTION_CALLING, LibraryFeature.TYPE_SAFETY},
                "pros": ["Official support", "Exceptional documentation", "OpenAI-compatible endpoints"],
                "cons": ["OpenAI-specific design"],
                "use_cases": ["OpenAI API usage", "OpenAI-compatible providers", "Production applications"]
            },
            "litellm": {
                "display_name": "LiteLLM",
                "description": "Universal LLM interface supporting 100+ providers through unified API",
                "category": LibraryCategory.MULTI_PROVIDER,
                "rating": 6.0,
                "features": {LibraryFeature.MULTI_PROVIDER, LibraryFeature.STREAMING, LibraryFeature.COST_TRACKING},
                "pros": ["Best multi-provider abstraction", "Excellent fallback support", "100+ providers"],
                "cons": ["Additional abstraction layer"],
                "use_cases": ["Multi-provider setups", "Provider failover", "Cost optimization"]
            },
            "instructor": {
                "display_name": "Instructor",
                "description": "Enhances OpenAI client with structured output using Pydantic models",
                "category": LibraryCategory.OPENAI_CLIENT,
                "rating": 6.0,
                "features": {LibraryFeature.STRUCTURED_OUTPUT, LibraryFeature.VALIDATION, LibraryFeature.TYPE_SAFETY},
                "pros": ["Essential for structured output", "Seamless Pydantic integration", "Automatic validation"],
                "cons": ["OpenAI-focused", "Additional complexity for simple use cases"],
                "use_cases": ["Structured output", "Data extraction", "Type-safe LLM responses"]
            },
            "mirascope": {
                "display_name": "Mirascope",
                "description": "Lightweight abstraction layer with decorator-based function calling",
                "category": LibraryCategory.MULTI_PROVIDER,
                "rating": 5.0,
                "features": {LibraryFeature.MULTI_PROVIDER, LibraryFeature.FUNCTION_CALLING, LibraryFeature.MONITORING},
                "pros": ["Clean decorator-based design", "Good observability features", "Multi-provider"],
                "cons": ["Smaller community", "Less mature ecosystem"],
                "use_cases": ["Code elegance", "Function calling", "Multi-provider with decorators"]
            },
            "pydantic_ai": {
                "display_name": "Pydantic AI",
                "description": "Modern framework by Pydantic team combining type safety with agent capabilities",
                "category": LibraryCategory.FRAMEWORK,
                "rating": 6.0,
                "features": {LibraryFeature.TYPE_SAFETY, LibraryFeature.VALIDATION, LibraryFeature.AGENT_FRAMEWORK, LibraryFeature.STRUCTURED_OUTPUT},
                "pros": ["Best balance of features and simplicity", "Type-safe by design", "Excellent Pydantic integration"],
                "cons": ["Relatively new", "Smaller ecosystem"],
                "use_cases": ["Type-safe AI applications", "Agent development", "Structured output"]
            },
            "langchain": {
                "display_name": "LangChain",
                "description": "Most comprehensive LLM framework with extensive ecosystem for chains, agents, RAG",
                "category": LibraryCategory.FRAMEWORK,
                "rating": 4.0,
                "features": {LibraryFeature.AGENT_FRAMEWORK, LibraryFeature.RAG_SUPPORT, LibraryFeature.MULTI_PROVIDER, LibraryFeature.TEMPLATING},
                "pros": ["Industry standard", "Comprehensive ecosystem", "Extensive documentation"],
                "cons": ["Significant learning curve", "Unnecessary complexity for simple cases", "Heavy dependencies"],
                "use_cases": ["Complex LLM applications", "RAG pipelines", "Agent frameworks"]
            },
            "llamaindex": {
                "display_name": "LlamaIndex",
                "description": "Specialized framework for RAG applications with 350+ data connectors",
                "category": LibraryCategory.FRAMEWORK,
                "rating": 4.0,
                "features": {LibraryFeature.RAG_SUPPORT, LibraryFeature.EMBEDDING_SUPPORT, LibraryFeature.MULTI_PROVIDER},
                "pros": ["Excellent for RAG", "350+ data connectors", "Production-ready"],
                "cons": ["Unnecessary complexity for basic completions", "RAG-focused"],
                "use_cases": ["RAG applications", "Document processing", "Knowledge bases"]
            },
            "aisuite": {
                "display_name": "AISuite",
                "description": "Minimal abstraction by Andrew Ng's team for simple multi-provider needs",
                "category": LibraryCategory.MULTI_PROVIDER,
                "rating": 5.0,
                "features": {LibraryFeature.MULTI_PROVIDER},
                "pros": ["Minimal overhead", "Maximum flexibility", "Simple multi-provider"],
                "cons": ["Lacks advanced features", "Basic functionality only"],
                "use_cases": ["Simple multi-provider", "Minimal overhead", "Basic completion needs"]
            },
            "guidance": {
                "display_name": "Guidance",
                "description": "Microsoft's library for constrained generation using context-free grammars",
                "category": LibraryCategory.SPECIALIZED,
                "rating": 5.0,
                "features": {LibraryFeature.STRUCTURED_OUTPUT, LibraryFeature.TEMPLATING},
                "pros": ["Unmatched for complex structured generation", "Grammar-based constraints"],
                "cons": ["Requires grammar understanding", "Complex setup"],
                "use_cases": ["Complex structured generation", "Grammar-based output", "Constrained generation"]
            },
            "outlines": {
                "display_name": "Outlines",
                "description": "Provider-agnostic structured text generation with mathematical guarantees",
                "category": LibraryCategory.SPECIALIZED,
                "rating": 6.0,
                "features": {LibraryFeature.STRUCTURED_OUTPUT, LibraryFeature.VALIDATION, LibraryFeature.MULTI_PROVIDER},
                "pros": ["Mathematical guarantees", "Provider-agnostic", "Excellent performance"],
                "cons": ["Learning curve for complex schemas"],
                "use_cases": ["Guaranteed structured outputs", "JSON/XML generation", "Data extraction"]
            },
            "portkey_ai": {
                "display_name": "Portkey AI",
                "description": "Enterprise AI gateway with fallbacks, load balancing, and observability",
                "category": LibraryCategory.SPECIALIZED,
                "rating": 6.0,
                "features": {LibraryFeature.MULTI_PROVIDER, LibraryFeature.CACHING, LibraryFeature.MONITORING, LibraryFeature.RATE_LIMITING},
                "pros": ["Essential for production", "Minimal overhead", "Massive reliability benefits"],
                "cons": ["Enterprise focus", "Additional service dependency"],
                "use_cases": ["Production deployments", "Enterprise applications", "High reliability needs"]
            }
        }
        
        # Create APILibrary objects
        for lib_name, data in library_data.items():
            try:
                library = APILibrary(
                    name=lib_name,
                    display_name=data["display_name"],
                    description=data["description"],
                    category=data["category"],
                    rating=data["rating"],
                    supported_features=data.get("features", set()),
                    pros=data.get("pros", []),
                    cons=data.get("cons", []),
                    use_cases=data.get("use_cases", []),
                    installation_command=f"pip install {lib_name}",
                    openai_compatible=lib_name in ["openai", "instructor", "litellm"],
                    async_support=LibraryFeature.ASYNC_SUPPORT in data.get("features", set()),
                    streaming_support=LibraryFeature.STREAMING in data.get("features", set()),
                    maintenance_status="active",
                    last_updated=datetime.utcnow()
                )
                libraries[lib_name] = library
            except Exception as e:
                logger.error(f"Error creating library {lib_name}: {e}")
                continue
        
        self._libraries = libraries
        logger.info(f"Loaded {len(libraries)} API libraries")
        return libraries
    
    def load_all_data(self) -> Tuple[Dict[str, Provider], Dict[str, List[Model]], Dict[str, APILibrary]]:
        """Load all data: providers, models, and libraries."""
        logger.info("Loading all provider, model, and library data...")
        
        providers = self.load_all_providers()
        models = self.load_all_models()
        libraries = self.load_api_libraries()
        
        return providers, models, libraries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        total_models = sum(len(models) for models in self._models.values())
        
        return {
            "providers": len(self._providers),
            "models": total_models,
            "libraries": len(self._libraries),
            "avg_models_per_provider": total_models / max(len(self._providers), 1),
            "capabilities": list(ModelCapability),
            "provider_tiers": list(ProviderTier),
            "library_categories": list(LibraryCategory)
        }