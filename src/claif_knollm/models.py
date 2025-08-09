#!/usr/bin/env python3
# this_file: src/claif_knollm/models.py

"""
Data models for the Knollm provider catalog.

This module defines Pydantic models for providers, models, and API libraries
based on the comprehensive data collected from 40+ LLM providers and analysis
of 15+ Python libraries for LLM API interaction.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from decimal import Decimal

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveFloat, PositiveInt


class ModelCapability(str, Enum):
    """Enumeration of LLM model capabilities."""
    
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    AUDIO_GENERATION = "audio_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    EMBEDDINGS = "embeddings"
    FINE_TUNING = "fine_tuning"
    STREAMING = "streaming"
    SYSTEM_MESSAGES = "system_messages"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"
    MATH = "math"
    CODING = "coding"


class ProviderTier(str, Enum):
    """Provider pricing and quality tiers."""
    
    FREE = "free"              # Free tier providers
    BUDGET = "budget"          # Low-cost providers
    STANDARD = "standard"      # Mid-tier providers
    PREMIUM = "premium"        # High-quality providers
    ENTERPRISE = "enterprise"  # Enterprise-grade providers


class AuthType(str, Enum):
    """Authentication types supported by providers."""
    
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH = "oauth"
    CUSTOM_HEADER = "custom_header"
    NO_AUTH = "no_auth"


class ModelModality(str, Enum):
    """Model input/output modalities."""
    
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"


class LibraryCategory(str, Enum):
    """Categories of Python libraries for LLM interaction."""
    
    HTTP_CLIENT = "http_client"          # Basic HTTP: requests, httpx, aiohttp
    OPENAI_CLIENT = "openai_client"      # OpenAI-focused: openai, litellm, instructor
    FRAMEWORK = "framework"              # Full frameworks: langchain, llamaindex, pydantic_ai
    SPECIALIZED = "specialized"          # Specialized: guidance, outlines, portkey-ai
    MULTI_PROVIDER = "multi_provider"    # Multi-provider: aisuite, mirascope


class ModelMetrics(BaseModel):
    """Performance and quality metrics for a model."""
    
    # Performance metrics
    avg_latency_ms: Optional[float] = Field(None, description="Average response latency in milliseconds")
    throughput_tokens_per_sec: Optional[float] = Field(None, description="Average throughput in tokens per second")
    
    # Quality metrics (0.0 to 1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score")
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy on benchmarks")
    helpfulness_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Helpfulness rating")
    safety_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Safety and alignment score")
    
    # Cost metrics
    cost_per_1k_input_tokens: Optional[Decimal] = Field(None, description="Cost per 1k input tokens in USD")
    cost_per_1k_output_tokens: Optional[Decimal] = Field(None, description="Cost per 1k output tokens in USD")
    cost_per_image: Optional[Decimal] = Field(None, description="Cost per image input in USD")
    
    # Usage metrics
    popularity_rank: Optional[int] = Field(None, description="Popularity ranking among all models")
    usage_volume: Optional[int] = Field(None, description="Estimated monthly usage volume")
    
    # Reliability metrics
    uptime_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Uptime percentage")
    error_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Error rate for requests")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When metrics were last updated")


class Provider(BaseModel):
    """A provider of LLM services."""
    
    # Basic information
    name: str = Field(..., description="Provider name (lowercase, no spaces)")
    display_name: str = Field(..., description="Human-readable provider name")
    description: Optional[str] = Field(None, description="Brief description of the provider")
    
    # API configuration
    base_url: str = Field(..., description="Base URL for the provider's API")
    auth_type: AuthType = Field(..., description="Authentication method required")
    auth_header: Optional[str] = Field(None, description="Custom authentication header name")
    api_version: Optional[str] = Field(None, description="API version identifier")
    
    # Capabilities and features
    supported_capabilities: Set[ModelCapability] = Field(default_factory=set, description="Capabilities supported by this provider")
    max_context_window: Optional[int] = Field(None, description="Maximum context window across all models")
    supports_streaming: bool = Field(False, description="Whether provider supports streaming responses")
    supports_function_calling: bool = Field(False, description="Whether provider supports function calling")
    supports_vision: bool = Field(False, description="Whether provider supports vision models")
    supports_multimodal: bool = Field(False, description="Whether provider supports multimodal models")
    
    # Business information
    tier: ProviderTier = Field(..., description="Provider tier (free, budget, standard, premium, enterprise)")
    pricing_model: Optional[str] = Field(None, description="Description of pricing model")
    free_tier_available: bool = Field(False, description="Whether provider offers free tier")
    
    # Status and reliability
    is_active: bool = Field(True, description="Whether provider is currently active")
    status_url: Optional[str] = Field(None, description="URL for provider status page")
    documentation_url: Optional[str] = Field(None, description="URL for provider documentation")
    
    # Metadata
    model_count: int = Field(0, description="Number of models offered by this provider")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When provider data was last updated")
    
    # Environmental requirements
    required_env_vars: List[str] = Field(default_factory=list, description="Required environment variables")
    optional_env_vars: List[str] = Field(default_factory=list, description="Optional environment variables")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        extra = "forbid"
    
    @validator('name')
    def validate_name(cls, v):
        """Validate provider name format."""
        if not v.islower() or ' ' in v:
            raise ValueError('Provider name must be lowercase with no spaces')
        return v
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Validate base URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Base URL must start with http:// or https://')
        return v.rstrip('/')


class Model(BaseModel):
    """An LLM model from a provider."""
    
    # Basic identification
    id: str = Field(..., description="Model identifier (as used in API calls)")
    display_name: Optional[str] = Field(None, description="Human-readable model name")
    provider: str = Field(..., description="Provider name this model belongs to")
    
    # Model specifications
    context_window: Optional[int] = Field(None, description="Maximum context window in tokens")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    input_modalities: Set[ModelModality] = Field(default_factory=set, description="Supported input modalities")
    output_modalities: Set[ModelModality] = Field(default_factory=set, description="Supported output modalities")
    
    # Capabilities
    capabilities: Set[ModelCapability] = Field(default_factory=set, description="Model capabilities")
    supports_system_messages: bool = Field(True, description="Whether model supports system messages")
    supports_streaming: bool = Field(False, description="Whether model supports streaming responses")
    supports_function_calling: bool = Field(False, description="Whether model supports function calling")
    supports_json_mode: bool = Field(False, description="Whether model supports JSON mode")
    
    # Model characteristics
    model_family: Optional[str] = Field(None, description="Model family (e.g., gpt-4, claude-3)")
    model_size: Optional[str] = Field(None, description="Model size indicator (small, medium, large)")
    training_data_cutoff: Optional[str] = Field(None, description="Training data cutoff date")
    release_date: Optional[datetime] = Field(None, description="Model release date")
    
    # Performance and metrics
    metrics: Optional[ModelMetrics] = Field(None, description="Performance and quality metrics")
    
    # Pricing information
    pricing_tier: Optional[ProviderTier] = Field(None, description="Pricing tier for this model")
    is_free: bool = Field(False, description="Whether this model is available for free")
    
    # Availability and status
    is_available: bool = Field(True, description="Whether model is currently available")
    is_deprecated: bool = Field(False, description="Whether model is deprecated")
    deprecation_date: Optional[datetime] = Field(None, description="When model will be deprecated")
    
    # Metadata
    tags: Set[str] = Field(default_factory=set, description="Tags for categorization and search")
    description: Optional[str] = Field(None, description="Model description")
    use_cases: List[str] = Field(default_factory=list, description="Recommended use cases")
    
    # Quality indicators
    popularity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Popularity score based on usage")
    quality_rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Overall quality rating")
    
    # Technical details
    architecture: Optional[str] = Field(None, description="Model architecture (transformer, etc.)")
    parameter_count: Optional[str] = Field(None, description="Approximate parameter count")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When model data was last updated")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        extra = "forbid"
    
    @validator('id')
    def validate_model_id(cls, v):
        """Validate model ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Model ID cannot be empty')
        return v.strip()
    
    @root_validator
    def validate_modalities(cls, values):
        """Validate modality consistency."""
        input_modalities = values.get('input_modalities', set())
        output_modalities = values.get('output_modalities', set())
        capabilities = values.get('capabilities', set())
        
        # If model has vision capability, should have image input modality
        if ModelCapability.VISION in capabilities and ModelModality.IMAGE not in input_modalities:
            input_modalities.add(ModelModality.IMAGE)
            values['input_modalities'] = input_modalities
        
        # If model has image generation, should have image output modality
        if ModelCapability.IMAGE_GENERATION in capabilities and ModelModality.IMAGE not in output_modalities:
            output_modalities.add(ModelModality.IMAGE)
            values['output_modalities'] = output_modalities
        
        return values


class LibraryFeature(str, Enum):
    """Features that API libraries may support."""
    
    ASYNC_SUPPORT = "async_support"
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    FUNCTION_CALLING = "function_calling"
    MULTI_PROVIDER = "multi_provider"
    CACHING = "caching"
    RETRY_LOGIC = "retry_logic"
    RATE_LIMITING = "rate_limiting"
    COST_TRACKING = "cost_tracking"
    MONITORING = "monitoring"
    TYPE_SAFETY = "type_safety"
    VALIDATION = "validation"
    TEMPLATING = "templating"
    PROMPT_ENGINEERING = "prompt_engineering"
    AGENT_FRAMEWORK = "agent_framework"
    RAG_SUPPORT = "rag_support"
    EMBEDDING_SUPPORT = "embedding_support"
    VISION_SUPPORT = "vision_support"
    AUDIO_SUPPORT = "audio_support"


class APILibrary(BaseModel):
    """A Python library for LLM API interaction."""
    
    # Basic information
    name: str = Field(..., description="Library name (as used in pip install)")
    display_name: Optional[str] = Field(None, description="Human-readable library name")
    description: str = Field(..., description="Brief description of the library")
    category: LibraryCategory = Field(..., description="Library category")
    
    # Quality and popularity
    rating: float = Field(..., ge=1.0, le=7.0, description="Rating from 1-7 stars")
    github_stars: Optional[int] = Field(None, description="GitHub stars count")
    pypi_downloads: Optional[int] = Field(None, description="Monthly PyPI downloads")
    
    # Features and capabilities
    supported_features: Set[LibraryFeature] = Field(default_factory=set, description="Features supported by this library")
    supported_providers: Set[str] = Field(default_factory=set, description="Providers this library supports")
    
    # Technical details
    requires_python: Optional[str] = Field(None, description="Minimum Python version required")
    dependencies: List[str] = Field(default_factory=list, description="Key dependencies")
    installation_command: str = Field(..., description="pip install command")
    
    # Documentation and examples
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    github_url: Optional[str] = Field(None, description="GitHub repository URL")
    example_code: Optional[str] = Field(None, description="Basic usage example")
    
    # Analysis from work/api_inference/
    pros: List[str] = Field(default_factory=list, description="Advantages of this library")
    cons: List[str] = Field(default_factory=list, description="Disadvantages of this library")
    use_cases: List[str] = Field(default_factory=list, description="Recommended use cases")
    
    # Compatibility and requirements
    openai_compatible: bool = Field(False, description="Whether library is OpenAI-compatible")
    async_support: bool = Field(False, description="Whether library supports async/await")
    streaming_support: bool = Field(False, description="Whether library supports streaming responses")
    
    # Maintenance and quality indicators
    last_release_date: Optional[datetime] = Field(None, description="Last release date")
    maintenance_status: Optional[str] = Field(None, description="Maintenance status (active, maintenance, deprecated)")
    test_coverage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Test coverage percentage")
    
    # Community and support
    community_size: Optional[int] = Field(None, description="Estimated community size")
    has_commercial_support: bool = Field(False, description="Whether commercial support is available")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When library data was last updated")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        extra = "forbid"
    
    @validator('name')
    def validate_name(cls, v):
        """Validate library name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Library name cannot be empty')
        return v.strip().lower()
    
    @validator('rating')
    def validate_rating(cls, v):
        """Validate rating is within acceptable range."""
        if v < 1.0 or v > 7.0:
            raise ValueError('Rating must be between 1.0 and 7.0')
        return round(v, 1)


class SearchFilter(BaseModel):
    """Filters for searching models and providers."""
    
    # Text search
    query: Optional[str] = Field(None, description="Text query for name/description search")
    
    # Provider filters
    providers: Optional[List[str]] = Field(None, description="Filter by specific providers")
    provider_tiers: Optional[List[ProviderTier]] = Field(None, description="Filter by provider tiers")
    exclude_providers: Optional[List[str]] = Field(None, description="Exclude specific providers")
    
    # Model capability filters
    required_capabilities: Optional[List[ModelCapability]] = Field(None, description="Must have all these capabilities")
    any_capabilities: Optional[List[ModelCapability]] = Field(None, description="Must have at least one of these capabilities")
    
    # Technical filters
    min_context_window: Optional[int] = Field(None, description="Minimum context window size")
    max_context_window: Optional[int] = Field(None, description="Maximum context window size")
    modalities: Optional[List[ModelModality]] = Field(None, description="Required modalities")
    
    # Cost filters
    max_cost_per_1k_tokens: Optional[Decimal] = Field(None, description="Maximum cost per 1k tokens")
    free_only: bool = Field(False, description="Only include free models")
    
    # Quality filters
    min_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum quality score")
    min_popularity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum popularity score")
    
    # Availability filters
    active_only: bool = Field(True, description="Only include active/available models")
    exclude_deprecated: bool = Field(True, description="Exclude deprecated models")
    
    # Sorting
    sort_by: Optional[str] = Field("name", description="Field to sort by")
    sort_order: str = Field("asc", regex="^(asc|desc)$", description="Sort order")
    
    # Pagination
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class SearchResult(BaseModel):
    """Result from a search operation."""
    
    models: List[Model] = Field(default_factory=list, description="Matching models")
    providers: List[Provider] = Field(default_factory=list, description="Matching providers")
    libraries: List[APILibrary] = Field(default_factory=list, description="Matching libraries")
    
    total_count: int = Field(0, description="Total number of results (before pagination)")
    page_size: int = Field(0, description="Number of results in this page")
    page_offset: int = Field(0, description="Offset of this page")
    
    search_time_ms: float = Field(0.0, description="Search time in milliseconds")
    filters_applied: SearchFilter = Field(..., description="Filters that were applied")
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class ProviderStatus(BaseModel):
    """Current status of a provider."""
    
    provider_name: str = Field(..., description="Provider name")
    is_healthy: bool = Field(..., description="Whether provider is currently healthy")
    status_message: Optional[str] = Field(None, description="Status message or error description")
    
    # Performance metrics
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time in milliseconds")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Success rate for requests")
    error_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Error rate for requests")
    
    # Rate limiting status
    rate_limit_status: Optional[str] = Field(None, description="Rate limiting status")
    requests_remaining: Optional[int] = Field(None, description="Requests remaining in current window")
    
    # Cost and usage
    estimated_monthly_cost: Optional[Decimal] = Field(None, description="Estimated monthly cost")
    usage_trend: Optional[str] = Field(None, description="Usage trend (increasing, stable, decreasing)")
    
    # Service status
    last_successful_request: Optional[datetime] = Field(None, description="Timestamp of last successful request")
    last_outage: Optional[datetime] = Field(None, description="Timestamp of last known outage")
    uptime_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Uptime percentage")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When status was last updated")
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"