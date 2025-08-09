"""
Claif Knollm - Comprehensive LLM Provider Catalog and Intelligent Routing.

This package provides the world's most complete catalog of LLM models and providers,
with intelligent routing and multi-provider capabilities for optimal performance,
cost, and reliability.
"""

from importlib import metadata

# Core classes
from .models import (
    Provider, Model, APILibrary, SearchFilter, SearchResult,
    ModelCapability, ProviderTier, ModelModality, LibraryCategory,
    LibraryFeature, ModelMetrics, ProviderStatus
)

from .registry import ModelRegistry, ProviderRegistry
from .data_loader import DataLoader
from .client import KnollmClient, RoutingStrategy, CompletionRequest, CompletionResponse

# Version
__version__ = metadata.version(__name__)

# Convenience imports
__all__ = [
    # Core client
    'KnollmClient',
    
    # Registries
    'ModelRegistry',
    'ProviderRegistry', 
    'DataLoader',
    
    # Data models
    'Provider',
    'Model', 
    'APILibrary',
    'SearchFilter',
    'SearchResult',
    'ModelMetrics',
    'ProviderStatus',
    
    # Request/Response models
    'CompletionRequest',
    'CompletionResponse',
    
    # Enums
    'ModelCapability',
    'ProviderTier',
    'ModelModality', 
    'LibraryCategory',
    'LibraryFeature',
    'RoutingStrategy',
    
    # Version
    '__version__'
]