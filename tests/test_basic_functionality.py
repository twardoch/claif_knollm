#!/usr/bin/env python3
# this_file: tests/test_basic_functionality.py

"""
Basic functionality tests for the Knollm package.

Tests core components to ensure they work correctly and can be imported.
"""

import pytest
from pathlib import Path
from decimal import Decimal

from claif_knollm import (
    ModelRegistry, ProviderRegistry, DataLoader,
    KnollmClient, RoutingStrategy,
    ModelCapability, ProviderTier, ModelModality,
    SearchFilter
)
from claif_knollm.models import Provider, Model, APILibrary


class TestImports:
    """Test that all main classes can be imported."""
    
    def test_main_imports(self):
        """Test that main classes can be imported without errors."""
        from claif_knollm import (
            ModelRegistry, ProviderRegistry, DataLoader,
            KnollmClient, RoutingStrategy,
            ModelCapability, ProviderTier
        )
        assert ModelRegistry is not None
        assert ProviderRegistry is not None
        assert DataLoader is not None
        assert KnollmClient is not None
        assert RoutingStrategy is not None
        assert ModelCapability is not None
        assert ProviderTier is not None


class TestDataModels:
    """Test the Pydantic data models."""
    
    def test_provider_creation(self):
        """Test creating a Provider instance."""
        provider = Provider(
            name="test_provider",
            display_name="Test Provider",
            base_url="https://api.test.com",
            auth_type="api_key",
            tier=ProviderTier.STANDARD,
            model_count=5
        )
        
        assert provider.name == "test_provider"
        assert provider.display_name == "Test Provider"
        assert provider.tier == ProviderTier.STANDARD
        assert provider.is_active is True
    
    def test_model_creation(self):
        """Test creating a Model instance.""" 
        model = Model(
            id="test-model-1",
            provider="test_provider",
            context_window=4096,
            capabilities={ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION},
            input_modalities={ModelModality.TEXT},
            output_modalities={ModelModality.TEXT}
        )
        
        assert model.id == "test-model-1"
        assert model.provider == "test_provider"
        assert model.context_window == 4096
        assert ModelCapability.TEXT_GENERATION in model.capabilities
    
    def test_search_filter_creation(self):
        """Test creating a SearchFilter instance."""
        search_filter = SearchFilter(
            query="gpt-4",
            required_capabilities=[ModelCapability.VISION],
            min_context_window=8000,
            max_cost_per_1k_tokens=Decimal("0.01"),
            limit=10
        )
        
        assert search_filter.query == "gpt-4"
        assert ModelCapability.VISION in search_filter.required_capabilities
        assert search_filter.min_context_window == 8000
        assert search_filter.limit == 10


class TestDataLoader:
    """Test the DataLoader functionality."""
    
    def test_data_loader_init(self):
        """Test DataLoader initialization."""
        # Use a mock work directory for testing
        test_work_dir = Path(__file__).parent / "test_data"
        loader = DataLoader(work_dir=test_work_dir)
        
        assert loader.work_dir == test_work_dir
        assert isinstance(loader._providers, dict)
        assert isinstance(loader._models, dict)
        assert isinstance(loader._libraries, dict)
    
    def test_capability_extraction(self):
        """Test capability extraction from model names."""
        loader = DataLoader()
        
        # Test vision capability extraction
        vision_caps = loader.extract_capabilities_from_name("gpt-4-vision-preview")
        assert ModelCapability.VISION in vision_caps
        assert ModelCapability.TEXT_GENERATION in vision_caps
        
        # Test code capability extraction
        code_caps = loader.extract_capabilities_from_name("codex-001")
        assert ModelCapability.CODE_GENERATION in code_caps
        
        # Test basic capabilities always present
        basic_caps = loader.extract_capabilities_from_name("basic-model")
        assert ModelCapability.TEXT_GENERATION in basic_caps
    
    def test_context_window_inference(self):
        """Test context window inference from model names."""
        loader = DataLoader()
        
        # Test specific patterns
        assert loader.infer_context_window("gpt-4-turbo") == 128000
        assert loader.infer_context_window("gpt-3.5-turbo") == 4096
        assert loader.infer_context_window("claude-3-opus") == 200000
        
        # Test unknown models
        assert loader.infer_context_window("unknown-model") is None
    
    def test_pricing_tier_inference(self):
        """Test pricing tier inference."""
        loader = DataLoader()
        
        # Test premium models
        assert loader.infer_pricing_tier("openai", "gpt-4") == ProviderTier.PREMIUM
        assert loader.infer_pricing_tier("anthropic", "claude-3-opus") == ProviderTier.PREMIUM
        
        # Test budget models
        assert loader.infer_pricing_tier("openai", "gpt-3.5-turbo") == ProviderTier.BUDGET
        
        # Test free providers
        assert loader.infer_pricing_tier("huggingface", "any-model") == ProviderTier.FREE


class TestModelRegistry:
    """Test the ModelRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test ModelRegistry initialization."""
        # This will load real data if available, or empty data if not
        registry = ModelRegistry()
        
        assert hasattr(registry, '_providers')
        assert hasattr(registry, '_models')
        assert hasattr(registry, '_model_index')
        assert isinstance(registry._providers, dict)
        assert isinstance(registry._models, dict)
        assert isinstance(registry._model_index, dict)
    
    def test_search_filter_validation(self):
        """Test SearchFilter validation."""
        # Valid search filter
        valid_filter = SearchFilter(
            query="test",
            limit=10,
            sort_order="asc"
        )
        assert valid_filter.query == "test"
        assert valid_filter.limit == 10
        assert valid_filter.sort_order == "asc"
        
        # Test invalid sort order should raise validation error
        with pytest.raises(ValueError):
            SearchFilter(sort_order="invalid")


class TestKnollmClient:
    """Test the KnollmClient functionality."""
    
    def test_client_initialization(self):
        """Test KnollmClient initialization."""
        client = KnollmClient(
            routing_strategy=RoutingStrategy.BALANCED,
            fallback_providers=["openai", "anthropic"]
        )
        
        assert client.router.strategy == RoutingStrategy.BALANCED
        assert client.fallback_providers == ["openai", "anthropic"]
        assert hasattr(client, 'registry')
        assert hasattr(client, 'provider_registry')
    
    def test_routing_strategies(self):
        """Test different routing strategies."""
        strategies = [
            RoutingStrategy.COST_OPTIMIZED,
            RoutingStrategy.SPEED_OPTIMIZED,
            RoutingStrategy.QUALITY_OPTIMIZED,
            RoutingStrategy.BALANCED,
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            client = KnollmClient(routing_strategy=strategy)
            assert client.router.strategy == strategy
    
    def test_cache_operations(self):
        """Test cache operations."""
        client = KnollmClient(enable_caching=True, cache_ttl=300)
        
        # Test cache is enabled
        assert client.enable_caching is True
        assert client.cache_ttl == 300
        
        # Test cache clearing
        client.clear_cache()
        assert len(client._response_cache) == 0


class TestEnums:
    """Test enum definitions."""
    
    def test_model_capabilities(self):
        """Test ModelCapability enum."""
        assert ModelCapability.TEXT_GENERATION.value == "text_generation"
        assert ModelCapability.VISION.value == "vision"
        assert ModelCapability.FUNCTION_CALLING.value == "function_calling"
        
        # Test all capabilities are strings
        for cap in ModelCapability:
            assert isinstance(cap.value, str)
            assert "_" in cap.value or cap.value.islower()
    
    def test_provider_tiers(self):
        """Test ProviderTier enum."""
        assert ProviderTier.FREE.value == "free"
        assert ProviderTier.PREMIUM.value == "premium"
        assert ProviderTier.ENTERPRISE.value == "enterprise"
        
        # Test ordering makes sense
        tiers = list(ProviderTier)
        assert ProviderTier.FREE in tiers
        assert ProviderTier.ENTERPRISE in tiers
    
    def test_routing_strategies(self):
        """Test RoutingStrategy enum.""" 
        assert RoutingStrategy.COST_OPTIMIZED.value == "cost_optimized"
        assert RoutingStrategy.QUALITY_OPTIMIZED.value == "quality_optimized"
        assert RoutingStrategy.BALANCED.value == "balanced"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_search_results(self):
        """Test handling of empty search results."""
        registry = ModelRegistry()
        
        # Search with impossible criteria
        search_filter = SearchFilter(
            query="definitely_nonexistent_model_12345",
            min_context_window=999999999,
            limit=1
        )
        
        result = registry.search_models(search_filter)
        assert len(result.models) == 0
        assert result.total_count == 0
    
    def test_invalid_model_names(self):
        """Test handling of invalid model names."""
        registry = ModelRegistry()
        
        # Search for non-existent model
        model = registry.get_model_by_id("nonexistent_model")
        assert model is None
    
    def test_data_validation(self):
        """Test data validation in models."""
        # Test invalid provider name (should be lowercase)
        with pytest.raises(ValueError):
            Provider(
                name="Invalid Name With Spaces",
                display_name="Test",
                base_url="https://test.com",
                auth_type="api_key",
                tier=ProviderTier.STANDARD
            )
        
        # Test invalid base URL
        with pytest.raises(ValueError):
            Provider(
                name="test",
                display_name="Test", 
                base_url="invalid_url",
                auth_type="api_key",
                tier=ProviderTier.STANDARD
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])