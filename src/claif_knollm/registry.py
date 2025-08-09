#!/usr/bin/env python3
# this_file: src/claif_knollm/registry.py

"""
Core registry for providers, models, and libraries with advanced search capabilities.

This module provides the central ModelRegistry and ProviderRegistry classes that
offer comprehensive search, filtering, and recommendation capabilities for the
complete catalog of LLM providers and models.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from decimal import Decimal
from collections import defaultdict
import re
import logging

from .models import (
    Provider, Model, APILibrary, SearchFilter, SearchResult, 
    ModelCapability, ProviderTier, ModelModality, LibraryCategory,
    LibraryFeature
)
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for LLM models with advanced search and filtering."""
    
    def __init__(self, data_loader: Optional[DataLoader] = None, cache_ttl: int = 3600):
        """Initialize the model registry.
        
        Args:
            data_loader: DataLoader instance, creates new one if None
            cache_ttl: Cache time-to-live in seconds
        """
        self.data_loader = data_loader or DataLoader()
        self.cache_ttl = cache_ttl
        
        # Data storage
        self._providers: Dict[str, Provider] = {}
        self._models: Dict[str, List[Model]] = {}
        self._libraries: Dict[str, APILibrary] = {}
        
        # Search indices for fast lookups
        self._model_index: Dict[str, Model] = {}  # model_id -> Model
        self._capability_index: Dict[ModelCapability, Set[str]] = defaultdict(set)
        self._provider_index: Dict[str, Set[str]] = defaultdict(set)  # provider -> model_ids
        self._modality_index: Dict[ModelModality, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> model_ids
        
        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._last_loaded: Optional[datetime] = None
        
        # Load data
        self.reload_data()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached result is still valid."""
        if key not in self._cache:
            return False
        
        _, cached_time = self._cache[key]
        return datetime.utcnow() - cached_time < timedelta(seconds=self.cache_ttl)
    
    def _cache_result(self, key: str, result: Any) -> Any:
        """Cache a result with timestamp."""
        self._cache[key] = (result, datetime.utcnow())
        return result
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if self._is_cache_valid(key):
            return self._cache[key][0]
        return None
    
    def reload_data(self) -> None:
        """Reload all data from the data loader."""
        logger.info("Reloading registry data...")
        start_time = time.time()
        
        try:
            providers, models, libraries = self.data_loader.load_all_data()
            
            self._providers = providers
            self._models = models
            self._libraries = libraries
            
            # Rebuild search indices
            self._rebuild_indices()
            
            # Clear cache
            self._cache.clear()
            self._last_loaded = datetime.utcnow()
            
            load_time = time.time() - start_time
            total_models = sum(len(model_list) for model_list in models.values())
            
            logger.info(
                f"Registry loaded: {len(providers)} providers, "
                f"{total_models} models, {len(libraries)} libraries "
                f"in {load_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error reloading registry data: {e}")
            raise
    
    def _rebuild_indices(self) -> None:
        """Rebuild search indices for fast lookups."""
        logger.debug("Rebuilding search indices...")
        
        # Clear existing indices
        self._model_index.clear()
        self._capability_index.clear()
        self._provider_index.clear()
        self._modality_index.clear()
        self._tag_index.clear()
        
        # Build indices
        for provider_name, model_list in self._models.items():
            for model in model_list:
                model_id = model.id
                
                # Model index
                self._model_index[model_id] = model
                
                # Provider index
                self._provider_index[provider_name].add(model_id)
                
                # Capability index
                for capability in model.capabilities:
                    self._capability_index[capability].add(model_id)
                
                # Modality index
                for modality in model.input_modalities.union(model.output_modalities):
                    self._modality_index[modality].add(model_id)
                
                # Tag index
                for tag in model.tags:
                    self._tag_index[tag].add(model_id)
        
        logger.debug(f"Built indices for {len(self._model_index)} models")
    
    def get_all_models(self) -> List[Model]:
        """Get all models across all providers."""
        cache_key = "all_models"
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        all_models = []
        for model_list in self._models.values():
            all_models.extend(model_list)
        
        return self._cache_result(cache_key, all_models)
    
    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Get a specific model by its ID."""
        return self._model_index.get(model_id)
    
    def get_models_by_provider(self, provider_name: str) -> List[Model]:
        """Get all models from a specific provider."""
        cache_key = f"provider_models:{provider_name}"
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        models = self._models.get(provider_name, [])
        return self._cache_result(cache_key, models)
    
    def find_models_by_capability(self, capabilities: List[ModelCapability]) -> List[Model]:
        """Find models that have all specified capabilities."""
        cache_key = f"capability_models:{'_'.join(sorted(str(c) for c in capabilities))}"
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        if not capabilities:
            return self.get_all_models()
        
        # Start with models that have the first capability
        model_ids = self._capability_index.get(capabilities[0], set()).copy()
        
        # Intersect with models that have remaining capabilities
        for capability in capabilities[1:]:
            model_ids &= self._capability_index.get(capability, set())
        
        models = [self._model_index[model_id] for model_id in model_ids if model_id in self._model_index]
        return self._cache_result(cache_key, models)
    
    def get_cheapest_models(self, limit: int = 10, capabilities: Optional[List[ModelCapability]] = None) -> List[Model]:
        """Get the cheapest models, optionally filtered by capabilities."""
        cache_key = f"cheapest_models:{limit}:{capabilities or 'all'}"
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        # Get candidate models
        if capabilities:
            candidates = self.find_models_by_capability(capabilities)
        else:
            candidates = self.get_all_models()
        
        # Filter models with pricing information
        priced_models = []
        for model in candidates:
            if model.metrics and model.metrics.cost_per_1k_input_tokens:
                priced_models.append(model)
            elif model.is_free:
                # Free models get a cost of 0
                priced_models.append(model)
        
        # Sort by cost (free models first)
        sorted_models = sorted(priced_models, key=lambda m: (
            0.0 if m.is_free else float(m.metrics.cost_per_1k_input_tokens or float('inf'))
        ))
        
        result = sorted_models[:limit]
        return self._cache_result(cache_key, result)
    
    def filter_by_context_window(self, min_context: int, max_context: Optional[int] = None) -> List[Model]:
        """Filter models by context window size."""
        cache_key = f"context_window:{min_context}:{max_context}"
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        filtered_models = []
        for model in self.get_all_models():
            if model.context_window is None:
                continue
            
            if model.context_window >= min_context:
                if max_context is None or model.context_window <= max_context:
                    filtered_models.append(model)
        
        return self._cache_result(cache_key, filtered_models)
    
    def search_models(self, search_filter: SearchFilter) -> SearchResult:
        """Comprehensive model search with advanced filtering."""
        start_time = time.time()
        
        # Start with all models
        candidates = set(self._model_index.keys())
        
        # Apply text query filter
        if search_filter.query:
            query_matches = self._search_by_text(search_filter.query)
            candidates &= query_matches
        
        # Apply provider filters
        if search_filter.providers:
            provider_matches = set()
            for provider in search_filter.providers:
                provider_matches.update(self._provider_index.get(provider, set()))
            candidates &= provider_matches
        
        if search_filter.exclude_providers:
            for provider in search_filter.exclude_providers:
                candidates -= self._provider_index.get(provider, set())
        
        # Apply capability filters
        if search_filter.required_capabilities:
            for capability in search_filter.required_capabilities:
                candidates &= self._capability_index.get(capability, set())
        
        if search_filter.any_capabilities:
            any_matches = set()
            for capability in search_filter.any_capabilities:
                any_matches.update(self._capability_index.get(capability, set()))
            candidates &= any_matches
        
        # Apply modality filters
        if search_filter.modalities:
            modality_matches = set()
            for modality in search_filter.modalities:
                modality_matches.update(self._modality_index.get(modality, set()))
            candidates &= modality_matches
        
        # Convert to Model objects and apply complex filters
        models = [self._model_index[model_id] for model_id in candidates if model_id in self._model_index]
        
        # Apply context window filters
        if search_filter.min_context_window:
            models = [m for m in models if m.context_window and m.context_window >= search_filter.min_context_window]
        
        if search_filter.max_context_window:
            models = [m for m in models if m.context_window and m.context_window <= search_filter.max_context_window]
        
        # Apply provider tier filters
        if search_filter.provider_tiers:
            models = [m for m in models if m.pricing_tier in search_filter.provider_tiers]
        
        # Apply cost filters
        if search_filter.max_cost_per_1k_tokens:
            models = [m for m in models if 
                     m.is_free or 
                     (m.metrics and m.metrics.cost_per_1k_input_tokens and 
                      m.metrics.cost_per_1k_input_tokens <= search_filter.max_cost_per_1k_tokens)]
        
        if search_filter.free_only:
            models = [m for m in models if m.is_free]
        
        # Apply quality filters
        if search_filter.min_quality_score:
            models = [m for m in models if 
                     m.metrics and m.metrics.quality_score and 
                     m.metrics.quality_score >= search_filter.min_quality_score]
        
        if search_filter.min_popularity_score:
            models = [m for m in models if 
                     m.popularity_score and m.popularity_score >= search_filter.min_popularity_score]
        
        # Apply availability filters
        if search_filter.active_only:
            models = [m for m in models if m.is_available]
        
        if search_filter.exclude_deprecated:
            models = [m for m in models if not m.is_deprecated]
        
        # Sort results
        models = self._sort_models(models, search_filter.sort_by, search_filter.sort_order)
        
        # Apply pagination
        total_count = len(models)
        offset = search_filter.offset
        limit = search_filter.limit
        
        if limit:
            models = models[offset:offset + limit]
        else:
            models = models[offset:]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            models=models,
            providers=[],  # Not searching providers in this method
            libraries=[],  # Not searching libraries in this method
            total_count=total_count,
            page_size=len(models),
            page_offset=offset,
            search_time_ms=search_time_ms,
            filters_applied=search_filter
        )
    
    def _search_by_text(self, query: str) -> Set[str]:
        """Search models by text query in name, description, and tags."""
        query_lower = query.lower()
        matches = set()
        
        for model_id, model in self._model_index.items():
            # Search in model ID
            if query_lower in model.id.lower():
                matches.add(model_id)
                continue
            
            # Search in display name
            if model.display_name and query_lower in model.display_name.lower():
                matches.add(model_id)
                continue
            
            # Search in description
            if model.description and query_lower in model.description.lower():
                matches.add(model_id)
                continue
            
            # Search in tags
            for tag in model.tags:
                if query_lower in tag.lower():
                    matches.add(model_id)
                    break
            
            # Search in model family
            if model.model_family and query_lower in model.model_family.lower():
                matches.add(model_id)
                continue
        
        return matches
    
    def _sort_models(self, models: List[Model], sort_by: str, sort_order: str) -> List[Model]:
        """Sort models by specified field and order."""
        reverse = (sort_order.lower() == 'desc')
        
        try:
            if sort_by == 'name':
                return sorted(models, key=lambda m: m.id.lower(), reverse=reverse)
            elif sort_by == 'provider':
                return sorted(models, key=lambda m: m.provider.lower(), reverse=reverse)
            elif sort_by == 'context_window':
                return sorted(models, key=lambda m: m.context_window or 0, reverse=reverse)
            elif sort_by == 'popularity':
                return sorted(models, key=lambda m: m.popularity_score or 0, reverse=reverse)
            elif sort_by == 'quality':
                return sorted(models, key=lambda m: (m.metrics.quality_score if m.metrics else 0) or 0, reverse=reverse)
            elif sort_by == 'cost':
                return sorted(models, key=lambda m: float(m.metrics.cost_per_1k_input_tokens or float('inf')) if m.metrics else float('inf'), reverse=reverse)
            elif sort_by == 'release_date':
                return sorted(models, key=lambda m: m.release_date or datetime.min, reverse=reverse)
            else:
                # Default to name sorting
                return sorted(models, key=lambda m: m.id.lower(), reverse=reverse)
        
        except Exception as e:
            logger.warning(f"Error sorting by {sort_by}: {e}, falling back to name sort")
            return sorted(models, key=lambda m: m.id.lower(), reverse=reverse)
    
    def find_optimal_model(
        self, 
        task_type: Optional[str] = None,
        max_cost_per_1k_tokens: Optional[Decimal] = None,
        min_context_window: Optional[int] = None,
        required_capabilities: Optional[List[ModelCapability]] = None,
        preferred_providers: Optional[List[str]] = None,
        quality_threshold: Optional[float] = None
    ) -> Optional[Model]:
        """Find the optimal model based on multiple criteria."""
        
        # Build search filter
        search_filter = SearchFilter(
            providers=preferred_providers,
            required_capabilities=required_capabilities,
            min_context_window=min_context_window,
            max_cost_per_1k_tokens=max_cost_per_1k_tokens,
            min_quality_score=quality_threshold,
            active_only=True,
            exclude_deprecated=True,
            sort_by='quality',
            sort_order='desc',
            limit=10
        )
        
        # Perform search
        result = self.search_models(search_filter)
        
        if not result.models:
            return None
        
        # Apply task-specific filtering
        candidates = result.models
        
        if task_type:
            task_filters = {
                'coding': [ModelCapability.CODE_GENERATION],
                'vision': [ModelCapability.VISION],
                'reasoning': [ModelCapability.REASONING],
                'math': [ModelCapability.MATH],
                'function_calling': [ModelCapability.FUNCTION_CALLING],
                'embeddings': [ModelCapability.EMBEDDINGS]
            }
            
            if task_type.lower() in task_filters:
                required_caps = task_filters[task_type.lower()]
                candidates = [m for m in candidates if all(cap in m.capabilities for cap in required_caps)]
        
        if not candidates:
            return None
        
        # Return the best candidate (first after sorting by quality)
        return candidates[0]
    
    def compare_models(
        self,
        models: List[str],
        criteria: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[Model, float]]:
        """Compare models across multiple criteria with weighted scoring."""
        
        if criteria is None:
            criteria = ['cost', 'speed', 'quality', 'context_window']
        
        if weights is None:
            weights = {criterion: 1.0 / len(criteria) for criterion in criteria}
        
        model_objects = []
        for model_id in models:
            model = self.get_model_by_id(model_id)
            if model:
                model_objects.append(model)
        
        if not model_objects:
            return []
        
        # Normalize criteria values and calculate weighted scores
        scored_models = []
        
        for model in model_objects:
            total_score = 0.0
            
            for criterion in criteria:
                weight = weights.get(criterion, 0.0)
                if weight == 0.0:
                    continue
                
                # Get normalized score for this criterion (0.0 to 1.0)
                score = self._get_normalized_score(model, criterion, model_objects)
                total_score += score * weight
            
            scored_models.append((model, total_score))
        
        # Sort by total score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models
    
    def _get_normalized_score(self, model: Model, criterion: str, all_models: List[Model]) -> float:
        """Get normalized score (0.0 to 1.0) for a model on a specific criterion."""
        
        if criterion == 'cost':
            # Lower cost is better
            if model.is_free:
                return 1.0
            if not model.metrics or not model.metrics.cost_per_1k_input_tokens:
                return 0.0
            
            costs = []
            for m in all_models:
                if m.is_free:
                    costs.append(0.0)
                elif m.metrics and m.metrics.cost_per_1k_input_tokens:
                    costs.append(float(m.metrics.cost_per_1k_input_tokens))
            
            if not costs or min(costs) == max(costs):
                return 0.5
            
            model_cost = 0.0 if model.is_free else float(model.metrics.cost_per_1k_input_tokens)
            return 1.0 - (model_cost - min(costs)) / (max(costs) - min(costs))
        
        elif criterion == 'speed':
            # Higher throughput is better
            if not model.metrics or not model.metrics.throughput_tokens_per_sec:
                return 0.5  # Neutral score
            
            throughputs = [
                m.metrics.throughput_tokens_per_sec for m in all_models 
                if m.metrics and m.metrics.throughput_tokens_per_sec
            ]
            
            if not throughputs or min(throughputs) == max(throughputs):
                return 0.5
            
            model_throughput = model.metrics.throughput_tokens_per_sec
            return (model_throughput - min(throughputs)) / (max(throughputs) - min(throughputs))
        
        elif criterion == 'quality':
            # Higher quality is better
            if not model.metrics or not model.metrics.quality_score:
                return 0.5  # Neutral score
            
            return model.metrics.quality_score
        
        elif criterion == 'context_window':
            # Higher context window is better
            if not model.context_window:
                return 0.0
            
            context_windows = [m.context_window for m in all_models if m.context_window]
            
            if not context_windows or min(context_windows) == max(context_windows):
                return 0.5
            
            return (model.context_window - min(context_windows)) / (max(context_windows) - min(context_windows))
        
        else:
            return 0.5  # Neutral score for unknown criteria
    
    def get_model_recommendations(self, user_requirements: Dict[str, Any]) -> List[Model]:
        """Get model recommendations based on user requirements."""
        
        # Extract requirements
        use_case = user_requirements.get('use_case', '').lower()
        budget = user_requirements.get('budget', 'standard')  # free, low, standard, high
        performance = user_requirements.get('performance', 'balanced')  # speed, quality, balanced
        capabilities = user_requirements.get('capabilities', [])
        
        # Map use cases to capabilities
        use_case_caps = {
            'chatbot': [ModelCapability.CHAT_COMPLETION],
            'coding': [ModelCapability.CODE_GENERATION, ModelCapability.CHAT_COMPLETION],
            'analysis': [ModelCapability.REASONING, ModelCapability.CHAT_COMPLETION],
            'vision': [ModelCapability.VISION, ModelCapability.CHAT_COMPLETION],
            'creative': [ModelCapability.TEXT_GENERATION],
            'embeddings': [ModelCapability.EMBEDDINGS],
            'function_calling': [ModelCapability.FUNCTION_CALLING, ModelCapability.CHAT_COMPLETION]
        }
        
        required_caps = list(capabilities)
        if use_case in use_case_caps:
            required_caps.extend(use_case_caps[use_case])
        
        # Build search filter based on budget
        search_filter = SearchFilter(
            required_capabilities=list(set(required_caps)) if required_caps else None,
            active_only=True,
            exclude_deprecated=True,
            limit=10
        )
        
        if budget == 'free':
            search_filter.free_only = True
        elif budget == 'low':
            search_filter.max_cost_per_1k_tokens = Decimal('0.01')
        elif budget == 'standard':
            search_filter.max_cost_per_1k_tokens = Decimal('0.05')
        # No cost limit for 'high' budget
        
        # Set sorting based on performance preference
        if performance == 'speed':
            search_filter.sort_by = 'speed'
        elif performance == 'quality':
            search_filter.sort_by = 'quality'
        else:  # balanced
            search_filter.sort_by = 'popularity'
        
        search_filter.sort_order = 'desc'
        
        # Perform search
        result = self.search_models(search_filter)
        return result.models
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about providers and models."""
        stats = {
            'total_providers': len(self._providers),
            'total_models': len(self._model_index),
            'models_by_provider': {},
            'capabilities_distribution': {},
            'tier_distribution': {},
            'modality_distribution': {}
        }
        
        # Models by provider
        for provider_name, models in self._models.items():
            stats['models_by_provider'][provider_name] = len(models)
        
        # Capability distribution
        for capability, model_ids in self._capability_index.items():
            stats['capabilities_distribution'][str(capability)] = len(model_ids)
        
        # Tier distribution
        tier_counts = defaultdict(int)
        for model in self._model_index.values():
            if model.pricing_tier:
                tier_counts[str(model.pricing_tier)] += 1
        stats['tier_distribution'] = dict(tier_counts)
        
        # Modality distribution
        for modality, model_ids in self._modality_index.items():
            stats['modality_distribution'][str(modality)] = len(model_ids)
        
        return stats


class ProviderRegistry:
    """Registry for provider-specific operations and management."""
    
    def __init__(self, model_registry: ModelRegistry):
        """Initialize with a model registry instance."""
        self.model_registry = model_registry
    
    def get_all_providers(self) -> List[Provider]:
        """Get all providers."""
        return list(self.model_registry._providers.values())
    
    def get_provider(self, name: str) -> Optional[Provider]:
        """Get a specific provider by name."""
        return self.model_registry._providers.get(name)
    
    def get_providers_by_tier(self, tier: ProviderTier) -> List[Provider]:
        """Get all providers in a specific tier."""
        return [p for p in self.get_all_providers() if p.tier == tier]
    
    def get_providers_with_capability(self, capability: ModelCapability) -> List[Provider]:
        """Get providers that support a specific capability."""
        return [p for p in self.get_all_providers() if capability in p.supported_capabilities]
    
    def recommend_provider(self, requirements: Dict[str, Any]) -> Optional[Provider]:
        """Recommend a provider based on requirements."""
        capabilities = requirements.get('capabilities', [])
        budget = requirements.get('budget', 'standard')
        reliability = requirements.get('reliability', 'standard')
        
        candidates = self.get_all_providers()
        
        # Filter by capabilities
        if capabilities:
            candidates = [
                p for p in candidates 
                if all(cap in p.supported_capabilities for cap in capabilities)
            ]
        
        # Filter by budget/tier
        if budget == 'free':
            candidates = [p for p in candidates if p.tier == ProviderTier.FREE]
        elif budget == 'low':
            candidates = [p for p in candidates if p.tier in [ProviderTier.FREE, ProviderTier.BUDGET]]
        elif budget == 'high':
            candidates = [p for p in candidates if p.tier in [ProviderTier.PREMIUM, ProviderTier.ENTERPRISE]]
        
        # Filter by reliability
        if reliability == 'high':
            candidates = [p for p in candidates if p.tier in [ProviderTier.PREMIUM, ProviderTier.ENTERPRISE]]
        
        candidates = [p for p in candidates if p.is_active]
        
        if not candidates:
            return None
        
        # Sort by model count and tier (more models and higher tier is better)
        tier_priority = {
            ProviderTier.FREE: 1,
            ProviderTier.BUDGET: 2, 
            ProviderTier.STANDARD: 3,
            ProviderTier.PREMIUM: 4,
            ProviderTier.ENTERPRISE: 5
        }
        
        candidates.sort(
            key=lambda p: (tier_priority.get(p.tier, 0), p.model_count),
            reverse=True
        )
        
        return candidates[0]