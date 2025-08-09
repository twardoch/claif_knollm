#!/usr/bin/env python3
# this_file: src/claif_knollm/client.py

"""
Knollm provider client with intelligent routing and multi-provider support.

This module implements the main client that integrates with the Claif ecosystem,
providing intelligent routing across multiple LLM providers based on cost,
performance, and availability optimization.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum

from .models import (
    Provider, Model, SearchFilter, ModelCapability, ProviderTier,
    ProviderStatus
)
from .registry import ModelRegistry, ProviderRegistry

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for selecting providers."""
    COST_OPTIMIZED = "cost_optimized"
    SPEED_OPTIMIZED = "speed_optimized" 
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"
    ADAPTIVE = "adaptive"


@dataclass
class CompletionRequest:
    """Request for a completion with routing metadata."""
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    
    # Routing preferences
    preferred_providers: Optional[List[str]] = None
    max_cost_per_1k_tokens: Optional[Decimal] = None
    min_context_window: Optional[int] = None
    required_capabilities: Optional[List[ModelCapability]] = None
    
    # Request metadata
    request_id: Optional[str] = None
    priority: int = 0  # Higher priority requests get better providers


@dataclass 
class CompletionResponse:
    """Response from a completion request."""
    content: str
    model: str
    provider: str
    usage: Dict[str, Any]
    cost: Optional[Decimal] = None
    response_time_ms: Optional[float] = None
    request_id: Optional[str] = None


class PerformanceTracker:
    """Tracks provider performance metrics."""
    
    def __init__(self, history_limit: int = 1000):
        self.history_limit = history_limit
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_request(
        self,
        provider: str,
        success: bool,
        response_time_ms: float,
        cost: Optional[Decimal] = None,
        error: Optional[str] = None
    ):
        """Record a request for performance tracking."""
        if provider not in self.metrics:
            self.metrics[provider] = []
        
        record = {
            'timestamp': datetime.utcnow(),
            'success': success,
            'response_time_ms': response_time_ms,
            'cost': cost,
            'error': error
        }
        
        self.metrics[provider].append(record)
        
        # Keep only recent history
        if len(self.metrics[provider]) > self.history_limit:
            self.metrics[provider] = self.metrics[provider][-self.history_limit:]
    
    def get_provider_stats(self, provider: str, window_hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for a provider."""
        if provider not in self.metrics:
            return {
                'success_rate': None,
                'avg_response_time_ms': None,
                'avg_cost': None,
                'request_count': 0
            }
        
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent_metrics = [
            m for m in self.metrics[provider] 
            if m['timestamp'] >= cutoff
        ]
        
        if not recent_metrics:
            return {
                'success_rate': None,
                'avg_response_time_ms': None, 
                'avg_cost': None,
                'request_count': 0
            }
        
        success_count = sum(1 for m in recent_metrics if m['success'])
        total_requests = len(recent_metrics)
        
        avg_response_time = sum(m['response_time_ms'] for m in recent_metrics) / total_requests
        
        costs = [m['cost'] for m in recent_metrics if m['cost'] is not None]
        avg_cost = sum(costs) / len(costs) if costs else None
        
        return {
            'success_rate': success_count / total_requests,
            'avg_response_time_ms': avg_response_time,
            'avg_cost': avg_cost,
            'request_count': total_requests
        }


class SmartRouter:
    """Intelligent router for selecting optimal providers."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        self.registry = registry
        self.strategy = strategy
        self.performance_tracker = performance_tracker or PerformanceTracker()
        
        # Provider health cache
        self._provider_health: Dict[str, ProviderStatus] = {}
        self._health_cache_ttl = 300  # 5 minutes
        
        # Round robin state
        self._round_robin_index = 0
    
    def select_provider_and_model(self, request: CompletionRequest) -> Optional[tuple[str, str]]:
        """Select optimal provider and model for a request.
        
        Returns:
            Tuple of (provider_name, model_id) or None if no suitable option found
        """
        # Get candidate models based on request requirements
        candidates = self._get_candidate_models(request)
        
        if not candidates:
            logger.warning("No candidate models found for request")
            return None
        
        # Apply routing strategy
        if self.strategy == RoutingStrategy.COST_OPTIMIZED:
            selected = self._select_by_cost(candidates)
        elif self.strategy == RoutingStrategy.SPEED_OPTIMIZED:
            selected = self._select_by_speed(candidates)
        elif self.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            selected = self._select_by_quality(candidates)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            selected = self._select_round_robin(candidates)
        elif self.strategy == RoutingStrategy.ADAPTIVE:
            selected = self._select_adaptive(candidates)
        else:  # BALANCED
            selected = self._select_balanced(candidates)
        
        if selected:
            logger.info(f"Selected provider: {selected[0]}, model: {selected[1]}")
            return selected
        
        logger.warning("No provider selected by routing strategy")
        return None
    
    def _get_candidate_models(self, request: CompletionRequest) -> List[Model]:
        """Get candidate models that meet request requirements."""
        search_filter = SearchFilter(
            providers=request.preferred_providers,
            required_capabilities=request.required_capabilities,
            min_context_window=request.min_context_window,
            max_cost_per_1k_tokens=request.max_cost_per_1k_tokens,
            active_only=True,
            exclude_deprecated=True
        )
        
        # If specific model requested, try to find it
        if request.model:
            model = self.registry.get_model_by_id(request.model)
            if model and model.is_available and not model.is_deprecated:
                return [model]
            else:
                logger.warning(f"Requested model {request.model} not available")
        
        # Search for suitable models
        result = self.registry.search_models(search_filter)
        return result.models
    
    def _select_by_cost(self, candidates: List[Model]) -> Optional[tuple[str, str]]:
        """Select the cheapest available option."""
        # Filter out unhealthy providers
        healthy_candidates = [
            m for m in candidates 
            if self._is_provider_healthy(m.provider)
        ]
        
        if not healthy_candidates:
            healthy_candidates = candidates  # Fallback to all candidates
        
        # Sort by cost (free first, then by cost)
        def cost_key(model: Model) -> float:
            if model.is_free:
                return 0.0
            elif model.metrics and model.metrics.cost_per_1k_input_tokens:
                return float(model.metrics.cost_per_1k_input_tokens)
            else:
                return float('inf')
        
        cheapest = min(healthy_candidates, key=cost_key)
        return (cheapest.provider, cheapest.id)
    
    def _select_by_speed(self, candidates: List[Model]) -> Optional[tuple[str, str]]:
        """Select the fastest available option."""
        # Use performance tracker data
        best_model = None
        best_speed = float('inf')
        
        for model in candidates:
            if not self._is_provider_healthy(model.provider):
                continue
            
            stats = self.performance_tracker.get_provider_stats(model.provider)
            avg_time = stats.get('avg_response_time_ms')
            
            if avg_time is not None and avg_time < best_speed:
                best_speed = avg_time
                best_model = model
            elif avg_time is None and model.metrics and model.metrics.avg_latency_ms:
                # Fallback to model metrics
                if model.metrics.avg_latency_ms < best_speed:
                    best_speed = model.metrics.avg_latency_ms
                    best_model = model
        
        if best_model:
            return (best_model.provider, best_model.id)
        
        # Fallback - just pick first healthy candidate
        for model in candidates:
            if self._is_provider_healthy(model.provider):
                return (model.provider, model.id)
        
        return None
    
    def _select_by_quality(self, candidates: List[Model]) -> Optional[tuple[str, str]]:
        """Select the highest quality option."""
        # Filter healthy providers and sort by quality
        healthy_candidates = [
            m for m in candidates 
            if self._is_provider_healthy(m.provider)
        ]
        
        if not healthy_candidates:
            healthy_candidates = candidates
        
        def quality_key(model: Model) -> float:
            if model.metrics and model.metrics.quality_score:
                return model.metrics.quality_score
            elif model.popularity_score:
                return model.popularity_score
            else:
                # Fallback to provider tier and model features
                tier_scores = {
                    ProviderTier.ENTERPRISE: 0.9,
                    ProviderTier.PREMIUM: 0.8,
                    ProviderTier.STANDARD: 0.6,
                    ProviderTier.BUDGET: 0.4,
                    ProviderTier.FREE: 0.2
                }
                base_score = tier_scores.get(model.pricing_tier, 0.5)
                
                # Bonus for more capabilities
                capability_bonus = len(model.capabilities) * 0.01
                return min(base_score + capability_bonus, 1.0)
        
        best_model = max(healthy_candidates, key=quality_key)
        return (best_model.provider, best_model.id)
    
    def _select_balanced(self, candidates: List[Model]) -> Optional[tuple[str, str]]:
        """Select using balanced cost/speed/quality optimization."""
        healthy_candidates = [
            m for m in candidates 
            if self._is_provider_healthy(m.provider)
        ]
        
        if not healthy_candidates:
            healthy_candidates = candidates
        
        # Calculate composite score for each candidate
        best_model = None
        best_score = -1.0
        
        for model in healthy_candidates:
            score = 0.0
            
            # Cost score (30% weight) - lower cost is better
            if model.is_free:
                cost_score = 1.0
            elif model.metrics and model.metrics.cost_per_1k_input_tokens:
                # Normalize cost (assuming max reasonable cost of $0.10)
                cost = float(model.metrics.cost_per_1k_input_tokens)
                cost_score = max(0.0, 1.0 - cost / 0.10)
            else:
                cost_score = 0.5  # Neutral for unknown cost
            
            score += cost_score * 0.3
            
            # Speed score (25% weight)
            stats = self.performance_tracker.get_provider_stats(model.provider)
            if stats['avg_response_time_ms']:
                # Normalize response time (assuming max reasonable time of 10 seconds)
                time_score = max(0.0, 1.0 - stats['avg_response_time_ms'] / 10000)
            elif model.metrics and model.metrics.avg_latency_ms:
                time_score = max(0.0, 1.0 - model.metrics.avg_latency_ms / 10000)
            else:
                time_score = 0.5  # Neutral for unknown speed
            
            score += time_score * 0.25
            
            # Quality score (35% weight)
            if model.metrics and model.metrics.quality_score:
                quality_score = model.metrics.quality_score
            elif model.popularity_score:
                quality_score = model.popularity_score
            else:
                # Fallback scoring
                tier_scores = {
                    ProviderTier.ENTERPRISE: 0.9,
                    ProviderTier.PREMIUM: 0.8,
                    ProviderTier.STANDARD: 0.6,
                    ProviderTier.BUDGET: 0.4,
                    ProviderTier.FREE: 0.2
                }
                quality_score = tier_scores.get(model.pricing_tier, 0.5)
            
            score += quality_score * 0.35
            
            # Reliability score (10% weight)
            if stats['success_rate'] is not None:
                reliability_score = stats['success_rate']
            else:
                reliability_score = 0.8  # Assume decent reliability for unknown providers
            
            score += reliability_score * 0.10
            
            if score > best_score:
                best_score = score
                best_model = model
        
        if best_model:
            return (best_model.provider, best_model.id)
        
        return None
    
    def _select_round_robin(self, candidates: List[Model]) -> Optional[tuple[str, str]]:
        """Select using round-robin strategy."""
        healthy_candidates = [
            m for m in candidates 
            if self._is_provider_healthy(m.provider)
        ]
        
        if not healthy_candidates:
            healthy_candidates = candidates
        
        if not healthy_candidates:
            return None
        
        # Select next in round-robin
        selected = healthy_candidates[self._round_robin_index % len(healthy_candidates)]
        self._round_robin_index += 1
        
        return (selected.provider, selected.id)
    
    def _select_adaptive(self, candidates: List[Model]) -> Optional[tuple[str, str]]:
        """Select using adaptive strategy based on recent performance."""
        # This is a simplified adaptive strategy
        # In a full implementation, this could use ML models
        
        healthy_candidates = [
            m for m in candidates 
            if self._is_provider_healthy(m.provider)
        ]
        
        if not healthy_candidates:
            healthy_candidates = candidates
        
        # Score based on recent performance
        best_model = None
        best_score = -1.0
        
        for model in healthy_candidates:
            stats = self.performance_tracker.get_provider_stats(model.provider, window_hours=1)
            
            if stats['request_count'] < 5:
                # Not enough recent data, use balanced scoring
                return self._select_balanced([model])
            
            # Adaptive scoring based on recent success rate and speed
            success_rate = stats['success_rate'] or 0.5
            avg_time = stats['avg_response_time_ms'] or 5000
            
            # Higher success rate and lower response time is better
            score = success_rate * (1.0 / (1.0 + avg_time / 1000))
            
            if score > best_score:
                best_score = score
                best_model = model
        
        if best_model:
            return (best_model.provider, best_model.id)
        
        # Fallback to balanced
        return self._select_balanced(healthy_candidates)
    
    def _is_provider_healthy(self, provider_name: str) -> bool:
        """Check if a provider is healthy and available."""
        # Check cache first
        if provider_name in self._provider_health:
            status = self._provider_health[provider_name]
            if datetime.utcnow() - status.last_updated < timedelta(seconds=self._health_cache_ttl):
                return status.is_healthy
        
        # Get recent performance stats
        stats = self.performance_tracker.get_provider_stats(provider_name, window_hours=1)
        
        if stats['request_count'] == 0:
            # No recent data, assume healthy
            return True
        
        success_rate = stats['success_rate']
        
        # Consider healthy if success rate > 80%
        is_healthy = success_rate is None or success_rate > 0.8
        
        # Cache the result
        self._provider_health[provider_name] = ProviderStatus(
            provider_name=provider_name,
            is_healthy=is_healthy,
            success_rate=success_rate,
            avg_response_time_ms=stats['avg_response_time_ms'],
            last_updated=datetime.utcnow()
        )
        
        return is_healthy


class KnollmClient:
    """Main Knollm client with intelligent routing and multi-provider support."""
    
    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        fallback_providers: Optional[List[str]] = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600
    ):
        """Initialize the Knollm client.
        
        Args:
            routing_strategy: Strategy for selecting providers
            fallback_providers: List of fallback providers in order of preference
            enable_caching: Whether to enable response caching
            cache_ttl: Cache time-to-live in seconds
        """
        # Initialize registry and routing components
        self.registry = ModelRegistry()
        self.provider_registry = ProviderRegistry(self.registry)
        self.performance_tracker = PerformanceTracker()
        self.router = SmartRouter(
            self.registry,
            routing_strategy,
            self.performance_tracker
        )
        
        self.fallback_providers = fallback_providers or []
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Response cache
        self._response_cache: Dict[str, tuple[CompletionResponse, datetime]] = {}
        
        # Provider clients would be initialized here in a full implementation
        # For now, this is a conceptual framework
        self._provider_clients: Dict[str, Any] = {}
        
        logger.info("Knollm client initialized")
    
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Create a completion using intelligent routing.
        
        Args:
            messages: List of messages for the completion
            model: Specific model to use (optional)
            **kwargs: Additional completion parameters
            
        Returns:
            CompletionResponse with result and metadata
        """
        request = CompletionRequest(
            messages=messages,
            model=model,
            **{k: v for k, v in kwargs.items() if hasattr(CompletionRequest, k)}
        )
        
        # Check cache first
        if self.enable_caching:
            cached_response = self._get_cached_response(request)
            if cached_response:
                logger.info("Returning cached response")
                return cached_response
        
        # Select provider and model
        provider_model = self.router.select_provider_and_model(request)
        
        if not provider_model:
            raise ValueError("No suitable provider found for request")
        
        provider_name, model_id = provider_model
        
        # Try primary selection
        try:
            response = await self._make_request(provider_name, model_id, request)
            
            # Cache successful response
            if self.enable_caching:
                self._cache_response(request, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed for {provider_name}/{model_id}: {e}")
            
            # Try fallback providers
            for fallback_provider in self.fallback_providers:
                if fallback_provider == provider_name:
                    continue  # Skip the one that just failed
                
                # Find a suitable model from the fallback provider
                fallback_models = self.registry.get_models_by_provider(fallback_provider)
                
                # Apply the same filters to find a suitable fallback model
                suitable_fallback = None
                for fallback_model in fallback_models:
                    if self._is_model_suitable(fallback_model, request):
                        suitable_fallback = fallback_model
                        break
                
                if suitable_fallback:
                    try:
                        logger.info(f"Trying fallback: {fallback_provider}/{suitable_fallback.id}")
                        response = await self._make_request(
                            fallback_provider,
                            suitable_fallback.id,
                            request
                        )
                        
                        if self.enable_caching:
                            self._cache_response(request, response)
                        
                        return response
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback failed for {fallback_provider}: {fallback_error}")
                        continue
            
            # All providers failed
            raise RuntimeError(f"All providers failed. Last error: {e}")
    
    async def _make_request(
        self,
        provider_name: str,
        model_id: str,
        request: CompletionRequest
    ) -> CompletionResponse:
        """Make a request to a specific provider.
        
        This is a placeholder method - in a real implementation,
        this would call the actual provider APIs.
        """
        start_time = time.time()
        
        try:
            # Placeholder implementation
            # In reality, this would use provider-specific clients
            logger.info(f"Making request to {provider_name}/{model_id}")
            
            # Simulate API call
            await asyncio.sleep(0.1)  # Simulate network delay
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Create mock response
            response = CompletionResponse(
                content="This is a mock response from the Knollm client.",
                model=model_id,
                provider=provider_name,
                usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                response_time_ms=response_time_ms,
                request_id=request.request_id
            )
            
            # Record successful request
            self.performance_tracker.record_request(
                provider_name,
                success=True,
                response_time_ms=response_time_ms
            )
            
            return response
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record failed request
            self.performance_tracker.record_request(
                provider_name,
                success=False,
                response_time_ms=response_time_ms,
                error=str(e)
            )
            
            raise
    
    def _is_model_suitable(self, model: Model, request: CompletionRequest) -> bool:
        """Check if a model meets the request requirements."""
        if not model.is_available or model.is_deprecated:
            return False
        
        if request.required_capabilities:
            if not all(cap in model.capabilities for cap in request.required_capabilities):
                return False
        
        if request.min_context_window:
            if not model.context_window or model.context_window < request.min_context_window:
                return False
        
        if request.max_cost_per_1k_tokens:
            if not model.is_free:
                if not model.metrics or not model.metrics.cost_per_1k_input_tokens:
                    return False  # Unknown cost, assume expensive
                if model.metrics.cost_per_1k_input_tokens > request.max_cost_per_1k_tokens:
                    return False
        
        return True
    
    def _generate_cache_key(self, request: CompletionRequest) -> str:
        """Generate a cache key for a request."""
        # Simplified cache key generation
        # In reality, this would be more sophisticated
        import hashlib
        
        key_data = {
            'messages': str(request.messages),
            'model': request.model,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, request: CompletionRequest) -> Optional[CompletionResponse]:
        """Get cached response if available and valid."""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self._response_cache:
            response, cached_time = self._response_cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(seconds=self.cache_ttl):
                return response
            else:
                # Cache expired
                del self._response_cache[cache_key]
        
        return None
    
    def _cache_response(self, request: CompletionRequest, response: CompletionResponse):
        """Cache a response."""
        cache_key = self._generate_cache_key(request)
        self._response_cache[cache_key] = (response, datetime.utcnow())
        
        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self._response_cache) > 1000:
            # Remove 20% of oldest entries
            sorted_items = sorted(
                self._response_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            
            items_to_remove = len(sorted_items) // 5
            for key, _ in sorted_items[:items_to_remove]:
                del self._response_cache[key]
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all providers."""
        providers = self.provider_registry.get_all_providers()
        stats = {}
        
        for provider in providers:
            provider_stats = self.performance_tracker.get_provider_stats(provider.name)
            stats[provider.name] = {
                'display_name': provider.display_name,
                'tier': provider.tier.value,
                'model_count': provider.model_count,
                'performance': provider_stats
            }
        
        return stats
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """Change the routing strategy."""
        self.router.strategy = strategy
        logger.info(f"Routing strategy changed to {strategy.value}")
    
    def add_fallback_provider(self, provider_name: str):
        """Add a fallback provider."""
        if provider_name not in self.fallback_providers:
            self.fallback_providers.append(provider_name)
            logger.info(f"Added fallback provider: {provider_name}")
    
    def clear_cache(self):
        """Clear the response cache."""
        self._response_cache.clear()
        logger.info("Response cache cleared")
    
    def reload_registry(self):
        """Reload the model and provider registry."""
        self.registry.reload_data()
        logger.info("Registry reloaded")