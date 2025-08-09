#!/usr/bin/env python3
# this_file: src/claif_knollm/cli.py

"""
Command-line interface for the Knollm provider catalog.

This module provides a rich CLI for searching, filtering, and exploring
the comprehensive catalog of LLM providers, models, and Python libraries.
"""

import sys
from typing import List, Optional, Dict, Any
from decimal import Decimal
import logging

try:
    import fire
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install with: pip install claif_knollm[cli]")
    sys.exit(1)

from .models import (
    SearchFilter, ModelCapability, ProviderTier, ModelModality,
    LibraryCategory, LibraryFeature
)
from .registry import ModelRegistry, ProviderRegistry
from .data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()


class KnollmCLI:
    """Comprehensive CLI for the Knollm LLM provider catalog."""
    
    def __init__(self):
        """Initialize the CLI with registry."""
        self.registry = None
        self.provider_registry = None
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the registry with loading indicator."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Loading Knollm registry...", total=None)
            try:
                self.registry = ModelRegistry()
                self.provider_registry = ProviderRegistry(self.registry)
            except Exception as e:
                console.print(f"[red]Error loading registry: {e}[/red]")
                sys.exit(1)
    
    def providers(self):
        """Provider management commands."""
        return ProviderCommands(self.provider_registry)
    
    def models(self):
        """Model search and discovery commands."""
        return ModelCommands(self.registry)
    
    def libraries(self):
        """API library information commands.""" 
        return LibraryCommands(self.registry)
    
    def stats(self):
        """Display comprehensive statistics about the catalog."""
        stats = self.registry.get_provider_stats()
        
        # Main statistics panel
        stats_text = f"""
**Total Providers:** {stats['total_providers']}
**Total Models:** {stats['total_models']}
**Average Models per Provider:** {stats['total_models'] / stats['total_providers']:.1f}
        """
        
        console.print(Panel(stats_text, title="📊 Catalog Statistics", border_style="blue"))
        
        # Top providers by model count
        top_providers = sorted(
            stats['models_by_provider'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        provider_table = Table(title="Top Providers by Model Count")
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Model Count", justify="right", style="green")
        
        for provider, count in top_providers:
            provider_table.add_row(provider, str(count))
        
        # Capability distribution
        cap_table = Table(title="Capability Distribution")
        cap_table.add_column("Capability", style="magenta")
        cap_table.add_column("Model Count", justify="right", style="green")
        
        for cap, count in sorted(stats['capabilities_distribution'].items()):
            cap_table.add_row(cap.replace('_', ' ').title(), str(count))
        
        console.print(Columns([provider_table, cap_table]))
        
        return stats


class ProviderCommands:
    """Commands for provider management."""
    
    def __init__(self, provider_registry: ProviderRegistry):
        self.provider_registry = provider_registry
    
    def list(self, tier: Optional[str] = None, capability: Optional[str] = None, active_only: bool = True):
        """List all providers with optional filtering.
        
        Args:
            tier: Filter by provider tier (free, budget, standard, premium, enterprise)
            capability: Filter by capability (vision, function_calling, etc.)
            active_only: Only show active providers
        """
        providers = self.provider_registry.get_all_providers()
        
        # Apply filters
        if tier:
            try:
                tier_enum = ProviderTier(tier.lower())
                providers = [p for p in providers if p.tier == tier_enum]
            except ValueError:
                console.print(f"[red]Invalid tier: {tier}. Valid tiers: {[t.value for t in ProviderTier]}[/red]")
                return
        
        if capability:
            try:
                cap_enum = ModelCapability(capability.lower())
                providers = [p for p in providers if cap_enum in p.supported_capabilities]
            except ValueError:
                console.print(f"[red]Invalid capability: {capability}[/red]")
                return
        
        if active_only:
            providers = [p for p in providers if p.is_active]
        
        if not providers:
            console.print("[yellow]No providers found matching criteria.[/yellow]")
            return
        
        # Create provider table
        table = Table(title=f"LLM Providers ({len(providers)} found)")
        table.add_column("Name", style="cyan")
        table.add_column("Tier", style="green")
        table.add_column("Models", justify="right", style="blue")
        table.add_column("Capabilities", style="magenta")
        table.add_column("Status", style="yellow")
        
        for provider in sorted(providers, key=lambda p: p.name):
            # Format capabilities
            caps = list(provider.supported_capabilities)[:3]  # Show first 3
            caps_str = ", ".join(cap.value.replace('_', ' ').title() for cap in caps)
            if len(provider.supported_capabilities) > 3:
                caps_str += f" (+{len(provider.supported_capabilities) - 3} more)"
            
            status = "✅ Active" if provider.is_active else "❌ Inactive"
            
            table.add_row(
                provider.display_name,
                provider.tier.value.title(),
                str(provider.model_count),
                caps_str,
                status
            )
        
        console.print(table)
        return providers
    
    def info(self, provider_name: str):
        """Show detailed information about a specific provider.
        
        Args:
            provider_name: Name of the provider to show info for
        """
        provider = self.provider_registry.get_provider(provider_name.lower())
        
        if not provider:
            console.print(f"[red]Provider '{provider_name}' not found.[/red]")
            # Show similar providers
            all_providers = self.provider_registry.get_all_providers()
            similar = [p for p in all_providers if provider_name.lower() in p.name.lower()]
            if similar:
                console.print(f"[yellow]Did you mean: {', '.join(p.name for p in similar[:3])}?[/yellow]")
            return
        
        # Main info panel
        info_text = f"""
**Display Name:** {provider.display_name}
**Base URL:** {provider.base_url}
**Authentication:** {provider.auth_type.value.replace('_', ' ').title()}
**Tier:** {provider.tier.value.title()}
**Model Count:** {provider.model_count}
**Free Tier:** {'Yes' if provider.free_tier_available else 'No'}
**Status:** {'Active' if provider.is_active else 'Inactive'}
        """
        
        if provider.description:
            info_text += f"\n**Description:** {provider.description}"
        
        console.print(Panel(info_text, title=f"🏢 {provider.display_name}", border_style="blue"))
        
        # Capabilities
        if provider.supported_capabilities:
            caps_text = "\n".join([
                f"• {cap.value.replace('_', ' ').title()}" 
                for cap in sorted(provider.supported_capabilities)
            ])
            console.print(Panel(caps_text, title="🚀 Capabilities", border_style="green"))
        
        # Technical details
        tech_details = []
        if provider.max_context_window:
            tech_details.append(f"**Max Context Window:** {provider.max_context_window:,} tokens")
        if provider.supports_streaming:
            tech_details.append("**Streaming:** Supported")
        if provider.supports_function_calling:
            tech_details.append("**Function Calling:** Supported") 
        if provider.supports_vision:
            tech_details.append("**Vision:** Supported")
        if provider.supports_multimodal:
            tech_details.append("**Multimodal:** Supported")
        
        if tech_details:
            console.print(Panel("\n".join(tech_details), title="⚙️ Technical Features", border_style="yellow"))
        
        return provider
    
    def recommend(self, **requirements):
        """Recommend a provider based on requirements.
        
        Args:
            **requirements: Requirements dict (capabilities, budget, reliability)
        """
        provider = self.provider_registry.recommend_provider(requirements)
        
        if not provider:
            console.print("[red]No provider found matching your requirements.[/red]")
            return
        
        console.print(f"[green]Recommended provider: {provider.display_name}[/green]")
        self.info(provider.name)
        
        return provider


class ModelCommands:
    """Commands for model search and discovery."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def search(
        self,
        query: Optional[str] = None,
        provider: Optional[str] = None,
        capability: Optional[str] = None,
        min_context: Optional[int] = None,
        max_cost: Optional[float] = None,
        free_only: bool = False,
        limit: int = 20
    ):
        """Search for models with various filters.
        
        Args:
            query: Text search in model names and descriptions
            provider: Filter by specific provider
            capability: Required capability
            min_context: Minimum context window size
            max_cost: Maximum cost per 1k tokens
            free_only: Only show free models
            limit: Maximum number of results
        """
        # Build search filter
        search_filter = SearchFilter(
            query=query,
            providers=[provider] if provider else None,
            required_capabilities=[ModelCapability(capability)] if capability else None,
            min_context_window=min_context,
            max_cost_per_1k_tokens=Decimal(str(max_cost)) if max_cost else None,
            free_only=free_only,
            limit=limit,
            sort_by='popularity',
            sort_order='desc'
        )
        
        # Perform search
        try:
            result = self.registry.search_models(search_filter)
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return
        
        if not result.models:
            console.print("[yellow]No models found matching criteria.[/yellow]")
            return
        
        # Display results
        table = Table(title=f"Model Search Results ({result.total_count} total, showing {len(result.models)})")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="green") 
        table.add_column("Context", justify="right", style="blue")
        table.add_column("Capabilities", style="magenta")
        table.add_column("Tier", style="yellow")
        
        for model in result.models:
            # Format capabilities (show first 3)
            caps = list(model.capabilities)[:3]
            caps_str = ", ".join(cap.value.replace('_', ' ').title() for cap in caps)
            if len(model.capabilities) > 3:
                caps_str += f" (+{len(model.capabilities) - 3})"
            
            # Format context window
            context_str = f"{model.context_window:,}" if model.context_window else "Unknown"
            
            # Format tier
            tier_str = model.pricing_tier.value.title() if model.pricing_tier else "Unknown"
            if model.is_free:
                tier_str = "🆓 Free"
            
            table.add_row(
                model.id,
                model.provider.title(),
                context_str,
                caps_str,
                tier_str
            )
        
        console.print(table)
        console.print(f"[dim]Search completed in {result.search_time_ms:.1f}ms[/dim]")
        
        return result.models
    
    def info(self, model_id: str):
        """Show detailed information about a specific model.
        
        Args:
            model_id: ID of the model to show info for
        """
        model = self.registry.get_model_by_id(model_id)
        
        if not model:
            console.print(f"[red]Model '{model_id}' not found.[/red]")
            # Search for similar models
            similar_result = self.registry.search_models(SearchFilter(
                query=model_id,
                limit=3
            ))
            if similar_result.models:
                console.print(f"[yellow]Similar models: {', '.join(m.id for m in similar_result.models)}[/yellow]")
            return
        
        # Main model info
        info_text = f"""
**Model ID:** {model.id}
**Provider:** {model.provider.title()}
**Display Name:** {model.display_name or model.id}
        """
        
        if model.description:
            info_text += f"\n**Description:** {model.description}"
        
        if model.model_family:
            info_text += f"\n**Family:** {model.model_family}"
        
        if model.context_window:
            info_text += f"\n**Context Window:** {model.context_window:,} tokens"
        
        if model.pricing_tier:
            tier_str = model.pricing_tier.value.title()
            if model.is_free:
                tier_str += " (Free)"
            info_text += f"\n**Pricing Tier:** {tier_str}"
        
        console.print(Panel(info_text, title=f"🤖 {model.display_name or model.id}", border_style="blue"))
        
        # Capabilities
        if model.capabilities:
            caps_text = "\n".join([
                f"• {cap.value.replace('_', ' ').title()}" 
                for cap in sorted(model.capabilities)
            ])
            console.print(Panel(caps_text, title="🚀 Capabilities", border_style="green"))
        
        # Modalities
        modalities_info = []
        if model.input_modalities:
            input_mods = ", ".join(mod.value.title() for mod in sorted(model.input_modalities))
            modalities_info.append(f"**Input:** {input_mods}")
        
        if model.output_modalities:
            output_mods = ", ".join(mod.value.title() for mod in sorted(model.output_modalities))
            modalities_info.append(f"**Output:** {output_mods}")
        
        if modalities_info:
            console.print(Panel("\n".join(modalities_info), title="📡 Modalities", border_style="yellow"))
        
        # Performance metrics
        if model.metrics:
            metrics_info = []
            if model.metrics.cost_per_1k_input_tokens:
                metrics_info.append(f"**Cost (1k input tokens):** ${model.metrics.cost_per_1k_input_tokens}")
            if model.metrics.cost_per_1k_output_tokens:
                metrics_info.append(f"**Cost (1k output tokens):** ${model.metrics.cost_per_1k_output_tokens}")
            if model.metrics.quality_score:
                metrics_info.append(f"**Quality Score:** {model.metrics.quality_score:.2f}/1.0")
            if model.metrics.avg_latency_ms:
                metrics_info.append(f"**Avg Latency:** {model.metrics.avg_latency_ms:.0f}ms")
            
            if metrics_info:
                console.print(Panel("\n".join(metrics_info), title="📊 Metrics", border_style="magenta"))
        
        # Tags
        if model.tags:
            tags_str = " ".join(f"#{tag}" for tag in sorted(model.tags))
            console.print(Panel(tags_str, title="🏷️ Tags", border_style="dim"))
        
        return model
    
    def cheapest(self, capability: Optional[str] = None, limit: int = 10):
        """Show the cheapest models, optionally filtered by capability.
        
        Args:
            capability: Filter by specific capability
            limit: Number of models to show
        """
        capabilities = None
        if capability:
            try:
                capabilities = [ModelCapability(capability)]
            except ValueError:
                console.print(f"[red]Invalid capability: {capability}[/red]")
                return
        
        models = self.registry.get_cheapest_models(limit=limit, capabilities=capabilities)
        
        if not models:
            console.print("[yellow]No models found with pricing information.[/yellow]")
            return
        
        table = Table(title=f"Cheapest Models ({len(models)} found)")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Cost (1k tokens)", justify="right", style="blue")
        table.add_column("Context", justify="right", style="yellow")
        
        for model in models:
            if model.is_free:
                cost_str = "FREE"
            elif model.metrics and model.metrics.cost_per_1k_input_tokens:
                cost_str = f"${model.metrics.cost_per_1k_input_tokens}"
            else:
                cost_str = "Unknown"
            
            context_str = f"{model.context_window:,}" if model.context_window else "Unknown"
            
            table.add_row(
                model.id,
                model.provider.title(),
                cost_str,
                context_str
            )
        
        console.print(table)
        return models
    
    def recommend(self, **requirements):
        """Recommend optimal models based on requirements.
        
        Args:
            **requirements: Requirements like task_type, budget, performance, etc.
        """
        recommendations = self.registry.get_model_recommendations(requirements)
        
        if not recommendations:
            console.print("[red]No models found matching your requirements.[/red]")
            return
        
        console.print(f"[green]Top {len(recommendations)} recommended models:[/green]")
        
        table = Table(title="Model Recommendations")
        table.add_column("Rank", justify="center", style="blue")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Match Score", justify="right", style="yellow")
        
        for i, model in enumerate(recommendations, 1):
            # Simple scoring based on matching requirements
            score = "⭐⭐⭐⭐⭐" if i <= 3 else "⭐⭐⭐⭐" if i <= 7 else "⭐⭐⭐"
            
            table.add_row(
                str(i),
                model.id,
                model.provider.title(),
                score
            )
        
        console.print(table)
        return recommendations
    
    def compare(self, models: List[str], criteria: Optional[List[str]] = None):
        """Compare multiple models across different criteria.
        
        Args:
            models: List of model IDs to compare
            criteria: List of criteria to compare (cost, speed, quality, context_window)
        """
        if not models or len(models) < 2:
            console.print("[red]Please provide at least 2 models to compare.[/red]")
            return
        
        if criteria is None:
            criteria = ['cost', 'quality', 'context_window']
        
        comparison = self.registry.compare_models(models, criteria)
        
        if not comparison:
            console.print("[red]Unable to compare models. Please check model IDs.[/red]")
            return
        
        table = Table(title="Model Comparison")
        table.add_column("Rank", justify="center", style="blue")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Overall Score", justify="right", style="yellow")
        
        for i, (model, score) in enumerate(comparison, 1):
            table.add_row(
                str(i),
                model.id,
                model.provider.title(),
                f"{score:.3f}"
            )
        
        console.print(table)
        
        # Show detailed comparison
        console.print("\n[bold]Detailed Comparison:[/bold]")
        for criterion in criteria:
            console.print(f"\n[bold]{criterion.title()}:[/bold]")
            for model, _ in comparison:
                if criterion == 'cost':
                    if model.is_free:
                        value = "FREE"
                    elif model.metrics and model.metrics.cost_per_1k_input_tokens:
                        value = f"${model.metrics.cost_per_1k_input_tokens}"
                    else:
                        value = "Unknown"
                elif criterion == 'quality':
                    value = f"{model.metrics.quality_score:.2f}" if model.metrics and model.metrics.quality_score else "Unknown"
                elif criterion == 'context_window':
                    value = f"{model.context_window:,}" if model.context_window else "Unknown"
                else:
                    value = "N/A"
                
                console.print(f"  • {model.id}: {value}")
        
        return comparison


class LibraryCommands:
    """Commands for API library information."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def list(self, category: Optional[str] = None, min_rating: Optional[float] = None):
        """List all API libraries with optional filtering.
        
        Args:
            category: Filter by library category
            min_rating: Minimum rating threshold
        """
        libraries = list(self.registry._libraries.values())
        
        # Apply filters
        if category:
            try:
                cat_enum = LibraryCategory(category.lower())
                libraries = [lib for lib in libraries if lib.category == cat_enum]
            except ValueError:
                console.print(f"[red]Invalid category: {category}[/red]")
                return
        
        if min_rating:
            libraries = [lib for lib in libraries if lib.rating >= min_rating]
        
        if not libraries:
            console.print("[yellow]No libraries found matching criteria.[/yellow]")
            return
        
        # Sort by rating
        libraries.sort(key=lambda x: x.rating, reverse=True)
        
        table = Table(title=f"Python LLM Libraries ({len(libraries)} found)")
        table.add_column("Library", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Rating", justify="center", style="yellow") 
        table.add_column("Features", style="magenta")
        table.add_column("Description", style="dim")
        
        for library in libraries:
            # Format rating as stars
            stars = "⭐" * int(library.rating)
            rating_str = f"{stars} ({library.rating})"
            
            # Format features (show first 3)
            features = list(library.supported_features)[:3]
            features_str = ", ".join(f.value.replace('_', ' ').title() for f in features)
            if len(library.supported_features) > 3:
                features_str += f" (+{len(library.supported_features) - 3})"
            
            # Truncate description
            desc = library.description[:60] + "..." if len(library.description) > 60 else library.description
            
            table.add_row(
                library.name,
                library.category.value.replace('_', ' ').title(),
                rating_str,
                features_str,
                desc
            )
        
        console.print(table)
        return libraries
    
    def info(self, library_name: str):
        """Show detailed information about a specific library.
        
        Args:
            library_name: Name of the library to show info for
        """
        library = self.registry._libraries.get(library_name.lower())
        
        if not library:
            console.print(f"[red]Library '{library_name}' not found.[/red]")
            # Show similar libraries
            similar = [lib for lib in self.registry._libraries.values() 
                      if library_name.lower() in lib.name.lower()]
            if similar:
                console.print(f"[yellow]Similar libraries: {', '.join(lib.name for lib in similar[:3])}[/yellow]")
            return
        
        # Main library info
        stars = "⭐" * int(library.rating)
        info_text = f"""
**Library:** {library.display_name or library.name}
**Category:** {library.category.value.replace('_', ' ').title()}
**Rating:** {stars} ({library.rating}/7)
**Installation:** {library.installation_command}
**Description:** {library.description}
        """
        
        console.print(Panel(info_text, title=f"📚 {library.display_name or library.name}", border_style="blue"))
        
        # Features
        if library.supported_features:
            features_text = "\n".join([
                f"• {feature.value.replace('_', ' ').title()}"
                for feature in sorted(library.supported_features)
            ])
            console.print(Panel(features_text, title="🚀 Features", border_style="green"))
        
        # Pros and Cons
        if library.pros or library.cons:
            pros_cons = []
            if library.pros:
                pros_cons.append("[green]**Pros:**[/green]")
                for pro in library.pros:
                    pros_cons.append(f"  ✅ {pro}")
            
            if library.cons:
                if library.pros:
                    pros_cons.append("")
                pros_cons.append("[red]**Cons:**[/red]")
                for con in library.cons:
                    pros_cons.append(f"  ❌ {con}")
            
            console.print(Panel("\n".join(pros_cons), title="⚖️ Pros & Cons", border_style="yellow"))
        
        # Use Cases
        if library.use_cases:
            use_cases_text = "\n".join([
                f"• {use_case}" for use_case in library.use_cases
            ])
            console.print(Panel(use_cases_text, title="💡 Use Cases", border_style="magenta"))
        
        # Code example
        if library.example_code:
            console.print(Panel(
                Syntax(library.example_code, "python", theme="monokai"),
                title="💻 Example Code",
                border_style="dim"
            ))
        
        return library
    
    def recommend(self, use_case: str, **preferences):
        """Recommend libraries based on use case and preferences.
        
        Args:
            use_case: The intended use case (e.g., 'simple', 'async', 'multi_provider', 'structured_output')
            **preferences: Additional preferences like min_rating, category, etc.
        """
        libraries = list(self.registry._libraries.values())
        
        # Filter by use case
        use_case_lower = use_case.lower()
        
        # Map use cases to library characteristics
        if 'simple' in use_case_lower or 'basic' in use_case_lower:
            # Prefer simple HTTP clients
            libraries = sorted(libraries, key=lambda x: (
                x.category == LibraryCategory.HTTP_CLIENT,
                x.rating
            ), reverse=True)
        
        elif 'async' in use_case_lower:
            # Prefer libraries with async support
            libraries = [lib for lib in libraries if LibraryFeature.ASYNC_SUPPORT in lib.supported_features]
            libraries.sort(key=lambda x: x.rating, reverse=True)
        
        elif 'multi' in use_case_lower and 'provider' in use_case_lower:
            # Prefer multi-provider libraries
            libraries = [lib for lib in libraries if lib.category == LibraryCategory.MULTI_PROVIDER]
            libraries.sort(key=lambda x: x.rating, reverse=True)
        
        elif 'structured' in use_case_lower or 'output' in use_case_lower:
            # Prefer libraries with structured output
            libraries = [lib for lib in libraries if LibraryFeature.STRUCTURED_OUTPUT in lib.supported_features]
            libraries.sort(key=lambda x: x.rating, reverse=True)
        
        elif 'framework' in use_case_lower or 'agent' in use_case_lower:
            # Prefer full frameworks
            libraries = [lib for lib in libraries if lib.category == LibraryCategory.FRAMEWORK]
            libraries.sort(key=lambda x: x.rating, reverse=True)
        
        else:
            # General recommendation - sort by rating
            libraries.sort(key=lambda x: x.rating, reverse=True)
        
        # Apply preference filters
        min_rating = preferences.get('min_rating')
        if min_rating:
            libraries = [lib for lib in libraries if lib.rating >= min_rating]
        
        category = preferences.get('category')
        if category:
            try:
                cat_enum = LibraryCategory(category.lower())
                libraries = [lib for lib in libraries if lib.category == cat_enum]
            except ValueError:
                pass
        
        # Show top recommendations
        top_libs = libraries[:5]
        
        if not top_libs:
            console.print("[red]No libraries found matching your criteria.[/red]")
            return
        
        console.print(f"[green]Top recommendations for '{use_case}':[/green]")
        
        table = Table(title="Library Recommendations")
        table.add_column("Rank", justify="center", style="blue")
        table.add_column("Library", style="cyan")
        table.add_column("Rating", justify="center", style="yellow")
        table.add_column("Why It's Recommended", style="green")
        
        for i, library in enumerate(top_libs, 1):
            stars = "⭐" * int(library.rating)
            rating_str = f"{stars} ({library.rating})"
            
            # Generate recommendation reason
            reasons = []
            if library.category == LibraryCategory.HTTP_CLIENT and 'simple' in use_case_lower:
                reasons.append("Simple HTTP interface")
            if LibraryFeature.ASYNC_SUPPORT in library.supported_features and 'async' in use_case_lower:
                reasons.append("Async support")
            if library.category == LibraryCategory.MULTI_PROVIDER:
                reasons.append("Multi-provider support")
            if LibraryFeature.STRUCTURED_OUTPUT in library.supported_features:
                reasons.append("Structured output")
            
            if not reasons:
                reasons.append(f"High rating ({library.rating})")
            
            table.add_row(
                str(i),
                library.name,
                rating_str,
                ", ".join(reasons)
            )
        
        console.print(table)
        return top_libs


def main():
    """Main CLI entry point."""
    try:
        fire.Fire(KnollmCLI)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("CLI error")


if __name__ == "__main__":
    main()