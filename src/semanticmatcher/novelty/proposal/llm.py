"""
LLM-based class proposal system for novel class discovery.

Uses litellm with structured output to generate meaningful class names
and descriptions for clusters of novel samples.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..schemas import (
    ClassProposal,
    NovelClassAnalysis,
    NovelSampleMetadata,
)
from semanticmatcher.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default LLM providers with fallback chain
DEFAULT_PROVIDERS = [
    "openrouter/anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "openrouter/openai/gpt-4o",
]

# Model-specific configuration
MODEL_CONFIGS = {
    "claude": {
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "gpt-4": {
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "default": {
        "max_tokens": 4096,
        "temperature": 0.3,
    },
}


class LLMClassProposer:
    """
    Propose new class names and descriptions using LLMs.

    Uses litellm for multi-provider support with automatic fallback.
    """

    def __init__(
        self,
        primary_model: Optional[str] = None,
        provider: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        Initialize LLM class proposer.

        Args:
            primary_model: Primary model to use (e.g., 'openrouter/anthropic/claude-sonnet-4')
            provider: Preferred provider when auto-selecting a default model
            fallback_models: Fallback models if primary fails
            api_keys: API keys for providers (e.g., {'openrouter': 'sk-...'})
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.primary_model = primary_model or os.getenv(
            "LLM_CLASS_PROPOSER_MODEL",
            self._default_model_for_provider(provider),
        )
        default_fallbacks = [
            model for model in DEFAULT_PROVIDERS if model != self.primary_model
        ]
        self.fallback_models = fallback_models or default_fallbacks
        self.api_keys = api_keys or self._get_api_keys_from_env()
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set API keys as environment variables for litellm
        for provider, key in self.api_keys.items():
            env_var = self._provider_to_env_var(provider)
            if env_var and not os.getenv(env_var):
                os.environ[env_var] = key

    def _default_model_for_provider(self, provider: Optional[str]) -> str:
        """Select a default model, optionally honoring a preferred provider."""
        if not provider:
            return DEFAULT_PROVIDERS[0]

        provider_prefixes = {
            "openrouter": "openrouter/",
            "anthropic": "anthropic/",
            "openai": "openai/",
        }
        prefix = provider_prefixes.get(provider.lower())
        if not prefix:
            return DEFAULT_PROVIDERS[0]

        for model in DEFAULT_PROVIDERS:
            if model.startswith(prefix):
                return model
        return DEFAULT_PROVIDERS[0]

    def _get_api_keys_from_env(self) -> Dict[str, str]:
        """Get API keys from environment variables."""
        keys = {}
        env_mappings = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }

        for provider, env_var in env_mappings.items():
            key = os.getenv(env_var)
            if key:
                keys[provider] = key

        return keys

    def _provider_to_env_var(self, provider: str) -> Optional[str]:
        """Convert provider name to environment variable name."""
        mappings = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        return mappings.get(provider)

    def propose_classes(
        self,
        novel_samples: List[NovelSampleMetadata],
        existing_classes: List[str],
        context: Optional[str] = None,
    ) -> NovelClassAnalysis:
        """
        Propose new classes based on novel samples.

        Args:
            novel_samples: List of detected novel samples
            existing_classes: List of existing class names
            context: Optional domain context

        Returns:
            NovelClassAnalysis with proposed classes
        """
        if not novel_samples:
            raise ValueError("novel_samples cannot be empty")

        logger.info(
            f"Proposing classes for {len(novel_samples)} novel samples "
            f"using model: {self.primary_model}"
        )

        # Group samples by cluster if available
        clustered_samples = self._group_by_cluster(novel_samples)

        # Build proposal prompt
        prompt = self._build_proposal_prompt(
            novel_samples,
            existing_classes,
            clustered_samples,
            context,
        )

        # Call LLM with fallback
        try:
            response, model_used = self._call_llm_with_fallback(prompt)
            analysis = self._parse_response(response, novel_samples, model_used)
            logger.info(
                f"Successfully proposed {len(analysis.proposed_classes)} classes"
            )
            return analysis
        except Exception as e:
            logger.error(f"Failed to generate LLM proposals: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(novel_samples, existing_classes)

    def _group_by_cluster(
        self, samples: List[NovelSampleMetadata]
    ) -> Dict[Optional[int], List[NovelSampleMetadata]]:
        """Group samples by cluster ID."""
        clusters: Dict[Optional[int], List[NovelSampleMetadata]] = {}
        for sample in samples:
            cluster_id = sample.cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sample)
        return clusters

    def _build_proposal_prompt(
        self,
        novel_samples: List[NovelSampleMetadata],
        existing_classes: List[str],
        clustered_samples: Dict[Optional[int], List[NovelSampleMetadata]],
        context: Optional[str],
    ) -> str:
        """Build the proposal prompt for the LLM."""
        # Format samples for the prompt
        sample_texts = [
            f"- {sample.text}"
            for sample in novel_samples[:20]  # Limit to 20
        ]

        if len(novel_samples) > 20:
            sample_texts.append(f"... and {len(novel_samples) - 20} more samples")

        samples_section = "\n".join(sample_texts)

        # Format existing classes
        existing_classes_section = ", ".join(existing_classes)

        # Build context section
        context_section = ""
        if context:
            context_section = f"\n\nDomain Context: {context}"

        # Build cluster section if applicable
        cluster_section = ""
        if clustered_samples and len(clustered_samples) > 1:
            cluster_info = []
            for cluster_id, samples in clustered_samples.items():
                cluster_name = (
                    f"Cluster {cluster_id}" if cluster_id is not None else "Unclustered"
                )
                sample_list = ", ".join([s.text[:50] for s in samples[:3]])
                cluster_info.append(f"- {cluster_name}: {sample_list}")
            cluster_section = "\n\nNatural clusters found:\n" + "\n".join(cluster_info)

        prompt = f"""You are analyzing text samples that don't fit well into existing categories.

Existing Classes: {existing_classes_section}{context_section}{cluster_section}

Novel Samples (detected as not fitting existing classes):
{samples_section}

Your task is to:
1. Analyze these samples to identify meaningful new categories
2. Propose concise, descriptive class names
3. Provide justifications for each proposal
4. Identify samples that should be rejected as noise

IMPORTANT RESPONSE FORMAT:
You must respond with a valid JSON object matching this schema:
{{
  "proposed_classes": [
    {{
      "name": "class name (2-4 words)",
      "description": "clear description of what this class represents",
      "confidence": 0.0-1.0,
      "sample_count": number of samples fitting this class,
      "example_samples": ["sample1", "sample2", "sample3"],
      "justification": "why this class makes sense",
      "suggested_parent": null or "parent class name if hierarchical"
    }}
  ],
  "rejected_as_noise": ["sample text to reject"],
  "analysis_summary": "brief summary of your analysis",
  "cluster_count": number of distinct clusters found
}}

Guidelines:
- Class names should be concise (2-4 words), descriptive, and follow naming conventions of existing classes
- Confidence should reflect how clearly the samples form a coherent category
- Only propose classes with at least 3 supporting samples
- Reject samples that appear to be noise, errors, or too diverse
- Return "proposed_classes": [] if no coherent new class should be created
- Consider hierarchical relationships if relevant to the domain

Provide your analysis as a JSON object:"""

        return prompt

    def _call_llm_with_fallback(self, prompt: str) -> tuple[str, str]:
        """Call LLM with automatic fallback on failure."""
        models_to_try = [self.primary_model] + self.fallback_models

        last_error = None
        for model in models_to_try:
            try:
                logger.info(f"Trying model: {model}")
                response = self._call_litellm(model, prompt)
                return response, model
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                last_error = e
                continue

        # All models failed
        error_msg = f"All LLM providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from last_error

    def _call_litellm(self, model: str, prompt: str) -> str:
        """Call litellm completion API."""
        try:
            from litellm import completion
        except ImportError:
            raise ImportError(
                "litellm is required for LLM class proposal. "
                "Install with: pip install litellm"
            )

        model_config = self._get_model_config(model)
        temperature = (
            self.temperature
            if self.temperature is not None
            else model_config["temperature"]
        )
        max_tokens = (
            self.max_tokens
            if self.max_tokens is not None
            else model_config["max_tokens"]
        )

        response = completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at categorizing text samples and proposing meaningful class names.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        # Determine model type from model name
        if "claude" in model.lower():
            return MODEL_CONFIGS["claude"]
        elif "gpt-4" in model.lower():
            return MODEL_CONFIGS["gpt-4"]
        else:
            return MODEL_CONFIGS["default"]

    def _parse_response(
        self,
        response: str,
        novel_samples: List[NovelSampleMetadata],
        model_used: Optional[str] = None,
    ) -> NovelClassAnalysis:
        """Parse structured LLM response into NovelClassAnalysis."""
        import json

        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block markers
            lines = response.split("\n")
            if lines[0].startswith("```json"):
                response = "\n".join(lines[1:-1])
            elif lines[0].startswith("```"):
                response = "\n".join(lines[1:-1])
            else:
                # Try to find JSON content
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    response = response[start:end]

        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e

        # Validate with Pydantic
        try:
            return NovelClassAnalysis(
                proposed_classes=[
                    ClassProposal(**proposal)
                    for proposal in data.get("proposed_classes", [])
                ],
                rejected_as_noise=data.get("rejected_as_noise", []),
                analysis_summary=data.get("analysis_summary", ""),
                cluster_count=data.get("cluster_count", 0),
                model_used=model_used or self.primary_model,
            )
        except ValidationError as e:
            raise ValueError(f"Response validation failed: {e}") from e

    def _create_fallback_analysis(
        self, novel_samples: List[NovelSampleMetadata], existing_classes: List[str]
    ) -> NovelClassAnalysis:
        """Create fallback analysis when LLM fails."""
        logger.warning("Creating fallback analysis due to LLM failure")

        # Simple fallback: group by predicted class
        predicted_groups: Dict[str, List[NovelSampleMetadata]] = {}
        for sample in novel_samples:
            pred_class = sample.predicted_class
            if pred_class not in predicted_groups:
                predicted_groups[pred_class] = []
            predicted_groups[pred_class].append(sample)

        proposals = []
        for pred_class, samples in predicted_groups.items():
            if len(samples) >= 3:
                proposals.append(
                    ClassProposal(
                        name=f"Novel {pred_class}",
                        description=f"Samples related to {pred_class} that don't fit existing categories",
                        confidence=0.5,
                        sample_count=len(samples),
                        example_samples=[s.text for s in samples[:3]],
                        justification=f"Grouped by predicted class '{pred_class}'",
                        suggested_parent=pred_class,
                    )
                )

        return NovelClassAnalysis(
            proposed_classes=proposals,
            rejected_as_noise=[],
            analysis_summary="Fallback analysis due to LLM failure. Samples grouped by predicted class.",
            cluster_count=len(proposals),
            model_used="fallback",
        )
