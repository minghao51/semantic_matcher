"""
File-based storage for novel class discovery results.

Provides utilities for saving and loading proposals in YAML format.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from .schemas import NovelClassDiscoveryReport
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def save_proposals(
    report: NovelClassDiscoveryReport,
    output_dir: Union[str, Path] = "./proposals",
    format: str = "yaml",
) -> str:
    """
    Save novel class discovery report to file.

    Args:
        report: Discovery report to save
        output_dir: Directory to save proposals in
        format: Output format ('yaml' or 'json')

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename. Timestamp-only names can collide when
    # several discovery reports are saved within the same second.
    timestamp = report.timestamp.strftime("%Y%m%d-%H%M%S")
    filename = f"discovery_{timestamp}_{report.discovery_id}.{format}"
    output_path = output_dir / filename

    # Convert to dict
    data = _report_to_dict(report)

    # Save based on format
    if format == "yaml":
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved discovery report to {output_path}")
    return str(output_path)


def load_proposals(path: Union[str, Path]) -> NovelClassDiscoveryReport:
    """
    Load novel class discovery report from file.

    Args:
        path: Path to proposal file

    Returns:
        NovelClassDiscoveryReport

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Proposal file not found: {path}")

    # Load based on format
    suffix = path.suffix.lower()
    if suffix in [".yaml", ".yml"]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Convert dict to report
    report = _dict_to_report(data)

    logger.info(f"Loaded discovery report from {path}")
    return report


def list_proposals(
    output_dir: Union[str, Path] = "./proposals",
    sort: str = "newest",
) -> List[Dict[str, Any]]:
    """
    List all discovery reports in output directory.

    Args:
        output_dir: Directory containing proposals
        sort: Sort order ('newest', 'oldest', 'name')

    Returns:
        List of proposal metadata dicts
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    proposals = []
    for path in output_dir.glob("discovery_*.yaml"):
        try:
            # Extract metadata from filename
            stem = path.stem  # e.g., "discovery_20250317-143000_ab12cd34"
            timestamp_str = stem.split("_", 2)[1]  # "20250317-143000"
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")

            proposals.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "timestamp": timestamp,
                    "format": "yaml",
                }
            )
        except (ValueError, IndexError):
            logger.warning(f"Could not parse filename: {path.name}")
            continue

    # Also check JSON files
    for path in output_dir.glob("discovery_*.json"):
        try:
            stem = path.stem
            timestamp_str = stem.split("_", 2)[1]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")

            proposals.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "timestamp": timestamp,
                    "format": "json",
                }
            )
        except (ValueError, IndexError):
            logger.warning(f"Could not parse filename: {path.name}")
            continue

    # Sort
    if sort == "newest":
        proposals.sort(key=lambda x: x["timestamp"], reverse=True)
    elif sort == "oldest":
        proposals.sort(key=lambda x: x["timestamp"])
    elif sort == "name":
        proposals.sort(key=lambda x: x["filename"])

    return proposals


def _report_to_dict(report: NovelClassDiscoveryReport) -> Dict[str, Any]:
    """Convert NovelClassDiscoveryReport to dict for serialization."""
    # Convert detection_config to dict, handling enums properly
    detection_config_dict = (
        report.detection_config.model_dump()
        if hasattr(report.detection_config, "model_dump")
        else dict(report.detection_config)
    )

    # Convert DetectionStrategy enums to strings in detection_config
    if "strategies" in detection_config_dict:
        detection_config_dict["strategies"] = [
            s.value if hasattr(s, "value") else str(s)
            for s in detection_config_dict["strategies"]
        ]

    # Convert novel_sample_report config to dict, handling enums properly
    novel_sample_config = (
        report.novel_sample_report.config.copy()
        if hasattr(report.novel_sample_report.config, "copy")
        else dict(report.novel_sample_report.config)
    )
    if "strategies" in novel_sample_config:
        novel_sample_config["strategies"] = [
            s.value if hasattr(s, "value") else str(s)
            for s in novel_sample_config["strategies"]
        ]

    data = {
        "discovery_id": report.discovery_id,
        "timestamp": report.timestamp.isoformat(),
        "matcher_config": report.matcher_config,
        "detection_config": detection_config_dict,
        "novel_sample_report": {
            "novel_samples": [
                {
                    "text": s.text,
                    "index": s.index,
                    "confidence": s.confidence,
                    "predicted_class": s.predicted_class,
                    "embedding_distance": s.embedding_distance,
                    "margin_score": s.margin_score,
                    "entropy_score": s.entropy_score,
                    "uncertainty_score": s.uncertainty_score,
                    "knn_novelty_score": s.knn_novelty_score,
                    "knn_mean_distance": s.knn_mean_distance,
                    "knn_nearest_distance": s.knn_nearest_distance,
                    "predicted_class_neighbor_ratio": s.predicted_class_neighbor_ratio,
                    "predicted_class_support": s.predicted_class_support,
                    "neighbor_labels": s.neighbor_labels,
                    "neighbor_distances": s.neighbor_distances,
                    "cluster_id": s.cluster_id,
                    "cluster_validation_passed": s.cluster_validation_passed,
                    "cluster_support_score": s.cluster_support_score,
                    "novelty_score": s.novelty_score,
                    "signals": s.signals,
                }
                for s in report.novel_sample_report.novel_samples
            ],
            # Convert DetectionStrategy enums to strings
            "detection_strategies": [
                s.value for s in report.novel_sample_report.detection_strategies
            ],
            "config": novel_sample_config,
            "signal_counts": report.novel_sample_report.signal_counts,
        },
    }

    # Add class proposals if available
    if report.class_proposals:
        data["class_proposals"] = {
            "proposed_classes": [
                {
                    "name": p.name,
                    "description": p.description,
                    "confidence": p.confidence,
                    "sample_count": p.sample_count,
                    "example_samples": p.example_samples,
                    "justification": p.justification,
                    "suggested_parent": p.suggested_parent,
                }
                for p in report.class_proposals.proposed_classes
            ],
            "rejected_as_noise": report.class_proposals.rejected_as_noise,
            "analysis_summary": report.class_proposals.analysis_summary,
            "cluster_count": report.class_proposals.cluster_count,
            "model_used": report.class_proposals.model_used,
        }

    # Add metadata
    data["metadata"] = report.metadata

    return data


def _dict_to_report(data: Dict[str, Any]) -> NovelClassDiscoveryReport:
    """Convert dict to NovelClassDiscoveryReport."""
    from .schemas import (
        ClassProposal,
        NovelClassAnalysis,
        NovelSampleMetadata,
        NovelSampleReport,
        DetectionStrategy,
    )

    # Parse timestamp
    timestamp = datetime.fromisoformat(data["timestamp"])

    # Reconstruct novel sample report
    novel_samples = [
        NovelSampleMetadata(**s) for s in data["novel_sample_report"]["novel_samples"]
    ]

    # Handle detection_strategies - convert from strings or enum values
    detection_strategies_raw = data["novel_sample_report"]["detection_strategies"]
    detection_strategies = []
    for s in detection_strategies_raw:
        if isinstance(s, str):
            detection_strategies.append(DetectionStrategy(s))
        elif isinstance(s, dict) and "value" in s:
            detection_strategies.append(DetectionStrategy(s["value"]))
        else:
            detection_strategies.append(DetectionStrategy(s))

    novel_sample_report = NovelSampleReport(
        novel_samples=novel_samples,
        detection_strategies=detection_strategies,
        config=data["novel_sample_report"]["config"],
        signal_counts=data["novel_sample_report"]["signal_counts"],
    )

    # Reconstruct class proposals if available
    class_proposals = None
    if "class_proposals" in data and data["class_proposals"]:
        proposal_data = data["class_proposals"]
        class_proposals = NovelClassAnalysis(
            proposed_classes=[
                ClassProposal(**p) for p in proposal_data["proposed_classes"]
            ],
            rejected_as_noise=proposal_data.get("rejected_as_noise", []),
            analysis_summary=proposal_data.get("analysis_summary", ""),
            cluster_count=proposal_data.get("cluster_count", 0),
            model_used=proposal_data.get("model_used", ""),
        )

    # Create report
    report = NovelClassDiscoveryReport(
        discovery_id=data["discovery_id"],
        timestamp=timestamp,
        matcher_config=data["matcher_config"],
        detection_config=data["detection_config"],
        novel_sample_report=novel_sample_report,
        class_proposals=class_proposals,
        metadata=data.get("metadata", {}),
    )

    return report


def export_summary(
    report: NovelClassDiscoveryReport,
    output_path: Union[str, Path],
    format: str = "markdown",
) -> None:
    """
    Export a human-readable summary of the discovery report.

    Args:
        report: Discovery report to export
        output_path: Path to save summary
        format: Output format ('markdown' or 'text')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build summary
    lines = [
        "# Novel Class Discovery Report",
        "",
        f"**Discovery ID:** {report.discovery_id}",
        f"**Timestamp:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Novel Samples Detected:** {len(report.novel_sample_report.novel_samples)}",
        f"- **Detection Strategies:** {', '.join([s.value for s in report.novel_sample_report.detection_strategies])}",
        "",
    ]

    if report.class_proposals:
        lines.extend(
            [
                "## Proposed Classes",
                "",
                f"**Number of Proposals:** {len(report.class_proposals.proposed_classes)}",
                f"**LLM Model Used:** {report.class_proposals.model_used}",
                "",
            ]
        )

        for i, proposal in enumerate(report.class_proposals.proposed_classes, 1):
            lines.extend(
                [
                    f"### {i}. {proposal.name}",
                    "",
                    f"**Description:** {proposal.description}",
                    f"**Confidence:** {proposal.confidence:.2%}",
                    f"**Sample Count:** {proposal.sample_count}",
                    "",
                    f"**Justification:** {proposal.justification}",
                    "",
                    "**Example Samples:**",
                ]
            )
            for example in proposal.example_samples:
                lines.append(f"- {example}")
            lines.append("")

        if report.class_proposals.rejected_as_noise:
            lines.extend(
                [
                    "## Rejected as Noise",
                    "",
                ]
            )
            for noise in report.class_proposals.rejected_as_noise:
                lines.append(f"- {noise}")
            lines.append("")

    lines.extend(
        [
            "## Novel Samples",
            "",
        ]
    )

    for sample in report.novel_sample_report.novel_samples[:20]:  # Limit to 20
        lines.extend(
            [
                f"### Sample {sample.index}",
                "",
                f"**Text:** {sample.text}",
                f"**Predicted Class:** {sample.predicted_class}",
                f"**Confidence:** {sample.confidence:.2%}",
                f"**Novelty Score:** {sample.novelty_score:.2%}"
                if sample.novelty_score is not None
                else "**Novelty Score:** n/a",
                f"**Signals:** {', '.join([k for k, v in sample.signals.items() if v])}",
                "",
            ]
        )

    if len(report.novel_sample_report.novel_samples) > 20:
        lines.append(
            f"... and {len(report.novel_sample_report.novel_samples) - 20} more samples"
        )

    # Write to file
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")

    logger.info(f"Exported summary to {output_path}")
