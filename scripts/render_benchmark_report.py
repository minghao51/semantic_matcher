"""Render benchmark JSON artifacts as compact markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: List[dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _render_model_benchmark(rows: List[dict]) -> str:
    sections = []
    by_section = {}
    for row in rows:
        by_section.setdefault(row.get("section", "unknown"), []).append(row)

    for section, section_rows in by_section.items():
        sections.append(f"## `{section}`")
        summary_columns = [
            "track",
            "mode",
            "model",
            "status",
            "throughput_qps",
            "avg_latency",
            "accuracy_split",
            "base_accuracy",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
        ]
        summary_columns = [
            col for col in summary_columns if any(col in row for row in section_rows)
        ]
        sections.append(_markdown_table(section_rows, summary_columns))

        perturbation_columns = [
            "model",
            "mode",
            "typo_accuracy",
            "remove_parenthetical_accuracy",
            "ampersand_expanded_accuracy",
            "first_clause_accuracy",
            "normalized_verbatim_accuracy",
        ]
        perturbation_columns = [
            col
            for col in perturbation_columns
            if any(col in row for row in section_rows)
        ]
        if len(perturbation_columns) > 2:
            sections.append("")
            sections.append("Perturbation breakdown:")
            sections.append(_markdown_table(section_rows, perturbation_columns))

    return "\n\n".join(sections)


def _render_speed_benchmark(rows: List[dict]) -> str:
    sections = []
    by_section = {}
    for row in rows:
        by_section.setdefault(row.get("section", "unknown"), []).append(row)

    for section, section_rows in by_section.items():
        sections.append(f"## `{section}`")
        columns = [
            "mode",
            "route",
            "construct_seconds",
            "fit_seconds",
            "cold_query_seconds",
            "match_seconds",
            "end_to_end_seconds",
            "qps",
            "avg_ms_per_query",
            "end_to_end_ms_per_query",
        ]
        columns = [col for col in columns if any(col in row for row in section_rows)]
        sections.append(_markdown_table(section_rows, columns))

    return "\n\n".join(sections)


def render(rows: List[dict]) -> str:
    if not rows:
        return "No benchmark rows found."
    if "route" in rows[0]:
        return _render_speed_benchmark(rows)
    return _render_model_benchmark(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Benchmark JSON artifact to render")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("Expected a JSON array of benchmark rows")

    print(render(payload))


if __name__ == "__main__":
    main()
