# Contributing

## Quick Start

Set up the development environment from the repo root:

```bash
uv sync --group dev
```

## Common Commands

```bash
uv run python -m pytest
uv run python -m pytest tests/test_core/test_matcher.py -q
uv run python -m build
uv run ruff check .
```

## Project Conventions

- PyPI distribution name: `semantic-matcher`
- Python import path: `semanticmatcher`
- Repo folder name: `semantic_matcher`
- Library/package code lives in `src/semanticmatcher/`

## Where Changes Go

- New library features: `src/semanticmatcher/`
- Tests for library changes: `tests/` (mirror package areas where practical)
- User-facing examples: `examples/`
- Exploratory scripts / benchmarks: `experiments/`
- Jupyter notebooks only: `notebooks/`
- Documentation updates: `docs/`

## CLI Mapping

- `semanticmatcher-ingest` entrypoint is implemented in `src/semanticmatcher/ingestion/cli.py`

## Notes for Contributors

- Prefer non-breaking changes to public imports in `semanticmatcher`.
- Keep docs and examples aligned with actual file locations when moving experiments.
- Avoid committing generated artifacts (`dist/`, `docs/build/`, checkpoints, `__pycache__`).
