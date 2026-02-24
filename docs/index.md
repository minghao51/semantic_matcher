# Documentation Index

This folder is organized by audience and task:

- New users: start with [`quickstart.md`](./quickstart.md)
- Contributors / maintainers: see [`architecture.md`](./architecture.md)
- Notebook experiment scripts: see [`country-classifier-scripts.md`](./country-classifier-scripts.md)

## Reading Paths

### I want to use the library

1. Install the package
2. Run a minimal example in [`quickstart.md`](./quickstart.md)
3. Pick a matcher strategy:
   - `EntityMatcher` (few-shot SetFit training)
   - `EmbeddingMatcher` (no training)

### I want to work on the codebase

1. Read the module map in [`architecture.md`](./architecture.md)
2. Check public exports in `semanticmatcher/__init__.py`
3. Run tests in `tests/`

### I want to reproduce country classifier experiments

1. Review script differences in [`country-classifier-scripts.md`](./country-classifier-scripts.md)
2. Run the script that matches your iteration depth (baseline / quick / advanced)

## Notes

- The package code lives in `semanticmatcher/` (not `src/`).
- Some backend integrations are documented as future/planned capabilities and may not be fully wired.
