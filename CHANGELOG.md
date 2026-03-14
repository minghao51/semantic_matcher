# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Async API
Comprehensive async/await support for high-concurrency scenarios:

- `fit_async()`: Async model training with all modes (zero-shot, head-only, full, hybrid, bert)
- `match_async()`: Async single and multiple matching with full mode routing support
- `match_batch_async()`: Async batch matching with:
  - Progress tracking via `on_progress` callback (sync and async callbacks supported)
  - Configurable batch_size for throughput optimization
  - Threshold override with automatic restoration
  - Cancellation support for graceful interruption
- `explain_match_async()`: Async debugging helper for match analysis
- `diagnose_async()`: Async diagnostics with suggestions
- **Context Manager Support**: `async with Matcher(...) as matcher:` for automatic resource cleanup
- **Lifecycle Methods**: `aclose()`, `__aenter__`, `__aexit__` for explicit resource management
- **Cancellation Support**: Graceful cancellation of long-running batch operations with proper cleanup
- **Progress Callbacks**: Both sync and async callbacks supported for progress reporting
- **EntityMatcher Async Methods**: `train_async()`, `match_async()`, `predict_async()`
- **EmbeddingMatcher Async Methods**: `build_index_async()`, `match_async()`
- **AsyncExecutor Helper Class**: Thread pool management with configurable worker count (default: CPU_COUNT * 2, max 32)

#### Documentation
- Added `docs/async-guide.md` with comprehensive async API documentation including:
  - When to use async vs sync
  - Basic and advanced usage patterns
  - FastAPI integration examples
  - Performance considerations and best practices
  - Troubleshooting guide
- Added `examples/async_examples.py` with 7 working examples:
  - Basic usage
  - Batch processing with progress tracking
  - Concurrent matchers
  - Training with async
  - Explain and diagnose
  - Cancellation
  - Threshold override
- Updated `README.md` with async API section and quick examples
- Updated `docs/index.md` to include async guide in documentation paths

### Changed

- Matcher class now has `_async_executor` attribute (lazily initialized)
- All async methods run CPU-bound operations in thread pool for non-blocking execution
- `match_batch_async()` includes explicit cancellation checks between batches

### Compatibility

- **100% backward compatible** - all existing sync code works unchanged
- Async methods are purely additive - no breaking changes to existing API
- Lazy executor initialization - no overhead for sync-only usage
- All 247 existing tests pass without modification

### Performance

- Async operations use thread pool executor with CPU_COUNT * 2 workers (capped at 32)
- Batch processing supports configurable batch_size for throughput/latency tradeoffs
- Progress reporting is non-blocking (handles both sync and async callbacks)
- Cancellation checks between batches for responsive interruption

### Testing

- 27 new async tests covering all async functionality
- Tests for lifecycle management, fitting, matching, batching, diagnostics, cancellation, and concurrency
- All 247 total tests pass (2 skipped)
- Verified backward compatibility with comprehensive sync API tests
