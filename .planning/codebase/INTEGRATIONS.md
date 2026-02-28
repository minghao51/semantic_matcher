# Integrations

## External APIs & Services

### Hugging Face (Primary)

**Purpose**: Model hosting and downloading

**Models Used:**
- Sentence transformer models (bi-encoders)
- Cross-encoder rerankers
- SetFit classification models

**Access Pattern:**
- Automatic download on first use via `sentence-transformers` library
- Models cached locally in `~/.cache/huggingface/` or `~/.cache/torch/sentence_transformers/`
- No API keys required (open-source models)

**Files:**
- `src/semanticmatcher/backends/sentencetransformer.py`
- `src/semanticmatcher/backends/reranker_st.py`
- `src/semanticmatcher/utils/embeddings.py` (ModelCache)

### External Data Sources (Ingestion)

**Currencies:**
- Source: `https://datahub.io/core/currency-codes/r/codes-all.csv`
- File: `src/semanticmatcher/ingestion/currencies.py`

**Industries:**
- Sources:
  - `https://raw.githubusercontent.com/erickogore/country-code-json/refs/heads/master/industry-codes.json`
  - `https://raw.githubusercontent.com/datasets/industry-codes/master/data/industry-codes.csv`
  - `https://www.bls.gov/cew/classifications/industry/sic-industry-titles.csv` (SIC codes)
- File: `src/semanticmatcher/ingestion/industries.py`

**Languages:**
- Source: `https://datahub.io/core/language-codes/r/language-codes-full.csv`
- File: `src/semanticmatcher/ingestion/languages.py`

**Occupations:**
- Sources:
  - `https://www.onetcenter.org/dl/30_2/occupation_data.zip` (O*NET)
  - `https://www.bls.gov/soc/2018/home.htm` (SOC codes)
- File: `src/semanticmatcher/ingestion/occupations.py`

**Products (UNSPSC):**
- Sources:
  - `https://unstats.un.org/unsd/services/v2/` (UN SPSC API)
  - `https://raw.githubusercontent.com/papermax/UNSPSC/master/UNSPSC_en.json`
- File: `src/semanticmatcher/ingestion/products.py`

**Timezones:**
- Source: IANA timezone database (via Python `zoneinfo`)
- File: `src/semanticmatcher/ingestion/timezones.py`

**Universities:**
- Source: Hardcoded fallback data (no external API)
- File: `src/semanticmatcher/ingestion/universities.py`

## Authentication

**Current Implementation:**
- No authentication required for Hugging Face models (public models)
- No API keys for data ingestion sources (all public CSV/JSON)

**Future:**
- LiteLLM backend stub exists (`src/semanticmatcher/backends/litellm.py`)
- Would support OpenAI/Anthropic APIs with API keys
- Currently not in active use

## Databases

**Not Used:**
- No database dependencies
- All entity data stored in-memory (Python lists/dicts)
- Processed data saved to `data/processed/` as JSON/CSV

## Caching

**Model Cache:**
- Implementation: `src/semanticmatcher/utils/embeddings.py`
- Thread-safe model caching with TTL and memory limits
- Prevents redundant model loading

**Data Cache:**
- Ingested datasets cached in `data/processed/`
- Avoids re-downloading from external sources

## Webhooks

**Not Used:**
- No webhook implementations
- All matching is synchronous

## File System

**Data Directories:**
- `data/raw/` - Downloaded source files
- `data/processed/` - Processed entity datasets
- `~/.cache/huggingface/` - Hugging Face model cache
- `~/.cache/torch/` - PyTorch model cache

## Security Considerations

**HTTP vs HTTPS:**
- All external URLs use HTTPS (secure)
- No hardcoded credentials in source code

**Input Validation:**
- Validation utilities in `src/semanticmatcher/utils/validation.py`
- Prevents injection attacks from user input

**Rate Limiting:**
- No built-in rate limiting for HTTP requests
- Assumes reasonable usage patterns for ingestion
