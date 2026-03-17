import json

import pandas as pd

from semanticmatcher.utils import benchmarks


def test_load_processed_sections_reads_csv_sections(tmp_path):
    section_dir = tmp_path / "languages"
    section_dir.mkdir()
    csv_path = section_dir / "languages.csv"
    csv_path.write_text(
        "id,name,aliases,type\n"
        "en,English,en|anglais,language\n"
        "fr,French,fr|francais,language\n",
        encoding="utf-8",
    )

    sections = benchmarks.load_processed_sections(
        processed_dir=tmp_path,
        max_entities_per_section=10,
        max_queries_per_section=5,
    )

    assert len(sections) == 1
    assert sections[0]["section"] == "languages/languages"
    assert sections[0]["entities"][0]["aliases"] == ["anglais"]
    assert sections[0]["queries"]
    assert sections[0]["training_data"]
    assert sections[0]["base_pairs"]
    assert sections[0]["train_pairs"]
    assert sections[0]["base_pairs"][0]["query"] not in {
        pair["query"] for pair in sections[0]["train_pairs"]
    }
    train_texts = {item["text"] for item in sections[0]["training_data"]}
    eval_texts = {
        item["query"] for item in sections[0]["val_pairs"] + sections[0]["test_pairs"]
    }
    assert train_texts.isdisjoint(eval_texts)


def test_load_processed_sections_skips_eval_for_single_text_entities(tmp_path):
    section_dir = tmp_path / "products"
    section_dir.mkdir()
    csv_path = section_dir / "products.csv"
    csv_path.write_text(
        "id,name,aliases,type\nsku-1,Widget,,product\n",
        encoding="utf-8",
    )

    sections = benchmarks.load_processed_sections(
        processed_dir=tmp_path,
        max_entities_per_section=10,
        max_queries_per_section=5,
    )

    assert len(sections) == 1
    assert sections[0]["training_data"] == [{"text": "Widget", "label": "sku-1"}]
    assert sections[0]["evaluation_pairs"]
    assert sections[0]["val_pairs"]
    assert sections[0]["test_pairs"] == []
    assert sections[0]["evaluation_pairs"][0]["query"] != "Widget"


def test_load_processed_sections_drops_shared_aliases(tmp_path):
    section_dir = tmp_path / "products"
    section_dir.mkdir()
    csv_path = section_dir / "products.csv"
    csv_path.write_text(
        "id,name,aliases,type\n"
        "sku-1,Widget,common|alpha,product\n"
        "sku-2,Gadget,common|beta,product\n",
        encoding="utf-8",
    )

    sections = benchmarks.load_processed_sections(
        processed_dir=tmp_path,
        max_entities_per_section=10,
        max_queries_per_section=5,
    )

    assert len(sections) == 1
    assert sections[0]["entities"][0]["aliases"] == ["alpha"]
    assert sections[0]["entities"][1]["aliases"] == ["beta"]
    assert all("common" not in pair["query"] for pair in sections[0]["base_pairs"])


def test_benchmark_embedding_models_expands_registry(monkeypatch):
    called = []

    class FakeEmbeddingMatcher:
        def __init__(self, entities, model_name, threshold):
            called.append(model_name)
            self.model_name = model_name

        def build_index(self, batch_size=None):
            return None

        def match(self, query, batch_size=None):
            if isinstance(query, list):
                return [{"id": "ID-0", "score": 0.9} for _ in query]
            return {"id": "ID-0", "score": 0.9}

    monkeypatch.setattr(benchmarks, "EmbeddingMatcher", FakeEmbeddingMatcher)

    results = benchmarks.benchmark_embedding_models(
        model_names=["potion-8m", "mpnet"],
        iterations=1,
        sections_data=[
            {
                "section": "custom/test",
                "entities": [
                    {"id": "ID-0", "name": "Alpha"},
                    {"id": "ID-1", "name": "Beta"},
                ],
                "queries": ["Alpha", "Beta"],
                "accuracy_pairs": [
                    {"query": "Alpha", "expected_id": "ID-0"},
                    {"query": "Beta", "expected_id": "ID-1"},
                ],
            }
        ],
    )

    assert called == ["potion-8m", "mpnet"]
    assert set(results["model"]) == {"potion-8m", "mpnet"}
    assert set(results["section"]) == {"custom/test"}


def test_benchmark_embedding_models_records_skips(monkeypatch):
    class FailingEmbeddingMatcher:
        def __init__(self, entities, model_name, threshold):
            self.model_name = model_name

        def build_index(self, batch_size=None):
            raise RuntimeError("missing model")

    monkeypatch.setattr(benchmarks, "EmbeddingMatcher", FailingEmbeddingMatcher)

    results = benchmarks.benchmark_embedding_models(
        model_names=["potion-8m"],
        iterations=1,
        sections_data=[
            {
                "section": "custom/test",
                "entities": [{"id": "ID-0", "name": "Alpha"}],
                "queries": ["Alpha"],
                "accuracy_pairs": [{"query": "Alpha", "expected_id": "ID-0"}],
            }
        ],
    )

    row = results.iloc[0]
    assert row["status"] == "skipped"
    assert "missing model" in row["skip_reason"]


def test_save_benchmark_report_writes_json(tmp_path):
    results = pd.DataFrame(
        [
            {
                "track": "embedding",
                "model": "potion-8m",
                "status": "ok",
                "throughput_qps": 10.0,
            }
        ]
    )

    output_path = tmp_path / "benchmarks.json"
    benchmarks.save_benchmark_report(results, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["model"] == "potion-8m"


def test_format_benchmark_summary_includes_skip_reason():
    results = pd.DataFrame(
        [
            {
                "track": "embedding",
                "section": "languages/languages",
                "model": "potion-8m",
                "backend": "static",
                "status": "skipped",
                "throughput_qps": None,
                "accuracy": None,
                "speedup_vs_minilm": None,
                "skip_reason": "unavailable",
            }
        ]
    )

    summary = benchmarks.format_benchmark_summary(results)

    assert "BENCHMARK RESULTS" in summary
    assert "languages/languages" in summary
    assert "unavailable" in summary
