"""
Hybrid Matching Demo

Demonstrates the three-stage waterfall pipeline:
1. Blocking (BM25) - Fast lexical filtering
2. Bi-Encoder Retrieval - Semantic similarity search
3. Cross-Encoder Reranking - Precise scoring
"""

import semanticmatcher as sm


def main():
    """Demonstrate hybrid matching pipeline."""

    # Sample product data
    products = [
        {
            "id": "p1",
            "name": "Apple iPhone 15 Pro",
            "text": "Apple iPhone 15 Pro 256GB with A17 chip",
        },
        {
            "id": "p2",
            "name": "Samsung Galaxy S24",
            "text": "Samsung Galaxy S24 Ultra Android smartphone",
        },
        {
            "id": "p3",
            "name": "Google Pixel 8",
            "text": "Google Pixel 8 Pro with Tensor G3",
        },
        {
            "id": "p4",
            "name": "Apple MacBook Pro",
            "text": "Apple MacBook Pro 14-inch M3 Max laptop",
        },
        {
            "id": "p5",
            "name": "Dell XPS 15",
            "text": "Dell XPS 15 laptop Windows 11",
        },
        {
            "id": "p6",
            "name": "Sony WH-1000XM5",
            "text": "Sony WH-1000XM5 noise cancelling headphones",
        },
        {
            "id": "p7",
            "name": "AirPods Pro 2",
            "text": "Apple AirPods Pro 2nd generation USB-C",
        },
        {
            "id": "p8",
            "name": "Samsung Galaxy Buds",
            "text": "Samsung Galaxy Buds2 Pro true wireless",
        },
    ]

    print("=" * 80)
    print("Hybrid Matching Pipeline Demo")
    print("=" * 80)
    print(f"Dataset size: {len(products)} products\n")

    # Initialize hybrid matcher with BM25 blocking
    print("Initializing HybridMatcher with BM25 blocking...")
    matcher = sm.HybridMatcher(
        entities=products,
        blocking_strategy=sm.BM25Blocking(),
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",  # Fast model
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast reranker
    )
    print("✓ Matcher initialized\n")

    # Test queries
    queries = [
        "iPhone phone",
        "laptop computer",
        "wireless earbuds",
        "Samsung smartphone",
    ]

    print("Running queries through three-stage pipeline...\n")

    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 40)

        results = matcher.match(
            query,
            blocking_top_k=5,
            retrieval_top_k=3,
            final_top_k=2,
        )

        for i, result in enumerate(results, 1):
            entity = next((p for p in products if p["id"] == result["id"]), None)
            if entity:
                print(f"  {i}. {entity['name']}")
                print(f"     Bi-Encoder Score: {result.get('score', 'N/A'):.3f}")
                print(
                    f"     Cross-Encoder Score: {result.get('cross_encoder_score', 'N/A'):.3f}"
                )
        print()

    # Demonstrate stage-by-stage
    print("=" * 80)
    print("Stage-by-Stage Breakdown for: 'iPhone phone'")
    print("=" * 80)

    query = "iPhone phone"

    # Stage 1: Blocking
    print("\nStage 1: BM25 Blocking")
    blocked = matcher.blocker.block(query, products, top_k=5)
    print(f"Candidates after blocking: {len(blocked)}")
    for p in blocked[:3]:
        print(f"  - {p['name']}")

    # Stage 2: Retrieval
    print("\nStage 2: Bi-Encoder Retrieval")
    retrieved = matcher.retriever.match(query, candidates=blocked, top_k=3)
    if retrieved and isinstance(retrieved, list) and len(retrieved) > 0:
        print(f"Candidates after retrieval: {len(retrieved)}")
        for r in retrieved[:3]:
            entity = next((p for p in products if p["id"] == r["id"]), None)
            if entity:
                print(f"  - {entity['name']} (score: {r['score']:.3f})")

    # Stage 3: Reranking
    print("\nStage 3: Cross-Encoder Reranking")
    if retrieved and isinstance(retrieved, list):
        final = matcher.reranker.rerank(query, retrieved, top_k=2)
        print(f"Final results: {len(final)}")
        for r in final:
            entity = next((p for p in products if p["id"] == r["id"]), None)
            if entity:
                print(f"  - {entity['name']} (score: {r['cross_encoder_score']:.3f})")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
