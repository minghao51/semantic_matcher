"""
Basic unified Matcher example.

This is the recommended async-first starting point for the public API.
"""

import asyncio

from novelentitymatcher import Matcher


async def main():
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
    ]

    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        queries = ["Deutschland", "America", "Frankreich"]
        for query in queries:
            print(f"{query} -> {await matcher.match_async(query)}")


if __name__ == "__main__":
    asyncio.run(main())
