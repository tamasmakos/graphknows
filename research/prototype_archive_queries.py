"""
Quick prototype: run ArchiveEngine against 4 test queries, print results with timing.
No agent, no LangGraph — just the engine directly.
"""
import asyncio, sys, time
sys.path.insert(0, "/root/graphknows/services")

from search.archive_engine import ArchiveEngine, ArchiveQuery

TESTS = [
    ArchiveQuery(
        query="ECB interest rate decisions inflation eurozone 2024",
        archives=["ecb_publications", "ec_press"],
        date_from="2023-01-01",
        limit_per_archive=4, top_k=6,
    ),
    ArchiveQuery(
        query="EU AI Act artificial intelligence regulation obligations",
        archives=["eurlex", "eprs_briefings", "ep_resolutions"],
        limit_per_archive=4, top_k=6,
    ),
    ArchiveQuery(
        query="migration asylum seekers border management EU policy",
        archives=["ep_debates", "council_conclusions", "eprs_briefings"],
        limit_per_archive=3, top_k=5,
    ),
    ArchiveQuery(
        query="UK post-Brexit trade agreement Northern Ireland protocol",
        archives=["uk_legislation", "ep_debates"],
        limit_per_archive=4, top_k=5,
    ),
]

async def main():
    engine = ArchiveEngine()
    total_start = time.time()

    for i, aq in enumerate(TESTS, 1):
        print(f"\n{'='*65}")
        print(f"Query {i}: \"{aq.query}\"")
        print(f"Archives: {aq.archives}")
        print("-" * 65)
        t0 = time.time()
        results = await engine.search(aq)
        elapsed = time.time() - t0

        if not results:
            print("  ⚠ No results returned.")
            continue

        for r in results:
            cache = "💾" if r.from_cache else "🌐"
            print(f"  {cache} [{r.score:.3f}] [{r.source}] {r.title[:70]}")
            print(f"     {r.date or 'n/d'} | {r.url[:70] if r.url else 'no url'}")
            print(f"     {r.excerpt[:120].strip()}…")

        cached = sum(1 for r in results if r.from_cache)
        print(f"\n  → {len(results)} results in {elapsed:.1f}s | {cached} from cache")

    print(f"\n{'='*65}")
    print(f"Total: {time.time()-total_start:.1f}s")

asyncio.run(main())
