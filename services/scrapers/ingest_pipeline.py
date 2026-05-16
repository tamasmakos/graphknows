"""
Hourly ingestion orchestrator — polls both scrapers and feeds new documents
into the GraphRAG pipeline (Neo4j + pgvector).

Designed to run as a cron job every hour:
  */60 * * * *  python /root/graphknows/scrapers/ingest_pipeline.py

Flow:
  1. Run EP debates scraper (incremental)
  2. Run EUR-Lex scraper (incremental)
  3. For each new document, run GraphGen extraction -> Neo4j/pgvector ingest
  4. Emit a summary report to logs/ingest_report.jsonl

The scraper outputs (data/raw/*/  *.json) serve as the queue.
Already-ingested files are tracked in data/ingested.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_pipeline")

import os

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "services"))

# RAW_DATA_DIR is the single canonical source store.
# In Docker, this is /data/raw (bind-mounted to ./data/raw on the host).
# On the host, falls back to <repo-root>/data/raw so the pipeline works
# without Docker as well.
_RAW_ROOT = Path(os.environ.get("RAW_DATA_DIR", str(ROOT / "data" / "raw")))

INGESTED_REGISTRY = ROOT / "data" / "ingested.json"
LOG_DIR = ROOT / "logs"
# All registered source directories — add new connectors here.
RAW_DIRS = [
    _RAW_ROOT / "ep_debates",
    _RAW_ROOT / "eurlex",
    _RAW_ROOT / "uk_legislation",
]


def _load_ingested() -> set:
    if INGESTED_REGISTRY.exists():
        try:
            return set(json.loads(INGESTED_REGISTRY.read_text()))
        except Exception:
            pass
    return set()


def _save_ingested(ingested: set) -> None:
    INGESTED_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    INGESTED_REGISTRY.write_text(json.dumps(sorted(ingested), indent=2))


def _collect_new_docs(ingested: set) -> List[Path]:
    """Find raw JSON docs not yet ingested."""
    new_docs = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            continue
        for f in sorted(raw_dir.glob("*.json")):
            if f.name == "state.json":
                continue
            if str(f) not in ingested:
                new_docs.append(f)
    return new_docs


def _ingest_document(doc_path: Path) -> Dict[str, Any]:
    """
    Ingest one raw document into the GraphRAG pipeline.
    Calls the graphrag ingest module with the document text.
    """
    try:
        record = json.loads(doc_path.read_text())
    except Exception as e:
        return {"path": str(doc_path), "status": "error", "error": f"JSON parse: {e}"}

    text = record.get("text", "")
    if not text or len(text) < 100:
        return {"path": str(doc_path), "status": "skipped", "reason": "too short"}

    # Import the graphrag ingest pipeline
    try:
        from graphrag.ingest import ingest_document
        result = ingest_document(
            text=text,
            metadata={
                "source": record.get("source", "unknown"),
                "source_type": record.get("source_type", "unknown"),
                "title": record.get("title", ""),
                "url": record.get("url", ""),
                "date": record.get("date") or record.get("published", ""),
                "doc_id": record.get("celex") or doc_path.stem,
            },
        )
        return {
            "path": str(doc_path),
            "status": "ok",
            "chunks": result.get("chunks_stored", 0),
            "entities": result.get("entities_stored", 0),
        }
    except ImportError:
        # GraphRAG ingest not yet wired — log and record for manual re-run
        logger.warning("graphrag.ingest.ingest_document not yet available — queuing %s", doc_path.name)
        return {"path": str(doc_path), "status": "queued", "reason": "ingest_not_wired"}
    except Exception as e:
        logger.error("Ingest failed for %s: %s", doc_path.name, e)
        return {"path": str(doc_path), "status": "error", "error": str(e)}


def _write_report(results: List[Dict[str, Any]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "ingest_report.jsonl"
    with log_file.open("a") as f:
        for r in results:
            r["timestamp"] = datetime.utcnow().isoformat()
            f.write(json.dumps(r) + "\n")


def run_scrapers() -> None:
    """Run all registered scrapers to pull new data."""
    logger.info("Running EP debates scraper...")
    try:
        from scrapers.ep_debates import run_incremental as ep_run
        ep_items = ep_run()
        logger.info("EP debates: %d new items", len(ep_items))
    except Exception as e:
        logger.error("EP debates scraper failed: %s", e)

    logger.info("Running EUR-Lex scraper...")
    try:
        from scrapers.eurlex import run_incremental as eur_run
        eur_items = eur_run(days=1)
        logger.info("EUR-Lex: %d new acts", len(eur_items))
    except Exception as e:
        logger.error("EUR-Lex scraper failed: %s", e)

    logger.info("Running UK Legislation scraper...")
    try:
        from scrapers.uk_legislation import run_incremental as ukleg_run
        ukleg_items = ukleg_run(days=1)
        logger.info("UK Legislation: %d new items", len(ukleg_items))
    except Exception as e:
        logger.error("UK Legislation scraper failed: %s", e)


def main() -> None:
    start = time.time()
    logger.info("=== Ingest pipeline starting ===")

    run_scrapers()

    ingested = _load_ingested()
    new_docs = _collect_new_docs(ingested)
    logger.info("Found %d new documents to ingest", len(new_docs))

    results = []
    for doc_path in new_docs:
        logger.info("Ingesting %s", doc_path.name)
        result = _ingest_document(doc_path)
        results.append(result)
        if result["status"] in ("ok", "queued", "skipped"):
            ingested.add(str(doc_path))
        time.sleep(0.2)

    _save_ingested(ingested)
    _write_report(results)

    ok = sum(1 for r in results if r["status"] == "ok")
    queued = sum(1 for r in results if r["status"] == "queued")
    errors = sum(1 for r in results if r["status"] == "error")
    elapsed = round(time.time() - start, 1)
    logger.info(
        "=== Done in %.1fs: %d ok, %d queued, %d errors ===",
        elapsed, ok, queued, errors,
    )


if __name__ == "__main__":
    main()
