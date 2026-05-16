"""
EP Debates Scraper — incremental poller for European Parliament plenary debate topics.

Source: EP Open Data Portal REST API (public, no auth required)
  https://data.europarl.europa.eu/api/v2/speeches

The EP API returns speech-level metadata. We group speeches by their plenary
agenda item (ITM) to produce one text document per debate topic:
  - Debate title (English)
  - All speakers' nationality + political group (when available)
  - The CRE agenda item ID as a stable unique key

This gives us high-quality political text with clear named entities
(treaty names, geopolitical actors, legislative acts, MEP names).

State: data/raw/ep_debates/state.json
Output: data/raw/ep_debates/{itm_id}.json

Usage:
  python scrapers/ep_debates.py           # incremental (fetches next page)
  python scrapers/ep_debates.py --reset   # start from beginning
  python scrapers/ep_debates.py --limit 200  # fetch up to N speeches
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import os

import requests

logger = logging.getLogger(__name__)

# RAW_DATA_DIR lets Docker and host agree on one canonical path.
# Docker: set RAW_DATA_DIR=/data/raw (bind-mounted to ./data/raw on host).
# Host dev: defaults to <repo-root>/data/raw.
_RAW_ROOT = Path(os.environ.get("RAW_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "raw")))
OUT_DIR = _RAW_ROOT / "ep_debates"
STATE_FILE = OUT_DIR / "state.json"
REQUEST_TIMEOUT = 20

EP_API = "https://data.europarl.europa.eu/api/v2"
HEADERS = {
    "User-Agent": "graphknows-scraper/1.0 (research)",
    "Accept": "application/ld+json",
}


def _fetch_speeches(offset: int = 0, limit: int = 100) -> Dict[str, Any]:
    r = requests.get(
        f"{EP_API}/speeches",
        headers=HEADERS,
        params={
            "format": "application/ld+json",
            "offset": offset,
            "limit": limit,
        },
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _extract_en(label: Any) -> str:
    if isinstance(label, dict):
        return label.get("en") or next(iter(label.values()), "")
    return str(label)


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"offset": 0, "seen_itms": [], "seen_ids": []}


def _save_state(state: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _speakers_for_speech(speech: Dict[str, Any]) -> List[str]:
    """Extract speaker names/groups from performed_by field."""
    speakers = []
    for performer in speech.get("performed_by", []):
        if isinstance(performer, dict):
            label = _extract_en(performer.get("label", ""))
            if label:
                speakers.append(label)
    return speakers


def _build_itm_document(
    itm_id: str,
    title: str,
    speeches: List[Dict[str, Any]],
    session_date: str,
) -> Dict[str, Any]:
    """Build a document from all speeches in one debate agenda item."""
    speakers = []
    for s in speeches:
        speakers.extend(_speakers_for_speech(s))
    speakers = list(dict.fromkeys(speakers))  # deduplicate preserving order

    # Build rich text: title + speaker mentions + topic context
    lines = [f"European Parliament Plenary Debate — {session_date}", "", f"Topic: {title}", ""]
    if speakers:
        lines.append(f"Speakers ({len(speakers)}): {', '.join(speakers[:10])}")
        lines.append("")
    # Add any sub-items' labels as additional context
    sub_titles: Set[str] = set()
    for s in speeches:
        sub = _extract_en(s.get("activity_label", ""))
        if sub and sub != title and len(sub) > 20:
            sub_titles.add(sub)
    if sub_titles:
        lines.append("Related agenda items:")
        for st in list(sub_titles)[:5]:
            lines.append(f"  - {st}")

    text = "\n".join(lines)
    return {
        "source": "ep_debates",
        "source_type": "parliamentary_debate",
        "session_date": session_date,
        "itm_id": itm_id,
        "title": title,
        "text": text,
        "speaker_count": len(speeches),
        "url": "https://www.europarl.europa.eu/plenary/en/plenary.html",
        "fetched_at": datetime.now().isoformat(),
        "char_count": len(text),
    }


def run_incremental(max_speeches: int = 300) -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    offset: int = state.get("offset", 0)
    seen_itms: Set[str] = set(state.get("seen_itms", []))
    seen_ids: Set[str] = set(state.get("seen_ids", []))

    # Accumulate speeches
    all_speeches: List[Dict[str, Any]] = []
    fetched = 0
    while fetched < max_speeches:
        batch = _fetch_speeches(offset=offset, limit=min(100, max_speeches - fetched))
        items = batch.get("data", [])
        if not items:
            # No more data — reset offset to restart cycle for next hourly run
            offset = 0
            break
        all_speeches.extend(items)
        fetched += len(items)
        offset += len(items)
        logger.info("Fetched %d speeches (total %d), offset now %d", len(items), fetched, offset)
        if len(items) < 100:
            offset = 0  # end of dataset, restart next run
            break
        time.sleep(0.2)

    # Group by ITM agenda item
    by_itm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    itm_titles: Dict[str, str] = {}
    itm_dates: Dict[str, str] = {}

    for speech in all_speeches:
        activity_id = speech.get("activity_id", "")
        if not activity_id or activity_id in seen_ids:
            continue
        seen_ids.add(activity_id)

        cre_docs = speech.get("recorded_in_a_realization_of", [])
        if isinstance(cre_docs, dict):
            cre_docs = [cre_docs]

        itm_id = None
        for cre in cre_docs:
            if isinstance(cre, dict):
                itm = cre.get("is_part_of", "")
                if itm and "ITM" in itm:
                    itm_id = itm
                    break

        if not itm_id:
            itm_id = f"MISC-{speech.get('activity_date', 'unknown')}"

        by_itm[itm_id].append(speech)

        # Extract date and title
        if itm_id not in itm_titles:
            title = _extract_en(speech.get("activity_label", ""))
            itm_titles[itm_id] = title
            itm_dates[itm_id] = speech.get("activity_date", "")[:10]

    # Write one document per new ITM
    results = []
    for itm_id, speeches in by_itm.items():
        if itm_id in seen_itms:
            continue
        title = itm_titles.get(itm_id, "")
        if not title or len(title) < 10:
            continue
        session_date = itm_dates.get(itm_id, "")
        doc = _build_itm_document(itm_id, title, speeches, session_date)
        if doc["char_count"] < 50:
            continue

        safe = re.sub(r"[^a-zA-Z0-9-]", "-", itm_id)[-80:]
        out = OUT_DIR / f"{safe}.json"
        if not out.exists():
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
            logger.info("Saved debate item %s: %s (%d chars)", itm_id, title[:60], doc["char_count"])
        results.append(doc)
        seen_itms.add(itm_id)

    # Save updated state
    state["offset"] = offset
    state["seen_itms"] = list(seen_itms)[-3000:]
    state["seen_ids"] = list(seen_ids)[-5000:]
    state["last_run"] = datetime.now().isoformat()
    _save_state(state)

    logger.info("EP debates: %d new debate topics saved", len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="EP Debates incremental scraper")
    parser.add_argument("--reset", action="store_true", help="Reset state and start from offset 0")
    parser.add_argument("--limit", type=int, default=300, help="Max speeches to fetch per run")
    args = parser.parse_args()

    if args.reset:
        STATE_FILE.unlink(missing_ok=True)
        print("State reset.")

    items = run_incremental(max_speeches=args.limit)
    print(f"Saved {len(items)} new debate items")
