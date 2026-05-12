"""
UK Legislation Scraper — incremental hourly poller for UK statutory instruments.

Source: legislation.gov.uk REST API (public, no auth required)
  Feed:    https://www.legislation.gov.uk/{type}/data.feed
  Text:    https://www.legislation.gov.uk/{type}/{year}/{num}/made/data.xml

Types targeted:
  uksi  = Statutory Instruments (delegated legislation, most policy-dense)
  ukpga = Public General Acts (primary legislation)

This provides a legislative source that contrasts with EP parliamentary debates:
  - Different document structure (enactment vs. debate)
  - Different entity profile (UK ministers, departments, statutory bodies)
  - Different vocabulary (statutory language vs. political speech)

This diversity is crucial for validating ontology injection across source types.

State: data/raw/uk_legislation/state.json  (last-seen URIs)
Output: data/raw/uk_legislation/{type}-{year}-{num}.json

Usage:
  python scrapers/uk_legislation.py              # incremental
  python scrapers/uk_legislation.py --days 7    # last N days
  python scrapers/uk_legislation.py --types ukpga  # specific type
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "uk_legislation"
STATE_FILE = OUT_DIR / "state.json"
REQUEST_TIMEOUT = 20

LEG_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "leg":  "http://www.legislation.gov.uk/namespaces/legislation",
    "ukm":  "http://www.legislation.gov.uk/namespaces/metadata",
}

# Types to scrape and their plain names
TARGET_TYPES: Dict[str, str] = {
    "uksi":  "statutory_instrument",   # UK delegated legislation
    "ukpga": "act_of_parliament",      # UK primary legislation
}

FEED_URL = "https://www.legislation.gov.uk/{leg_type}/data.feed?results-count=25"
HEADERS = {
    "User-Agent": "graphknows-scraper/1.0 (research; contact: graphknows@example.com)",
    "Accept": "application/atom+xml, text/xml",
}


def _fetch(url: str, xml: bool = True) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code in (404, 403, 410):
            logger.debug("Skip %s (HTTP %s)", url, r.status_code)
            return None
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        logger.debug("Fetch error %s: %s", url, e)
        return None


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"seen_uris": []}


def _save_state(state: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _out_path(leg_type: str, doc_id: str) -> Path:
    safe_id = doc_id.replace("/", "-").strip("-")
    return OUT_DIR / f"{safe_id}.json"


def _extract_text_from_xml(content: bytes) -> str:
    """Extract plain readable text from legislation.gov.uk XML."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return ""
    parts = []
    # Walk all elements, collecting text
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in {"Text", "Para", "P1para", "P2para", "P3para", "Pblock",
                   "Title", "Subtitle", "EnactingText", "IntroductoryText",
                   "Body", "BlockAmendment"}:
            text = (elem.text or "").strip()
            if text and len(text) > 5:
                parts.append(text)
            # Also grab tail text
            tail = (elem.tail or "").strip()
            if tail and len(tail) > 5:
                parts.append(tail)
    result = "\n\n".join(parts)
    return result[:60000]


def _parse_feed(content: bytes) -> List[Dict[str, str]]:
    """Parse Atom feed, return list of {uri, title, updated, data_url}."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    entries = []
    for entry in root.findall("atom:entry", LEG_NS):
        title_el = entry.find("atom:title", LEG_NS)
        updated_el = entry.find("atom:updated", LEG_NS)
        # Find XML data link
        data_url = ""
        for link in entry.findall("atom:link", LEG_NS):
            href = link.get("href", "")
            if href.endswith("data.xml"):
                data_url = href
                break
        # Fall back to alternate link
        if not data_url:
            alt = entry.find("atom:link[@rel='alternate']", LEG_NS)
            if alt is not None:
                base = alt.get("href", "")
                if base and not base.endswith(".pdf"):
                    data_url = base.rstrip("/") + "/data.xml"
        id_el = entry.find("atom:id", LEG_NS)
        uri = (id_el.text or "").strip() if id_el is not None else ""
        title = (title_el.text or "").strip() if title_el is not None else ""
        updated = (updated_el.text or "")[:10] if updated_el is not None else ""
        if data_url and not data_url.endswith(".pdf"):
            entries.append({
                "uri": uri,
                "title": title,
                "updated": updated,
                "data_url": data_url,
            })
    return entries


def _doc_id_from_url(data_url: str) -> str:
    """Extract e.g. 'uksi-2026-485' from data URL."""
    m = re.search(r"/(uk\w+)/(\d{4})/(\d+)/", data_url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return re.sub(r"[^a-zA-Z0-9-]", "-", data_url.split("/")[-3:-1][0])


def fetch_document(
    leg_type: str,
    entry: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Download and parse one UK legislation document."""
    data_url = entry["data_url"]
    content = _fetch(data_url)
    if not content:
        return None
    text = _extract_text_from_xml(content)
    if len(text) < 100:
        logger.debug("Too short, skipping %s", data_url)
        return None
    doc_id = _doc_id_from_url(data_url)
    record = {
        "source": "uk_legislation",
        "source_type": TARGET_TYPES.get(leg_type, "legislation"),
        "leg_type": leg_type,
        "doc_id": doc_id,
        "title": entry["title"],
        "text": text,
        "url": data_url,
        "updated": entry["updated"],
        "fetched_at": datetime.utcnow().isoformat(),
        "char_count": len(text),
    }
    out = _out_path(leg_type, doc_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    logger.info("Saved %s %s (%d chars)", leg_type, doc_id, len(text))
    return record


def run_incremental(
    leg_types: Optional[List[str]] = None,
    days: int = 1,
) -> List[Dict[str, Any]]:
    """Fetch only new documents since last run."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    seen: Set[str] = set(state.get("seen_uris", []))
    leg_types = leg_types or list(TARGET_TYPES.keys())
    all_new = []

    for leg_type in leg_types:
        feed_url = FEED_URL.format(leg_type=leg_type)
        content = _fetch(feed_url)
        if not content:
            logger.warning("Could not fetch feed for %s", leg_type)
            continue
        entries = _parse_feed(content)
        logger.info("%s feed: %d entries", leg_type, len(entries))
        for entry in entries:
            uri = entry["uri"]
            data_url = entry["data_url"]
            if uri in seen or data_url in seen:
                continue
            out = _out_path(leg_type, _doc_id_from_url(data_url))
            if out.exists():
                seen.add(uri)
                seen.add(data_url)
                continue
            record = fetch_document(leg_type, entry)
            if record:
                all_new.append(record)
                seen.add(uri)
                seen.add(data_url)
            time.sleep(0.5)

    state["seen_uris"] = list(seen)[-1000:]
    state["last_run"] = datetime.utcnow().isoformat()
    _save_state(state)
    logger.info("UK Legislation incremental run: %d new docs", len(all_new))
    return all_new


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="UK Legislation incremental scraper")
    parser.add_argument("--days", type=int, default=1, help="Lookback days (default: 1)")
    parser.add_argument("--types", nargs="+", default=None,
                        help="Legislation types (uksi, ukpga)")
    args = parser.parse_args()
    items = run_incremental(leg_types=args.types, days=args.days)
    print(f"Fetched {len(items)} new UK legislation documents")
