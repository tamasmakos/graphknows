"""
EUR-Lex Legislation Scraper — incremental hourly poller for EU legislation.

Source: EUR-Lex SPARQL / RSS feeds + direct document download (public, no auth)
  Primary feed: https://eur-lex.europa.eu/oj/daily-view/L-series/default.html?ojDate=...
  SPARQL API:   https://publications.europa.eu/webapi/rdf/sparql
  RSS feed:     https://eur-lex.europa.eu/oj/direct-access.html?ojType=L (OJ L-series)

Strategy:
  1. Poll the EUR-Lex OJ L-series RSS atom feed for new acts published today/yesterday
  2. For each new act, download the plain-text version via EUR-Lex HTML endpoint
  3. Extract meaningful legislative text (preamble + operative provisions)
  4. Save with metadata for downstream ingestion

State: data/raw/eurlex/state.json  — stores last-seen CELEX numbers
Output: data/raw/eurlex/{CELEX}.json

Document types targeted (most policy-rich):
  R = Regulations
  L = Directives
  D = Decisions

Usage:
  python scrapers/eurlex.py           # incremental
  python scrapers/eurlex.py --days 7  # fetch acts from last N days
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import os

import requests

logger = logging.getLogger(__name__)

# RAW_DATA_DIR lets Docker and host agree on one canonical path.
# Docker: set RAW_DATA_DIR=/data/raw (bind-mounted to ./data/raw on host).
# Host dev: defaults to <repo-root>/data/raw.
_RAW_ROOT = Path(os.environ.get("RAW_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "raw")))
OUT_DIR = _RAW_ROOT / "eurlex"
STATE_FILE = OUT_DIR / "state.json"
REQUEST_TIMEOUT = 25
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml;q=0.9",
}

# EUR-Lex atom feed for new OJ publications — L-series (legislation)
# This is the daily official journal atom feed, reliably updated
OJ_ATOM_TMPL = (
    "https://eur-lex.europa.eu/oj/direct-access.html"
    "?ojType=L&ojDate={date}&format=atom"
)
# Fallback: use the SPARQL endpoint
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

# EUR-Lex HTML text endpoint for a CELEX number
EURLEX_HTML_TMPL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=CELEX:{celex}&from=EN"
)

# Atom namespace
NS = {"atom": "http://www.w3.org/2005/Atom"}

# Doc type codes we want
TARGET_TYPES = {"R", "L", "D"}   # Regulation, Directive, Decision


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"seen_celex": [], "last_run": None}


def _save_state(state: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _fetch(url: str, **kw) -> Optional[requests.Response]:
    try:
        headers = dict(HEADERS)
        headers.update(kw.pop("headers", {}))
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, **kw)
        if r.status_code in (404, 403):
            return None
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        logger.debug("Fetch error %s: %s", url, e)
        return None


def _parse_atom_entries(xml_bytes: bytes) -> List[Dict[str, str]]:
    """Parse EUR-Lex Atom feed, return list of {celex, title, url, published}."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    entries = []
    for entry in root.findall("atom:entry", NS):
        celex_el = entry.find("atom:id", NS)
        title_el = entry.find("atom:title", NS)
        link_el = entry.find("atom:link[@rel='alternate']", NS)
        pub_el = entry.find("atom:published", NS)
        celex = (celex_el.text or "").strip() if celex_el is not None else ""
        title = (title_el.text or "").strip() if title_el is not None else ""
        url = link_el.get("href", "") if link_el is not None else ""
        published = (pub_el.text or "")[:10] if pub_el is not None else ""
        # Extract CELEX from ID if needed
        m = re.search(r"CELEX[:/]([0-9A-Za-z]+)", celex + url)
        if m:
            celex = m.group(1)
        entries.append({"celex": celex, "title": title, "url": url, "published": published})
    return entries


def _extract_legislative_text(html: str) -> str:
    """Extract plain text from EUR-Lex HTML document."""
    # Remove script/style tags
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.I)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Extract from "THE EUROPEAN PARLIAMENT..." onward (skip nav/header)
    m = re.search(r"(THE EUROPEAN PARLIAMENT|THE COUNCIL OF|COMMISSION REGULATION)", text, re.I)
    if m:
        text = text[m.start():]
    # Trim to ~50k chars (typical for a regulation)
    return text[:50000]


def _celex_doc_type(celex: str) -> Optional[str]:
    """Extract document type code from CELEX number."""
    # CELEX format: [sector][year][doc_type][number]
    # e.g. 32023R0001 => R (Regulation), 32023L0001 => L (Directive)
    m = re.match(r"[0-9]([0-9]{4})([A-Z])[0-9]+", celex)
    if m:
        return m.group(2)
    return None


def _type_name(code: str) -> str:
    return {"R": "regulation", "L": "directive", "D": "decision"}.get(code, "legislation")


def _out_path(celex: str) -> Path:
    return OUT_DIR / f"{celex}.json"


def fetch_celex(celex: str, title: str, published: str, source_url: str) -> Optional[Dict[str, Any]]:
    """Download and parse one EUR-Lex document by CELEX number."""
    url = EURLEX_HTML_TMPL.format(celex=celex)
    r = _fetch(url, headers={"Accept-Language": "en"})
    if not r:
        logger.debug("Could not fetch CELEX %s", celex)
        return None

    text = _extract_legislative_text(r.text)
    if len(text) < 200:
        logger.debug("Too short, skipping CELEX %s", celex)
        return None

    doc_type_code = _celex_doc_type(celex) or "R"
    record = {
        "source": "eurlex",
        "source_type": _type_name(doc_type_code),
        "celex": celex,
        "doc_type_code": doc_type_code,
        "title": title,
        "text": text,
        "url": source_url or url,
        "published": published,
        "fetched_at": datetime.utcnow().isoformat(),
        "char_count": len(text),
    }
    _out_path(celex).write_text(json.dumps(record, ensure_ascii=False, indent=2))
    logger.info("Saved EUR-Lex %s (%s, %d chars)", celex, _type_name(doc_type_code), len(text))
    return record


def _sparql_recent_acts(days: int = 1) -> List[Dict[str, str]]:
    """
    Query SPARQL for recent OJ L-series acts as fallback when atom feed is unavailable.
    Returns list of {celex, title, published}.
    """
    since = (date.today() - timedelta(days=days)).isoformat()
    query = f"""
    PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT DISTINCT ?celex ?title ?date WHERE {{
      ?doc cdm:resource_legal_id_celex ?celex ;
           cdm:work_date_document ?date .
      OPTIONAL {{ ?doc cdm:work_title_oj ?title . }}
      FILTER(?date >= "{since}"^^xsd:date)
      FILTER(REGEX(?celex, "^3[0-9]{{4}}[RLD][0-9]+$"))
    }}
    ORDER BY DESC(?date)
    LIMIT 50
    """
    try:
        r = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "application/sparql-results+json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for binding in data.get("results", {}).get("bindings", []):
            celex = binding.get("celex", {}).get("value", "")
            title = binding.get("title", {}).get("value", "")
            pub = binding.get("date", {}).get("value", "")[:10]
            if celex:
                results.append({"celex": celex, "title": title, "published": pub, "url": ""})
        return results
    except Exception as e:
        logger.warning("SPARQL fallback failed: %s", e)
        return []


def run_incremental(days: int = 1) -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    seen: Set[str] = set(state.get("seen_celex", []))

    # Try atom feed first
    candidates: List[Dict[str, str]] = []
    for d in range(days):
        target = date.today() - timedelta(days=d)
        atom_url = OJ_ATOM_TMPL.format(date=target.strftime("%Y%m%d"))
        r = _fetch(atom_url)
        if r and r.content:
            entries = _parse_atom_entries(r.content)
            candidates.extend(entries)
            logger.debug("Atom feed %s: %d entries", target, len(entries))
        time.sleep(0.5)

    # Fallback to SPARQL if atom returned nothing
    if not candidates:
        logger.info("Atom feed returned no entries, trying SPARQL fallback")
        candidates = _sparql_recent_acts(days=max(days, 2))

    new_records = []
    for entry in candidates:
        celex = entry["celex"]
        if not celex or celex in seen:
            continue
        doc_type = _celex_doc_type(celex)
        if doc_type not in TARGET_TYPES:
            continue
        if _out_path(celex).exists():
            seen.add(celex)
            continue
        record = fetch_celex(celex, entry["title"], entry["published"], entry["url"])
        if record:
            new_records.append(record)
            seen.add(celex)
        time.sleep(1.0)

    state["seen_celex"] = list(seen)[-500:]   # keep last 500 to bound size
    state["last_run"] = datetime.utcnow().isoformat()
    _save_state(state)
    logger.info("EUR-Lex incremental run: %d new acts", len(new_records))
    return new_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="EUR-Lex legislation incremental scraper")
    parser.add_argument("--days", type=int, default=1, help="Lookback window in days (default: 1)")
    args = parser.parse_args()
    items = run_incremental(days=args.days)
    print(f"Fetched {len(items)} new EUR-Lex acts")
