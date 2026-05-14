"""Archive Registry — declarative catalogue of all queryable public archives.

Each entry declares what the archive covers, what doc types it holds,
which jurisdictions, and how good it is at different query types.
This is what the agent reads when deciding which archives to query.

The agent never guesses adapter IDs — it reads this registry and emits
a structured ArchiveQuery (see archive_engine.py) specifying exactly
which archives to hit and with what filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DocType = Literal[
    "legislation", "regulation", "directive",
    "parliamentary_debate", "parliamentary_resolution",
    "committee_report", "press_release", "policy_brief",
    "court_decision", "budget", "treaty",
]

Jurisdiction = Literal["EU", "UK", "DE", "FR", "UN", "OECD", "GLOBAL"]

Strength = Literal["primary", "secondary", "weak"]


@dataclass(frozen=True)
class ArchiveEntry:
    """Declarative description of one queryable archive.

    Attributes:
        id: Adapter key used at query time (matches ADAPTER_REGISTRY in agent).
        name: Human-readable name shown in agent reasoning.
        description: One-line description the agent uses for archive selection.
        jurisdictions: Which political jurisdictions this archive covers.
        doc_types: Document types available in this archive.
        temporal_coverage: Free-text note on date range (e.g. "2004–present").
        query_strengths: What kind of queries this archive is best at.
        language: Primary language of documents.
        rate_limit_rps: Requests per second to stay polite.
        requires_auth: Whether a token/cookie is needed.
    """
    id: str
    name: str
    description: str
    jurisdictions: list[Jurisdiction]
    doc_types: list[DocType]
    temporal_coverage: str
    query_strengths: list[str]
    language: str = "en"
    rate_limit_rps: float = 1.0
    requires_auth: bool = False


# ── Registry ──────────────────────────────────────────────────────────────────
# Add new adapters here. The agent sees this list verbatim.

REGISTRY: list[ArchiveEntry] = [

    ArchiveEntry(
        id="eprs_briefings",
        name="European Parliament Research Service (EPRS)",
        description=(
            "Policy briefings, impact assessments, and in-depth analyses published "
            "by the European Parliament Research Service. Covers all major EU policy "
            "areas. Best for high-quality, synthesised EU policy overviews."
        ),
        jurisdictions=["EU"],
        doc_types=["policy_brief", "committee_report"],
        temporal_coverage="2014–present",
        query_strengths=[
            "EU policy overview", "legislative impact", "EU institutional analysis",
            "European Parliament positions",
        ],
        language="en",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="eurlex",
        name="EUR-Lex (EU Legislation Database)",
        description=(
            "Official EU legislation: regulations, directives, decisions, treaties, "
            "and international agreements. Indexed by CELEX number. Authoritative "
            "primary source for any EU legal act."
        ),
        jurisdictions=["EU"],
        doc_types=["legislation", "regulation", "directive", "treaty"],
        temporal_coverage="1952–present",
        query_strengths=[
            "EU law text", "specific regulation lookup", "CELEX retrieval",
            "legislative history", "directive transposition",
        ],
        language="multilingual",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="ep_debates",
        name="European Parliament Plenary Debates",
        description=(
            "Verbatim transcripts of European Parliament plenary debates. "
            "Records MEP speeches, voting explanations, and committee interventions. "
            "Best for understanding political positions and parliamentary dynamics."
        ),
        jurisdictions=["EU"],
        doc_types=["parliamentary_debate"],
        temporal_coverage="2004–present",
        query_strengths=[
            "MEP positions", "political debate", "party group stance",
            "parliament vote", "EU political dynamics",
        ],
        language="multilingual",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="ep_resolutions",
        name="European Parliament Resolutions",
        description=(
            "Non-legislative resolutions adopted by the European Parliament. "
            "Expresses parliament's official position on policy, external affairs, "
            "and institutional matters."
        ),
        jurisdictions=["EU"],
        doc_types=["parliamentary_resolution"],
        temporal_coverage="2004–present",
        query_strengths=[
            "EP official position", "human rights", "foreign policy stance",
            "non-legislative EU policy",
        ],
        language="multilingual",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="council_conclusions",
        name="Council of the EU — Conclusions & Press Releases",
        description=(
            "Formal conclusions, press releases, and outcome documents from "
            "Council of the EU meetings. Reflects member state positions and "
            "intergovernmental agreements."
        ),
        jurisdictions=["EU"],
        doc_types=["parliamentary_resolution", "press_release"],
        temporal_coverage="2010–present",
        query_strengths=[
            "Council position", "member state agreement", "EU summit outcome",
            "intergovernmental negotiations",
        ],
        language="en",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="ec_press",
        name="European Commission Press Releases",
        description=(
            "Official press releases, statements, and communications from the "
            "European Commission. Covers policy announcements, infringement "
            "proceedings, and institutional news."
        ),
        jurisdictions=["EU"],
        doc_types=["press_release"],
        temporal_coverage="2010–present",
        query_strengths=[
            "Commission announcement", "policy launch", "infringement case",
            "EU executive position", "Commissioner statement",
        ],
        language="en",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="ecb_publications",
        name="European Central Bank Publications",
        description=(
            "ECB economic bulletins, working papers, speeches, and policy reports. "
            "Authoritative source for eurozone monetary policy, financial stability, "
            "and macroeconomic analysis."
        ),
        jurisdictions=["EU"],
        doc_types=["policy_brief", "press_release"],
        temporal_coverage="1999–present",
        query_strengths=[
            "monetary policy", "eurozone economy", "inflation", "banking regulation",
            "financial stability", "ECB decision",
        ],
        language="en",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="uk_legislation",
        name="UK Legislation (legislation.gov.uk)",
        description=(
            "Acts of Parliament, Statutory Instruments, and secondary legislation "
            "for England, Wales, Scotland, and Northern Ireland. Official text "
            "including amendments."
        ),
        jurisdictions=["UK"],
        doc_types=["legislation", "regulation"],
        temporal_coverage="1267–present",
        query_strengths=[
            "UK law", "Act of Parliament", "statutory instrument",
            "UK regulatory framework", "post-Brexit legislation",
        ],
        language="en",
        rate_limit_rps=1.0,
    ),

    ArchiveEntry(
        id="wikipedia_eu",
        name="Wikipedia — EU & Political Topics",
        description=(
            "Wikipedia articles on EU institutions, political figures, policies, "
            "and events. Useful for background context, entity disambiguation, "
            "and overview of complex topics. Secondary source only."
        ),
        jurisdictions=["EU", "GLOBAL"],
        doc_types=["policy_brief"],
        temporal_coverage="2001–present",
        query_strengths=[
            "background context", "entity overview", "historical context",
            "concept explanation",
        ],
        language="en",
        rate_limit_rps=2.0,
    ),

]

# Fast lookup by id
REGISTRY_BY_ID: dict[str, ArchiveEntry] = {e.id: e for e in REGISTRY}


# ── Selection helpers (used by archive_engine.py) ─────────────────────────────

def get_archive(archive_id: str) -> ArchiveEntry | None:
    """Return archive entry by id, or None if not found."""
    return REGISTRY_BY_ID.get(archive_id)


def registry_summary() -> str:
    """Compact summary of all archives for injection into the agent system prompt.

    Returns one line per archive: id | name | jurisdictions | doc_types | strengths
    """
    lines = ["Available archives (use these ids in search_archives):"]
    for e in REGISTRY:
        juris = ", ".join(e.jurisdictions)
        dtypes = ", ".join(e.doc_types[:3])
        strengths = "; ".join(e.query_strengths[:3])
        lines.append(
            f"  • {e.id:25s} | {juris:12s} | {dtypes}\n"
            f"    → best for: {strengths}"
        )
    return "\n".join(lines)
