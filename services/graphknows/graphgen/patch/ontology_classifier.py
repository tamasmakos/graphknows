"""Rule-based ontology class classifier for legacy and unknown entities."""

from __future__ import annotations

import re
from typing import Optional

PERSON_TOKENS = {
    "MR",
    "MS",
    "MRS",
    "DR",
    "PRESIDENT",
    "MINISTER",
    "COMMISSIONER",
    "CHANCELLOR",
    "PRIME_MINISTER",
}

POLITICAL_PARTY_PATTERNS = (
    r"_PARTY$",
    r"\bPARTY\b",
)

ORG_PATTERNS = (
    r"_COMMISSION$",
    r"_PARLIAMENT$",
    r"_UNION$",
    r"_COUNCIL$",
    r"_AGENCY$",
    r"_COURT$",
    r"_ORGANIZATION$",
    r"\bNATO\b",
    r"\bUNICEF\b",
)

LOCATION_TOKENS = {
    "AUSTRIA",
    "BELGIUM",
    "BULGARIA",
    "CROATIA",
    "CYPRUS",
    "CZECHIA",
    "DENMARK",
    "ESTONIA",
    "FINLAND",
    "FRANCE",
    "GERMANY",
    "GREECE",
    "HUNGARY",
    "IRELAND",
    "ITALY",
    "LATVIA",
    "LITHUANIA",
    "LUXEMBOURG",
    "MALTA",
    "NETHERLANDS",
    "POLAND",
    "PORTUGAL",
    "ROMANIA",
    "SLOVAKIA",
    "SLOVENIA",
    "SPAIN",
    "SWEDEN",
    "EUROPE",
    "REGION",
    "COUNTRY",
    "CITY",
    "CAPITAL",
    "BRUSSELS",
    "PARIS",
    "BERLIN",
    "ROME",
    "MADRID",
    "WARSAW",
    "LISBON",
    "VIENNA",
    "ATHENS",
    "DUBLIN",
    "AMSTERDAM",
    "HELSINKI",
    "STOCKHOLM",
    "PRAGUE",
    "BUDAPEST",
    "SOFIA",
    "BUCHAREST",
    "RIGA",
    "VILNIUS",
    "TALLINN",
    "LJUBLJANA",
    "BRATISLAVA",
    "ZAGREB",
    "NICOSIA",
    "VALLETTA",
    "LUXEMBOURG_CITY",
}

POLICY_TOKENS = {
    "POLICY",
    "REGULATION",
    "DIRECTIVE",
    "TREATY",
}


def _normalize(name: str) -> str:
    """Normalize raw node names for rule matching."""
    normalized = name.strip().upper()
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized


def _is_date_like(name: str) -> bool:
    """Detect date and year-like strings."""
    if re.search(r"\d{4}-\d{2}-\d{2}", name):
        return True
    if re.search(r"\d{4}_\d{2}_\d{2}", name):
        return True
    if re.search(r"\b\d{4}\b", name):
        return True
    if "_YEAR_" in name or name.endswith("_YEAR"):
        return True
    return False


def classify_name(name: str, default_to_concept: bool = True) -> Optional[str]:
    """Classify node name into ontology class.

    Args:
        name: Node name to classify.

    Returns:
        Ontology class string.
    """
    if not name:
        return "CONCEPT" if default_to_concept else None

    normalized = _normalize(name)

    for token in PERSON_TOKENS:
        if re.search(rf"(^|_){re.escape(token)}($|_)", normalized):
            return "PERSON"

    for pattern in POLITICAL_PARTY_PATTERNS:
        if re.search(pattern, normalized):
            return "POLITICAL_PARTY"

    for pattern in ORG_PATTERNS:
        if re.search(pattern, normalized):
            return "ORGANIZATION"

    if any(token in normalized for token in LOCATION_TOKENS):
        return "LOCATION"

    if _is_date_like(normalized):
        return "EVENT"

    if any(token in normalized for token in POLICY_TOKENS):
        return "POLICY"

    if default_to_concept:
        return "CONCEPT"
    return None


def classify_by_name(name: str, default_to_concept: bool = True) -> Optional[str]:
    """Backward-compatible alias for classify_name."""
    return classify_name(name, default_to_concept=default_to_concept)
