"""Conservative entity surface-form canonicalization helpers.

These helpers are intentionally modest. They aim to improve graph cleanliness
without introducing broad ontology matching or aggressive semantic collapse.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

ROLE_TERMS = {
    "PRIME_MINISTER",
    "PRESIDENT",
    "CHANCELLOR",
    "COMMISSIONER",
    "MINISTER",
    "SPEAKER",
    "CHAIR",
    "LEADER",
}

CONCEPT_LIKE_TERMS = {
    "MIGRATION",
    "HEALTH",
    "CLIMATE",
    "SECURITY",
    "SANCTIONS",
    "AID",
    "POLICY_CHANGES",
    "WHATEVER_IT_TAKES",
    "ENERGY_INDEPENDENCE",
    "ENERGY_DEPENDENCE",
}

NAMED_ENTITY_HINTS = {
    "EUROPEAN_CENTRAL_BANK",
    "EUROPEAN_PARLIAMENT",
    "MARIO_DRAGHI",
    "ROBERTA_METSOLA",
    "UKRAINE",
    "ITALY",
    "EU",
    "EUROPE",
}

OPPOSING_TERM_PAIRS = {
    ("DEPENDENCE", "INDEPENDENCE"),
}

DEMONYM_SUFFIXES = (
    "AN",
    "IAN",
    "EAN",
    "ESE",
    "ISH",
    "IC",
)


def normalize_surface_form(text: str) -> str:
    if not text:
        return ""
    normalized = str(text).strip().upper()
    normalized = re.sub(r"[^A-Z0-9\s_]", " ", normalized)
    normalized = re.sub(r"[\s_]+", "_", normalized).strip("_")
    return normalized


def _strip_plural_suffix(token: str) -> str:
    if token.endswith("S") and len(token) > 3:
        return token[:-1]
    return token


def _looks_like_demonym_form(token: str) -> bool:
    singular = _strip_plural_suffix(token)
    return any(singular.endswith(suffix) for suffix in DEMONYM_SUFFIXES)


def _likely_country_demonym_conflict(token_a: str, token_b: str) -> bool:
    base_a = _strip_plural_suffix(token_a)
    base_b = _strip_plural_suffix(token_b)
    if base_a == base_b:
        return False

    for shorter, longer in ((base_a, base_b), (base_b, base_a)):
        if not _looks_like_demonym_form(longer):
            continue
        stem = shorter[:-1] if shorter.endswith(("A", "E", "Y")) else shorter
        if len(stem) < 4:
            continue
        if longer.startswith(stem):
            return True
    return False


def _likely_region_adjectival_conflict(tokens_a: tuple[str, ...], tokens_b: tuple[str, ...]) -> bool:
    if len(tokens_a) != len(tokens_b) or len(tokens_a) < 2:
        return False
    if tokens_a[:-1] != tokens_b[:-1]:
        return False
    return _likely_country_demonym_conflict(tokens_a[-1], tokens_b[-1])



def surface_forms_conflict(a: str, b: str) -> bool:
    norm_a = normalize_surface_form(a)
    norm_b = normalize_surface_form(b)
    if not norm_a or not norm_b or norm_a == norm_b:
        return False

    tokens_a = tuple(token for token in norm_a.split("_") if token)
    tokens_b = tuple(token for token in norm_b.split("_") if token)
    token_set_a = set(tokens_a)
    token_set_b = set(tokens_b)

    for left, right in OPPOSING_TERM_PAIRS:
        if (left in token_set_a and right in token_set_b) or (right in token_set_a and left in token_set_b):
            return True

    if len(tokens_a) == 1 and len(tokens_b) == 1 and _likely_country_demonym_conflict(tokens_a[0], tokens_b[0]):
        return True

    if _likely_region_adjectival_conflict(tokens_a, tokens_b):
        return True

    return False


def classify_surface_form(text: str) -> str:
    normalized = normalize_surface_form(text)
    if not normalized:
        return "unknown"
    if normalized in ROLE_TERMS:
        return "role_artifact"
    if normalized in CONCEPT_LIKE_TERMS:
        return "concept_like"
    if normalized in NAMED_ENTITY_HINTS:
        return "named_entity"
    if len(normalized) <= 3:
        return "unknown"
    if "_" in normalized and all(token[:1].isalpha() for token in normalized.split("_") if token):
        return "named_entity"
    return "concept_like"


def are_potential_aliases(a: str, b: str) -> bool:
    norm_a = normalize_surface_form(a)
    norm_b = normalize_surface_form(b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True

    class_a = classify_surface_form(norm_a)
    class_b = classify_surface_form(norm_b)
    if "role_artifact" in {class_a, class_b} and class_a != class_b:
        return False
    if {norm_a, norm_b} == {"EU", "EUROPE"}:
        return False
    if surface_forms_conflict(norm_a, norm_b):
        return False

    ratio = SequenceMatcher(None, norm_a.replace("_", " "), norm_b.replace("_", " ")).ratio()
    if ratio >= 0.92:
        return True

    tokens_a = set(norm_a.split("_"))
    tokens_b = set(norm_b.split("_"))
    if tokens_a and tokens_b and tokens_a == tokens_b:
        return True

    return False
