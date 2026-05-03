"""
NER Validation Layer — cross-validates GLiNER2 extraction against spaCy en_core_web_sm.

spaCy serves as an *external oracle*:
  - It has its own type system (PERSON, ORG, GPE, LAW, EVENT, ...)
  - We map its types to our ontology buckets for comparison
  - Discrepancy = candidate for review, not automatic rejection

Usage:
  from graphrag.ner_validator import NERValidator
  validator = NERValidator()
  report = validator.validate(text, gliner_entities)
  # report['agreement_rate'], report['missing_in_gliner'], report['extra_in_gliner']
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# spaCy label -> our ontology bucket mapping
SPACY_TO_BUCKET: Dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",       # geopolitical entity
    "LOC": "LOCATION",
    "LAW": "POLICY",
    "EVENT": "EVENT",
    "FAC": "LOCATION",       # facilities
    "NORP": "ORGANIZATION",  # nationalities/political groups
    "PRODUCT": "CONCEPT",
    "WORK_OF_ART": "CONCEPT",
    "LANGUAGE": "CONCEPT",
}


class NERValidator:
    """Cross-validates GLiNER2 entity extraction against spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy
            self._nlp = spacy.load(model_name)
            self._available = True
            logger.info("NERValidator ready: %s", model_name)
        except Exception as e:
            logger.warning("NERValidator unavailable (%s): %s", model_name, e)
            self._available = False
            self._nlp = None

    @property
    def available(self) -> bool:
        return self._available

    def _spacy_entities(self, text: str) -> List[Dict[str, str]]:
        if not self._available:
            return []
        doc = self._nlp(text)
        out = []
        for ent in doc.ents:
            bucket = SPACY_TO_BUCKET.get(ent.label_, None)
            if bucket:
                out.append({
                    "text": ent.text.strip(),
                    "spacy_label": ent.label_,
                    "bucket": bucket,
                })
        return out

    def validate(
        self,
        text: str,
        gliner_entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare GLiNER2 entities against spaCy.

        Returns a dict with:
          agreement_rate       float 0-1 (how many spaCy entities GLiNER also found)
          spacy_entities       list of spaCy finds
          gliner_surface_set   set of surface strings from GLiNER
          missing_in_gliner    spaCy entities not found by GLiNER (false negatives)
          extra_in_gliner      GLiNER entities not found by spaCy (may be correct for legal NER)
          bucket_mismatches    where same surface text got different buckets
        """
        spacy_ents = self._spacy_entities(text)
        gliner_surfaces = {
            e.get("text", "").strip().lower()
            for e in gliner_entities
            if e.get("text", "").strip()
        }
        gliner_bucket_map = {
            e.get("text", "").strip().lower(): e.get("ontology_label", "UNKNOWN")
            for e in gliner_entities
        }

        missing = []
        bucket_mismatches = []
        for se in spacy_ents:
            surf = se["text"].lower()
            if surf not in gliner_surfaces:
                missing.append(se)
            else:
                g_bucket = gliner_bucket_map.get(surf, "UNKNOWN")
                if g_bucket != se["bucket"]:
                    bucket_mismatches.append({
                        "text": se["text"],
                        "spacy_bucket": se["bucket"],
                        "gliner_bucket": g_bucket,
                    })

        spacy_surfaces = {se["text"].lower() for se in spacy_ents}
        extra = [
            e for e in gliner_entities
            if e.get("text", "").strip().lower() not in spacy_surfaces
        ]

        total_spacy = len(spacy_ents)
        found = total_spacy - len(missing)
        agreement = found / total_spacy if total_spacy > 0 else 1.0

        return {
            "agreement_rate": round(agreement, 3),
            "total_spacy": total_spacy,
            "total_gliner": len(gliner_entities),
            "found_by_both": found,
            "missing_in_gliner": missing,
            "extra_in_gliner": extra,
            "bucket_mismatches": bucket_mismatches,
            "spacy_entities": spacy_ents,
        }

    def validate_batch(
        self,
        chunks: List[Dict[str, Any]],
        gliner_results: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Aggregate validation across multiple chunks."""
        all_missing: List[Dict] = []
        all_extra: List[Dict] = []
        all_mismatches: List[Dict] = []
        rates: List[float] = []

        for chunk, ents in zip(chunks, gliner_results):
            text = chunk.get("text", "")
            r = self.validate(text, ents)
            rates.append(r["agreement_rate"])
            all_missing.extend(r["missing_in_gliner"])
            all_extra.extend(r["extra_in_gliner"])
            all_mismatches.extend(r["bucket_mismatches"])

        return {
            "mean_agreement_rate": round(sum(rates) / len(rates), 3) if rates else 0.0,
            "total_missing": len(all_missing),
            "total_extra": len(all_extra),
            "total_bucket_mismatches": len(all_mismatches),
            "missing_surface_counts": _count_surfaces(all_missing, "text"),
            "mismatch_details": all_mismatches[:20],
        }


def _count_surfaces(items: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        surf = item.get(key, "").strip()
        if surf:
            counts[surf] = counts.get(surf, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1])[:30])
