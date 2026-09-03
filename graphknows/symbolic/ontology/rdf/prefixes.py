"""Authoritative prefix table + CURIE minting. Shipped, not scraped.

Document prefix bindings are serialization sugar with no semantics and no
round-trip stability: merging one folder of RDF assets made rdflib invent
``default1``..``default15``, ``MilitaryRanksOntology1`` and ``dc11``. None of
those are usable as identifiers, so nothing here reads them. CURIEs come from,
in order:

1. :data:`BOOTSTRAP` — reserved + well-known namespaces (this file)
2. ``vann:preferredNamespacePrefix`` — the ontology's own declared preference
3. deterministic fallback — ``<last-segment>_<4 hex of the namespace IRI>``

Rule 3 is deliberately ugly: an unrecognised namespace should look
unrecognised, and the hash keeps it stable across runs and corpora instead of
order-dependent — a prefix that changes between two ingests of the same corpus
silently forks every CURIE-keyed row written from it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

# --- reserved / normative (never re-bindable) ------------------------------
RESERVED: dict[str, str] = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://www.w3.org/2001/XMLSchema#": "xsd",
    "http://www.w3.org/XML/1998/namespace": "xml",
    "http://www.w3.org/2001/XMLSchema-instance": "xsi",
}

# --- well-known bootstrap --------------------------------------------------
WELL_KNOWN: dict[str, str] = {
    "http://www.w3.org/2004/02/skos/core#": "skos",
    "http://www.w3.org/2008/05/skos-xl#": "skosxl",
    "http://purl.org/dc/terms/": "dcterms",
    "http://purl.org/dc/elements/1.1/": "dce",  # legacy DC 1.1 — NOT dcterms
    "http://purl.org/dc/dcam/": "dcam",
    "http://purl.org/dc/dcmitype/": "dcmitype",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "https://schema.org/": "schema",
    "http://schema.org/": "schema",  # both forms occur in the wild
    "http://www.w3.org/ns/prov#": "prov",
    "http://www.w3.org/ns/shacl#": "sh",
    "http://www.w3.org/ns/shex#": "shex",
    "http://www.w3.org/ns/dcat#": "dcat",
    "http://rdfs.org/ns/void#": "void",
    "http://purl.org/vocab/vann/": "vann",
    "http://www.w3.org/2003/06/sw-vocab-status/ns#": "vs",
    "http://www.w3.org/ns/org#": "org",
    "http://www.w3.org/2006/time#": "time",
    "http://www.w3.org/2006/vcard/ns#": "vcard",
    "http://www.w3.org/2003/01/geo/wgs84_pos#": "geo",
    "https://www.w3.org/2003/01/geo/wgs84_pos#": "geo",
    "http://www.opengis.net/ont/geosparql#": "geosparql",
    "http://purl.org/linked-data/cube#": "qb",
    "http://www.w3.org/ns/sosa/": "sosa",
    "http://www.w3.org/ns/ssn/": "ssn",
    "http://www.w3.org/ns/oa#": "oa",
    "http://www.w3.org/ns/ldp#": "ldp",
    "http://www.w3.org/ns/sparql-service-description#": "sd",
    "http://www.w3.org/ns/r2rml#": "rr",
    "http://www.w3.org/ns/csvw#": "csvw",
    "https://www.w3.org/ns/activitystreams#": "as",
    "http://www.w3.org/ns/odrl/2/": "odrl",
    "http://creativecommons.org/ns#": "cc",
    "http://usefulinc.com/ns/doap#": "doap",
    "http://purl.org/ontology/bibo/": "bibo",
    "http://rdfs.org/sioc/ns#": "sioc",
    "http://purl.org/goodrelations/v1#": "gr",
    "http://www.w3.org/2003/11/swrl#": "swrl",
    "http://www.w3.org/2003/11/swrlb#": "swrlb",
    "http://purl.obolibrary.org/obo/": "obo",  # FLAT: obo:BFO_0000001
    "http://www.geneontology.org/formats/oboInOwl#": "oboInOwl",
    "http://www.wikidata.org/entity/": "wd",
    "http://www.wikidata.org/prop/direct/": "wdt",
    "http://www.wikidata.org/prop/": "p",
    "http://www.wikidata.org/prop/statement/": "ps",
    "http://www.wikidata.org/prop/qualifier/": "pq",
    "http://dbpedia.org/ontology/": "dbo",
    "http://dbpedia.org/resource/": "dbr",
    "http://dbpedia.org/property/": "dbp",
}

# --- corpus-local -----------------------------------------------------------
# The two CCO IRI schemes are the SAME ontology in two releases; named apart so
# the split shows up in every CURIE instead of being implied. The `wwww.` hosts
# are typos IN THE SOURCE FILES, kept as distinct namespaces under honest names
# — aliasing them onto the correct host would silently merge different IRIs.
CORPUS: dict[str, str] = {
    "https://www.commoncoreontologies.org/": "cco",
    "http://www.ontologyrepository.com/CommonCoreOntologies/": "ccov1",
    "http://wwww.ontologyrepository.com/CommonCoreOntologies/": "ccov1_typohost",
    "https://www.commoncoreontologies.org/mro/": "ccomro",
    "http://www.ontologylibrary.mil/CommonCore/Mid/MilitaryRanksOntology/": "mro",
    "http://wwww.ontologylibrary.mil/CommonCore/Mid/MilitaryRanksOntology/": "mro_typohost",
    "http://www.ontologylibrary.mil/CommonCore/Mid/JointDoctrineOntology#": "jdo",
    "http://www.ontologylibrary.mil/CommonCore/": "mil",  # catch-all, longest match wins
    "http://wwww.ontologylibrary.mil/CommonCore/": "mil_typohost",
    "http://www.github/argumentsontology": "argo",  # IRI has no valid host, as shipped
}

BOOTSTRAP: dict[str, str] = {**WELL_KNOWN, **CORPUS, **RESERVED}

_SEG = re.compile(r"[^A-Za-z0-9]+")
# PN_LOCAL may not end in ':' or contain whitespace; a local name that does is a
# source defect, not something to normalise away.
_BAD_LOCAL = re.compile(r"[\s]|:$")


def split_iri(iri: str) -> tuple[str, str]:
    """Split an IRI at the last gen-delim into ``(namespace, local name)``.

    Not the inverse of CURIE expansion — CURIE expansion is pure concatenation
    and never re-parses, but going IRI -> CURIE requires choosing a split point.
    """
    cut = max(iri.rfind("#"), iri.rfind("/"))
    return (iri[: cut + 1], iri[cut + 1 :]) if cut >= 0 else (iri, "")


class Prefixes:
    """Namespace IRI -> prefix, resolved once per corpus and then frozen."""

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self._map = dict(BOOTSTRAP)
        self._map.update(extra or {})
        self._minted: dict[str, str] = {}
        self._known: list[str] = []
        self.unknown: set[str] = set()
        self.defects: list[str] = []
        self._reindex()

    @property
    def table(self) -> dict[str, str]:
        """The resolved namespace -> prefix map (bootstrap + vann + corpus)."""
        return dict(self._map)

    def _reindex(self) -> None:
        # longest namespace first: /obo/ must not win over a longer known root
        self._known = sorted(self._map, key=len, reverse=True)

    def split(self, iri: str) -> tuple[str, str]:
        """Return ``(namespace, local)``, table match first, gen-delim only as fallback.

        A namespace is NOT derivable from an IRI: ``.../MilitaryRanksOntology/SSD/x``
        gen-delim-splits into a namespace that does not exist. Local names may
        legally contain ``/``, so a known root claims the whole remainder.
        """
        for ns in self._known:
            if iri.startswith(ns) and len(iri) > len(ns):
                return ns, iri[len(ns) :]
        return split_iri(iri)

    def learn_vann(self, graph: Any) -> int:
        """Adopt ``vann:preferredNamespacePrefix`` declarations, returning the count.

        The one declaration an ontology makes about its OWN prefix, as opposed to
        the per-document bindings a serializer invents. Needs the ``ontology``
        extra; every other method here is stdlib.
        """
        try:
            from rdflib import URIRef
        except ImportError as exc:
            from graphknows.exceptions import MissingExtraError

            raise MissingExtraError("Reading vann prefix declarations", "ontology") from exc

        pnp = URIRef("http://purl.org/vocab/vann/preferredNamespacePrefix")
        pnu = URIRef("http://purl.org/vocab/vann/preferredNamespaceUri")
        adopted = 0
        for subj, pfx in graph.subject_objects(pnp):
            declared = next(iter(graph.objects(subj, pnu)), None) or subj
            ns = str(declared)
            if not ns.endswith(("#", "/")):
                ns += "#"
            if ns not in self._map:
                self._map[ns] = str(pfx)
                adopted += 1
        self._reindex()
        return adopted

    def prefix(self, ns: str) -> str:
        """Prefix for a namespace: the table, then a stable hashed fallback."""
        if ns in self._map:
            return self._map[ns]
        if ns not in self._minted:
            stem = _SEG.sub("_", split_iri(ns.rstrip("#/"))[1]).strip("_").lower() or "ns"
            # sha1 of the namespace: a naming device, not a security boundary —
            # what matters is that the same namespace mints the same prefix in
            # every process and corpus.
            tag = hashlib.sha1(ns.encode(), usedforsecurity=False).hexdigest()[:4]
            self._minted[ns] = f"{stem}_{tag}"
            self.unknown.add(ns)
        return self._minted[ns]

    def curie(self, iri: Any) -> str:
        """CURIE for an IRI, recording (never repairing) a malformed local name."""
        ns, local = self.split(str(iri))
        if _BAD_LOCAL.search(local):
            self.defects.append(f"malformed local name: {iri}")
        return f"{self.prefix(ns)}:{local}"

    def near_duplicates(self, namespaces: Iterable[str]) -> list[tuple[str, ...]]:
        """Namespaces identical once repeated characters collapse — i.e. typos.

        Caught ``wwww.ontologylibrary.mil`` (four w's) sitting next to the real
        host in the CCO corpus, which is why rdflib had minted a ``...Ontology1``.
        Run it against the shipped table too, not only the namespaces seen: the
        correct spelling may exist nowhere in the data.
        """
        buckets: dict[str, set[str]] = {}
        for ns in namespaces:
            buckets.setdefault(re.sub(r"(.)\1+", r"\1", ns), set()).add(ns)
        return [tuple(sorted(v)) for v in buckets.values() if len(v) > 1]
