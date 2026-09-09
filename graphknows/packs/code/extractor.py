"""The code pack's extractor: facts read off the syntax, with no model (FR-030)."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from hashlib import sha256

# _byte_span is the core's own rule for citing a surface; a second copy here
# would be a second rule, and evidence has to agree across extractors.
from graphknows.ingestion.extraction.protocol import PackGuidance, _byte_span
from graphknows.models.fact import Fact, Mention
from graphknows.models.segment import Segment, SegmentKind
from graphknows.packs.code.pack import symbol_id

_CALLS = "code:calls"
_IMPORTS = "code:imports"
_DEFINES = "code:defines"
_TESTS = "code:tests"

_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True, slots=True)
class _Symbol:
    """A code symbol as this extractor names it: a kind and a bare name."""

    kind: str
    name: str

    @property
    def id(self) -> str:
        return symbol_id(self.kind, self.name)


@dataclass(frozen=True, slots=True)
class _Claim:
    """One relation the syntax asserts: *subject* ``predicate`` *object*."""

    predicate: str
    object: _Symbol


class CodeExtractor:
    """Turns one code segment into the mentions and facts its syntax evidences.

    Deterministic by construction: the syntax tree says who calls, imports,
    defines and tests whom, so no model is loaded and no provider is called.
    """

    name = "code"
    version = "1"

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def extract(self, segment: Segment, pack: PackGuidance) -> tuple[list[Mention], list[Fact]]:
        """The mentions and facts *segment* evidences, or nothing when it is not code."""
        tree = _parse(segment)
        if tree is None:
            return [], []
        subject = _subject(tree, segment.path)
        claims = list(_claims(tree, subject))
        admits = pack.hygiene()
        cited = {
            symbol.id: self._mention(symbol, segment)
            for symbol in [subject, *(claim.object for claim in claims)]
            if admits(symbol.name)
        }
        # A body calling one helper twice asserts the relation once.
        linked = {
            (claim.predicate, claim.object.id): claim
            for claim in claims
            if subject.id in cited and claim.object.id in cited
        }
        facts = [self._fact(claim, segment, subject) for claim in linked.values()]
        return list(cited.values()), facts

    def _mention(self, symbol: _Symbol, segment: Segment) -> Mention:
        """*symbol* as a mention, cited where it is written, else at the whole segment.

        A segment does not always spell its own symbol — a module segment names
        no module — and the segment is the evidence for it either way (FR-008).
        """
        span = _byte_span(segment.text, symbol.name) or (0, len(segment.text.encode()))
        return Mention(
            segment_id=segment.id,
            entity_id=symbol.id,
            surface=symbol.name,
            span=span,
            label=symbol.kind,
            extractor=self.name,
        )

    def _fact(self, claim: _Claim, segment: Segment, subject: _Symbol) -> Fact:
        """*claim* as a fact anchored to the segment that evidences it."""
        key = f"{segment.id}|{subject.id}|{claim.predicate}|{claim.object.id}"
        return Fact(
            id=sha256(key.encode()).hexdigest()[:16],
            subject=subject.id,
            predicate=claim.predicate,
            object=claim.object.id,
            extractor=self.name,
            extractor_version=self.version,
        )


def _parse(segment: Segment) -> ast.Module | None:
    """*segment* as a syntax tree, or ``None`` when it is not parseable code.

    A segment cut mid-file is indented and may be syntactically incomplete; that
    is a segment this extractor has nothing to say about, never an ingest error.
    """
    if segment.kind is not SegmentKind.code:
        return None
    try:
        return ast.parse(textwrap.dedent(segment.text))
    except SyntaxError:
        return None


def _subject(tree: ast.Module, path: str) -> _Symbol:
    """The symbol the segment is about — its single definition, else the module."""
    node = tree.body[0] if len(tree.body) == 1 else None
    name = path.rsplit(".", 1)[-1] or (node.name if isinstance(node, _DEFINITION) else "")
    if isinstance(node, ast.ClassDef):
        return _Symbol("class", name)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return _Symbol("function", name)
    return _Symbol("module", name or path)


def _claims(tree: ast.Module, subject: _Symbol) -> list[_Claim]:
    """Every relation the tree asserts about *subject*."""
    top = tree.body[0] if len(tree.body) == 1 else None
    invocation = _TESTS if subject.name.startswith("test_") else _CALLS
    claims: list[_Claim] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (called := _called_name(node)) is not None:
            claims.append(_Claim(invocation, _Symbol("function", called)))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            claims += [_Claim(_IMPORTS, _Symbol("module", name)) for name in _imported(node)]
        elif isinstance(node, _DEFINITION) and node is not top:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            claims.append(_Claim(_DEFINES, _Symbol(kind, node.name)))
    return claims


def _called_name(node: ast.Call) -> str | None:
    """The bare name of what *node* calls, or ``None`` for a call with no name."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return node.func.attr if isinstance(node.func, ast.Attribute) else None


def _imported(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The modules *node* brings into the namespace; a relative import names none."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module] if node.module and not node.level else []
