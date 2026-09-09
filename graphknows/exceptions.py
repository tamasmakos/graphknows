"""Public exception hierarchy for graphknows.

Catch :class:`GraphKnowsError` to handle any error raised by the library.
"""

from __future__ import annotations


class GraphKnowsError(Exception):
    """Base class for all graphknows errors."""


class ConfigurationError(GraphKnowsError):
    """Settings are invalid or incomplete (bad env vars, missing secrets)."""


class StoreError(GraphKnowsError):
    """A storage backend operation failed (connection, query, or transport)."""


class MissingExtraError(GraphKnowsError, ImportError):
    """A feature was used whose optional dependency extra is not installed.

    Only raise this for a dependency that really is behind an extra declared in
    ``pyproject.toml`` — the message tells the user to run
    ``pip install 'graphknows[<extra>]'``, and naming an extra that does not
    exist sends them to a command that installs nothing. For a *core* dependency
    that has somehow gone missing, raise :class:`BrokenInstallError` instead.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working.
    """

    #: Extras declared in pyproject.toml. Kept here so the test suite can assert
    #: that every call site names a real one.
    KNOWN_EXTRAS = frozenset({"langgraph", "observability", "ontology"})

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires the '{extra}' extra. "
            f"Install it with:  pip install 'graphknows[{extra}]'"
        )
        self.feature = feature
        self.extra = extra


class BrokenInstallError(GraphKnowsError, ImportError):
    """A *core* dependency is missing — the installation itself is broken.

    Distinct from :class:`MissingExtraError`: there is no extra to install, so
    the remedy is to repair the environment. This is what a stale container
    image looks like from the inside — the code is current, the site-packages
    are not — so the message names the offending distribution explicitly.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working.
    """

    def __init__(self, feature: str, distribution: str, hint: str = "") -> None:
        super().__init__(
            f"{feature} needs the '{distribution}' package, which is a core "
            f"dependency of graphknows but is not importable. The install is "
            f"incomplete — reinstall with:  pip install --force-reinstall graphknows"
            f"\n(If you are in a container, the image is stale relative to the "
            f"source it is running: rebuild it.)" + (f"\n{hint}" if hint else "")
        )
        self.feature = feature
        self.distribution = distribution
        self.hint = hint


class SchemaVersionMismatchError(StoreError):
    """A namespace's recorded schema version is not the running one.

    Raised before any read or write, so a graph built by a different schema is
    never touched. No migration is attempted and nothing is dropped
    automatically — the operator decides, so the message names the remedy: a
    rebuild from the original sources.

    ``recorded`` is ``None`` for a pre-existing database that carries no stamp
    at all, which is a mismatch too (FR-014) — it predates stamping and is not
    assumed compatible.

    Subclasses :class:`StoreError` so existing store-failure handlers keep
    working.
    """

    def __init__(
        self,
        namespace: str,
        database: str,
        recorded: str | None,
        running: str,
    ) -> None:
        super().__init__(
            f"Namespace {namespace or '(default)'} (database '{database}') was built by "
            f"schema version {recorded or 'none recorded — it predates schema stamping'}, "
            f"but this code runs schema version {running}. Refusing to read or write it.\n"
            f"Rebuild it from its original sources: drop the namespace with "
            f"Memory.drop_namespace() (MCP tool 'memory_drop_namespace'), or for the "
            f"default namespace Memory.purge_memory() (MCP tool 'memory_purge'), then "
            f"re-ingest."
        )
        self.namespace = namespace
        self.database = database
        self.recorded = recorded
        self.running = running


class MissingModelError(GraphKnowsError, OSError):
    """A spaCy model is not installed.

    Distinct from :class:`MissingExtraError` and :class:`BrokenInstallError`:
    a spaCy model is not a PyPI distribution and not behind an extra — PyPI
    rejects direct-URL requirements, so it can only ever be a documented
    post-install download (``python -m spacy download <model>``). Neither
    ``pip install 'graphknows[x]'`` nor a reinstall is the remedy.

    Subclasses :class:`OSError` so existing handlers around ``spacy.load``,
    which raises :class:`OSError`, keep working.
    """

    def __init__(self, model: str) -> None:
        super().__init__(
            f"spaCy model '{model}' is not installed. "
            f"Install it with:  python -m spacy download {model}"
        )
        self.model = model


class PackConflictError(ConfigurationError):
    """Two loaded packs claim the same concept URI or predicate id.

    Loading is all-or-nothing (FR-025): the message names both packs and the
    contested key so the operator can drop one of them, and no pack is left
    loaded.
    """

    def __init__(self, key: str, first_pack: str, second_pack: str) -> None:
        super().__init__(
            f"Packs '{first_pack}' and '{second_pack}' both claim '{key}'. "
            f"No pack was loaded — remove one of them or rename the key."
        )
        self.key = key
        self.first_pack = first_pack
        self.second_pack = second_pack
