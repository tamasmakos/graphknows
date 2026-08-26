"""``ancestors()`` must follow EVERY ``broader`` parent, not just the first.

CCO concepts are genuinely multi-parent: the bundled overlay makes ``Store`` a
``Facility`` AND an ``Organization``. A walk that keeps only ``broader[0]``
silently drops a whole branch of the hierarchy, so generalisation never reaches
``Organization``, ``Group of Agents`` or anything above them — the label
vocabulary offered to the extractor shrinks with no error anywhere.
"""

from __future__ import annotations

from graphknows.symbolic.ontology.skos import Concept, ancestors


def _scheme(parents: dict[str, tuple[str, ...]]) -> dict[str, Concept]:
    """A minimal in-memory concept scheme: label -> its ``broader`` labels."""
    return {
        label: Concept(
            uri=f"http://example.org/{label.replace(' ', '_')}",
            pref_label=label,
            definition="",
            kind="class",
            broader=broader,
        )
        for label, broader in parents.items()
    }


_STORE = _scheme(
    {
        "Store": ("Facility", "Organization"),
        "Facility": ("Material Artifact",),
        "Organization": ("Group of Agents",),
        "Material Artifact": ("material entity",),
        "Group of Agents": ("object aggregate",),
        "material entity": (),
        "object aggregate": (),
    }
)


class TestAncestors:
    """The transitive ``broader`` closure multi-hop generalisation walks."""

    def test_multi_parent_concept_reaches_both_branches(self) -> None:
        """``Store`` generalises through Facility AND Organization."""
        assert set(ancestors(_STORE, "Store")) == {
            "Facility",
            "Organization",
            "Material Artifact",
            "Group of Agents",
            "material entity",
            "object aggregate",
        }

    def test_diamond_yields_each_ancestor_once(self) -> None:
        """A shared grandparent is reported once, not once per path."""
        scheme = _scheme(
            {
                "Store": ("Facility", "Organization"),
                "Facility": ("continuant",),
                "Organization": ("continuant",),
                "continuant": (),
            }
        )
        result = ancestors(scheme, "Store")
        assert len(result) == len(set(result))
        assert set(result) == {"Facility", "Organization", "continuant"}

    def test_cycle_terminates_without_repeating(self) -> None:
        """A -> B -> A must not loop or list a label twice."""
        result = ancestors(_scheme({"A": ("B",), "B": ("A",)}), "A")
        assert "B" in result
        assert len(result) == len(set(result))

    def test_depth_one_returns_only_direct_parents(self) -> None:
        """``depth`` still bounds how far the walk generalises."""
        assert set(ancestors(_STORE, "Store", depth=1)) == {"Facility", "Organization"}
