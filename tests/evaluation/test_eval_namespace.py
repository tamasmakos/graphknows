"""`--namespace` must reach BOTH the run and the reset.

The defect this prevents: `--reset` built its config from the default
`LocomoConfig()`, so `--reset --namespace foo` would drop the *default*
namespace's databases — potentially someone else's baseline — while the run
wrote to `foo`. A reset that clears the wrong graph is worse than no flag.
"""

from __future__ import annotations

import pytest

from evaluation.locomo.cli import build_parser
from evaluation.locomo.config import LocomoConfig

pytestmark = pytest.mark.unit


def test_the_flag_exists_and_defaults_to_empty() -> None:
    args = build_parser().parse_args([])
    assert args.namespace == ""


def test_the_flag_is_parsed() -> None:
    args = build_parser().parse_args(["--namespace", "run_a"])
    assert args.namespace == "run_a"


def test_an_empty_flag_leaves_the_default_namespace() -> None:
    cfg = LocomoConfig()
    assert cfg.namespace == "eval_locomo"


def test_the_reset_config_is_built_from_the_run_namespace() -> None:
    """Read the source: reset must copy args.namespace before dropping anything."""
    import inspect

    from evaluation.locomo import cli

    src = inspect.getsource(cli.amain)
    reset_block = src[src.index("if args.reset:") :]
    assert "args.namespace" in reset_block.split("if args.compare:")[0], (
        "--reset ignores --namespace and would drop the default namespace's databases"
    )


def test_every_run_call_site_forwards_it() -> None:
    import inspect

    from evaluation.locomo import cli

    src = inspect.getsource(cli.amain)
    assert src.count("namespace=args.namespace") == 3, (
        "a _run_one call site does not forward --namespace, so it would silently "
        "write to the default namespace"
    )
