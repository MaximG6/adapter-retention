"""Tests for the count-word gate.

Built from an external review's meta-note: every count word in the body is a claim, and
the claim audit cannot see it, because a count word is not a printed measurement and the
thing it counts is a structure. Four instances were live when this was written.

The gate gets the standard treatment: fed the exact sentences that shipped, and fed
correct prose, and required to distinguish them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _load("countcheck")

HAVE = {
    "weight-space adapters": 9, "behavioural adapters": 6, "practice entries": 7,
    "registered predictions": 9, "figures": 12, "body tables": 2,
}


def test_fires_on_the_sentences_that_shipped() -> None:
    shipped = [
        "validated within 2.3% on six published adapters and across three decades",
        "Fifteen practices each evidenced by an error of ours",
        "We pre-registered four predictions before the runs they concern",
        "Eight Taboo adapters x four precisions x 32 prompts",
    ]
    for text in shipped:
        assert cc.check(text, HAVE), f"gate did not fire on {text!r}"


def test_passes_the_corrected_sentences() -> None:
    corrected = [
        "validated within 2.3% on nine published adapters and across three decades",
        "seven practices each evidenced by an error of ours",
        "We pre-registered nine predictions before the runs they concern",
        "Six Taboo adapters x four precisions x 32 prompts x two conditions",
    ]
    for text in corrected:
        assert not cc.check(text, HAVE), f"gate fired on correct prose: {text!r}"


def test_a_cardinal_inside_a_compound_is_not_a_count_word() -> None:
    """`rank-32 taboo adapters` is not a claim that there are 32 taboo adapters, and
    the first version of this gate said it was. A gate that fires on correct input is
    the failure this project has recorded seven times."""
    assert not cc.check("the same six rank-32 taboo adapters in both panels", HAVE)


def test_the_shipped_body_agrees_with_its_structures() -> None:
    have = cc.structures()
    for path in [ROOT / "paper" / "tex" / "main.tex"] + sorted(
            (ROOT / "paper").glob("*.md")):
        bad = cc.check(path.read_text(encoding="utf-8"), have)
        assert not bad, f"{path.name}: {[(b[0], b[1], b[2]) for b in bad]}"
