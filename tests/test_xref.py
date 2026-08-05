"""Tests for the cross-reference checker.

Section C.5 of the manuscript records a cross-reference checker that reported nine false
positives: its pattern required whitespace after the section number, and every heading in
the paper uses a period. A checker that is confidently wrong is worse than none, because
its author learns to discount it.

So this one is required to distinguish a reference known to be broken from one known to
be good, in both directions, and the period case is pinned explicitly.
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


xref = _load("xref")

MAIN = r"""
\section{Introduction}
Text referring to \S2 and \S2.1 and Figure~1.
\section{Method}
\subsection{Setting}
\subsection{Statistics}
\begin{figure}\end{figure}
\begin{equation}x=1\end{equation}
"""

APX = r"""
\appendix
\section{The Tool}
\subsection{Accuracy}
\section{Full Tables}
"""


def test_structure_numbers_like_latex() -> None:
    secs, apps, titles = xref.structure(MAIN, APX)
    assert secs == {"1", "2", "2.1", "2.2"}
    assert apps == {"A", "A.1", "B"}
    assert titles["2.1"] == "Setting"


def test_good_reference_resolves() -> None:
    assert xref.check(MAIN, APX) == []


def test_known_bad_section_is_flagged() -> None:
    bad = xref.check(MAIN + r"\newline See \S7.4 for detail.", APX)
    assert [t for k, t, _ in bad if k == "section"] == ["7.4"]


def test_known_bad_appendix_is_flagged() -> None:
    bad = xref.check(MAIN + r"\newline listed in Appendix~E.", APX)
    assert [t for k, t, _ in bad if k == "appendix"] == ["E"]


def test_reference_ending_a_sentence_is_still_resolved() -> None:
    """The exact defect of the earlier checker: a pattern demanding whitespace after the
    number treats "\\S2.1." as unresolved because a period follows."""
    ok = xref.check(MAIN + r"\newline as shown in \S2.1.", APX)
    assert ok == [], f"period after a valid reference was misread: {ok}"


def test_reference_in_parentheses_is_resolved() -> None:
    assert xref.check(MAIN + r"\newline (\S2.2)", APX) == []


def test_figure_over_count_is_flagged() -> None:
    bad = xref.check(MAIN + r"\newline see Figure~9", APX)
    assert [t for k, t, _ in bad if k == "figure"] == ["9"]


def test_appendix_subsection_is_checked() -> None:
    assert xref.check(MAIN + r"\newline Appendix~A.1", APX) == []
    bad = xref.check(MAIN + r"\newline Appendix~A.7", APX)
    assert [t for k, t, _ in bad if k == "appendix"] == ["A.7"]


def test_equation_reference_resolves() -> None:
    assert xref.check(MAIN + r"\newline Equation~(1)", APX) == []
    bad = xref.check(MAIN + r"\newline Equation~(4)", APX)
    assert [t for k, t, _ in bad if k == "equation"] == ["4"]


def test_the_built_paper_has_no_unresolved_references() -> None:
    """The gate itself. Fails while any reference in the built document dangles."""
    main = (ROOT / "paper" / "tex" / "main.tex").read_text(encoding="utf-8")
    apx = (ROOT / "paper" / "tex" / "appendices.tex").read_text(encoding="utf-8")
    bad = xref.check(main, apx)
    summary = sorted({f"{k} {t}" for k, t, _ in bad})
    assert not bad, f"{len(bad)} unresolved: {summary[:20]}"


def test_bare_appendix_reference_is_checked() -> None:
    """The M17 class: "see D.1.2" is the same reference as "Appendix D.1.2".

    Both the appendix-letter remapper and this gate keyed on the literal word
    "Appendix", so four live references in the reproduction appendix pointed at the
    safety-adapter appendix and nothing objected. Pinned in both directions.
    """
    main = ("\\section{One}\n\\appendix\n\\section{Alpha}\n"
            "\\subsection{First}\n")
    bad = xref.check(main, "The breakdown is in A.9.")
    assert [t for k, t, _ in bad if k == "bare-appendix"] == ["A.9"]
    assert not [t for k, t, _ in xref.check(main, "The breakdown is in A.1.")
                if k == "bare-appendix"]


def test_bare_pattern_does_not_fire_on_non_references() -> None:
    """Version strings, model names, paths and decimals are not appendix references."""
    main = ("\\section{One}\n\\appendix\n\\section{Alpha}\n"
            "\\subsection{First}\n")
    text = "Qwen3-8B.1 and v2.1 and results/D.7 and 0.15 and torch 2.11.0"
    assert not [t for k, t, _ in xref.check(main, text) if k == "bare-appendix"]


def test_markdown_appendix_labels_match_the_latex_numbering() -> None:
    """The appendix markdown writes its own labels and the arXiv build lets LaTeX count.
    Those agree only while the labels are sequential from 1, so inserting a section in the
    middle silently shifts every later one. It happened twice -- a `B.6b` between B.6 and
    B.7, and a `D.7a` -- and `check` could not see either, because every shifted target
    still exists and every reference still resolves, to the wrong section."""
    assert not xref.numbering_drift()


def test_the_numbering_gate_fires_on_an_inserted_subsection() -> None:
    """Against the state that shipped: one heading inserted between B.6 and B.7, and
    every later label is one short. Fed the real Appendix B with that insertion restored,
    the gate must flag the insertion and every subsection after it."""
    src = (ROOT / "paper" / "appendix-B-tables.md").read_text(encoding="utf-8")
    assert not xref.drift_in("b.md", src), "the committed appendix is sequential"

    lines = src.splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith("## B.7 "))
    broken = "\n".join(lines[:at] + ["## B.6b An inserted section", ""] + lines[at:])
    bad = xref.drift_in("b.md", broken)
    assert [b[1] for b in bad][0] == "B.6b"
    assert len(bad) == 8, f"one insertion should shift seven later labels, got {bad}"


def test_the_numbering_gate_passes_a_sequential_appendix() -> None:
    """A gate that fires on correct input is the failure this project has recorded
    repeatedly, so the pass direction is a test too."""
    good = "\n".join(f"## C.{i} Heading {i}" for i in range(1, 6))
    assert not xref.drift_in("c.md", good)
