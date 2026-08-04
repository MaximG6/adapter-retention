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
