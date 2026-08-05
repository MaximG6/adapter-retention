"""Resolve every cross-reference in the built paper against its actual structure.

The appendices were written against a section numbering that later changed, and about
forty references went stale: section 8 cites "Appendix E" for a table in Appendix C,
Appendix A cites subsections of sections that have none. Nothing noticed, because a
dangling reference in LaTeX is not an error -- it is a sentence that reads fine and
points nowhere.

This builds the document's real structure by numbering the LaTeX exactly as LaTeX does,
then resolves every reference found in the text against it.

The known trap, from this project's own record: a previous cross-reference checker
reported nine false positives because its pattern required whitespace after the section
number where the text uses a period. So the matcher here is tested in both directions --
it must flag a reference known to be broken and pass one known to be good -- and those
are regression tests, not a one-time check.

Usage:
    python analysis/xref.py            # report
    python analysis/xref.py --strict   # exit 1 on any unresolved reference
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXDIR = REPO_ROOT / "paper" / "tex"

#: Body sections are \section in main.tex; appendices restart lettering after \appendix.
_SEC = re.compile(r"^\\(section|subsection|subsubsection)\*?\{(.*)\}\s*$")
_APPENDIX = re.compile(r"^\\appendix\b")


def structure(main: str, appendices: str) -> tuple[set[str], set[str], dict[str, str]]:
    """(section labels, appendix labels, label -> title) as LaTeX will number them."""
    secs: set[str] = set()
    apps: set[str] = set()
    titles: dict[str, str] = {}
    in_appendix = False
    n = [0, 0, 0]           # section, subsection, subsubsection counters
    letter = 0

    for line in (main + "\n" + appendices).splitlines():
        if _APPENDIX.match(line):
            in_appendix = True
            n = [0, 0, 0]
            continue
        m = _SEC.match(line.strip())
        if not m:
            continue
        kind, title = m.group(1), m.group(2)
        if kind == "section":
            n[1] = n[2] = 0
            if in_appendix:
                letter += 1
                label = chr(ord("A") + letter - 1)
                apps.add(label)
            else:
                n[0] += 1
                label = str(n[0])
                secs.add(label)
            titles[label] = title
        elif kind == "subsection":
            n[1] += 1
            n[2] = 0
            head = chr(ord("A") + letter - 1) if in_appendix else str(n[0])
            label = f"{head}.{n[1]}"
            (apps if in_appendix else secs).add(label)
            titles[label] = title
        else:
            n[2] += 1
            head = chr(ord("A") + letter - 1) if in_appendix else str(n[0])
            label = f"{head}.{n[1]}.{n[2]}"
            (apps if in_appendix else secs).add(label)
            titles[label] = title
    return secs, apps, titles


#: A reference is the marker plus its target. The target ends at the first character that
#: cannot be part of one: crucially a period FOLLOWED BY whitespace ends a sentence, but a
#: period between digits does not. Requiring whitespace after the number -- the mistake
#: the earlier checker made -- misses every reference that ends a sentence.
REFS = [
    ("section", re.compile(r"(?:\\S\{\}|\\S|§)\s?(\d+(?:\.\d+)*)")),
    ("appendix", re.compile(r"Appendix~?\s*([A-Z])(?:\.(\d+(?:\.\d+)*))?\b")),
    # A BARE appendix reference -- "see D.1.2", "the breakdown in B.4". Same kind of
    # reference, same need to resolve, and it was outside this gate entirely because the
    # pattern above requires the literal word "Appendix". The reproduction appendix
    # shipped with four live references to D.1/D.1.1/D.1.2/D.6, every one of which
    # resolved to a different appendix, and nothing here objected.
    #
    # The lookbehind excludes a letter that is part of a word, a path, a version string
    # or a decimal, so `Qwen3-8B.1` and `v2.1` are not references. Requiring the letter
    # to be a capital in A-G excludes ordinary sentence-initial words.
    ("bare-appendix", re.compile(r"(?<![\w./~-])([A-G])(\.\d+(?:\.\d+)*)\b")),
    # "Fig 8" is the same reference as "Figure 8" and was outside this gate, which is how
    # two references to the pre-cut figure numbering survived in the practice appendix.
    ("figure", re.compile(r"\bFig(?:ure)?~?\.?\s*(\d+)\b")),
    ("table", re.compile(r"Table~?\s*(\d+)\b")),
    ("equation", re.compile(r"Equation~?\s*\((\d+)\)")),
    ("fw", re.compile(r"\bFW-(\d+)\b")),
]


def counts(main: str, appendices: str) -> dict[str, int]:
    both = main + "\n" + appendices
    return {
        "figure": len(re.findall(r"\\begin\{figure\*?\}", both)),
        "table": len(re.findall(r"\\begin\{table\*?\}", both)),
        "equation": len(re.findall(r"\\begin\{equation\}", both)),
        "fw": len(set(re.findall(r"\*\*FW-(\d+)", both) + re.findall(r"FW-(\d+)\*\*", both)
                      + re.findall(r"textbf\{FW-(\d+)", both))),
    }


def check(main: str, appendices: str) -> list[tuple[str, str, str]]:
    """Unresolved references as (kind, target, context)."""
    secs, apps, _ = structure(main, appendices)
    cnt = counts(main, appendices)
    text = main + "\n" + appendices
    bad: list[tuple[str, str, str]] = []
    for kind, pat in REFS:
        for m in pat.finditer(text):
            if kind == "section":
                target = m.group(1)
                ok = target in secs
            elif kind == "appendix":
                target = m.group(1) + (f".{m.group(2)}" if m.group(2) else "")
                ok = target in apps
            elif kind == "bare-appendix":
                target = m.group(1) + m.group(2)
                ok = target in apps
            elif kind in ("figure", "table", "equation"):
                target = m.group(1)
                ok = 1 <= int(target) <= max(cnt[kind], 0)
            else:
                target = m.group(1)
                ok = 1 <= int(target) <= max(cnt["fw"], 1)
            if not ok:
                lo = max(0, m.start() - 55)
                ctx = " ".join(text[lo:m.end() + 25].split())
                bad.append((kind, target, ctx))
    return bad


def main_() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    main = (TEXDIR / "main.tex").read_text(encoding="utf-8")
    apx = (TEXDIR / "appendices.tex").read_text(encoding="utf-8")
    secs, apps, _ = structure(main, apx)
    bad = check(main, apx)
    print(f"structure: {len(secs)} section labels, {len(apps)} appendix labels")
    print(f"           sections   {' '.join(sorted(secs, key=_key))}")
    print(f"           appendices {' '.join(sorted(apps, key=_key))}")
    if not bad:
        print("all cross-references resolve")
        return 0
    print(f"\n{len(bad)} unresolved references:")
    seen = set()
    for kind, target, ctx in bad:
        k = (kind, target)
        if k in seen:
            continue
        seen.add(k)
        n = sum(1 for a, b, _ in bad if (a, b) == k)
        print(f"  {kind:>9} {target:<8} x{n:<3} ...{ctx}")
    return 1 if args.strict else 0


def _key(s: str):
    return [int(p) if p.isdigit() else p for p in s.split(".")]


if __name__ == "__main__":
    raise SystemExit(main_())
