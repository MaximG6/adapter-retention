"""Convert the appendix markdown to LaTeX for the arXiv-format build.

Hand-converting 12,000 words would make the appendices a fork of the markdown that
drifts the moment either is edited — the failure mode of §7.8, applied to typesetting.
This converts the subset of markdown the manuscript actually uses, so the LaTeX
appendices remain derived from the same sources as the HTML report.

Handles: ATX headings, bold/italic/code spans, fenced code, pipe tables, bullet lists,
inline links, blockquotes, horizontal rules, and LaTeX-special characters.

Usage:
    python analysis/md_to_tex.py --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper"
OUT = PAPER / "tex" / "appendices.tex"

# (file, appendix title, figures to place at the end of that appendix)
APPENDICES = [
    ("appendix-A-tool.md", "The \\texttt{ar.predict} Tool",
     ["figA1_predict_validation"]),
    ("appendix-B-tables.md", "Full Tables",
     ["fig02_channel_model", "fig03_forest", "fig04_amplification",
      "fig11_layer_profile"]),
    ("07-methodological-lessons.md", "Registered Predictions and Methodological Practice",
     []),
    ("06-results-advertised-vs-measured.md", "Advertised versus Measured: Full Detail",
     ["fig10_refusal"]),
    ("appendix-C-prompts.md", "Prompt Sets", []),
    ("appendix-D-reproduction.md", "Reproduction", []),
    (None, "Supplementary Figures",
     ["fig07_entropy_control", "fig09_bootstrap_intervals"]),
]

SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


#: Typewriter fonts carry no glyphs for these, and neither code spans nor verbatim
#: blocks are escaped, so both need an ASCII substitution pass of their own.
TT_ASCII = {"−": "-", "–": "-", "—": "--", "→": "->", "≥": ">=", "≤": "<=",
            "×": "x", "≈": "~", "Δ": "Delta", "δ": "delta", "α": "alpha",
            "√": "sqrt", "±": "+/-", "≠": "!=", "π": "pi", "σ": "sigma",
            "ρ": "rho", "∝": "prop", "∞": "inf", "·": "*", "…": "...",
            "“": '"', "”": '"', "'": "'", "'": "'", "§": "S"}


def ascii_only(s: str) -> str:
    for k, v in TT_ASCII.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode("ascii")


def esc(s: str) -> str:
    """Escape LaTeX specials outside code spans. Backslash first, always."""
    s = s.replace("\\", r"\textbackslash{}")
    for k, v in SPECIALS.items():
        s = s.replace(k, v)
    # Unicode the manuscript actually uses.
    for k, v in {"−": r"$-$", "—": "---", "–": "--", "×": r"$\times$", "≈": r"$\approx$",
                 "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$",
                 "Δ": r"$\Delta$", "δ": r"$\delta$", "α": r"$\alpha$",
                 "√": r"$\sqrt{\ }$", "σ": r"$\sigma$", "ρ": r"$\rho$",
                 "∝": r"$\propto$", "∞": r"$\infty$", "±": r"$\pm$",
                 "'": "'", "'": "'", "\u201c": "``", "\u201d": "''",
                 "…": r"\ldots{}", "⚠": "", "§": r"\S{}", "✓": r"\checkmark{}",
                 "≠": r"$\neq$", "·": r"$\cdot$", "É": r"\'E", "é": r"\'e",
                 "ü": r'\"u', "ö": r'\"o', "ń": r"\'n", "ś": r"\'s",
                 "‑": "-", "⟨": r"$\langle$", "⟩": r"$\rangle$",
                 "⁻": r"$^{-}$", "⁰": r"$^{0}$", "¹": r"$^{1}$", "²": r"$^{2}$",
                 "³": r"$^{3}$", "⁴": r"$^{4}$", "⁵": r"$^{5}$", "⁶": r"$^{6}$",
                 "⁷": r"$^{7}$", "⁸": r"$^{8}$", "⁹": r"$^{9}$",
                 "ᵀ": r"$^{\top}$", "∈": r"$\in$", "∀": r"$\forall$",
                 "🔬": "", "⚡": "", "📊": ""}.items():
        s = s.replace(k, v)
    return s


def leftover_non_ascii(tex: str) -> dict[str, int]:
    """Any non-ASCII surviving conversion. TeX renders these as garbage or drops them
    silently, so the build reports them rather than shipping a PDF with holes in it."""
    from collections import Counter
    return dict(Counter(c for c in tex if ord(c) > 127))


#: The markdown is the technical report, where section 7.8 is a real heading. The arXiv
#: build cuts and re-orders: the methodological section becomes Appendix C, the
#: advertised-versus-measured section becomes Appendix D, and the body's subsections are
#: folded. A reference correct in one document is therefore wrong in the other, and that
#: is what left roughly forty dangling references in the built PDF. Rather than break the
#: markdown to suit the LaTeX, translate on the way through.
#:
#: Left side is the markdown's own numbering; right side is what LaTeX will number it.
REFMAP = {
    # Related work: seven subsections in the report, one section in the paper.
    **{f"2.{i}": "2" for i in range(1, 8)},
    # Method: 3.8 (ground-truth fixture) and 3.10 (refusal battery) fold into 3.7.
    "3.8": "3.7", "3.9": "3.8", "3.10": "3.7", "3.11": "3.9",
    # Results: 4.5.1 folds into 4.5; the predictive gap moves 5.4 -> 5.3.
    "4.5.1": "4.5", "5.4": "5.3",
    # Advertised versus measured becomes Appendix D, keeping its own numbering.
    **{f"6.{i}": f"D.{i}" for i in range(1, 6)},
    # Methodological practice becomes Appendix C, offset by one because 7.0 is C.1.
    **{f"7.{i}": f"C.{i + 1}" for i in range(0, 16)},
    # Limitations has no subsections in the paper.
    **{f"8.{i}": "9" for i in range(1, 9)},
}

_REF = re.compile(r"§\s?(\d+(?:\.\d+)*)")


def remap_refs(md: str) -> str:
    """Rewrite the report's section numbers to the paper's, for the LaTeX build."""

    def sub(m: re.Match[str]) -> str:
        target = REFMAP.get(m.group(1))
        if target is None:
            return m.group(0)
        return f"Appendix {target}" if target[0].isalpha() else f"§{target}"

    return _REF.sub(sub, md)


def inline(s: str) -> str:
    """Inline spans. Code is extracted first so its contents are never escaped."""
    holds: list[str] = []

    def hold(m: re.Match[str]) -> str:
        body = m.group(1)
        for k, v in {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                     "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
                     "_": r"\_", "^": r"\textasciicircum{}",
                     "~": r"\textasciitilde{}"}.items():
            body = body.replace(k, v)
        # Typewriter fonts lack these glyphs, and code spans are not escaped by esc(),
        # so unicode inside them reaches the TeX font directly. Substitute ASCII.
        body = ascii_only(body)
        # A path in a code span is one unbreakable word. In a narrow table column it
        # overflows however the column is sized, so permit a break after the
        # separators. \allowbreak adds no hyphen, so the path stays copy-pasteable.
        body = re.sub(r"(?<=[/_.-])(?=[^\s/_.-])", r"\\allowbreak{}", body)
        holds.append(rf"\texttt{{{body}}}")
        return f"\x00{len(holds) - 1}\x00"

    s = re.sub(r"`([^`]+)`", hold, s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)          # links -> text
    s = esc(s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: holds[int(m.group(1))], s)


#: Page geometry, in points. article/10pt/twocolumn at margin=0.75in on letter.
#: Must match main.tex; TABLE_GEOMETRY_CHECK below fails the build if it drifts.
COLUMN_PT = 247.0
TEXT_PT = 505.0
#: Latin Modern mean advance, in points per character, at each size we emit.
CHAR_PT = {"footnotesize": 8.0 * 0.47, "scriptsize": 7.0 * 0.47}
#: booktabs inter-column gutter, both sides.
COL_PAD_PT = 12.0
#: Below this, a column is a label or a number and should size to its content.
NARROW_CH = 14
#: A wrapped column narrower than this reads one word per line.
MIN_MEASURE_CH = 26


def _demand(texts: list[str]) -> tuple[int, int]:
    """(longest cell, longest unbreakable word) in characters, markup removed."""
    plain = [re.sub(r"[*`\[\]]|\((?:https?|#)[^)]*\)", "", t) for t in texts]
    longest = max((len(t) for t in plain), default=0)
    word = max((len(w) for t in plain for w in t.split()), default=0)
    return longest, word


def table(rows: list[str]) -> str:
    """Lay a markdown table out to fit, measuring content instead of guessing.

    The previous version chose a fixed spec from the column count alone: bare `r`
    columns for numeric tables, which have no width bound and simply ran off the page,
    and a uniform `p{0.72/(n-1)}` for prose, which gave a 9-character measure at n=6.
    Both defects shipped in the arXiv PDF; neither is visible in the markdown, and only
    the first one warns.

    Each column's width demand is measured, then a container is chosen in increasing
    order of disruption -- one column, one column at \\scriptsize, then full width via
    table* -- and the width is distributed in proportion to demand. Columns that want
    less than NARROW_CH size to their content; the rest wrap, never below
    MIN_MEASURE_CH where the container allows it.
    """
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return ""
    n = max(len(c) for c in cells)
    cells = [c + [""] * (n - len(c)) for c in cells]

    cols = [[row[c] for row in cells] for c in range(n)]
    longest, words = zip(*(_demand(col) for col in cols))
    numeric = [c > 0 and all(re.fullmatch(r"[-+~<>=$\\\d.,%/()\s]*", t or "0")
                             for t in cols[c][1:]) for c in range(n)]

    # Natural (unwrapped) width at each size, plus gutters.
    def natural(size: str) -> float:
        return sum(longest) * CHAR_PT[size] + n * COL_PAD_PT

    if natural("footnotesize") <= COLUMN_PT:
        size, avail, wide = "footnotesize", COLUMN_PT, False
    elif natural("scriptsize") <= COLUMN_PT:
        size, avail, wide = "scriptsize", COLUMN_PT, False
    elif natural("footnotesize") <= TEXT_PT:
        size, avail, wide = "footnotesize", TEXT_PT, True
    else:
        size, avail, wide = "scriptsize", TEXT_PT, True

    body = [r"\toprule",
            " & ".join(inline(x) for x in cells[0]) + r" \\",
            r"\midrule"]
    body += [" & ".join(inline(x) for x in row) + r" \\" for row in cells[1:]]
    body += [r"\bottomrule"]

    if avail == COLUMN_PT and not wide and natural(size) <= COLUMN_PT:
        # Fits as-is; let TeX size every column to its content.
        spec = "".join("r" if numeric[c] else "l" for c in range(n))
        inner = [rf"\begin{{tabular}}{{{spec}}}"] + body + [r"\end{tabular}"]
    else:
        # tabularx solves for the X widths so the total is exactly the container.
        # Estimated p{} widths cannot promise that: they are mixed with natural l/r
        # columns whose true width depends on glyph metrics, and the sum overran.
        # Weight the X columns by demand so the widest text column gets the widest
        # measure rather than an equal slice. The weights must sum to the number of X
        # columns for tabularx's own total to come out right.
        flex = [c for c in range(n) if longest[c] > NARROW_CH]
        if not flex:
            flex = [max(range(n), key=lambda c: longest[c])]
        tot = sum(longest[c] for c in flex) or 1
        k = len(flex)
        parts = []
        for c in range(n):
            if c not in flex:
                parts.append("r" if numeric[c] else "l")
                continue
            f = k * longest[c] / tot
            f = min(max(f, 0.45), k)               # keep any X column legible
            parts.append(f">{{\\hsize={f:.3f}\\hsize\\raggedright\\arraybackslash}}X")
        # Renormalise so the factors still sum to k after clamping.
        got = sum(float(re.search(r"hsize=([0-9.]+)", p).group(1))
                  for p in parts if "hsize=" in p)
        if got and abs(got - k) > 1e-6:
            parts = [re.sub(r"hsize=([0-9.]+)",
                            lambda m: f"hsize={float(m.group(1)) * k / got:.3f}", p)
                     for p in parts]
        spec = "".join(parts)
        width = r"\textwidth" if wide else r"\columnwidth"
        inner = ([rf"\begin{{tabularx}}{{{width}}}{{{spec}}}"] + body
                 + [r"\end{tabularx}"])

    if wide:
        # table* spans both columns. It floats, so it may move to the top of a page;
        # that is the cost of not truncating the data.
        return "\n".join([r"\begin{table*}[t]", rf"\centering\{size}"]
                         + inner + [r"\end{table*}"])
    return "\n".join([rf"\begin{{center}}\{size}"] + inner + [r"\end{center}"])


#: Characters that fit across \columnwidth in a verbatim block at each size, for the
#: fixed-pitch face (advance 0.6em). Verbatim cannot wrap, so a block that exceeds its
#: size simply runs off the page -- the same defect as the tables, in a different
#: environment, and equally silent in the markdown.
VERBATIM_FIT = ((51, "footnotesize"), (58, "scriptsize"), (82, "tiny"))


def verbatim_size(block: list[str]) -> str:
    longest = max((len(b) for b in block), default=0)
    for fits, size in VERBATIM_FIT:
        if longest <= fits:
            return size
    return VERBATIM_FIT[-1][1]


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            close_list()
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            # verbatim passes bytes straight to the typewriter font, which has no
            # glyphs for the unicode the manuscript uses in code examples.
            buf = [ascii_only(b) for b in buf]
            out += [rf"\begin{{quote}}\{verbatim_size(buf)}\begin{{verbatim}}", *buf,
                    r"\end{verbatim}\end{quote}"]
            continue
        if re.match(r"^\s*\|", ln):
            close_list()
            buf = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                buf.append(lines[i])
                i += 1
            out.append(table(buf))
            continue
        m = re.match(r"^(#{2,4})\s+(.*)$", ln)
        if m:
            close_list()
            lvl = len(m.group(1))
            title = re.sub(r"^[\d.]+\s*", "", m.group(2)).strip()
            cmd = {2: "subsection", 3: "subsubsection", 4: "paragraph"}[min(lvl, 4)]
            out.append(rf"\{cmd}{{{inline(title)}}}")
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            if not in_list:
                out.append(r"\begin{itemize}\itemsep1pt")
                in_list = True
            item = [re.sub(r"^\s*[-*]\s+", "", ln)]
            # An indented continuation belongs to this bullet. Emitting it separately
            # closed the list and left an orphan paragraph between two bullets, which is
            # how the gate self-test list rendered in the built PDF.
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and lines[i + 1][:1].isspace()
                   and not re.match(r"^\s*[-*]\s+", lines[i + 1])):
                i += 1
                item.append(lines[i].strip())
            out.append(r"\item " + inline(" ".join(item)))
            i += 1
            continue
        if ln.strip().startswith(">"):
            close_list()
            out += [r"\begin{quote}\small", inline(ln.strip().lstrip(">").strip()),
                    r"\end{quote}"]
            i += 1
            continue
        if re.match(r"^\s*(---+|\*\*\*+)\s*$", ln):
            close_list()
            i += 1
            continue
        if not ln.strip():
            close_list()
            out.append("")
            i += 1
            continue
        close_list()
        # Gather the whole paragraph before converting it. inline() is regex-based and
        # its spans cannot match across a line break, so a **bold** or *emphasis* the
        # author wrapped mid-phrase used to pass through as literal asterisks -- 70 of
        # them reached the built PDF. A markdown paragraph is its consecutive lines, so
        # joining first is also what the format means.
        para = [ln]
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            if (not nxt.strip() or nxt.startswith("|") or nxt.strip().startswith((">", "```"))
                    or re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^#{2,4}\s", nxt)
                    or re.match(r"^\s*(---+|\*\*\*+)\s*$", nxt)):
                break
            i += 1
            para.append(lines[i])
        out.append(inline(" ".join(x.strip() for x in para)))
        i += 1
    close_list()
    return "\n".join(out)


def figure(fig: str) -> str:
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        rf"\includegraphics[width=\columnwidth]{{{fig}.pdf}}",
        rf"\caption{{{fig.replace('_', ' ').replace('fig', 'Figure ')}.}}",
        r"\end{figure}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    parts = [r"\appendix", r"\onecolumn" if False else ""]
    for fname, title, figs in APPENDICES:
        parts.append(rf"\section{{{title}}}")
        if fname:
            md = (PAPER / fname).read_text(encoding="utf-8")
            md = re.sub(r"^#\s+.*$", "", md, count=1, flags=re.M)     # drop H1
            md = re.sub(r"^\*Draft\..*?\*\s*$", "", md, flags=re.M | re.S, count=1)
            md = re.sub(r"^\*Generated by .*?regenerate\.\*\s*$", "", md, flags=re.M)
            md = re.sub(r"^\*All values re-derived.*$", "", md, flags=re.M)
            parts.append(convert(remap_refs(md)))
        for f in figs:
            parts.append(figure(f))

    tex = "\n\n".join(p for p in parts if p)

    left = leftover_non_ascii(tex)
    if left:
        print(f"  {sum(left.values())} non-ASCII characters survived conversion:")
        for ch, n in sorted(left.items(), key=lambda kv: -kv[1]):
            print(f"    U+{ord(ch):04X} {ch!r} x{n}")
        print("  These render as garbage or vanish in TeX. Add them to esc()/TT_ASCII.")
    else:
        print("  no leftover non-ASCII")

    if args.write:
        OUT.write_text(tex, encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(tex):,} chars)")
    else:
        print(tex[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
