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
        holds.append(rf"\texttt{{{body}}}")
        return f"\x00{len(holds) - 1}\x00"

    s = re.sub(r"`([^`]+)`", hold, s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)          # links -> text
    s = esc(s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: holds[int(m.group(1))], s)


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return ""
    n = max(len(c) for c in cells)
    cells = [c + [""] * (n - len(c)) for c in cells]
    # First column left, remainder right unless clearly prose.
    prose = any(len(x) > 28 for row in cells for x in row[1:])
    spec = "p{0.30\\columnwidth}" + ("p{0.55\\columnwidth}" if prose and n == 2
                                     else "r" * (n - 1))
    if n > 2 and prose:
        spec = "l" + "p{{{:.2f}\\columnwidth}}".format(0.72 / (n - 1)) * (n - 1)
    out = [r"\begin{center}\footnotesize", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    out.append(" & ".join(inline(x) for x in cells[0]) + r" \\")
    out.append(r"\midrule")
    for row in cells[1:]:
        out.append(" & ".join(inline(x) for x in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


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
            out += [r"\begin{quote}\footnotesize\begin{verbatim}", *buf,
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
            out.append(r"\item " + inline(re.sub(r"^\s*[-*]\s+", "", ln)))
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
        out.append(inline(ln))
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
            parts.append(convert(md))
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
