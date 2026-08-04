"""Build the arXiv-format PDF: figures in paper mode, appendices from markdown, Tectonic.

Kept separate from build_pdf.py (the HTML-derived circulation artifact), which is
retained deliberately: the two serve different purposes and the markdown remains the
single source both derive from.

Requires `tectonic` on PATH or at --tectonic. Tectonic is a single ~20 MB binary that
fetches TeX packages on demand, chosen over a full TeXLive install.

Usage:
    python analysis/build_arxiv_pdf.py --tectonic <path-to-tectonic>
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXDIR = REPO_ROOT / "paper" / "tex"
FIGPAPER = REPO_ROOT / "paper" / "figures-paper"
OUT = REPO_ROOT / "paper" / "adapter-retention-arxiv.pdf"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


#: An Overfull box wider than this is content running off the page rather than a
#: typesetting nicety. Tables that overflowed the arXiv PDF for a whole draft cycle ran
#: 19-299pt over; ordinary prose boxes that no reader would notice run under 5pt.
OVERFULL_LIMIT_PT = 6.0

_OVERFULL = re.compile(r"([\w.]+):(\d+): Overfull .hbox \(([0-9.]+)pt too wide")


def check_overfull(log: str, limit_pt: float) -> bool:
    """Fail the build on any Overfull box beyond `limit_pt`.

    Same principle as the figure cross-checks and the non-ASCII gate: the failure is
    invisible in the source and silent in the output, so the build has to be the thing
    that notices. Tectonic reports an alignment's overfull box at the line that CLOSES
    it, not at the offending row, so line numbers point at \\end{center}.
    """
    worst: dict[tuple[str, int], float] = {}
    for m in _OVERFULL.finditer(log):
        key = (m.group(1), int(m.group(2)))
        worst[key] = max(worst.get(key, 0.0), float(m.group(3)))
    over = sorted(((v, f, l) for (f, l), v in worst.items() if v > limit_pt),
                  reverse=True)
    if not over:
        n = len(worst)
        print(f"     no overfull box beyond {limit_pt:.0f}pt "
              f"({n} under it)")
        return True
    print(f"     {len(over)} overfull boxes beyond {limit_pt:.0f}pt "
          f"-- content is running off the page:", file=sys.stderr)
    for amt, f, l in over[:12]:
        print(f"       {amt:>7.1f}pt  {f}:{l}", file=sys.stderr)
    if len(over) > 12:
        print(f"       ... and {len(over) - 12} more", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=shutil.which("tectonic"))
    ap.add_argument("--overfull-pt", type=float, default=OVERFULL_LIMIT_PT,
                    help="fail the build on an overfull box wider than this")
    args = ap.parse_args()
    if not args.tectonic:
        print("tectonic not found; pass --tectonic <path>", file=sys.stderr)
        return 1

    env = dict(os.environ, AR_FIG_PAPER="1", PYTHONPATH=str(REPO_ROOT / "src"))
    FIGPAPER.mkdir(parents=True, exist_ok=True)

    print("1/3  regenerating figures in paper mode (headers suppressed)")
    stub = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from pathlib import Path\n"
        "import fig01_erasure_vs_survival as a, fig05_06_08 as b, fig_secondary as c\n"
        "for m in (a, b, c): m.FIGDIR = Path(r'%s')\n"
        "sys.argv = ['x']\n"
        "a.main(); b.main(); c.main()\n" % (REPO_ROOT / "analysis", FIGPAPER)
    )
    r = run([sys.executable, "-c", stub], env=env)
    if r.returncode:
        print(r.stdout[-3000:] + r.stderr[-3000:], file=sys.stderr)
        return 1
    if "MISMATCH" in r.stdout:
        print("figure cross-check failed:\n" + r.stdout, file=sys.stderr)
        return 1
    print(f"     {len(list(FIGPAPER.glob('*.pdf')))} vector figures, cross-checks passed")

    print("2/3  converting appendices from markdown")
    r = run([sys.executable, str(REPO_ROOT / "analysis" / "md_to_tex.py"), "--write"],
            env=env)
    print("     " + r.stdout.strip().replace("\n", "\n     "))
    if "survived conversion" in r.stdout:
        print("non-ASCII survived conversion; fix before shipping", file=sys.stderr)
        return 1

    print("3/3  running tectonic")
    r = run([args.tectonic, "-X", "compile", "main.tex"], cwd=TEXDIR)
    bad = [l for l in (r.stdout + r.stderr).splitlines()
           if "Missing character" in l or "could not represent" in l]
    if bad:
        print(f"     {len(bad)} unicode errors:", file=sys.stderr)
        for l in bad[:5]:
            print("       " + l, file=sys.stderr)
        return 1

    if not check_overfull(r.stdout + r.stderr, args.overfull_pt):
        return 1

    # Standing gate, not a one-time fix: sections move between drafts and a dangling
    # reference is not a LaTeX error, just a sentence pointing nowhere.
    sys.path.insert(0, str(REPO_ROOT / "analysis"))
    import xref

    unresolved = xref.check((TEXDIR / "main.tex").read_text(encoding="utf-8"),
                            (TEXDIR / "appendices.tex").read_text(encoding="utf-8"))
    if unresolved:
        uniq = sorted({f"{k} {t}" for k, t, _ in unresolved})
        print(f"     {len(unresolved)} unresolved cross-references: {', '.join(uniq[:12])}",
              file=sys.stderr)
        return 1
    print("     all cross-references resolve")

    # Table-to-table agreement. The prose-level check passed 18/18 while two tables
    # printed the same interval with different values, because both were tables.
    import tablecheck

    clash = tablecheck.disagreements()
    if clash:
        print(f"     {len(clash)} cells disagree between artifacts:", file=sys.stderr)
        for (a, b), entries in clash[:8]:
            print(f"       {a} x {b}: "
                  + "; ".join(f"{v} ({o})" for v, o in entries), file=sys.stderr)
        return 1
    print("     all cross-table cells agree")
    built = TEXDIR / "main.pdf"
    if not built.exists():
        print((r.stdout + r.stderr)[-3000:], file=sys.stderr)
        return 1
    shutil.copy(built, OUT)
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(str(OUT)).pages)
    except Exception:
        pages = -1
    print(f"     wrote {OUT.relative_to(REPO_ROOT)} "
          f"({OUT.stat().st_size / 1024 / 1024:.2f} MB, {pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
