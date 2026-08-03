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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=shutil.which("tectonic"))
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
