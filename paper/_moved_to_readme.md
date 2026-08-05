### D.1.1 If your GPU is not Blackwell — read this first

Every measurement script resolves its device through `ar.device.require_cuda`, which
**defaults to a capability floor of sm_120** and raises rather than falling back. That
default is a property of *our* machine, not of the science: an RTX 5090 under a pre-cu128
torch build imports cleanly and then produces garbage, so the floor guards against that
specific failure.

**On any other card the default floor will raise.** That is intended — but it is a
one-variable fix. Lower the floor explicitly:

```bash
export AR_MIN_CAPABILITY=8.0     # Ampere (A100), Ada (4090), Hopper (H100)
```

The run still raises if no visible device clears the floor you set, so this is an opt-in,
never a silent relaxation. Among qualifying devices the **largest-memory** one is chosen
(ties break on lower index), so 8B BF16 loads land on the biggest card without anything
naming a device index. The resolved device name and capability are written into every
`manifest.json`.

BF16 is required throughout, so sm_80 (Ampere) is the practical minimum.

### D.1.2 On Windows, enable long paths before cloning — the clone fails otherwise

Some record paths reach **157 characters**, over the 260-character Windows `MAX_PATH`
budget once a clone directory is prepended. Without long-path support, `git clone`
reports *"Filename too long"* and leaves an **incomplete checkout** — the clone appears
to succeed and the failure is in the second line of output:

```
fatal: cannot create directory at
  'results/raw/phase0/public_adapter/...': Filename too long
warning: Clone succeeded, but checkout failed.
```

Enable it once, globally:

```bash
git config --global core.longpaths true
```

**This is verified, not anticipated** — cloning this repository on Windows 11 without it
fails at exactly the path above. With it, the clone completes and everything in this
appendix runs.

One residual symptom if you clone with `-c core.longpaths=true` rather than setting it
globally: the setting applies to the clone but not to the resulting repository, so
`git status` afterwards reports spurious `M`/`D` entries for the deepest record files.
They are artifacts of `git` being unable to stat those paths, not real changes —
`git -c core.longpaths=true status` shows a clean tree. Setting the option globally
avoids this entirely.

Linux and macOS are unaffected.

### D.6.1 There are two figure directories, and both are current

`paper/figures/` and `paper/figures-paper/` hold the same twelve figures under the same
names. Neither is stale.

| directory | built by | consumed by | difference |
|---|---|---|---|
| `paper/figures/` | the three figure scripts, run directly | the technical-report PDF, and the README | each figure carries its own title and caption text |
| `paper/figures-paper/` | `analysis/build_arxiv_pdf.py` | the LaTeX build, `paper/tex/` | in-figure headers suppressed, because LaTeX sets the captions |

```bash
# paper/figures/  -- default mode
PYTHONPATH=src python analysis/fig01_erasure_vs_survival.py
PYTHONPATH=src python analysis/fig05_06_08.py
PYTHONPATH=src python analysis/fig_secondary.py

# paper/figures-paper/  -- regenerated as step 1 of the arXiv build
PYTHONPATH=src python analysis/build_arxiv_pdf.py --tectonic <path>
```

**Do not run the figure scripts directly with `AR_FIG_PAPER=1`.** The variable controls
the *style* only; the output directory is `FIGDIR`, which those scripts always set to
`paper/figures/`. `build_arxiv_pdf.py` is what redirects them, by rebinding `FIGDIR`
before calling each `main()`. Setting the variable by hand therefore writes header-less
figures over the default set, and the damage is silent — the cross-checks still pass,
because the data behind the figures is unchanged.

Each script cross-checks its own figures against an independent recomputation from the
raw records, in whichever mode it runs. A figure that disagrees with the data fails the
build rather than being written.

**Reproducibility of the figure files themselves.** The PNGs are byte-identical on
regeneration. The vector PDFs are not: matplotlib embeds a `/CreationDate`, so two builds
of the same figure differ in exactly those five bytes and are identical once it is
stripped. Compare figure PDFs with the timestamp removed, or compare the PNGs.

## D.9 Known rough edges

Recorded because a reproducer will hit them, and finding them undocumented wastes an
afternoon.

- `snapshot_download` has been observed to exit 0 while leaving 0-byte files. Our
  download path fetches file-by-file and asserts sizes against the Hub's metadata.
  If a model load fails oddly, check for 0-byte files in `~/.cache/huggingface/hub`.
- On Windows, `Set-Content -Encoding utf8` writes a BOM that breaks Python source files.
  Irrelevant to running the code; relevant if you edit it from PowerShell.
- `analysis/appendix_tables.py` prints Unicode (α, Δ) that the Windows console codepage
  cannot encode. Use `--write` rather than piping stdout.
- The `latentqa` adapter's repo name is long and unlabelled in raw records; it is mapped
  to a short name in `analysis/appendix_tables.py`.
