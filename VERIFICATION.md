# Reproducibility & Verification Guide

This repository is a self-contained reproducibility archive for the paper
*Stable Architecture, Rotating Hubs (2008–2025)*. Following the steps below
reproduces every published number on a clean Windows machine.

## What `verify.py` checks

`verify.py` performs four independent layers of checking:

1. **Integrity** — sha256 of every input data file and result table against `CHECKSUMS.txt`.
2. **Published values** — the stored result tables match the numbers reported in Section 4 (relative tolerance `1e-3`).
3. **Live re-estimation** — the fixed-effects (FE) ladder is recomputed from the raw derived data and matched to the canonical coefficients (tolerance `1e-9`). *This proves the computation itself reproduces, not merely that the stored bytes are intact.*
4. **Full pipeline (optional, `--full`)** — Track A and Track B are re-run end to end.

If only the `sha256` checks fail while layers 2–3 pass, the analysis is correct and only byte-level integrity (typically line endings) was affected.

## Prerequisites

- Windows 10/11
- Anaconda or Miniconda
- Git for Windows

## Environment (pinned — required)

The loose bounds in `requirements.txt` are **not** sufficient for reproduction; the versions below are required (see Troubleshooting for why):

- Python 3.10
- `numpy==1.26.4`, `scipy==1.13.1` (mandatory — other combinations crash)
- `wcwidth>=0.8.2`, `prettytable==3.18.0`

A full pinned lock is provided in `requirements-lock.txt`.

## One-click verification (recommended)

1. Clone with line-ending conversion **disabled** (so checksums match):

   ```
   git clone -c core.autocrlf=false https://github.com/dongwoo2022008/stable-architecture-rotating-hubs.git
   cd stable-architecture-rotating-hubs
   ```

2. Open **Anaconda Prompt** (not a plain `cmd`), then run:

   ```
   verify_windows.bat
   ```

   Choose **[1] Quick** (~1–2 min) or **[2] Full** (~30–60 min).

3. Success is printed as:

   ```
   ALL CHECKS PASSED - results match the published canonical values.
   ```

## Manual alternative (no batch)

```
conda create -n sarh_verify python=3.10 -y
conda activate sarh_verify
pip install -r requirements-lock.txt
python verify.py            # quick (~1-2 min)
python verify.py --full     # full end-to-end re-run (~30-60 min)
```

## Troubleshooting (known pitfalls)

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: Numba needs NumPy 2.0 or less` | numpy 2.x installed | pin `numpy==1.26.4` |
| Process/kernel dies silently; `Windows fatal exception: 0xc06d007f` in `scipy.linalg.lstsq` | scipy BLAS/LAPACK linkage broken by mixed pip/conda installs | pin `scipy==1.13.1`, force-reinstall |
| `AttributeError: module 'wcwidth' has no attribute 'width'` | `wcwidth` too old (0.2.x) for `prettytable` 3.18 | `wcwidth>=0.8.2` |
| `N sha256 CHECKS FAILED` (all `.csv`; the `.parquet` passes) | git converted `LF`→`CRLF` on checkout | clone with `-c core.autocrlf=false`, or `git config core.autocrlf false && git checkout --force` |
| `python` prints nothing / `conda` not recognized | Windows Store `python` stub, or conda not on PATH | use **Anaconda Prompt**; the batch calls the env interpreter by full path |

> The verification must be run on the **pristine clone** (before re-running the pipelines). Re-running Track A/B overwrites the result tables and will change their sha256 by design; use `verify.py --full`, which checks the pristine files first and then re-runs.
