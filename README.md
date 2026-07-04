# Stable Architecture, Rotating Hubs: Network Position and Migration Attraction in South Korea's Interregional Migration System, 2008–2025

Replication and verification package. Every table and figure in the manuscript
is reproducible from this repository; `python verify.py` confirms it automatically.

## Quick start
```bash
pip install -r requirements.txt
python verify.py            # ~1-2 min: checksums + published values + live FE re-estimation
python scripts/make_figures.py          # regenerate figures from results/tables
python verify.py --full     # optional: end-to-end re-run of both pipelines (~1 h)
```

## What reproduces what

| Manuscript object | Script | Output |
|---|---|---|
| Table 4-1, Supp. Tables S1–S5, MK tests | `scripts/network/trackA_unified_pipeline.py` | `results/tables/unified_network_stats_*`, `trackA_concentration_hub_stats`, `moran_global_*`, `lisa_cluster_counts_*`, `lisa_assignments_*` |
| Table 4-2 (OLS + FE ladder), Table 4-3 LSDV column | `scripts/panel/trackB_canonical_models.py` §1 | `trackB_canonical_FE_lag1.csv` |
| Table 4-3 (system GMM, 4 variants) | same, §2 (console) | canonical summary: `trackB_canonical_GMM.csv` |
| Table 4-4 (SDM direct/indirect/total) | same, §3 | `trackB_canonical_SDM_effects.csv` |
| Table 4-5 (KNN-5, subsamples) | same, §4 | `trackB_canonical_robust_hetero.csv` |
| Figures 1-1, 4-1…4-6, S2, S4–S6 | `scripts/make_figures.py` | `figures/*.png` |
| Figures S1, S3 (Moran scatter, Louvain maps) | node-level recomputation via Track A pipeline | reference renders in `figures/` |
| Derived datasets themselves | `scripts/preprocessing/step1_preprocessing.py` | `data/derived/track_A_*.csv`, `track_B_*.csv` (byte-identical) |

## Key design choices
- **Lagged PageRank.** Contemporaneous PageRank is computed from the same flows
  that define net migration and is mechanically simultaneous with it; all models
  use the one-year lag.
- **Canonical sample (Track B).** Balanced panel of 223 municipalities × 16 years
  (2009–2024, N = 3,568). Jeonju (52110) is recovered by imputing its official
  administrative area (205.53 km²); see CANONICAL.md for identity values.
- **Spatial weights.** Queen contiguity (`data/spatial_weights/queen_w_229_fixed.*`,
  row-standardized, mean 5.19 neighbors, no islands); KNN-5 for robustness.
- **Inference.** 9,999 permutations for Moran/LISA (seed 42); Windmeijer-corrected
  two-step system GMM with collapsed instruments; 2,000 Monte Carlo draws for
  SDM effects (seed 42).
- **COVID window.** 2020–2022 throughout (dummy, interactions, figure shading).

## Data
All inputs are official aggregate statistics (Statistics Korea/KOSIS, Ministry of
the Interior and Safety, HIRA), harmonized to 229 municipalities on 2025
boundaries. `data/source/migration_long.parquet` (72 MB) ships via Git LFS.
See `data/source/DATA_LINEAGE.md` for the full source-to-derived lineage,
including the exact extent of interpolation applied during dataset construction.

## Environment
Python ≥ 3.10: pandas pyarrow networkx python-louvain geopandas libpysal esda
spreg linearmodels pydynpd pymannkendall scipy statsmodels matplotlib
