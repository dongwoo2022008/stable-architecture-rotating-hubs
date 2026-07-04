# Data lineage: source → derived → results

```
KOSIS Intra-national Migration Statistics (API)      MOIS/KOSIS/HIRA regional indicators
        │                                                     │
        ▼                                                     ▼
data/source/migration_long.parquet          data/source/analysis_dataset_FINAL_v4.csv
  (OD flows, sgg level, 2008–2025;            (covariate master, 229 municipalities ×
   age_group/sex disaggregation included)      2008–2025, 64 variables incl. network
        │                                      metrics computed from the OD networks)
        │                                                     │
        │            scripts/preprocessing/step1_preprocessing.py
        │                                                     │
        │                             ┌───────────────────────┴──────────────┐
        ▼                             ▼                                      ▼
 (read directly by          data/derived/track_A_2008_2025.csv   data/derived/track_B_2009_2024.csv
  Track A pipeline                    │                                      │
  for topology/Louvain)     scripts/network/trackA_…py           scripts/panel/trackB_…py
                                      ▼                                      ▼
                             results/tables/* (Track A)          results/tables/* (Track B)
```

Reproduction of the derived layer is **byte-identical**: running
`step1_preprocessing.py` on `analysis_dataset_FINAL_v4.csv` regenerates
`track_A_2008_2025.csv` (4,122 × 108) and `track_B_2009_2024.csv` exactly
(verified 2026-07-04).

## Sources of the covariate master
| Domain | Variables (examples) | Source |
|---|---|---|
| Migration & population | in/out/net rates, population, age structure | Statistics Korea (KOSIS), resident registration |
| Network metrics | pagerank, in/out-strength, betweenness, closeness | computed from the OD networks (this project) |
| Fiscal | fiscal_indep | MOIS local finance statistics |
| Health | doctor_per1000, hospital beds | HIRA / KOSIS |
| Living conditions | culture_facility_count, childcare_pk, housing | KOSIS regional indicators |
| Geography | area_km2, seoul_dist_km | MOIS / geometric computation |

## Interpolation disclosure (exact extent)
During Step-1 preprocessing, variables with ≤5% missingness were linearly
interpolated **within municipality** (limit_direction='both'). Within the
223-municipality estimation sample this affects:

- the eight core controls: **224 of 28,544 cells (0.78%)**; per-variable maximum
  aging_ratio 2.07%, minimum youth_ratio/fertility 0.45%;
- the network measure: **19 cells** of `pagerank` (2008–2012) for two
  municipalities affected by administrative reorganization (여주 41670,
  인천 미추홀 28177), which propagate to `pagerank_lag1` for 2009–2013;
- the dependent variable `net_rate`: **no imputed values** (0 cells).

No winsorization is applied anywhere. Jeonju's administrative area
(205.53 km²), missing from the source extract, is imputed in-script by the
Track B model code (see CANONICAL.md).

Variables with >30% missingness were left untouched (they are not used in
this paper). The raw 1.46 GB CSV export underlying the parquet is retained
locally; the parquet is the distribution root and carries identical content
for the fields used here.
