# OD (Origin-Destination) Migration Matrix

## Files

| File | Size | Description |
|------|------|-------------|
| `migration_long.parquet` | 72 MB | Long-format OD migration panel (all years, all age groups, both sexes) |
| `od_flow_2025_all.json` | 2.4 MB | Raw OD flow JSON for 2025 (all ages, all sexes) — Statistics Korea API format |

## Schema: `migration_long.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `origin_code` | str | 5-digit SGG code of origin |
| `dest_code` | str | 5-digit SGG code of destination |
| `year` | int | Year (2008–2025) |
| `age_group` | str | Age group label (e.g., `0-4`, `20-24`, `total`) |
| `sex` | str | `M` / `F` / `total` |
| `flow` | int | Number of migrants (origin → destination) |

## Notes

- `migration_long.csv` (1.46 GB uncompressed) is stored locally only; use the `.parquet` version for analysis
- For RQ3 (age-specific networks), filter by `age_group` and `sex`
- Source: Statistics Korea (KOSIS) — Intra-national migration statistics by SGG
- **Required for RQ3 & RQ4**: age-group OD matrices are embedded in this file
