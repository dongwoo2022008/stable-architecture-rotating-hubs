# -*- coding: utf-8 -*-
"""Track B canonical models (final, 2026-07-04).

Re-estimation underlying Section 4.2, with the key regressor entered as
pagerank_lag1 (contemporaneous PageRank is mechanically simultaneous with
net migration). Interactions are recomputed in-script: the stored
pagerank_lag1_x_covid column is contaminated in non-COVID years - do not use.

Canonical sample: balanced panel, 223 municipalities x 16 years (N = 3,568),
with Jeonju (52110) recovered by imputing its official administrative area.

  1. FE ladder (two-way FE, entity-clustered SE) : trackB_canonical_FE_lag1.csv
  2. Two-step system GMM (Windmeijer, collapsed) : console output
  3. Panel SDM (ML, Elhorst) + LeSage-Pace effects: trackB_canonical_SDM_effects.csv
  4. Subsample heterogeneity                      : trackB_hetero_rerun.csv

Run from repository root:  python scripts/panel/trackB_canonical_models.py
(runtime: FE ~1 min; GMM ~2 min; SDM with 2,000 MC draws ~30-60 min)
"""
import ast, warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from libpysal.weights import W as LW
from scipy.stats import norm
import spreg

warnings.filterwarnings("ignore")
OUT = "results/tables/"
Y = "net_rate"
CORE = ["ln_pop_density","youth_ratio","aging_ratio","fiscal_indep","doctor_per1000",
        "fertility","culture_facility_count","childcare_pk"]

def load():
    df = pd.read_csv("data/derived/track_B_2009_2024.csv")
    # Jeonju (52110): area_km2 missing in the source extract -> impute official
    # administrative area (205.53 km^2) and rebuild density terms (canonical v2)
    jj = df.region_code == 52110
    df.loc[jj, "area_km2"] = 205.53
    df.loc[jj, "pop_density"] = df.loc[jj, "population"] / 205.53
    df.loc[jj, "ln_pop_density"] = np.log(df.loc[jj, "pop_density"])
    df["prl_x_covid"] = df.pagerank_lag1 * df.covid_dummy   # recomputed (canonical)
    df["prl_x_metro"] = df.pagerank_lag1 * df.metro_dummy
    need = [Y,"pagerank_lag1"]+CORE+["seoul_dist_km","covid_dummy","region_code",
            "year","metro_dummy","net_rate_lag1","prl_x_covid","prl_x_metro","city_size"]
    dm = df[need].dropna(subset=[c for c in need if c != "city_size"])
    cnt = dm.groupby("region_code").size()
    return dm[dm.region_code.isin(cnt[cnt == 16].index)]   # balanced 223 x 16 (N=3,568)

def fe(dfe, xs):
    return PanelOLS(dfe[Y], dfe[xs], entity_effects=True, time_effects=True,
        drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

def main():
    dm = load()
    assert dm.region_code.nunique() == 223 and len(dm) == 3568, \
        f"sample mismatch: {dm.region_code.nunique()} x -> {len(dm)} (expected 223 / 3,568)"
    dfe = dm.set_index(["region_code","year"])
    # ---------- 1. FE ladder (+ pooled OLS benchmark) ----------
    import statsmodels.api as sm
    Xo = dm[["pagerank_lag1"]+CORE+["seoul_dist_km","covid_dummy"]]
    ro = sm.OLS(dm[Y], sm.add_constant(Xo)).fit(cov_type="HC3")
    rows = [dict(model="OLS", variable=v, coef=ro.params[v], se=ro.bse[v],
                 p=ro.pvalues[v]) for v in Xo.columns]
    rows += [dict(model="OLS", variable="_N", coef=int(ro.nobs), se=np.nan, p=np.nan),
             dict(model="OLS", variable="_R2", coef=ro.rsquared, se=np.nan, p=np.nan)]
    specs = {"FE-M1": ["pagerank_lag1"], "FE-M2": ["pagerank_lag1"]+CORE,
        "FE-M3": ["pagerank_lag1"]+CORE+["prl_x_covid"],
        "FE-M4": ["pagerank_lag1"]+CORE+["prl_x_covid","prl_x_metro"],
        "LSDV":  ["net_rate_lag1","pagerank_lag1"]+CORE}
    for nm, xs in specs.items():
        r = fe(dfe, xs)
        for v in r.params.index:
            rows.append(dict(model=nm, variable=v, coef=r.params[v],
                             se=r.std_errors[v], p=r.pvalues[v]))
        rows.append(dict(model=nm, variable="_N", coef=int(r.nobs), se=np.nan, p=np.nan))
        rows.append(dict(model=nm, variable="_R2", coef=float(r.rsquared), se=np.nan, p=np.nan))
        print(nm, "prl=%.1f p=%.4f" % (r.params["pagerank_lag1"],
                                        r.pvalues["pagerank_lag1"]))
    pd.DataFrame(rows).to_csv(OUT+"trackB_canonical_FE_lag1.csv", index=False)

    # ---------- 2. System GMM (pydynpd) ----------
    from pydynpd import regression as dyn
    d2 = dm.copy(); d2["id"] = d2.region_code
    base = "net_rate L1.net_rate pagerank_lag1 " + " ".join(CORE)
    iv = " iv(" + " ".join(CORE) + ")"
    variants = {
        "SYS-GMM (base)":      base+" | gmm(net_rate, 2:5) gmm(pagerank_lag1, 1:4)"+iv+" | timedumm collapse",
        "DIFF-GMM":            base+" | gmm(net_rate, 2:5) gmm(pagerank_lag1, 1:4)"+iv+" | timedumm collapse nolevel",
        "SYS-GMM (short IV)":  base+" | gmm(net_rate, 2:3) gmm(pagerank_lag1, 1:2)"+iv+" | timedumm collapse",
        "SYS-GMM (prl endog)": base+" | gmm(net_rate, 2:6) gmm(pagerank_lag1, 2:5)"+iv+" | timedumm collapse"}
    for nm, cmd in variants.items():
        print("---", nm); dyn.abond(cmd, d2, ["id","year"])
    # coefficient / Hansen / AR(2) values are printed to console;
    # the canonical summary is results/tables/trackB_canonical_GMM.csv

    # ---------- 3. Panel SDM + LeSage-Pace effects ----------
    regions = sorted(dm.region_code.unique())
    nb = pd.read_csv("data/spatial_weights/queen_w_229_neighbors.csv")
    nb["neighbor_codes"] = nb["neighbor_codes"].apply(ast.literal_eval)
    neigh = {r.panel_code: [c for c in r.neighbor_codes if c in set(regions)]
             for r in nb.itertuples() if r.panel_code in set(regions)}
    w = LW(neigh, silence_warnings=True); w.transform = "r"
    order = w.id_order
    dm2 = dm.copy()
    dm2["rk"] = dm2.region_code.map({c: i for i, c in enumerate(order)})
    dm2 = dm2.sort_values(["year","rk"])
    Wd = np.array(w.full()[0])
    Xv = ["pagerank_lag1"] + CORE
    X = dm2[Xv].values
    WX = np.vstack([Wd @ dm2[dm2.year == t][Xv].values
                    for t in sorted(dm2.year.unique())])
    years = sorted(dm2.year.unique())
    D = np.column_stack([(dm2.year == t).astype(float).values for t in years[1:]])
    y = dm2[Y].values.reshape(-1, 1)
    np.random.seed(42)
    sdm = spreg.Panel_FE_Lag(y, np.hstack([X, WX, D]), w, vm=True)
    rho = float(sdm.rho); k = len(Xv); n = Wd.shape[0]
    S = np.linalg.inv(np.eye(n) - rho * Wd)
    draws = np.random.multivariate_normal(sdm.betas.flatten(), sdm.vm, 2000)
    erows = []
    for i, v in enumerate(Xv):
        M = S @ (np.eye(n)*sdm.betas[i][0] + Wd*sdm.betas[k+i][0])
        d0 = np.trace(M)/n; t0 = M.sum()/n
        dd, ii, tt = [], [], []
        for dr in draws:
            Sr = np.linalg.inv(np.eye(n) - dr[-1]*Wd)
            Mr = Sr @ (np.eye(n)*dr[i] + Wd*dr[k+i])
            d = np.trace(Mr)/n; t = Mr.sum()/n
            dd.append(d); ii.append(t-d); tt.append(t)
        p = lambda sim, est: 2*(1-norm.cdf(abs(est/np.std(sim))))
        erows.append(dict(variable=v, direct=d0, direct_p=p(dd, d0),
            indirect=t0-d0, indirect_p=p(ii, t0-d0), total=t0, total_p=p(tt, t0)))
    pd.DataFrame(erows).to_csv(OUT+"trackB_SDM_rerun.csv", index=False)
    print("SDM rho=%.4f (p=%.4f)" % (rho, sdm.z_stat[-1][1]))

    # ---------- 4. Subsample heterogeneity ----------
    hrows = []
    for lbl, sub in [("capital", dm[dm.metro_dummy == 1]),
                     ("non-capital", dm[dm.metro_dummy == 0])] + \
                    [(f"size:{cs}", dm[dm.city_size == cs])
                     for cs in sorted(dm.city_size.dropna().unique())]:
        r = fe(sub.set_index(["region_code","year"]),
               ["pagerank_lag1"]+CORE+["prl_x_covid"])
        hrows.append(dict(split=lbl, coef=r.params["pagerank_lag1"],
            se=r.std_errors["pagerank_lag1"], p=r.pvalues["pagerank_lag1"],
            N=int(r.nobs)))
        print(lbl, hrows[-1])
    pd.DataFrame(hrows).to_csv(OUT+"trackB_hetero_rerun.csv", index=False)
    # KNN-5 robustness: build centroids from data/shapefile/sgg229.geojson and
    # KNN.from_dataframe(..., k=5); canonical values are stored in
    # results/tables/trackB_canonical_robust_hetero.csv (row 'KNN-5 SDM').

if __name__ == "__main__":
    main()
