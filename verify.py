# -*- coding: utf-8 -*-
"""One-click verification of the canonical results (Paper: Stable Architecture,
Rotating Hubs, 2008-2025).

  python verify.py            # quick (~1-2 min): checksums + key published values
                              #   + live re-estimation of the FE ladder from raw data
  python verify.py --full     # additionally re-runs both pipelines end-to-end
                              #   (Track A ~15 min; Track B SDM ~30-60 min)

Exits with code 0 and prints ALL CHECKS PASSED if every published number is
reproduced; any deviation is reported with expected vs. obtained values.
"""
import sys, hashlib, subprocess
import numpy as np
import pandas as pd

TOL = 1e-3           # relative tolerance for stored-table comparisons
FAIL = []

def ck(label, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else f"  [{detail}]"))
    if not ok: FAIL.append(label)

def close(a, b, tol=TOL):
    a, b = float(a), float(b)
    return abs(a-b) <= tol*max(1.0, abs(b))

# ---------- 0. checksums ----------
print("== 0. data checksums =====================================")
for line in open("CHECKSUMS.txt"):
    h, path = line.split()
    with open(path, "rb") as f:
        ck(f"sha256 {path}", hashlib.sha256(f.read()).hexdigest() == h)

# ---------- 1. canonical identity values (guards against stale copies) ----------
print("== 1. canonical identity (v2: 223 municipalities, N=3,568) ==")
fe = pd.read_csv("results/tables/trackB_canonical_FE_lag1.csv")
g  = lambda m, v: fe[(fe.model == m) & (fe.variable == v)].iloc[0]
ck("FE-M2 pagerank_lag1 = 3648.8102", close(g("FE-M2","pagerank_lag1").coef, 3648.810202787355, 1e-6))
ck("FE-M2 SE            = 634.6202",  close(g("FE-M2","pagerank_lag1").se,   634.6202105837472, 1e-6))
ck("LSDV pagerank_lag1  = -2658.18 (NOT -2660.5 = deprecated 222-sample run)",
   close(g("LSDV","pagerank_lag1").coef, -2658.184573299107, 1e-6))
ck("N = 3,568", int(g("FE-M2","_N").coef) == 3568)
gm = pd.read_csv("results/tables/trackB_canonical_GMM.csv")
e  = gm[gm.model.str.contains("endog")].iloc[0]
ck("GMM endog prl = -1910.37 (NOT -1952.99 = deprecated)", close(e.prl, -1910.37, 1e-3))
ck("DIFF-GMM Hansen p = 0.010", close(gm[gm.model=="DIFF-GMM"].hansen_p.iloc[0], 0.010, 1e-6))

# ---------- 2. published Section-4 values vs stored tables ----------
print("== 2. published values vs results/tables =================")
net = pd.read_csv("results/tables/unified_network_stats_2008_2025.csv")
ck("density 0.933 -> 0.920", round(net.density.iloc[0],3)==0.933 and round(net.density.iloc[-1],3)==0.920)
ck("reciprocity 0.964 -> 0.954", round(net.reciprocity.iloc[0],3)==0.964 and round(net.reciprocity.iloc[-1],3)==0.954)
ck("modularity 0.426 -> 0.400", round(net.Q.iloc[0],3)==0.426 and round(net.Q.iloc[-1],3)==0.400)
ck("communities 8-9 every year", set(net.ncomm.unique()) == {8,9})
S = pd.read_csv("results/tables/trackA_concentration_hub_stats.csv")
ck("Jaccard 2025 = 0.43", round(S.jaccard.iloc[-1],2)==0.43)
ck("Kendall tau (top-20) 2025 = 0.53", round(S.kendall_top20.iloc[-1],2)==0.53)
ck("Freeman 2008 5.94e-4 / peak 2024 7.67e-4",
   round(S.freeman.iloc[0]*1e4,2)==5.94 and round(S.freeman[S.year==2024].iloc[0]*1e4,2)==7.67)
mo = pd.read_csv("results/tables/moran_global_2008_2025.csv")
ck("Moran significant in 11 of 18 (5%)", int((mo.p_sim < 0.05).sum())==11)
ck("Moran 2011 peak 0.147", round(mo[mo.year==2011].moran_I.iloc[0],3)==0.147)
li = pd.read_csv("results/tables/lisa_cluster_counts_2008_2025.csv")
r08=li[li.year==2008].iloc[0]; r25=li[li.year==2025].iloc[0]
ck("LISA 2008 = 31 (10/10/6/5)", (r08.total,r08.HH,r08.LL,r08.HL,r08.LH)==(31,10,10,6,5))
ck("LISA 2025 = 22 (4/7/5/6)",  (r25.total,r25.HH,r25.LL,r25.HL,r25.LH)==(22,4,7,5,6))
sd = pd.read_csv("results/tables/trackB_canonical_SDM_effects.csv").set_index("variable")
ck("SDM direct 3224 / indirect 937 (ns)",
   close(sd.loc["pagerank_lag1"].direct, 3224.283, 1e-3) and close(sd.loc["pagerank_lag1"].indirect, 937.46, 1e-2))
he = pd.read_csv("results/tables/trackB_canonical_robust_hetero.csv")
ck("KNN-5 direct 3463.3", close(he[he.split=="KNN-5 SDM"].coef.iloc[0], 3463.289, 1e-3))
ck("medium cities 5188.2", close(he[he.group=="medium"].coef.iloc[0], 5188.164, 1e-3))

# ---------- 3. live re-estimation of the FE ladder from raw data ----------
print("== 3. live FE re-estimation from data/derived =============")
from linearmodels.panel import PanelOLS
df = pd.read_csv("data/derived/track_B_2009_2024.csv")
jj = df.region_code == 52110
df.loc[jj,"area_km2"]=205.53
df.loc[jj,"pop_density"]=df.loc[jj,"population"]/205.53
df.loc[jj,"ln_pop_density"]=np.log(df.loc[jj,"pop_density"])
CORE=["ln_pop_density","youth_ratio","aging_ratio","fiscal_indep","doctor_per1000",
      "fertility","culture_facility_count","childcare_pk"]
df["prl_x_covid"]=df.pagerank_lag1*df.covid_dummy
need=["net_rate","pagerank_lag1"]+CORE+["seoul_dist_km","covid_dummy","region_code",
      "year","metro_dummy","net_rate_lag1","prl_x_covid"]
dm=df[need].dropna(); cnt=dm.groupby("region_code").size()
dm=dm[dm.region_code.isin(cnt[cnt==16].index)]
ck("live sample = 223 x 16 = 3,568", dm.region_code.nunique()==223 and len(dm)==3568,
   f"{dm.region_code.nunique()} x -> {len(dm)}")
dfe=dm.set_index(["region_code","year"])
for nm, xs, exp in [("FE-M1",["pagerank_lag1"],3639.331508596904),
                    ("FE-M2",["pagerank_lag1"]+CORE,3648.810202787355),
                    ("FE-M3",["pagerank_lag1"]+CORE+["prl_x_covid"],3761.945773097784)]:
    r=PanelOLS(dfe["net_rate"],dfe[xs],entity_effects=True,time_effects=True,
        drop_absorbed=True).fit(cov_type="clustered",cluster_entity=True)
    ck(f"live {nm} prl = {exp:.4f}", close(r.params["pagerank_lag1"], exp, 1e-9),
       f"got {r.params['pagerank_lag1']:.6f}")

# ---------- 4. optional full pipeline re-run ----------
if "--full" in sys.argv:
    print("== 4. full pipeline re-run (this takes a while) ==========")
    subprocess.run([sys.executable,"scripts/network/trackA_unified_pipeline.py"],check=True)
    subprocess.run([sys.executable,"scripts/panel/trackB_canonical_models.py"],check=True)

print("="*60)
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED:"); [print("  -",f) for f in FAIL]; sys.exit(1)
print("ALL CHECKS PASSED - results match the published canonical values.")
