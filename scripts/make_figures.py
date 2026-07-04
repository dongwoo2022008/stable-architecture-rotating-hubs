# -*- coding: utf-8 -*-
"""Regenerate the paper figures from results/tables + data (grayscale main set).

  python scripts/make_figures.py        -> figures/*.png

Covers Figures 1-1, 4-1..4-6 and Supplementary S2, S4, S5, S6.
S1 (Moran scatter) and S3 (Louvain maps) require node-level recomputation and
are produced by scripts/network/trackA_unified_pipeline.py + this module's
--heavy flag counterparts; reference renders for all figures ship in figures/.
"""
import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

FIG = "figures/"
plt.rcParams.update({"font.size": 12})
GRAY = dict(covid=dict(color="0.9"))

net = pd.read_csv("results/tables/unified_network_stats_2008_2025.csv")
S   = pd.read_csv("results/tables/trackA_concentration_hub_stats.csv")
mo  = pd.read_csv("results/tables/moran_global_2008_2025.csv")
li  = pd.read_csv("results/tables/lisa_cluster_counts_2008_2025.csv")
la  = pd.read_csv("results/tables/lisa_assignments_2008_2025.csv")
fe  = pd.read_csv("results/tables/trackB_canonical_FE_lag1.csv")
gmm = pd.read_csv("results/tables/trackB_canonical_GMM.csv")
sdm = pd.read_csv("results/tables/trackB_canonical_SDM_effects.csv", index_col=0)
het = pd.read_csv("results/tables/trackB_canonical_robust_hetero.csv")
YR  = net.year.values

def shade(ax):
    ax.axvspan(2019.5, 2022.5, color="0.9", zorder=0)

# ---------- Figure 4-1: concentration (3 panels) ----------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
ax = axes[0]
ax.plot(YR, S.top20_instr, "o-", color="k", label="Top-20 in-strength")
ax.plot(YR, S.top10pct_pr, "s--", color="0.5", label="Top-10% PageRank")
ax.set_ylim(20, 35); ax.set_ylabel("Share (%)"); ax.set_title("(a) Top-hub shares")
ax.legend(fontsize=9); shade(ax)
ax = axes[1]
ax.plot(YR, S.hhi, "o-", color="k"); ax.set_ylim(0.005, 0.010)
ax.set_ylabel("HHI"); ax.set_title("(b) HHI of PageRank distribution"); shade(ax)
ax = axes[2]
a, b = np.polynomial.polynomial.polyfit(YR, S.freeman, 1)
ax.plot(YR, S.freeman, "s-", color="k")
ax.plot(YR, a + b*YR, "--", color="0.5",
        label="Trend (slope = +%.1f×10$^{-6}$/yr)" % (b*1e6))
ax.set_ylim(0, 0.0010); ax.set_ylabel("Freeman centralization")
ax.set_title("(c) Freeman degree centralization"); ax.legend(fontsize=9); shade(ax)
for ax in axes: ax.set_xlabel("Year"); ax.grid(ls="--", alpha=.4)
plt.tight_layout(); plt.savefig(FIG+"Figure_4-1.png", dpi=200); plt.close()

# ---------- Figure 4-2: hub stability ----------
fig, ax = plt.subplots(figsize=(11.8, 4.6))
ax.plot(YR, S.jaccard, "o-", color="k", label="Jaccard similarity (top-20 vs. 2008)")
ax.plot(YR, S.kendall_top20, "s--", color="0.5", label="Kendall's τ (2008 top-20 hubs)")
a, b = np.polynomial.polynomial.polyfit(YR, S.jaccard, 1)
ax.plot(YR, a + b*YR, ":", color="0.35", label="Jaccard trend (slope = %.4f/yr)" % b)
for x, y in [(2008, S.jaccard.iloc[0]), (2018, S.jaccard[S.year==2018].iloc[0]),
             (2025, S.jaccard.iloc[-1])]:
    ax.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(4, 6), fontsize=9)
ax.set_xlabel("Year"); ax.set_ylabel("Similarity vs. 2008")
ax.legend(fontsize=9, loc="lower left"); ax.grid(ls="--", alpha=.4); shade(ax)
plt.tight_layout(); plt.savefig(FIG+"Figure_4-2.png", dpi=200); plt.close()

# ---------- Figure 4-3: rank shift (top-10 bump chart) ----------
tA = pd.read_csv("data/derived/track_A_2008_2025.csv",
                 usecols=["region_code","name","year","pagerank"])
t8  = tA[tA.year==2008].nlargest(20,"pagerank").reset_index(drop=True)
t25 = tA[tA.year==2025].nlargest(20,"pagerank").reset_index(drop=True)
r8  = {r.region_code: i+1 for i, r in t8.head(10).iterrows()}
r25 = {r.region_code: i+1 for i, r in t25.head(10).iterrows()}
EN = {41110:"Suwon",41590:"Hwaseong",11710:"Songpa-gu",41460:"Yongin",41280:"Goyang",
      41130:"Seongnam",41190:"Bucheon",11680:"Gangnam-gu",11620:"Gwanak-gu",11500:"Gangseo-gu",
      28260:"Incheon Seo-gu",41220:"Pyeongtaek",44130:"Cheonan"}
names = {c: EN.get(c, str(c)) for c in tA.region_code.unique()}
fig, ax = plt.subplots(figsize=(9.5, 7.5))
codes = sorted(set(r8) | set(r25), key=lambda c: r8.get(c, 10.8+list(r25).index(c)*0.9 if c in r25 else 14))
slot = 10.8
for c in list(r8) + [c for c in r25 if c not in r8]:
    y1 = r8.get(c); y2 = r25.get(c)
    if y1 is None: y1 = slot; slot += 0.9
    if y2 is None: y2 = slot; slot += 0.9
    rose = (c in r25) and (y2 <= (r8.get(c) or 99))
    col = "k" if rose else "0.65"
    ax.plot([0, 1], [y1, y2], "-o", color=col, lw=2.2 if rose else 1.6, ms=7)
    nm = names.get(c, str(c))
    l8 = f"{nm} ({int(y1)})" if c in r8 else f"{nm} (new)"
    l25 = f"({int(y2)}) {nm}" if c in r25 else f"(out) {nm}"
    ax.text(-0.03, y1, l8, ha="right", va="center", fontsize=10, color=col)
    ax.text(1.03, y2, l25, ha="left", va="center", fontsize=10, color=col)
ax.axhline(10.4, ls=":", color="0.6"); ax.text(0.5, 10.35, "top-10 boundary",
    ha="center", va="bottom", color="0.6", fontsize=9)
ax.set_xlim(-0.55, 1.55); ax.set_ylim(0.3, slot); ax.invert_yaxis()
ax.set_xticks([0, 1]); ax.set_xticklabels(["2008", "2025"], fontsize=12, fontweight="bold")
ax.set_yticks(range(1, 11)); ax.set_ylabel("PageRank rank")
for s in ["top", "right", "bottom"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(FIG+"Figure_4-3.png", dpi=200); plt.close()

# ---------- Figure 4-4: global Moran trend ----------
fig, ax = plt.subplots(figsize=(11.8, 4.6))
sig = mo.p_sim < 0.05
ax.plot(mo.year, mo.moran_I, "-", color="k")
ax.plot(mo.year[sig], mo.moran_I[sig], "o", color="k",
        label="Significant (pseudo p < 0.05)")
ax.plot(mo.year[~sig], mo.moran_I[~sig], "o", mfc="white", mec="k",
        label="Not significant")
ax.axhline(mo.E_I.iloc[0], ls="--", color="0.5", label="E[I] = −0.0044")
for yv in [2011, 2021, 2022]:
    v = mo[mo.year==yv].moran_I.iloc[0]
    ax.annotate(("%.3f" % v).replace("-", "−"), (yv, v),
                textcoords="offset points", xytext=(-8, 8), fontsize=9)
ax.set_xlabel("Year"); ax.set_ylabel("Global Moran's I")
ax.legend(fontsize=9); ax.grid(ls="--", alpha=.4); shade(ax)
plt.tight_layout(); plt.savefig(FIG+"Figure_4-4.png", dpi=200); plt.close()

# ---------- Figure 4-5: LISA maps (GeoDa palette) ----------
try:
    import geopandas as gpd
    gd = gpd.read_file("data/shapefile/sgg229.geojson")
    code_col = "panel" if "panel" in gd.columns else [c for c in gd.columns if "code" in c.lower()][0]
    COL = {"HH":"#e31a1c","LL":"#1f78b4","HL":"#fb9a99","LH":"#a6cee3","NS":"#efefef"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    for ax, yr, tag in zip(axes, [2008, 2016, 2025], "abc"):
        lab = la[la.year==yr].set_index("region_code").cluster
        g = gd.copy(); g["c"] = g[code_col].astype(int).map(lab).fillna("NS")
        g.plot(ax=ax, color=g["c"].map(COL), edgecolor="white", linewidth=.2)
        ax.set_title(f"({tag}) {yr}"); ax.axis("off")
    import matplotlib.patches as mp
    for ax, yr in zip(axes, [2008, 2016, 2025]):
        cnt = la[la.year==yr].cluster.value_counts()
        ax.legend(handles=[mp.Patch(fc=COL[k],
            label=f"{'Not sig.' if k=='NS' else k} (n={int(cnt.get(k,0))})")
            for k in ["HH","LL","HL","LH","NS"]], loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG+"Figure_4-5.png", dpi=200); plt.close()
except Exception as e:
    print("Figure 4-5 skipped:", e)

# ---------- Figure 4-6: SDM decomposition ----------
z = lambda p: norm.isf(p/2)
fig, axes = plt.subplots(1, 2, figsize=(12.85, 5.04))
s = sdm.loc["pagerank_lag1"]
vals=[s.direct,s.indirect,s.total]; ps=[s.direct_p,s.indirect_p,s.total_p]
ses=[abs(v)/z(p) for v,p in zip(vals,ps)]
ax=axes[0]
ax.bar(["Direct","Indirect","Total"],vals,color=["#222","#aaa","#666"],
       yerr=[1.96*e for e in ses],capsize=6)
ax.axhline(0,color="k",lw=1); ax.set_title("(a) Lagged PageRank")
ax.set_ylabel("Effect on net migration rate"); ax.grid(axis="y",ls="--",alpha=.5)
ax=axes[1]
ctrl=[("ln_pop_density","ln pop. density"),("youth_ratio","Youth ratio"),
 ("aging_ratio","Aging ratio"),("fiscal_indep","Fiscal independence"),
 ("doctor_per1000","Doctors per 1,000"),("fertility","Fertility"),
 ("culture_facility_count","Cultural facilities"),("childcare_pk","Childcare per 1,000")]
yy=np.arange(len(ctrl))[::-1]; h=0.38
for i,(v,lab) in enumerate(ctrl):
    r=sdm.loc[v]
    ax.barh(yy[i]+h/2,r.direct,height=h,color="#222",label="Direct" if i==0 else None)
    ax.barh(yy[i]-h/2,r.indirect,height=h,color="#aaa",label="Indirect" if i==0 else None)
    for val,pv,off in [(r.direct,r.direct_p,h/2),(r.indirect,r.indirect_p,-h/2)]:
        if pv<0.05:
            ax.text(val+(0.6 if val>=0 else -0.6),yy[i]+off,"*",va="center",
                    ha="left" if val>=0 else "right",fontsize=14)
ax.set_yticks(yy); ax.set_yticklabels([l for _,l in ctrl])
ax.axvline(0,color="k",lw=1); ax.set_title("(b) Control variables")
ax.set_xlabel("Effect on net migration rate"); ax.grid(axis="x",ls="--",alpha=.5)
ax.legend(loc="lower left",framealpha=0.95)
plt.tight_layout(); plt.savefig(FIG+"Figure_4-6.png",dpi=200); plt.close()

# ---------- Supplementary S2 (topology), S4 (Q), S5 (LISA counts), S6 (spec chart) ----------
fig,axes=plt.subplots(1,2,figsize=(13,4.3))
ax=axes[0]
ax.plot(YR,net.density,"o-",color="k",label="Density (directed)")
ax.plot(YR,net.reciprocity,"s--",color="0.4",label="Reciprocity (directed)")
ax.plot(YR,net.clustering,"^:",color="0.6",label="Avg clustering (undirected)")
ax.set_ylim(0.90,1.00); ax.set_title("(a) Density, reciprocity, clustering"); ax.legend(fontsize=8)
ax=axes[1]; ax.plot(YR,net.apl,"o-",color="k"); ax.set_ylim(1.00,1.05)
ax.set_title("(b) Average path length (undirected)")
for ax in axes: ax.set_xlabel("Year"); ax.grid(ls="--",alpha=.4); shade(ax)
plt.tight_layout(); plt.savefig(FIG+"Figure_S2.png",dpi=200); plt.close()

fig,ax=plt.subplots(figsize=(11.8,4.3))
ax.plot(YR,net.Q,"o-",color="k")
a,b=np.polynomial.polynomial.polyfit(YR,net.Q,1)
ax.plot(YR,a+b*YR,":",color="0.35",label="Trend (slope = %.4f/yr)"%b)
ax.axhline(net.Q.mean(),ls="--",color="0.6",label="Period mean = %.3f"%net.Q.mean())
ax.set_xlabel("Year"); ax.set_ylabel("Modularity (Q)"); ax.legend(fontsize=9)
ax.grid(ls="--",alpha=.4); shade(ax)
plt.tight_layout(); plt.savefig(FIG+"Figure_S4.png",dpi=200); plt.close()

COL={"HH":"#e31a1c","LL":"#1f78b4","HL":"#fb9a99","LH":"#a6cee3"}
fig,ax=plt.subplots(figsize=(11.8,4.6))
for k in ["HH","LL","HL","LH"]:
    ax.plot(li.year,li[k],"o-",color=COL[k],label=k)
ax.plot(li.year,li.total,"s--",color="0.4",label="Total significant")
ax.set_xlabel("Year"); ax.set_ylabel("Number of municipalities")
ax.legend(fontsize=9,ncol=5); ax.grid(ls="--",alpha=.4); shade(ax)
plt.tight_layout(); plt.savefig(FIG+"Figure_S5.png",dpi=200); plt.close()

gfe=lambda m: fe[(fe.model==m)&(fe.variable=="pagerank_lag1")].iloc[0]
knn=het[het.split=="KNN-5 SDM"].iloc[0]
hh=lambda s: het[(het.split!="KNN-5 SDM")&(het.group==s)].iloc[0]
rows=[("Pooled OLS",gfe("OLS").coef,gfe("OLS").se,"o","#111"),
 ("FE-M1 (two-way FE)",gfe("FE-M1").coef,gfe("FE-M1").se,"o","#111"),
 ("FE-M2 (+controls)",gfe("FE-M2").coef,gfe("FE-M2").se,"o","#111"),
 ("FE-M3 (+COVID int.)",gfe("FE-M3").coef,gfe("FE-M3").se,"o","#111"),
 ("FE-M4 (+metro int.)",gfe("FE-M4").coef,gfe("FE-M4").se,"o","#111"),
 ("Dynamic LSDV",gfe("LSDV").coef,gfe("LSDV").se,"s","#777"),
 ("System GMM (base)",gmm.iloc[0].prl,gmm.iloc[0].prl_se,"s","#777"),
 ("System GMM (short IV)",gmm.iloc[2].prl,gmm.iloc[2].prl_se,"s","#777"),
 ("System GMM (endog.)",gmm.iloc[3].prl,gmm.iloc[3].prl_se,"s","#777"),
 ("SDM direct (Queen)",sdm.loc["pagerank_lag1"].direct,
  abs(sdm.loc["pagerank_lag1"].direct)/z(sdm.loc["pagerank_lag1"].direct_p),"D","#111"),
 ("SDM direct (KNN-5)",knn.coef,knn.se,"D","#111"),
 ("FE-M3: capital region",hh("capital").coef,hh("capital").se,"^","#555"),
 ("FE-M3: non-capital",hh("non-capital").coef,hh("non-capital").se,"^","#555"),
 ("FE-M3: medium cities",hh("medium").coef,hh("medium").se,"^","#555"),
 ("FE-M3: large cities",hh("large").coef,hh("large").se,"^","#555"),
 ("FE-M3: small rural counties",hh("small").coef,hh("small").se,"^","#555")]
fig,ax=plt.subplots(figsize=(10.36,7.03))
ypos=np.arange(len(rows))[::-1]
for (lab,c,se,mk_,col),yv in zip(rows,ypos):
    ax.errorbar(float(c),yv,xerr=1.96*float(se),fmt=mk_,color=col,ecolor=col,ms=9,capsize=3)
ax.axvline(0,color="0.3",lw=1.2)
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Coefficient of lagged PageRank (95% CI)")
ax.grid(axis="y",ls="--",alpha=.4); ax.grid(axis="x",ls="--",alpha=.3)
xmax=ax.get_xlim()[1]
for a0,t in [(0,"Static panel"),(5,"Dynamic panel"),(9,"Spatial (direct effect)"),(11,"Subsamples")]:
    ax.text(xmax*0.98,ypos[a0],t,ha="right",va="center",color="0.4",fontsize=12)
    if a0>0: ax.axhline(ypos[a0]+0.5,color="0.85",lw=1)
plt.tight_layout(); plt.savefig(FIG+"Figure_S6.png",dpi=200); plt.close()

print("figures regenerated ->", FIG)
