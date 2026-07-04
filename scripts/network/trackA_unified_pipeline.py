# -*- coding: utf-8 -*-
"""Track A canonical pipeline (final, 2026-07-04).

Unified 229-municipality pipeline underlying Section 4.1:
  1. yearly directed/undirected flow-weighted networks from migration_long.parquet
  2. topology (density, reciprocity, APL, clustering) + Louvain (rs=42)
  3. concentration & hub stability from track_A_2008_2025.csv
  4. global Moran's I and LISA (Queen fixed weights, 9,999 permutations, seed=42)
  5. Mann-Kendall trend tests

Run from repository root:  python scripts/network/trackA_unified_pipeline.py
Outputs -> results/tables/   (runtime: ~10-20 min, dominated by 9,999-perm LISA)
"""
import ast, json, warnings
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import pyarrow.parquet as pq
from collections import Counter
from scipy.stats import kendalltau
from libpysal.weights import W as LW
import esda
import pymannkendall as mk

warnings.filterwarnings("ignore")
YEARS = range(2008, 2026)
PERM = 9999
SEED = 42
OUT = "results/tables/"

# ---------- crosswalk: raw sgg codes -> 229 harmonized panel codes ----------
CROSSWALK = {28170:28177,41111:41110,41113:41110,41115:41110,41117:41110,41131:41130,
 41133:41130,41135:41130,41171:41170,41173:41170,41192:41190,41194:41190,41195:41190,
 41196:41190,41197:41190,41199:41190,41271:41270,41273:41270,41281:41280,41283:41280,
 41285:41280,41287:41280,41461:41460,41463:41460,41465:41460,41710:41630,41730:41670,
 41750:41590,41790:41610,41810:41650,42110:51110,42130:51130,42150:51150,42170:51170,
 42190:51190,42210:51210,42230:51230,42720:51720,42730:51730,42750:51750,42760:51760,
 42770:51770,42780:51780,42790:51790,42800:51800,42810:51810,42820:51820,42830:51830,
 43111:43110,43112:43110,43113:43110,43114:43110,43710:43110,44131:44130,44133:44130,
 44730:36110,44830:44270,45111:52110,45113:52110,45130:52130,45140:52140,45180:52180,
 45190:52190,45210:52210,45710:52710,45720:52720,45730:52730,45740:52740,45750:52750,
 45770:52770,45790:52790,45800:52800,47111:47110,47113:47110,47720:27720,48121:48110,
 48123:48110,48125:48110,48127:48110,48129:48110,48160:48110,48190:48110,49110:50110,
 49130:50130,49710:50110,49720:50130,52111:52110,52113:52110}

def load_od():
    t = pq.read_table("data/source/migration_long.parquet",
        filters=[("ageGroup","=","전체"),("gender","=","전체"),
                 ("dataset","=","flow"),("unit","=","sgg")])
    return t.to_pandas()[["year","ori","des","flow"]]

def main():
    m229 = sorted(pd.read_csv("data/shapefile/sgg_code_mapping_229.csv").panel_code)
    mset = set(m229)
    od = load_od()
    od["o"] = od.ori.map(lambda c: CROSSWALK.get(c, c))
    od["d"] = od.des.map(lambda c: CROSSWALK.get(c, c))
    od = od[(od.o != od.d) & od.o.isin(mset) & od.d.isin(mset)]

    # ---------- 1-2. yearly networks: topology + Louvain ----------
    rows = []
    for yr in YEARS:
        dd = od[od.year == yr].groupby(["o","d"], as_index=False).flow.sum()
        DG = nx.DiGraph(); DG.add_nodes_from(m229)
        DG.add_weighted_edges_from(dd[["o","d","flow"]].values)
        UG = nx.Graph(); UG.add_nodes_from(m229)
        for r in dd.itertuples():
            a, b = min(r.o, r.d), max(r.o, r.d)
            w0 = UG[a][b]["weight"] if UG.has_edge(a, b) else 0
            UG.add_edge(a, b, weight=w0 + r.flow)
        giant = max(nx.connected_components(UG), key=len)
        part = community_louvain.best_partition(UG, weight="weight", random_state=SEED)
        sizes = sorted(Counter(part.values()).values(), reverse=True)
        rows.append(dict(year=yr, n=DG.number_of_nodes(), E=DG.number_of_edges(),
            density=DG.number_of_edges()/(229*228), reciprocity=nx.reciprocity(DG),
            apl=nx.average_shortest_path_length(UG.subgraph(giant)),
            clustering=nx.average_clustering(UG),
            Q=community_louvain.modularity(part, UG, weight="weight"),
            ncomm=len(sizes), sizes=";".join(map(str, sizes))))
        print(yr, round(rows[-1]["density"],4), round(rows[-1]["Q"],4))
    pd.DataFrame(rows).to_csv(OUT+"unified_network_stats_2008_2025.csv", index=False)

    # ---------- 3. concentration & hub stability (track_A node metrics) ----------
    tA = pd.read_csv("data/derived/track_A_2008_2025.csv",
        usecols=["region_code","year","net_rate","pagerank","in_strength","deg_cent"])
    d08 = tA[tA.year == 2008].set_index("region_code").pagerank
    top20_08 = set(d08.nlargest(20).index)
    srows = []
    for yr in YEARS:
        d = tA[tA.year == yr].set_index("region_code")
        pr = d.pagerank / d.pagerank.sum(); n = len(d)
        c = d.deg_cent.values
        top20 = set(pr.nlargest(20).index)
        srows.append(dict(year=yr, hhi=float(pr @ pr),
            top10pct_pr=float(pr.nlargest(round(n*0.1)).sum())*100,
            top20_instr=float(d.in_strength.nlargest(20).sum()/d.in_strength.sum())*100,
            freeman=float((c.max()-c).sum()/((n-1)*(n-2))), sd_pr=float(pr.std(ddof=1)),
            kendall=float(kendalltau(d08.values, d.pagerank.reindex(d08.index).values)[0]),
            kendall_top20=float(kendalltau(d08[list(top20_08)].values,
                d.pagerank[list(top20_08)].values)[0]),
            jaccard=len(top20_08 & top20)/len(top20_08 | top20)))
    pd.DataFrame(srows).to_csv(OUT+"trackA_concentration_hub_stats.csv", index=False)

    # ---------- 4. Moran / LISA ----------
    nb = pd.read_csv("data/spatial_weights/queen_w_229_neighbors.csv")
    nb["neighbor_codes"] = nb["neighbor_codes"].apply(ast.literal_eval)
    w = LW({r.panel_code: list(r.neighbor_codes) for r in nb.itertuples()},
           silence_warnings=True)
    w.transform = "r"
    grows, crows, arows = [], [], []
    for yr in YEARS:
        y = (tA[tA.year == yr].set_index("region_code")
             .reindex(w.id_order).net_rate.values)
        g = esda.Moran(y, w, permutations=PERM)
        lm = esda.Moran_Local(y, w, permutations=PERM, seed=SEED)
        sig = lm.p_sim < 0.05
        lab = np.where(~sig, "NS", np.where(lm.q == 1, "HH",
              np.where(lm.q == 2, "LH", np.where(lm.q == 3, "LL", "HL"))))
        grows.append(dict(year=yr, moran_I=g.I, E_I=g.EI, p_sim=g.p_sim))
        crows.append(dict(year=yr, **{k: int((lab == k).sum())
                     for k in ["HH","LL","HL","LH"]}, total=int(sig.sum())))
        arows += [dict(year=yr, region_code=rc, cluster=lb)
                  for rc, lb in zip(w.id_order, lab)]
    pd.DataFrame(grows).to_csv(OUT+"moran_global_2008_2025.csv", index=False)
    pd.DataFrame(crows).to_csv(OUT+"lisa_cluster_counts_2008_2025.csv", index=False)
    pd.DataFrame(arows).to_csv(OUT+"lisa_assignments_2008_2025.csv", index=False)

    # ---------- 5. Mann-Kendall ----------
    net = pd.DataFrame(rows); S = pd.DataFrame(srows)
    for name, series in [("density", net.density), ("reciprocity", net.reciprocity),
        ("apl", net.apl), ("clustering", net.clustering), ("Q", net.Q),
        ("top20_instr", S.top20_instr), ("hhi", S.hhi), ("freeman", S.freeman),
        ("jaccard", S.jaccard)]:
        r = mk.original_test(series.values)
        print(f"MK {name:12s} trend={r.trend:10s} p={r.p:.4f}")

if __name__ == "__main__":
    main()
