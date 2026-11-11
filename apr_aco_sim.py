#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apr_aco_sim.py  -  Re-implementation of the simulation described in
"Adaptive Pheromone-Based Communication Protocol for UAV Networks in Complex
Urban Environments: Design, Simulation, and Statistical Evaluation of APR-ACO".

Everything the paper specifies is implemented as written (Tables 2, Sections 3-4).
Everything the paper does NOT specify is marked  # ASSUMPTION  and is exposed in
`Params`, so you can change it.  Because the original code was not provided, the
absolute numbers will not be identical to the paper's; what matters is whether the
*conclusions* (PDR equal, latency lower, overhead higher) reproduce.

Requirements:  pip install numpy scipy networkx matplotlib pandas

Usage examples
--------------
  python apr_aco_sim.py checks                 # sanity checks + paper-number audit (fast)
  python apr_aco_sim.py main --runs 30         # Table 3, Figs 1-4
  python apr_aco_sim.py topology               # Fig 5
  python apr_aco_sim.py scalability            # Table 4, Fig 6
  python apr_aco_sim.py ablation               # Table 5, Fig 7
  python apr_aco_sim.py sensitivity            # Tables 6-7, Fig 8
  python apr_aco_sim.py all --workers 4        # everything
  python apr_aco_sim.py main --oracle          # adds control baselines (see README in chat)
  python apr_aco_sim.py lq                     # tune on seeds A, test on unseen seeds B
  python apr_aco_sim.py main --scenario paper  # the paper's original 50 UAVs @ 50-150 m (no obstruction)
"""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

SRC, DST = 0, 1  # node indices of the fixed source / destination pair


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    # --- Table 2 (as published) ---
    area: float = 1000.0
    n_buildings: int = 200
    bld_side: tuple = (15.0, 60.0)
    bld_height: tuple = (10.0, 40.0)
    n_uav: int = 50
    alt_min: float = 50.0
    alt_max: float = 150.0
    gm_alpha: float = 0.75          # Gauss-Markov memory (the paper also calls it alpha!)
    speed_mean: float = 12.0
    freq_hz: float = 5.9e9
    tx_dbm: float = 13.0            # 20 mW
    rate_bps: float = 6e6
    range_m: float = 300.0
    nlos_db: float = 22.0
    sens_dbm: float = -85.0
    max_retx: int = 3
    pkt_bytes: int = 1500
    pkts_per_step: int = 8
    steps: int = 50
    n_ants: int = 6
    rho: float = 0.35
    alpha_ph: float = 1.0           # pheromone exponent
    beta_ph: float = 2.0            # heuristic exponent
    Q: float = 100.0
    wH: float = 1.0
    wL: float = 40.0
    wD: float = 0.08
    route_timeout: int = 6
    dsr_alternates: int = 3
    # --- NOT specified in the paper (ASSUMPTIONS) ---
    dt: float = 1.0                 # ASSUMPTION: 1 step = 1 s (this is what makes 8*1500*8 bit/s = 0.096 Mbps)
    src_pos: tuple = (35.0, 35.0, 100.0)    # ASSUMPTION: "near opposite corners, ~1300 m apart"
    dst_pos: tuple = (965.0, 965.0, 100.0)  # ASSUMPTION: both hover (do not move)
    speed_sd: float = 2.0           # ASSUMPTION: Gauss-Markov speed sigma (m/s)
    dir_sd: float = 0.4             # ASSUMPTION: heading sigma (rad)
    pitch_sd: float = 0.15          # ASSUMPTION: pitch sigma (rad)
    logistic_k: float = 0.5         # ASSUMPTION: logistic slope (1/dB), midpoint = sensitivity
    mac_overhead_s: float = 0.003   # ASSUMPTION: DIFS+backoff+ACK+processing per attempt
    tau0: float = 0.1               # ASSUMPTION: initial pheromone
    tau_min: float = 0.01           # ASSUMPTION: pheromone floor (keeps 1/(tau+eps) finite)
    eps: float = 1e-6               # ASSUMPTION: epsilon of the Dijkstra weight
    max_hops: int = 15              # ASSUMPTION: ant hop budget
    lq_gamma: float = 0.0           # EXTENSION (not in paper): link-quality exponent in the ant's next-hop choice; 0 = paper

    @property
    def t_att(self) -> float:
        """Time of one transmission attempt on one hop (s)."""
        return self.pkt_bytes * 8 / self.rate_bps + self.mac_overhead_s


# ----------------------------------------------------------------------------
# World: buildings + Gauss-Markov 3D mobility + per-step connectivity graphs
# ----------------------------------------------------------------------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gauss_markov_3d(p: Params, rng) -> np.ndarray:
    n, T, a = p.n_uav, p.steps, p.gm_alpha
    A = p.area
    pos = np.empty((T, n, 3))
    x = rng.uniform(0, A, n)
    y = rng.uniform(0, A, n)
    z = rng.uniform(p.alt_min, p.alt_max, n)
    s = np.clip(rng.normal(p.speed_mean, p.speed_sd, n), 1, None)
    th = rng.uniform(0, 2 * np.pi, n)
    th_bar = th.copy()
    ph = rng.normal(0, p.pitch_sd, n)
    k = math.sqrt(1 - a * a)
    for t in range(T):
        pos[t, :, 0], pos[t, :, 1], pos[t, :, 2] = x, y, z
        pos[t, SRC] = p.src_pos
        pos[t, DST] = p.dst_pos
        # drift mean heading toward the centre when close to a border
        near = (x < 80) | (x > A - 80) | (y < 80) | (y > A - 80)
        to_c = np.arctan2(A / 2 - y, A / 2 - x)
        mean_th = np.where(near, to_c, th_bar)
        s = a * s + (1 - a) * p.speed_mean + k * p.speed_sd * rng.standard_normal(n)
        s = np.clip(s, 0.5, None)
        th = th + (1 - a) * _wrap(mean_th - th) + k * p.dir_sd * rng.standard_normal(n)
        ph = a * ph + k * p.pitch_sd * rng.standard_normal(n)
        x = x + s * np.cos(ph) * np.cos(th) * p.dt
        y = y + s * np.cos(ph) * np.sin(th) * p.dt
        z = z + s * np.sin(ph) * p.dt
        # reflect at the borders
        for arr, lo, hi in ((x, 0, A), (y, 0, A), (z, p.alt_min, p.alt_max)):
            lo_m, hi_m = arr < lo, arr > hi
            arr[lo_m] = 2 * lo - arr[lo_m]
            arr[hi_m] = 2 * hi - arr[hi_m]
        ph = np.where((z <= p.alt_min + 1) | (z >= p.alt_max - 1), -ph, ph)
    return pos


def nlos_mask(P1, P2, bmin, bmax, chunk=400) -> np.ndarray:
    """True where the 3-D segment P1->P2 intersects any building box (slab method, chunked for memory)."""
    out = np.zeros(len(P1), bool)
    for a in range(0, len(P1), chunk):
        q1, q2 = P1[a:a + chunk], P2[a:a + chunk]
        d = q2 - q1
        d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        t1 = (bmin[None, :, :] - q1[:, None, :]) / d[:, None, :]
        t2 = (bmax[None, :, :] - q1[:, None, :]) / d[:, None, :]
        tmin = np.minimum(t1, t2).max(axis=2)
        tmax = np.maximum(t1, t2).min(axis=2)
        out[a:a + chunk] = ((tmin <= tmax) & (tmax >= 0) & (tmin <= 1)).any(axis=1)
    return out


def build_graph(p: Params, P, bmin, bmax):
    n = len(P)
    iu, ju = np.triu_indices(n, 1)
    d = np.linalg.norm(P[iu] - P[ju], axis=1)
    m = d <= p.range_m
    iu, ju, d = iu[m], ju[m], d[m]
    nl = nlos_mask(P[iu], P[ju], bmin, bmax) if len(iu) else np.zeros(0, bool)
    fspl = 20 * np.log10(np.maximum(d, 1.0)) + 20 * np.log10(p.freq_hz) - 147.55
    rx = p.tx_dbm - fspl - p.nlos_db * nl
    pa = 1.0 / (1.0 + np.exp(-p.logistic_k * (rx - p.sens_dbm)))   # per-attempt success
    A = 1 + p.max_retx
    ph = 1 - (1 - pa) ** A                                           # per-hop success incl. ARQ
    exp_tx = np.where(pa < 1e-9, A, (1 - (1 - pa) ** A) / np.maximum(pa, 1e-12))
    dly = exp_tx * p.t_att * 1000.0                                  # expected hop delay, ms
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(
        (int(i), int(j), {"d": float(dd), "p": float(a_), "ph": float(h_), "dly": float(y_)})
        for i, j, dd, a_, h_, y_ in zip(iu, ju, d, pa, ph, dly)
    )
    return G, len(d), int(nl.sum())


class World:
    def __init__(self, p: Params, seed: int):
        rng = np.random.default_rng(seed)
        nb = p.n_buildings
        cx, cy = rng.uniform(0, p.area, nb), rng.uniform(0, p.area, nb)
        w, dp = rng.uniform(*p.bld_side, nb), rng.uniform(*p.bld_side, nb)
        h = rng.uniform(*p.bld_height, nb)
        self.bmin = np.stack([cx - w / 2, cy - dp / 2, np.zeros(nb)], 1)
        self.bmax = np.stack([cx + w / 2, cy + dp / 2, h], 1)
        self.pos = gauss_markov_3d(p, rng)
        self.graphs, links, nlos = [], 0, 0
        for t in range(p.steps):
            G, nl_, nn_ = build_graph(p, self.pos[t], self.bmin, self.bmax)
            self.graphs.append(G)
            links += nl_
            nlos += nn_
        self.nlos_frac = nlos / max(links, 1)


# ----------------------------------------------------------------------------
# Routing protocols.  step(G, pos, rng) -> (route|None, control_packets, extra_delay_s)
# ----------------------------------------------------------------------------
def first_broken(G, route):
    for i, (u, v) in enumerate(zip(route[:-1], route[1:])):
        if not G.has_edge(u, v):
            return i
    return None


class AODV:
    def __init__(self, p):
        self.p, self.route, self.age = p, None, 0

    def step(self, G, pos, rng):
        p, ovh, extra, need = self.p, 0, 0.0, False
        if self.route is None or self.age >= p.route_timeout:
            need = True
        else:
            b = first_broken(G, self.route)
            if b is not None:
                ovh += max(b, 1)          # RERR travelling back to the source
                need = True
        if need:
            comp = nx.node_connected_component(G, SRC)
            ovh += len(comp)              # RREQ flood: every node rebroadcasts once
            if DST in comp:
                path = nx.shortest_path(G, SRC, DST)       # hop-count metric
                ovh += len(path) - 1                        # RREP
                extra = 2 * (len(path) - 1) * p.t_att       # discovery latency
                self.route, self.age = path, 0
            else:
                self.route = None
        if self.route is not None:
            self.age += 1
        return self.route, ovh, extra


class DSR:
    def __init__(self, p):
        self.p, self.route, self.cache, self.age = p, None, [], 0

    def _discover(self, G):
        p = self.p
        comp = nx.node_connected_component(G, SRC)
        ovh = len(comp)
        if DST not in comp:
            self.route, self.cache = None, []
            return ovh, 0.0
        first = nx.shortest_path(G, SRC, DST)
        paths, H = [first], G.copy()
        for _ in range(p.dsr_alternates):
            interior = paths[-1][1:-1]
            if not interior:
                break
            H.remove_nodes_from(interior)
            if not nx.has_path(H, SRC, DST):
                break
            paths.append(nx.shortest_path(H, SRC, DST))
        ovh += sum(len(q) - 1 for q in paths)         # one reply per discovered path
        self.route, self.cache, self.age = first, paths[1:], 0
        return ovh, 2 * (len(first) - 1) * p.t_att

    def step(self, G, pos, rng):
        p, ovh, extra = self.p, 0, 0.0
        if self.route is None or self.age >= p.route_timeout:
            ovh, extra = self._discover(G)
        else:
            b = first_broken(G, self.route)
            if b is not None:
                ovh += max(b, 1)
                alt = next((q for q in self.cache if first_broken(G, q) is None), None)
                if alt is not None:
                    self.route = alt
                else:
                    o2, extra = self._discover(G)
                    ovh += o2
        if self.route is not None:
            self.age += 1
        return self.route, ovh, extra


class APRACO:
    def __init__(self, p):
        self.p, self.tau = p, {}

    @staticmethod
    def _key(u, v):
        return (u, v) if u < v else (v, u)

    def step(self, G, pos, rng):
        p, tau = self.p, self.tau
        # (1) evaporation on every known edge
        for k in tau:
            tau[k] = max(p.tau_min, (1 - p.rho) * tau[k])
        ovh = 0
        dst_pos = pos[DST]
        for _ in range(p.n_ants):
            path, visited, cur = [SRC], {SRC}, SRC
            while cur != DST and len(path) - 1 < p.max_hops:
                cands = [n for n in G[cur] if n not in visited]
                if not cands:
                    break
                w = np.array([
                    tau.get(self._key(cur, n), p.tau0) ** p.alpha_ph
                    * (1.0 / (1.0 + np.linalg.norm(pos[n] - dst_pos))) ** p.beta_ph
                    * G[cur][n]["ph"] ** p.lq_gamma
                    for n in cands
                ])
                nxt = cands[rng.choice(len(cands), p=w / w.sum())]
                path.append(nxt)
                visited.add(nxt)
                cur = nxt
                ovh += 1                                   # forward-ant transmission
            if cur == DST:
                ovh += len(path) - 1                       # backward ant retraces the path
                edges = list(zip(path[:-1], path[1:]))
                P_k = float(np.prod([G[u][v]["ph"] for u, v in edges]))
                D_k = float(sum(G[u][v]["dly"] for u, v in edges))
                L_k = p.wH * (len(path) - 1) + p.wL * (1 - P_k) * 100 + p.wD * D_k
                dtau = p.Q / max(L_k, 1e-9)
                for u, v in edges:
                    k = self._key(u, v)
                    tau[k] = tau.get(k, p.tau0) + dtau
        # (2) data route: pheromone-weighted Dijkstra over the CURRENT graph G(t)
        if nx.has_path(G, SRC, DST):
            route = nx.shortest_path(
                G, SRC, DST,
                weight=lambda u, v, d: 1.0 / (tau.get(self._key(u, v), p.tau0) + p.eps))
        else:
            route = None
        return route, ovh, 0.0


class OracleSP:
    """CONTROL BASELINE (not in the paper): per-step global shortest path, no ants, 0 overhead.
    weight='hop'   -> hop count recomputed every step (no route caching)
    weight='delay' -> expected hop delay (link-quality aware)"""
    def __init__(self, p, weight="delay"):
        self.weight = weight

    def step(self, G, pos, rng):
        if not nx.has_path(G, SRC, DST):
            return None, 0, 0.0
        w = None if self.weight == "hop" else (lambda u, v, d: d["dly"])
        return nx.shortest_path(G, SRC, DST, weight=w), 0, 0.0


def make_proto(kind, p):
    return {"aodv": AODV, "dsr": DSR, "apr": APRACO,
            "oracle_hop": lambda q: OracleSP(q, "hop"),
            "oracle_delay": lambda q: OracleSP(q, "delay")}[kind](p)


# ----------------------------------------------------------------------------
# Simulation driver
# ----------------------------------------------------------------------------
def deliver(p, G, route, rng, extra_s):
    hops = list(zip(route[:-1], route[1:]))
    ok_n, lat = 0, []
    for _ in range(p.pkts_per_step):
        t, ok = extra_s, True
        for u, v in hops:
            pa = G[u][v]["p"]
            for _a in range(1 + p.max_retx):
                t += p.t_att
                if rng.random() < pa:
                    break
            else:
                ok = False
                break
        if ok:
            ok_n += 1
            lat.append(t * 1000.0)
    return ok_n, lat


def run_protocol(p, world, proto, rng):
    T = p.steps
    out = {k: np.zeros(T) for k in ("deliv", "lat_sum", "lat_n", "ovh", "hops")}
    for t in range(T):
        G = world.graphs[t]
        route, ovh, extra = proto.step(G, world.pos[t], rng)
        out["ovh"][t] = ovh
        if route is None:
            continue
        n_ok, lat = deliver(p, G, route, rng, extra)
        out["deliv"][t], out["lat_sum"][t], out["lat_n"][t] = n_ok, sum(lat), len(lat)
        out["hops"][t] = len(route) - 1
    out["sent"] = np.full(T, p.pkts_per_step, float)
    return out


def _worker(args):
    p, seed, specs = args
    world = World(p, seed)
    res = {"_nlos": world.nlos_frac}
    for name, kind, over in specs:
        q = replace(p, **over)
        res[name] = run_protocol(q, world, make_proto(kind, q), np.random.default_rng(seed * 7919 + 13))
    return res


def run_experiment(p, runs, specs, seed0=1000, workers=1):
    tasks = [(p, seed0 + r, specs) for r in range(runs)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_worker, tasks))
    return [_worker(t) for t in tasks]


def to_frame(p, res, specs) -> pd.DataFrame:
    rows = []
    for r, out in enumerate(res):
        for name, _, _ in specs:
            o = out[name]
            dl, n_lat = o["deliv"].sum(), o["lat_n"].sum()
            rows.append(dict(
                run=r, proto=name, pdr=dl / o["sent"].sum(),
                latency_ms=o["lat_sum"].sum() / n_lat if n_lat > 0 else np.nan,
                throughput_mbps=dl * p.pkt_bytes * 8 / (p.steps * p.dt) / 1e6,
                overhead=o["ovh"].mean(),
                hops=o["hops"][o["hops"] > 0].mean() if (o["hops"] > 0).any() else np.nan))
    return pd.DataFrame(rows)


def series(res, name, what):
    """(runs, steps) array of a per-step metric."""
    rows = []
    for out in res:
        o = out[name]
        if what == "pdr":
            rows.append(o["deliv"] / o["sent"])
        elif what == "lat":
            with np.errstate(invalid="ignore", divide="ignore"):
                rows.append(np.where(o["lat_n"] > 0, o["lat_sum"] / o["lat_n"], np.nan))
        else:
            rows.append(o[what])
    return np.array(rows)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def mean_sd_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, sd, m - h, m + h


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def cliffs_delta(x, y):
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def stats_table(df, metrics, ref="APR-ACO", others=("AODV", "DSR")):
    protos = list(df["proto"].unique())
    rows, p_mwu_all, p_wil_all = [], [], []
    for m in metrics:
        piv = df.pivot(index="run", columns="proto", values=m)
        row = {"metric": m}
        for c in protos:
            mu, sd, lo, hi = mean_sd_ci(piv[c])
            row[c] = f"{mu:.4g} ± {sd:.3g} [{lo:.4g}, {hi:.4g}]"
        groups = [piv[c].dropna().values for c in protos]
        try:
            row["KW_p"] = stats.kruskal(*groups).pvalue
        except ValueError:
            row["KW_p"] = np.nan
        full = piv.dropna()
        try:
            row["Friedman_p(paired)"] = stats.friedmanchisquare(*[full[c] for c in protos]).pvalue
        except ValueError:
            row["Friedman_p(paired)"] = np.nan
        for o in others:
            x, y = piv[ref].dropna().values, piv[o].dropna().values
            try:
                pm = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                d = cliffs_delta(x, y)
            except ValueError:
                pm, d = np.nan, np.nan
            both = piv[[ref, o]].dropna()
            try:
                pw = stats.wilcoxon(both[ref], both[o]).pvalue
            except ValueError:
                pw = np.nan
            row[f"MWU_p_vs_{o}"], row[f"Wilcoxon_p_vs_{o}"], row[f"CliffD_vs_{o}"] = pm, pw, d
            p_mwu_all.append(pm)
            p_wil_all.append(pw)
        rows.append(row)
    t = pd.DataFrame(rows)
    # Holm correction across all (metric x baseline) paired tests
    k = len(others)
    adj = holm(np.nan_to_num(p_wil_all, nan=1.0))
    for j, o in enumerate(others):
        t[f"Wilcoxon_Holm_vs_{o}"] = [adj[i * k + j] for i in range(len(metrics))]
    return t


def fmt_table(df: pd.DataFrame) -> str:
    with pd.option_context("display.max_columns", None, "display.width", 250,
                           "display.float_format", lambda v: f"{v:.3g}"):
        return df.to_string(index=False)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
COLORS = {"APR-ACO": "#d62728", "AODV": "#1f77b4", "DSR": "#2ca02c",
          "Oracle-hop": "#7f7f7f", "Oracle-delay": "#9467bd"}


def band(ax, y, label, color):
    m, s = np.nanmean(y, 0), np.nanstd(y, 0, ddof=1)
    x = np.arange(1, len(m) + 1)
    ax.plot(x, m, label=label, color=color)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  saved", path)


def protocol_specs(oracle: bool):
    s = [("APR-ACO", "apr", {}), ("AODV", "aodv", {}), ("DSR", "dsr", {})]
    if oracle:
        s += [("Oracle-hop", "oracle_hop", {}), ("Oracle-delay", "oracle_delay", {})]
    return s


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def cmd_main(p, a):
    specs = protocol_specs(a.oracle)
    runs = a.runs
    print(f"[main] {runs} runs x {p.steps} steps, {p.n_uav} UAVs")
    res = run_experiment(p, runs, specs, a.seed, a.workers)
    df = to_frame(p, res, specs)
    df.to_csv(os.path.join(a.out, "main_runs.csv"), index=False)
    others = tuple(n for n, _, _ in specs[1:])
    tab = stats_table(df, ["pdr", "latency_ms", "throughput_mbps", "overhead"], others=others)
    tab.to_csv(os.path.join(a.out, "table3_main.csv"), index=False)
    print("\nTable 3 (mean ± SD [95% CI]; p-values)\n" + fmt_table(tab))
    nl = np.mean([r["_nlos"] for r in res])
    print(f"\nShare of in-range links that are NLOS-obstructed (buildings): {100 * nl:.3f} %")

    for what, ylabel, fname, title in (
            ("pdr", "PDR", "fig1_pdr.png", "Fig.1 PDR per step"),
            ("lat", "Latency (ms)", "fig2_latency.png", "Fig.2 Latency per step"),
            ("ovh", "Control packets / step", "fig3_overhead.png", "Fig.3 Overhead per step")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for n, _, _ in specs:
            band(ax, series(res, n, what), n, COLORS.get(n))
        ax.set(xlabel="Step", ylabel=ylabel, title=title)
        ax.legend()
        save(fig, a.out, fname)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, m, lab in zip(axs, ("pdr", "latency_ms", "overhead"), ("PDR", "Latency (ms)", "Overhead")):
        ax.boxplot([df[df.proto == n][m].dropna() for n, _, _ in specs],
                   tick_labels=[n for n, _, _ in specs])
        ax.set_title(lab)
    save(fig, a.out, "fig4_distributions.png")

    # --- Section 5.3: PDR-latency relationship (APR-ACO step series) ---
    pdr_t = np.nanmean(series(res, "APR-ACO", "pdr"), 0)
    lat_t = np.nanmean(series(res, "APR-ACO", "lat"), 0)
    ok = ~np.isnan(pdr_t) & ~np.isnan(lat_t)
    pr, pp = stats.pearsonr(pdr_t[ok], lat_t[ok])
    sr, sp = stats.spearmanr(pdr_t[ok], lat_t[ok])
    print(f"\nSection 5.3  step-series (n={ok.sum()}): Pearson r={pr:.3f} (p={pp:.3f}); "
          f"Spearman r={sr:.3f} (p={sp:.3f})")
    # run-level view (independent observations, unlike the autocorrelated step series)
    sub = df[df.proto == "APR-ACO"].dropna()
    rr, rp = stats.spearmanr(sub["pdr"], sub["latency_ms"])
    print(f"             run-level (n={len(sub)}):        Spearman r={rr:.3f} (p={rp:.3f})")
    return res


def cmd_topology(p, a):
    seed = a.seed
    w = World(p, seed)
    proto = APRACO(p)
    rng = np.random.default_rng(seed)
    t_show, route = None, None
    for t in range(p.steps):
        r, _, _ = proto.step(w.graphs[t], w.pos[t], rng)
        if r is not None and len(r) >= 4:
            t_show, route = t, r
    if route is None:
        print("[topology] no connected step found; try another --seed")
        return
    G, P = w.graphs[t_show], w.pos[t_show]
    fig, ax = plt.subplots(figsize=(7, 7))
    for lo, hi in zip(w.bmin, w.bmax):
        ax.add_patch(plt.Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], color="#bbbbbb"))
    for u, v in G.edges:
        ax.plot(*zip(P[u, :2], P[v, :2]), color="#cccccc", lw=0.6, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], s=14, color="k", zorder=3)
    ax.plot(P[route, 0], P[route, 1], "r-", lw=2.2, zorder=4, label=f"APR-ACO route ({len(route) - 1} hops)")
    ax.scatter(*P[SRC, :2], marker="s", s=90, color="b", zorder=5, label="Source")
    ax.scatter(*P[DST, :2], marker="^", s=110, color="g", zorder=5, label="Destination")
    ax.set(xlim=(0, p.area), ylim=(0, p.area), aspect="equal", title=f"Fig.5 topology, step {t_show + 1}")
    ax.legend(loc="upper left")
    save(fig, a.out, "fig5_topology.png")


def cmd_scalability(p, a):
    specs = protocol_specs(a.oracle)
    runs = a.runs or 6
    sizes = [10, 25, 50, 75, 100, 150]
    rows = []
    print(f"[scalability] {runs} runs x 20 steps for N in {sizes}")
    for n in sizes:
        q = replace(p, n_uav=n, steps=20)
        df = to_frame(q, run_experiment(q, runs, specs, a.seed, a.workers), specs)
        g = df.groupby("proto")
        for proto, sub in g:
            rows.append(dict(
                n_uav=n, proto=proto,
                pdr=f"{sub.pdr.mean():.3f} ± {sub.pdr.std(ddof=1):.2f}",
                latency_ms=round(sub.latency_ms.mean(), 1) if sub.latency_ms.notna().any() else "n/a",
                overhead=round(sub.overhead.mean(), 1),
                _pdr=sub.pdr.mean(), _lat=sub.latency_ms.mean(), _ovh=sub.overhead.mean(),
                _pdr_sd=sub.pdr.std(ddof=1), _lat_sd=sub.latency_ms.std(ddof=1), _ovh_sd=sub.overhead.std(ddof=1)))
        print("  done N =", n)
    t = pd.DataFrame(rows)
    t.drop(columns=[c for c in t if c.startswith("_")]).to_csv(os.path.join(a.out, "table4_scalability.csv"), index=False)
    print("\nTable 4\n" + fmt_table(t[["n_uav", "proto", "pdr", "latency_ms", "overhead"]]))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, lab in zip(axs, ("pdr", "lat", "ovh"), ("PDR", "Latency (ms)", "Overhead (pkts/step)")):
        for n, _, _ in specs:
            s = t[t.proto == n]
            ax.errorbar(s.n_uav, s[f"_{key}"], s[f"_{key}_sd"], marker="o", capsize=3, label=n, color=COLORS.get(n))
        ax.set(xlabel="UAVs", ylabel=lab)
    axs[0].legend()
    save(fig, a.out, "fig6_scalability.png")


def cmd_ablation(p, a):
    runs = a.runs or 12
    q = replace(p, steps=40)
    specs = [("Full APR-ACO", "apr", {}), ("No evaporation (rho=0)", "apr", {"rho": 0.0}),
             ("No loss term (wL=0)", "apr", {"wL": 0.0}), ("No delay term (wD=0)", "apr", {"wD": 0.0}),
             ("1 ant/step", "apr", {"n_ants": 1})]
    if a.oracle:
        specs += [("Oracle-delay (no ants)", "oracle_delay", {}), ("Oracle-hop (no ants)", "oracle_hop", {}),
                  ("AODV", "aodv", {})]
    print(f"[ablation] {runs} runs x 40 steps")
    df = to_frame(q, run_experiment(q, runs, specs, a.seed, a.workers), specs)
    full = df[df.proto == "Full APR-ACO"].set_index("run")
    rows = []
    for n, _, _ in specs:
        sub = df[df.proto == n].set_index("run")
        row = dict(variant=n)
        for m in ("pdr", "latency_ms", "overhead"):
            row[m] = f"{sub[m].mean():.3f} ± {sub[m].std(ddof=1):.2f}"
        if n != "Full APR-ACO":      # paired test against the full protocol (same worlds)
            for m in ("pdr", "latency_ms"):
                try:
                    row[f"Wilcoxon_p_{m}"] = stats.wilcoxon(sub[m], full[m]).pvalue
                except ValueError:
                    row[f"Wilcoxon_p_{m}"] = np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "table5_ablation.csv"), index=False)
    print("\nTable 5 (the paper reports NO significance test here; paired Wilcoxon p added)\n" + fmt_table(t))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m, lab in zip(axs, ("pdr", "latency_ms", "overhead"), ("PDR", "Latency (ms)", "Overhead")):
        vals = [df[df.proto == n][m] for n, _, _ in specs]
        ax.bar(range(len(specs)), [v.mean() for v in vals], yerr=[v.std(ddof=1) for v in vals], capsize=3)
        ax.set_xticks(range(len(specs)))
        ax.set_xticklabels([n for n, _, _ in specs], rotation=35, ha="right", fontsize=8)
        ax.set_title(lab)
    save(fig, a.out, "fig7_ablation.png")


def cmd_sensitivity(p, a):
    runs = a.runs or 8
    q = replace(p, steps=35)
    rhos, ants = [0.1, 0.2, 0.35, 0.5, 0.7], [1, 3, 6, 10, 15]
    specs = [(f"rho={r}", "apr", {"rho": r}) for r in rhos] + [(f"Na={k}", "apr", {"n_ants": k}) for k in ants]
    print(f"[sensitivity] {runs} runs x 35 steps")
    df = to_frame(q, run_experiment(q, runs, specs, a.seed, a.workers), specs)
    g = df.groupby("proto")[["pdr", "latency_ms", "overhead"]].mean()
    t6 = g.loc[[f"rho={r}" for r in rhos]]
    t7 = g.loc[[f"Na={k}" for k in ants]]
    t6.to_csv(os.path.join(a.out, "table6_rho.csv"))
    t7.to_csv(os.path.join(a.out, "table7_ants.csv"))
    print("\nTable 6 (rho)\n" + fmt_table(t6.reset_index()))
    print("\nTable 7 (N_a)\n" + fmt_table(t7.reset_index()))
    print("\nNOTE: rho=0.35 (Table 6) and Na=6 (Table 7) are the SAME configuration -> identical rows here "
          "because the runs share seeds; in the paper they differ (PDR 0.934 vs 0.886).")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    ax.plot(rhos, t6.pdr, "o-", color="C0")
    ax.set(xlabel="rho", ylabel="PDR")
    ax2 = ax.twinx()
    ax2.plot(rhos, t6.latency_ms, "s--", color="C3")
    ax2.set_ylabel("Latency (ms)")
    ax = axs[1]
    ax.plot(ants, t7.overhead, "o-", color="C1")
    ax.set(xlabel="N_a", ylabel="Overhead")
    ax2 = ax.twinx()
    ax2.plot(ants, t7.pdr, "s--", color="C0")
    ax2.set_ylabel("PDR")
    save(fig, a.out, "fig8_sensitivity.png")


def cmd_lq(p, a):
    """Tune the link-quality exponent on TUNE seeds, then test once on disjoint TEST seeds."""
    tune_seed, test_seed = a.seed, a.seed + 4000
    runs_t, runs_f = 12, a.runs
    gammas = [0.0, 1.0, 2.0, 4.0, 8.0]
    specs = [(f"gamma={g}", "apr", {"lq_gamma": g}) for g in gammas]
    print(f"[lq] TUNING on seeds {tune_seed}..{tune_seed + runs_t - 1}")
    df = to_frame(p, run_experiment(p, runs_t, specs, tune_seed, a.workers), specs)
    g = df.groupby("proto")[["pdr", "latency_ms", "overhead"]].mean().loc[[n for n, _, _ in specs]]
    print(fmt_table(g.reset_index()))
    best = float(g["pdr"].idxmax().split("=")[1])
    print(f"-> selected gamma = {best} (max PDR on tuning seeds)")
    specs = [("APR-ACO (paper)", "apr", {}), (f"APR-ACO-LQ (gamma={best})", "apr", {"lq_gamma": best}),
             ("AODV", "aodv", {}), ("DSR", "dsr", {}), ("Oracle-delay", "oracle_delay", {})]
    print(f"\n[lq] FINAL TEST on unseen seeds {test_seed}..{test_seed + runs_f - 1}")
    res = run_experiment(p, runs_f, specs, test_seed, a.workers)
    df = to_frame(p, res, specs)
    df.to_csv(os.path.join(a.out, "lq_test_runs.csv"), index=False)
    tab = stats_table(df, ["pdr", "latency_ms", "overhead"], ref=specs[1][0],
                      others=("APR-ACO (paper)", "AODV", "DSR", "Oracle-delay"))
    tab.to_csv(os.path.join(a.out, "lq_test_table.csv"), index=False)
    cols = ["metric"] + [n for n, _, _ in specs]
    print(fmt_table(tab[cols]))
    print(fmt_table(tab[["metric"] + [c for c in tab.columns if c.startswith("Wilcoxon_Holm")]]))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m, lab in zip(axs, ("pdr", "latency_ms", "overhead"), ("PDR", "Latency (ms)", "Overhead")):
        ax.boxplot([df[df.proto == n][m].dropna() for n, _, _ in specs], tick_labels=[n.replace(" (", "\n(") for n, _, _ in specs])
        ax.set_title(lab)
        ax.tick_params(axis="x", labelsize=7)
    save(fig, a.out, "fig9_lq_test.png")


# ----------------------------------------------------------------------------
# Sanity checks + audit of the numbers printed in the paper
# ----------------------------------------------------------------------------
def cmd_checks(p, a):
    def chk(ok, label, detail=""):
        print(("  [OK]   " if ok else "  [FAIL] ") + label + (f"  ->  {detail}" if detail else ""))

    print("\n=== A. Model consistency (paper's own parameters) ===")
    buildings_matter = p.alt_min <= p.bld_height[1]
    chk(buildings_matter,
        f"UAV altitude ({p.alt_min:.0f}-{p.alt_max:.0f} m) vs building height ({p.bld_height[0]:.0f}-{p.bld_height[1]:.0f} m)",
        "buildings can block links" if buildings_matter else
        "all UAVs fly ABOVE every building => no link can ever be blocked => the NLOS model is dead code")
    for lo, hi, tag in ((p.alt_min, p.alt_max, "paper altitude"), (10, 35, "low altitude 10-35 m")):
        q = replace(p, alt_min=lo, alt_max=hi, steps=10)
        fr = np.mean([World(q, 500 + s).nlos_frac for s in range(4)])
        print(f"        measured NLOS share of in-range links, {tag:>22}: {100 * fr:6.2f} %")
    bits = p.pkts_per_step * p.pkt_bytes * 8
    print(f"  offered load = {bits} bit/step = {bits / p.dt / 1e6:.3f} Mbps  "
          f"=> throughput = PDR x {bits / p.dt / 1e6:.3f}  (not an independent metric)")
    print(f"  paper Table 3: 0.889 x {bits / p.dt / 1e6:.3f} = {0.889 * bits / p.dt / 1e6:.4f} Mbps (paper: 0.0853) -> throughput is just PDR rescaled")
    fspl300 = 20 * math.log10(300) + 20 * math.log10(p.freq_hz) - 147.55
    print(f"  link budget at 300 m: Rx = {p.tx_dbm - fspl300:.1f} dBm (margin {p.tx_dbm - fspl300 - p.sens_dbm:+.1f} dB); "
          f"with 22 dB NLOS penalty a link survives only within ~{10 ** ((p.tx_dbm - p.sens_dbm - p.nlos_db - 47.9) / 20):.0f} m")

    print("\n=== B. Arithmetic of numbers printed in the paper ===")
    n = 30
    tc = stats.t.ppf(0.975, n - 1)
    for lab, m, sd, ci, tol in (("PDR APR-ACO", .889, .108, (.849, .929), .002),
                                ("Latency APR-ACO", 55.24, 9.77, (51.59, 58.89), .05),
                                ("Latency AODV", 73.44, 13.50, (68.39, 78.48), .05),
                                ("Latency DSR", 69.41, 12.54, (64.73, 74.09), .05)):
        h = tc * sd / math.sqrt(n)
        chk(abs(m - h - ci[0]) < tol and abs(m + h - ci[1]) < tol, f"95% CI {lab}", f"recomputed [{m - h:.3f}, {m + h:.3f}] vs paper {ci}")
    chk(abs((55.24 - 73.44) / 73.44 + 0.248) < .002 and abs((55.24 - 69.41) / 69.41 + 0.204) < .002, "latency reductions -24.8% / -20.4%")
    chk(abs(97.09 / 22.02 - 4.4) < .05 and abs(97.09 / 18.33 - 5.3) < .05, "overhead ratios 4.4x / 5.3x")
    nn = 50
    for lab, r, pp in (("Pearson", .008, .96), ("Spearman", -.058, .69)):
        tt = r * math.sqrt((nn - 2) / (1 - r * r))
        pv = 2 * stats.t.sf(abs(tt), nn - 2)
        chk(abs(pv - pp) < .01, f"{lab} r={r} -> p (n=50)", f"recomputed p={pv:.3f}")
    pmin = stats.mannwhitneyu(np.arange(30), np.arange(100, 130)).pvalue
    chk(True, "MWU p=3.02e-11 (overhead) is the smallest p possible for 30 vs 30 (complete separation)", f"scipy gives {pmin:.2e}")

    print("\n=== C. Statements in the text that the paper's own tables contradict ===")
    t4 = {50: dict(apr=54.8, aodv=78.5, dsr=74.0), 150: dict(apr=49.6, aodv=83.5, dsr=76.0)}
    for N, d in t4.items():
        print(f"        Table 4, N={N}: APR-AODV = {d['apr'] - d['aodv']:+.1f} ms, APR-DSR = {d['apr'] - d['dsr']:+.1f} ms")
    chk(False, "Sec 5.4 text: 'from -4.4 ms vs DSR at 50 UAVs to -26.4 ms vs AODV at 150'",
        "Table 4 gives -19.2 (DSR) / -23.7 (AODV) at 50, and -26.4 (DSR) / -33.9 (AODV) at 150; -4.4 is the AODV-DSR gap, and -26.4 is vs DSR")
    pdr6, lat6 = [.940, .944, .934, .913, .901], [48.97, 49.05, 52.25, 56.81, 61.78]
    chk(all(np.diff(pdr6) <= 0) and all(np.diff(lat6) >= 0), "Sec 5.6: 'monotonically improve both PDR and latency' as rho falls",
        f"PDR rho 0.1->0.2 rises ({pdr6[0]}->{pdr6[1]}), latency 48.97->49.05 rises: not monotone")
    chk(False, "Table 6 (rho=0.35) vs Table 7 (Na=6): same config (rho=.35, Na=6, 8 runs, 35 steps)",
        "PDR 0.934 vs 0.886, latency 52.25 vs 50.47, overhead ~100-102 vs 92.78")
    chk(False, "Same config across studies: main 0.889/55.2 ms, ablation 0.890/59.2 ms, sens. 0.934/52.3 & 0.886/50.5",
        "run-to-run noise (~±0.05 PDR, ~±4 ms) is as large as the effects reported in Tables 5-7, and no test is given there")
    chk(abs((104.89 - 17.60) / 104.89 - .83) < .005, "ablation: 83% overhead reduction with 1 ant")
    chk(False, "Abstract: 'A previously assumed inverse relationship between PDR and latency (Pearson r = 0.01, p = 0.96).'",
        "sentence has no verb / conclusion (incomplete)")
    chk(False, "Sec 3.2: 'see the sensitivity analysis later in this section' for the weights",
        "Sec 5.6 sweeps only rho and Na; w_H, w_L, w_D, Q, alpha, beta are never explored")
    chk(False, "Sec 3.1 cites [12] for 3D Gauss-Markov; Table 1 cites [12] for geographic routing",
        "[12] is a 2012 Zone Routing Protocol paper")
    chk(False, "References [15]-[30] are never cited in the text", "16 uncited references")
    print()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["checks", "main", "topology", "scalability", "ablation", "sensitivity", "lq", "all"],
                    help="what to run (default: all)")
    ap.add_argument("--runs", type=int, default=None, help="Monte-Carlo runs (default: paper's value per experiment)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--uavs", type=int, default=None)
    ap.add_argument("--alt-min", type=float, default=None)
    ap.add_argument("--alt-max", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True,
                    help="include per-step global shortest-path control baselines (default on; --no-oracle to disable)")
    ap.add_argument("--scenario", choices=["urban", "paper"], default="urban",
                    help="urban: 100 UAVs at 15-80 m (buildings matter, default); paper: 50 UAVs at 50-150 m (buildings never block)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="fewer runs for a fast smoke test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p = Params()
    if a.scenario == "urban":
        p = replace(p, n_uav=100, alt_min=15.0, alt_max=80.0)
    over = {k: v for k, v in (("steps", a.steps), ("n_uav", a.uavs), ("alt_min", a.alt_min), ("alt_max", a.alt_max)) if v is not None}
    p = replace(p, **over)
    if a.runs is None:
        a.runs = 8 if a.quick else 30
    aux_runs = a.runs if (a.runs != 30 or a.quick) else None   # let aux experiments use their own defaults

    t0 = time.time()
    if a.cmd == "checks":
        cmd_checks(p, a)
    if a.cmd in ("main", "all"):
        cmd_main(p, a)
    if a.cmd in ("topology", "all"):
        cmd_topology(p, a)
    sub = argparse.Namespace(**{**vars(a), "runs": aux_runs if a.quick else None})
    if a.cmd in ("scalability", "all"):
        cmd_scalability(p, sub)
    if a.cmd in ("ablation", "all"):
        cmd_ablation(p, sub)
    if a.cmd in ("sensitivity", "all"):
        cmd_sensitivity(p, sub)
    if a.cmd in ("lq", "all"):
        cmd_lq(p, a)
    print(f"\nFinished in {time.time() - t0:.0f} s. Output folder: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
