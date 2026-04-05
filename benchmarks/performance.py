"""
Performance profiling: identify bottlenecks at scale.

Tests with increasing panel sizes to measure time and memory for each
pipeline stage. Outputs an HTML report.

Usage:
    uv run python benchmarks/performance.py
"""

import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from conftest import gen_data
from py2sdid.panel import prepare_panel
from py2sdid.first_stage import estimate_first_stage
from py2sdid.effects import compute_effects
from py2sdid.inference import compute_se_did2s, compute_se_bjs
import py2sdid

SEP = "=" * 72


def profile_stage(name, fn):
    """Run fn, return (result, elapsed_sec, peak_mb)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024
    return result, elapsed, peak_mb


def profile_panel_size(n, n_periods=31, seed=42):
    """Profile all stages for a given panel size."""
    print(f"\n{'─' * 72}")
    print(f"  N={n:,} units, T={n_periods}, obs={n * n_periods:,}")
    print(f"{'─' * 72}")

    # Generate data
    _, t_gen, m_gen = profile_stage("gen_data", lambda: gen_data(
        n=n, te1=2.0, te2=3.0, seed=seed, panel=(1990, 1990 + n_periods - 1)))
    print(f"  gen_data:        {t_gen:7.2f}s  {m_gen:7.1f} MB")

    df = gen_data(n=n, te1=2.0, te2=3.0, seed=seed, panel=(1990, 1990 + n_periods - 1))

    # Stage 1: prepare_panel
    panel, t1, m1 = profile_stage("prepare_panel", lambda: prepare_panel(
        df, yname="dep_var", idname="unit", tname="year", gname="g", cluster_var="state"))
    print(f"  prepare_panel:   {t1:7.2f}s  {m1:7.1f} MB")

    # Stage 2: first_stage
    fs, t2, m2 = profile_stage("first_stage", lambda: estimate_first_stage(panel))
    print(f"  first_stage:     {t2:7.2f}s  {m2:7.1f} MB")

    # Stage 3: compute_effects
    eff, t3, m3 = profile_stage("effects", lambda: compute_effects(panel, fs))
    n_horizons = len(eff.horizons)
    print(f"  compute_effects: {t3:7.2f}s  {m3:7.1f} MB  ({n_horizons} horizons)")

    # Stage 4a: did2s SEs
    _, t4a, m4a = profile_stage("se_did2s", lambda: compute_se_did2s(panel, fs, eff))
    print(f"  se_did2s:        {t4a:7.2f}s  {m4a:7.1f} MB")

    # Stage 4b: BJS SEs
    _, t4b, m4b = profile_stage("se_bjs", lambda: compute_se_bjs(panel, fs, eff))
    print(f"  se_bjs:          {t4b:7.2f}s  {m4b:7.1f} MB")

    # Full pipeline
    _, t_full_ts, m_full_ts = profile_stage("ts_did", lambda: py2sdid.ts_did(
        df, yname="dep_var", idname="unit", tname="year", gname="g",
        cluster_var="state", verbose=False))
    print(f"  ts_did (total):  {t_full_ts:7.2f}s  {m_full_ts:7.1f} MB")

    _, t_full_bjs, m_full_bjs = profile_stage("bjs_did", lambda: py2sdid.bjs_did(
        df, yname="dep_var", idname="unit", tname="year", gname="g",
        cluster_var="state", verbose=False))
    print(f"  bjs_did (total): {t_full_bjs:7.2f}s  {m_full_bjs:7.1f} MB")

    # No-SE path
    _, t_no_se, _ = profile_stage("ts_did(se=F)", lambda: py2sdid.ts_did(
        df, yname="dep_var", idname="unit", tname="year", gname="g",
        cluster_var="state", se=False, verbose=False))
    print(f"  ts_did(se=F):    {t_no_se:7.2f}s")

    return {
        "n": n, "T": n_periods, "obs": n * n_periods, "horizons": n_horizons,
        "prepare_panel": t1, "first_stage": t2, "effects": t3,
        "se_did2s": t4a, "se_bjs": t4b,
        "ts_did": t_full_ts, "bjs_did": t_full_bjs, "no_se": t_no_se,
        "mem_did2s": m4a, "mem_bjs": m4b, "mem_total_ts": m_full_ts,
    }


def main():
    print(SEP)
    print("  py2sdid Performance Profile")
    print(SEP)

    sizes = [500, 1000, 3000, 5000, 10000, 20000]
    results = []
    for n in sizes:
        try:
            r = profile_panel_size(n)
            results.append(r)
        except Exception as e:
            print(f"  FAILED at N={n}: {e}")
            break

    # Summary table
    print(f"\n{SEP}")
    print("  Summary")
    print(SEP)
    print(f"{'N':>8s} {'obs':>10s} {'horizons':>8s} "
          f"{'panel':>7s} {'1st stg':>7s} {'effects':>7s} "
          f"{'se_ts':>7s} {'se_bjs':>7s} {'TOTAL':>7s} "
          f"{'mem_ts':>7s}")
    print(f"{'─'*8} {'─'*10} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7}")
    for r in results:
        print(f"{r['n']:8,d} {r['obs']:10,d} {r['horizons']:8d} "
              f"{r['prepare_panel']:7.2f} {r['first_stage']:7.2f} {r['effects']:7.2f} "
              f"{r['se_did2s']:7.2f} {r['se_bjs']:7.2f} {r['ts_did']:7.2f} "
              f"{r['mem_total_ts']:6.1f}M")

    # Identify bottlenecks
    if results:
        last = results[-1]
        stages = [
            ("prepare_panel", last["prepare_panel"]),
            ("first_stage", last["first_stage"]),
            ("effects", last["effects"]),
            ("se_did2s", last["se_did2s"]),
            ("se_bjs", last["se_bjs"]),
        ]
        stages.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBottleneck at N={last['n']:,d}: {stages[0][0]} ({stages[0][1]:.2f}s)")
        for name, t in stages:
            pct = t / last["ts_did"] * 100 if last["ts_did"] > 0 else 0
            print(f"  {name:20s} {t:7.2f}s  ({pct:5.1f}%)")

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
