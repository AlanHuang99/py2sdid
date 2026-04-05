"""
Validation: Compare ts_did vs bjs_did (internal) and Python vs R (external).

Generates synthetic panel data, runs all estimators, and compares:
  1. Point estimates (ts_did vs bjs_did — must be identical)
  2. Standard errors (ts_did vs bjs_did — should be very close)
  3. Point estimates vs R did2s (should match to 4+ decimals)
  4. Point estimates vs R didimputation (should match to 4+ decimals)
  5. Standard errors vs R (should be within a few percent)

Usage:
    uv run python benchmarks/validation.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

# Add parent to path for local dev
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import py2sdid
from conftest import gen_data


SEED = 42
SEPARATOR = "=" * 70


def find_r():
    for r_bin in ["/usr/bin/R", "/opt/R/current/bin/R", "R"]:
        try:
            result = subprocess.run(
                [r_bin, "-e", 'library(did2s); library(didimputation); cat("OK")'],
                capture_output=True, text=True, timeout=30,
            )
            if "OK" in result.stdout:
                return r_bin
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def run_r(r_bin, script):
    result = subprocess.run(
        [r_bin, "--vanilla", "--slave", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed:\n{result.stderr}")
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            return json.loads(line)
    raise RuntimeError(f"No JSON in R output:\n{result.stdout}")


def main():
    print(SEPARATOR)
    print("py2sdid Validation Benchmark")
    print(SEPARATOR)

    # ── Generate data ──────────────────────────────────────────────────
    configs = [
        {"name": "Homogeneous (te=2)", "n": 900, "te1": 2.0, "te2": 2.0,
         "te_m1": 0.0, "te_m2": 0.0},
        {"name": "Heterogeneous (te1=2, te2=3)", "n": 900, "te1": 2.0, "te2": 3.0,
         "te_m1": 0.0, "te_m2": 0.0},
        {"name": "Dynamic (te_m=0.2)", "n": 900, "te1": 1.0, "te2": 1.0,
         "te_m1": 0.2, "te_m2": 0.2},
        {"name": "Large (n=3000)", "n": 3000, "te1": 2.0, "te2": 2.0,
         "te_m1": 0.0, "te_m2": 0.0},
    ]

    r_bin = find_r()
    has_r = r_bin is not None

    for cfg in configs:
        print(f"\n{'─' * 70}")
        print(f"Config: {cfg['name']}")
        print(f"{'─' * 70}")

        df = gen_data(
            n=cfg["n"], te1=cfg["te1"], te2=cfg["te2"],
            te_m1=cfg["te_m1"], te_m2=cfg["te_m2"], seed=SEED,
        )

        # ── 1. ts_did vs bjs_did point estimates ───────────────────────
        t0 = time.perf_counter()
        r_ts = py2sdid.ts_did(df, yname="dep_var", idname="unit", tname="year",
                              gname="g", cluster_var="state", se=True, verbose=False)
        t_ts = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_bjs = py2sdid.bjs_did(df, yname="dep_var", idname="unit", tname="year",
                                gname="g", cluster_var="state", se=True, verbose=False)
        t_bjs = time.perf_counter() - t0

        att_diff = abs(r_ts.att_avg - r_bjs.att_avg)
        se_ratio = r_ts.att_avg_se / r_bjs.att_avg_se if r_bjs.att_avg_se > 0 else float("nan")

        print(f"\n  ts_did:  ATT={r_ts.att_avg:+.6f}  SE={r_ts.att_avg_se:.6f}  ({t_ts:.3f}s)")
        print(f"  bjs_did: ATT={r_bjs.att_avg:+.6f}  SE={r_bjs.att_avg_se:.6f}  ({t_bjs:.3f}s)")
        print(f"  Point estimate diff: {att_diff:.2e}  {'PASS' if att_diff < 1e-10 else 'FAIL'}")
        print(f"  SE ratio (ts/bjs):   {se_ratio:.6f}  {'PASS' if 0.8 < se_ratio < 1.2 else 'WARN'}")

        # ── 2. Event study comparison ──────────────────────────────────
        r_ts_es = py2sdid.ts_did(df, yname="dep_var", idname="unit", tname="year",
                                 gname="g", cluster_var="state",
                                 se=True, verbose=False)
        r_bjs_es = py2sdid.bjs_did(df, yname="dep_var", idname="unit", tname="year",
                                   gname="g", cluster_var="state",
                                   se=True, verbose=False)

        h_ts = r_ts_es.att_by_horizon
        h_bjs = r_bjs_es.att_by_horizon
        max_est_diff = 0.0
        se_ratios = []
        for row_ts, row_bjs in zip(h_ts.iter_rows(named=True), h_bjs.iter_rows(named=True)):
            est_diff = abs(row_ts["estimate"] - row_bjs["estimate"])
            max_est_diff = max(max_est_diff, est_diff)
            if row_ts["se"] and row_bjs["se"] and row_bjs["se"] > 0:
                se_ratios.append(row_ts["se"] / row_bjs["se"])

        print(f"\n  Event-study (ts vs bjs):")
        print(f"    Max |est diff|:   {max_est_diff:.2e}  {'PASS' if max_est_diff < 1e-10 else 'FAIL'}")
        if se_ratios:
            print(f"    Median SE ratio:  {np.median(se_ratios):.6f}")
            print(f"    SE ratio range:   [{min(se_ratios):.4f}, {max(se_ratios):.4f}]")

        # ── 3. Python vs R ────────────────────────────────────────────
        if has_r:
            csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
            df.write_csv(csv_path)

            try:
                r_did2s = run_r(r_bin, f"""
                    library(did2s); library(jsonlite)
                    df <- read.csv("{csv_path}")
                    df$treat <- as.integer(df$g > 0 & df$year >= df$g)
                    est <- did2s(df, yname="dep_var",
                        first_stage = ~ 0 | unit + year,
                        second_stage = ~ i(treat, ref=FALSE),
                        treatment="treat", cluster_var="state", verbose=FALSE)
                    cat(toJSON(list(att=unbox(coef(est)[[1]]),
                        se=unbox(sqrt(diag(est$cov.scaled))[[1]]))))
                """)

                r_bjs_r = run_r(r_bin, f"""
                    library(didimputation); library(jsonlite)
                    df <- read.csv("{csv_path}")
                    est <- did_imputation(data=df, yname="dep_var", gname="g",
                        tname="year", idname="unit", cluster_var="state")
                    cat(toJSON(list(att=unbox(est$estimate),
                        se=unbox(est$std.error))))
                """)

                py_vs_r_did2s = abs(r_ts.att_avg - r_did2s["att"])
                py_vs_r_bjs = abs(r_bjs.att_avg - r_bjs_r["att"])
                se_vs_r_did2s = r_ts.att_avg_se / r_did2s["se"] if r_did2s["se"] > 0 else float("nan")
                se_vs_r_bjs = r_bjs.att_avg_se / r_bjs_r["se"] if r_bjs_r["se"] > 0 else float("nan")

                print(f"\n  Python vs R:")
                print(f"    ATT vs R did2s:          diff={py_vs_r_did2s:.2e}  {'PASS' if py_vs_r_did2s < 0.01 else 'FAIL'}")
                print(f"    ATT vs R didimputation:  diff={py_vs_r_bjs:.2e}  {'PASS' if py_vs_r_bjs < 0.01 else 'FAIL'}")
                print(f"    SE ratio vs R did2s:     {se_vs_r_did2s:.4f}")
                print(f"    SE ratio vs R bjs:       {se_vs_r_bjs:.4f}")
            except Exception as e:
                print(f"\n  R comparison failed: {e}")
            finally:
                import os
                os.unlink(csv_path)
        else:
            print("\n  [R not available — skipping R comparison]")

    print(f"\n{SEPARATOR}")
    print("Validation complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
