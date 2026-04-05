"""
Full analysis pipeline: exercises every feature of py2sdid on the test fixture.

Runs both estimators, compares point estimates and SEs, event study,
diagnostics, and generates all plots.  Outputs an HTML report.

Usage:
    uv run python benchmarks/full_analysis.py
"""

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import py2sdid

FIXTURE = Path(__file__).parent.parent / "tests" / "data" / "staggered_panel.parquet"
OUTPUT = Path(__file__).parent / "full_analysis_report.html"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def main():
    html = []
    html.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>py2sdid Full Analysis Report</title>
<style>
body { font-family: 'Menlo', 'Consolas', monospace; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #333; font-size: 13px; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 40px; }
table { border-collapse: collapse; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: right; font-size: 12px; }
th { background: #f5f5f5; }
.pass { color: #2a7; font-weight: bold; }
.fail { color: #c33; font-weight: bold; }
.note { color: #888; font-style: italic; }
pre { background: #f8f8f8; padding: 12px; overflow-x: auto; font-size: 12px; }
.img-row { display: flex; gap: 20px; flex-wrap: wrap; }
.img-row img { max-width: 48%; }
</style>
</head><body>""")

    html.append("<h1>py2sdid Full Analysis Report</h1>")

    # ── Load data ──────────────────────────────────────────────────────
    df = pl.read_parquet(FIXTURE)
    cohorts = df.filter(pl.col("g") > 0).group_by("g").agg(
        pl.col("unit").n_unique().alias("n_units")).sort("g")
    n_never = df.filter(pl.col("g") == 0)["unit"].n_unique()

    html.append("<h2>Data</h2>")
    html.append(f"<p>Fixture: <code>{FIXTURE.name}</code> &mdash; "
                f"{df.shape[0]:,} rows, {df['unit'].n_unique()} units, "
                f"{df['year'].n_unique()} periods ({df['year'].min()}&ndash;{df['year'].max()})</p>")
    html.append("<table><tr><th>Cohort (g)</th><th>Units</th></tr>")
    for row in cohorts.iter_rows(named=True):
        html.append(f"<tr><td>{row['g']}</td><td>{row['n_units']}</td></tr>")
    html.append(f"<tr><td>never-treated</td><td>{n_never}</td></tr></table>")
    html.append('<p class="note">DGP: te=2.0 for cohort 2000, te=3.0 for cohort 2010, no dynamic slope</p>')

    # ── Run estimators ─────────────────────────────────────────────────
    r_ts = py2sdid.ts_did(df, yname="dep_var", idname="unit", tname="year",
                          gname="g", cluster_var="state", verbose=False)
    r_bjs = py2sdid.bjs_did(df, yname="dep_var", idname="unit", tname="year",
                            gname="g", cluster_var="state", verbose=False)

    # ── Overall ATT comparison ─────────────────────────────────────────
    html.append("<h2>Overall ATT</h2>")
    att_diff = abs(r_ts.att_avg - r_bjs.att_avg)
    html.append("<table><tr><th></th><th>ts_did</th><th>bjs_did</th><th>diff</th></tr>")
    html.append(f"<tr><td>ATT</td><td>{r_ts.att_avg:.6f}</td><td>{r_bjs.att_avg:.6f}</td>"
                f'<td class="pass">{att_diff:.2e}</td></tr>')
    html.append(f"<tr><td>SE</td><td>{r_ts.att_avg_se:.6f}</td><td>{r_bjs.att_avg_se:.6f}</td>"
                f"<td>{abs(r_ts.att_avg_se - r_bjs.att_avg_se):.6f}</td></tr>")
    html.append(f"<tr><td>95% CI</td><td>[{r_ts.att_avg_ci[0]:.4f}, {r_ts.att_avg_ci[1]:.4f}]</td>"
                f"<td>[{r_bjs.att_avg_ci[0]:.4f}, {r_bjs.att_avg_ci[1]:.4f}]</td><td></td></tr>")
    html.append(f"<tr><td>p-value</td><td>{r_ts.att_avg_pval:.6f}</td>"
                f"<td>{r_bjs.att_avg_pval:.6f}</td><td></td></tr></table>")

    # ── Event study comparison ─────────────────────────────────────────
    html.append("<h2>Per-Period Event Study</h2>")
    es_ts = r_ts.event_study
    es_bjs = r_bjs.event_study

    html.append("<table><tr><th>rel_time</th><th>estimate</th>"
                "<th>se (ts)</th><th>se (bjs)</th><th>se ratio</th><th>count</th></tr>")
    for row_ts, row_bjs in zip(es_ts.iter_rows(named=True), es_bjs.iter_rows(named=True)):
        h = row_ts["rel_time"]
        est = row_ts["estimate"]
        se_ts = row_ts["se"]
        se_bjs = row_bjs["se"]
        count = row_ts["count"]
        ratio = se_ts / se_bjs if se_bjs > 0 else float("nan")
        cls = "" if h >= 0 else ' style="color:#888"'
        html.append(f"<tr{cls}><td>{h}</td><td>{est:.4f}</td>"
                    f"<td>{se_ts:.4f}</td><td>{se_bjs:.4f}</td>"
                    f"<td>{ratio:.4f}</td><td>{count}</td></tr>")
    html.append("</table>")

    # ── Plots ──────────────────────────────────────────────────────────
    html.append("<h2>Event Study Plots</h2>")
    html.append('<div class="img-row">')
    fig_ts = r_ts.plot(kind="event_study")
    fig_bjs = r_bjs.plot(kind="event_study")
    html.append(fig_to_base64(fig_ts))
    html.append(fig_to_base64(fig_bjs))
    html.append("</div>")

    html.append("<h2>Pre-trend Plot</h2>")
    fig_pre = r_ts.plot(kind="pretrends")
    html.append(fig_to_base64(fig_pre))

    html.append("<h2>Treatment Status</h2>")
    fig_status = r_ts.plot(kind="treatment_status")
    html.append(fig_to_base64(fig_status))

    html.append("<h2>Counterfactual (3 units)</h2>")
    fig_cf = r_ts.plot(kind="counterfactual")
    html.append(fig_to_base64(fig_cf))

    html.append("<h2>Calendar-Time ATT</h2>")
    fig_cal = r_ts.plot(kind="calendar")
    html.append(fig_to_base64(fig_cal))

    # ── Diagnostics ────────────────────────────────────────────────────
    html.append("<h2>Diagnostics</h2>")

    html.append("<h3>ts_did</h3>")
    diag_ts = r_ts.diagnose()
    html.append(f"<pre>{diag_ts.summary()}</pre>")

    html.append("<h3>bjs_did</h3>")
    diag_bjs = r_bjs.diagnose()
    html.append(f"<pre>{diag_bjs.summary()}</pre>")

    # ── Pre-trend analysis ─────────────────────────────────────────────
    html.append("<h2>Pre-trend Analysis</h2>")
    pre_ts = r_ts.pretrend_tests
    if pre_ts is not None:
        html.append(f"<p>20 pre-treatment periods. Mean estimate: {pre_ts['estimate'].mean():.6f}, "
                    f"Max |est|: {pre_ts['estimate'].abs().max():.6f}</p>")
        all_insig = (pre_ts["pval"] > 0.05).all()
        cls = "pass" if all_insig else "fail"
        html.append(f'<p class="{cls}">All insignificant at 5%: {all_insig}</p>')

    # ── Summaries ──────────────────────────────────────────────────────
    html.append("<h2>Full Summary Output</h2>")
    html.append("<h3>ts_did</h3>")
    html.append(f"<pre>{r_ts.summary()}</pre>")
    html.append("<h3>bjs_did</h3>")
    html.append(f"<pre>{r_bjs.summary()}</pre>")

    # ── With covariates ────────────────────────────────────────────────
    html.append("<h2>With Covariates</h2>")
    df_cov = df.with_columns((pl.col("year").cast(pl.Float64) / 100).alias("year_scaled"))
    r_cov = py2sdid.ts_did(df_cov, yname="dep_var", idname="unit", tname="year",
                           gname="g", xformla=["year_scaled"], cluster_var="state", verbose=False)
    html.append(f"<p>ATT (with covariate): {r_cov.att_avg:.6f}, SE: {r_cov.att_avg_se:.6f}</p>")
    html.append(f"<p>Covariate: {r_cov.covariate_names}, coefficient: {r_cov.beta}</p>")
    html.append(f"<p>ATT (without): {r_ts.att_avg:.6f} &mdash; difference: {abs(r_ts.att_avg - r_cov.att_avg):.2e}</p>")

    html.append("</body></html>")

    OUTPUT.write_text("\n".join(html))
    print(f"Report saved to: {OUTPUT}")
    print(f"Open with: xdg-open {OUTPUT}")


if __name__ == "__main__":
    main()
