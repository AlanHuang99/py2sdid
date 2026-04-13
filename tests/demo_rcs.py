"""
Demo script: run RCS estimation on fixture data, print coefficients, save plots.

Usage:
    uv run python tests/demo_rcs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def main():
    from py2sdid import ts_did, bjs_did

    # Load fixture data
    ind_data = pl.read_parquet(DATA_DIR / "rcs_individual.parquet")
    agg_data = pl.read_parquet(DATA_DIR / "rcs_aggregated.parquet")
    with open(DATA_DIR / "rcs_metadata.json") as f:
        meta = json.load(f)

    print("=" * 70)
    print("RCS FIXTURE DATA")
    print("=" * 70)
    print(f"Groups: {meta['n_groups']}")
    print(f"Individuals per cell: {meta['n_individuals_per_cell']}")
    print(f"Panel: {meta['panel']}")
    print(f"Cohorts: {meta['cohorts']}")
    print(f"True TE: {meta['te']}")
    print(f"True TE slope: {meta['te_m']}")
    print(f"Individual-level: {ind_data.shape}")
    print(f"Aggregated: {agg_data.shape}")
    print()

    # ---------------------------------------------------------------
    # 1. Individual-level RCS — ts_did
    # ---------------------------------------------------------------
    print("=" * 70)
    print("1. INDIVIDUAL-LEVEL RCS — ts_did (Gardner 2021)")
    print("=" * 70)
    r_ind_ts = ts_did(
        ind_data,
        yname="dep_var",
        idname="individual_id",
        tname="year",
        gname="g",
        dataset_type="rcs",
        groupname="group",
        verbose=True,
    )
    print()
    print(r_ind_ts.summary())
    print()

    # Save event study plot
    fig = r_ind_ts.plot(kind="event_study")
    fig.savefig(OUT_DIR / "rcs_individual_event_study.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'rcs_individual_event_study.png'}")
    plt.close(fig)

    # Save pretrends plot
    fig = r_ind_ts.plot(kind="pretrends")
    fig.savefig(OUT_DIR / "rcs_individual_pretrends.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'rcs_individual_pretrends.png'}")
    plt.close(fig)

    # Save counterfactual plot
    fig = r_ind_ts.plot(kind="counterfactual")
    fig.savefig(OUT_DIR / "rcs_individual_counterfactual.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'rcs_individual_counterfactual.png'}")
    plt.close(fig)

    # Save treatment status plot
    fig = r_ind_ts.plot(kind="treatment_status")
    fig.savefig(OUT_DIR / "rcs_individual_treatment_status.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'rcs_individual_treatment_status.png'}")
    plt.close(fig)

    print()

    # ---------------------------------------------------------------
    # 2. Individual-level RCS — bjs_did
    # ---------------------------------------------------------------
    print("=" * 70)
    print("2. INDIVIDUAL-LEVEL RCS — bjs_did (Borusyak et al. 2024)")
    print("=" * 70)
    r_ind_bjs = bjs_did(
        ind_data,
        yname="dep_var",
        idname="individual_id",
        tname="year",
        gname="g",
        dataset_type="rcs",
        groupname="group",
        verbose=True,
    )
    print()
    print(r_ind_bjs.summary())
    print()

    fig = r_ind_bjs.plot(kind="event_study")
    fig.savefig(OUT_DIR / "rcs_individual_bjs_event_study.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'rcs_individual_bjs_event_study.png'}")
    plt.close(fig)

    print()

    # ---------------------------------------------------------------
    # 3. Aggregated RCS — ts_did
    # ---------------------------------------------------------------
    print("=" * 70)
    print("3. AGGREGATED RCS — ts_did (Gardner 2021)")
    print("=" * 70)
    r_agg_ts = ts_did(
        agg_data,
        yname="dep_var",
        idname="group",
        tname="year",
        gname="g",
        dataset_type="rcs",
        verbose=True,
    )
    print()
    print(r_agg_ts.summary())
    print()

    fig = r_agg_ts.plot(kind="event_study")
    fig.savefig(OUT_DIR / "rcs_aggregated_event_study.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'rcs_aggregated_event_study.png'}")
    plt.close(fig)

    fig = r_agg_ts.plot(kind="pretrends")
    fig.savefig(OUT_DIR / "rcs_aggregated_pretrends.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'rcs_aggregated_pretrends.png'}")
    plt.close(fig)

    print()

    # ---------------------------------------------------------------
    # 4. Diagnostics
    # ---------------------------------------------------------------
    print("=" * 70)
    print("4. DIAGNOSTICS — Individual-level RCS")
    print("=" * 70)
    diag = r_ind_ts.diagnose()
    print(diag.summary())
    print()

    print("=" * 70)
    print("All plots saved to:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
