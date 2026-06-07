from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSET_PATH = ROOT / "assets" / "predictability_gradient.json"


def install_css() -> None:
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400;1,500&display=swap" rel="stylesheet">
<style>
:root {
  --paper: #f5f1e8;
  --paper-deep: #ece6d6;
  --ink: #1f1d18;
  --ink-soft: #5a554a;
  --ink-faint: #94907f;
  --rule: #d6cdb6;
  --rule-soft: #e6dfca;
  --accent: #a8521a;
  --accent-tint: #fdf6e8;
  --pos: #3f6b3a;
  --o2: #3a7d6e;
  --focused-strip: #ede4cd;
  --serif: 'IBM Plex Serif', Georgia, serif;
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--paper) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], footer, #MainMenu { visibility: hidden; }
.stApp { background: var(--paper); }
.block-container { max-width: 100% !important; padding: 0 0 4rem 0 !important; }
.main .block-container > div:first-child { padding-top: 0; }
body, p, div, span, li, label, .stMarkdown { font-family: var(--sans) !important; color: var(--ink); }
h1, h2, h3, h4 { font-family: var(--serif) !important; color: var(--ink); letter-spacing: -0.01em; }
code, pre { font-family: var(--mono) !important; color: var(--accent); }
.pg-kicker { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); letter-spacing: .05em; text-transform: uppercase; }
.pg-hero { padding: 38px 34px 24px; border-bottom: 1px solid var(--rule); margin-bottom: 0; background: var(--paper); }
.pg-hero h1 { font-size: clamp(2.5rem, 6vw, 5.4rem); line-height: .9; margin: .22rem 0 .75rem; max-width: 1040px; font-weight: 500; }
.pg-hero p { max-width: 860px; color: var(--ink-soft); font-size: 1.06rem; line-height: 1.55; }
.pg-card { border: 1px solid var(--rule); background: var(--paper); padding: 16px 18px; border-radius: 2px; min-height: 112px; transition: border-color 120ms; }
.pg-card:hover { border-color: var(--ink); }
.pg-card .num { font-family: var(--serif); color: var(--ink); font-size: 32px; font-weight: 500; font-variant-numeric: tabular-nums; line-height: 1; }
.pg-card .label { color: var(--ink-soft); font-family: var(--mono); font-size: 10px; letter-spacing: .05em; text-transform: uppercase; margin-top: .5rem; }
.pg-pill { display: inline-block; border: 1px solid var(--accent); color: var(--accent); padding: 1px 6px; margin: 0 6px 7px 0; border-radius: 2px; font-size: 11px; font-family: var(--mono); }
.pg-callout { border: 1px solid var(--rule); background: var(--focused-strip); padding: 14px 16px; margin: 1rem 0; border-radius: 2px; color: var(--ink); font-family: var(--serif); font-style: italic; }
.pg-table { width: 100%; border-collapse: collapse; font-family: var(--sans); background: var(--paper); border: 1px solid var(--rule); }
.pg-table th { background: var(--paper-deep); border-bottom: 1px solid var(--rule); padding: 8px 12px; text-align: left; font-family: var(--mono); font-size: 10px; font-weight: 500; color: var(--ink-soft); letter-spacing: .05em; text-transform: uppercase; white-space: nowrap; }
.pg-table td { padding: 10px 12px; border-bottom: 1px solid var(--rule-soft); font-size: 12px; color: var(--ink); vertical-align: middle; }
.pg-table tr:last-child td { border-bottom: none; }
.pg-table tr:hover { background: #ede5cd; }
[data-baseweb="tab-list"] { background: var(--paper) !important; border-bottom: 1px solid var(--rule) !important; padding: 0 28px !important; gap: 0 !important; }
[data-baseweb="tab"] { font-family: var(--sans) !important; font-size: 13px !important; color: var(--ink-faint) !important; padding: 12px 18px !important; height: auto !important; background: transparent !important; }
[data-baseweb="tab"] p { color: var(--ink-faint) !important; }
[data-baseweb="tab"][aria-selected="true"] p { color: var(--ink) !important; font-weight: 500 !important; }
[data-baseweb="tab-highlight"] { background: var(--ink) !important; height: 2px !important; }
[data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-panel"] { padding: 0 28px !important; }
.stCaptionContainer, .stCaptionContainer p { color: var(--ink-soft) !important; font-family: var(--sans) !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def load_asset() -> dict:
    if ASSET_PATH.exists():
        return json.loads(ASSET_PATH.read_text())
    return {
        "paper": {"n_genomes": 19592, "n_proteins": 82000000, "n_traits": 21, "embedding": "ESM-2 t30 150M"},
        "headline_gradient": [
            {"split": "species", "class": "compositional", "delta_f1": 0.021, "std": 0.002},
            {"split": "species", "class": "machinery", "delta_f1": 0.083, "std": 0.012},
            {"split": "genus", "class": "compositional", "delta_f1": 0.016, "std": 0.004},
            {"split": "genus", "class": "machinery", "delta_f1": 0.067, "std": 0.010},
            {"split": "family", "class": "compositional", "delta_f1": 0.009, "std": 0.002},
            {"split": "family", "class": "machinery", "delta_f1": 0.010, "std": 0.003},
        ],
        "trait_classes": {"compositional": [], "machinery": [], "excluded": []},
        "head_deltas": [],
        "attention": {"animal": {}, "human": {}, "genes": []},
        "comparison_rows": [],
    }


def metric_card(value: str, label: str) -> str:
    return f"<div class='pg-card'><div class='num'>{value}</div><div class='label'>{label}</div></div>"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def gradient_chart(df: pd.DataFrame):
    base = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("split:N", sort=["species", "genus", "family"], title=None),
            xOffset=alt.XOffset("class:N"),
            y=alt.Y("delta_f1:Q", title="Attention - mean macro-F1"),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(domain=["compositional", "machinery"], range=["#3a7d6e", "#a8521a"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["split", "class", alt.Tooltip("delta_f1:Q", format=".3f"), alt.Tooltip("std:Q", format=".3f")],
        )
        .properties(height=330)
    )
    err = (
        alt.Chart(df)
        .mark_errorbar()
        .encode(
            x=alt.X("split:N", sort=["species", "genus", "family"]),
            xOffset=alt.XOffset("class:N"),
            y=alt.Y("low:Q"),
            y2="high:Q",
            color=alt.Color("class:N", scale=alt.Scale(domain=["compositional", "machinery"], range=["#3a7d6e", "#a8521a"])),
        )
    )
    return (base + err).configure_view(stroke="#d6cdb6").configure_axis(
        labelColor="#5a554a",
        titleColor="#5a554a",
        gridColor="#e6dfca",
        domainColor="#d6cdb6",
        tickColor="#d6cdb6",
    ).configure_legend(labelColor="#1f1d18", titleColor="#5a554a")


def render_table(rows: list[dict], cols: list[str]) -> None:
    html_rows = []
    for row in rows:
        html_rows.append("<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in cols) + "</tr>")
    st.markdown(
        "<table class='pg-table'><tr>"
        + "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
        + "</tr>"
        + "".join(html_rows)
        + "</table>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="predictability-gradient", page_icon=None, layout="wide")
    install_css()
    asset = load_asset()
    paper = asset["paper"]

    st.markdown(
        """
<div class='pg-hero'>
  <div class='pg-kicker'>Paper companion / mechanism exhibit</div>
  <h1>When does attention help?</h1>
  <p>A compact interactive view of the predictability-gradient result:
  attention-pooling helps gene-localized machinery traits, contributes little to
  diffuse compositional traits, and loses its advantage under family-level shift.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card(f"{paper['n_genomes']:,}", "BacDive genomes"), unsafe_allow_html=True)
    c2.markdown(metric_card("82M", "per-protein embeddings"), unsafe_allow_html=True)
    c3.markdown(metric_card(str(paper["n_traits"]), "prediction heads"), unsafe_allow_html=True)
    c4.markdown(metric_card("3 x 3", "splits x seeds"), unsafe_allow_html=True)

    tabs = st.tabs(["Gradient", "Trait Classes", "Attention Spotlight", "VFDB + Ablation", "Benchmark Context"])

    with tabs[0]:
        df = pd.DataFrame(asset["headline_gradient"])
        df["low"] = df["delta_f1"] - df["std"]
        df["high"] = df["delta_f1"] + df["std"]
        st.altair_chart(gradient_chart(df), use_container_width=True)
        st.markdown(
            "<div class='pg-callout'>The species-level machinery gain is about four times the compositional gain "
            "(+0.083 vs +0.021 F1). At family holdout, the gap shrinks to +0.001, localizing the bottleneck to "
            "cross-clade generalization.</div>",
            unsafe_allow_html=True,
        )
        if asset.get("head_deltas"):
            top = pd.DataFrame(asset["head_deltas"]).head(10)
            top["delta_f1_mean"] = top["delta_f1_mean"].map(lambda x: f"{x:+.3f}")
            top["delta_f1_std"] = top["delta_f1_std"].map(lambda x: f"{x:.3f}")
            render_table(top.to_dict("records"), ["split", "head", "class", "delta_f1_mean", "delta_f1_std", "n_seeds"])

    with tabs[1]:
        classes = asset["trait_classes"]
        left, right = st.columns(2)
        with left:
            st.subheader("Compositional")
            st.markdown("".join(f"<span class='pg-pill'>{x}</span>" for x in classes["compositional"]), unsafe_allow_html=True)
            st.caption("Diffuse or bulk cellular signals where mean-pooling should be competitive.")
        with right:
            st.subheader("Machinery")
            st.markdown("".join(f"<span class='pg-pill'>{x}</span>" for x in classes["machinery"]), unsafe_allow_html=True)
            st.caption("Gene-localized or pathway-localized signals where attention has something to select.")
        st.markdown(
            "<div class='pg-callout'>The partition is biological, not result-driven: two metadata heads "
            "(isolation source and country) are excluded from the gradient analysis.</div>",
            unsafe_allow_html=True,
        )

    with tabs[2]:
        animal = asset["attention"]["animal"]
        cols = st.columns(4)
        cols[0].markdown(metric_card(f"{animal.get('auroc', 0):.2f}", "animal pathogenicity AUROC"), unsafe_allow_html=True)
        cols[1].markdown(metric_card(f"{animal.get('median_entropy', 0):.2f}", "median attention entropy"), unsafe_allow_html=True)
        cols[2].markdown(metric_card(pct(animal.get("top5_attention_mass", 0)), "attention in top 5 proteins"), unsafe_allow_html=True)
        cols[3].markdown(metric_card("~3,800", "proteins per genome"), unsafe_allow_html=True)
        st.markdown("".join(f"<span class='pg-pill'>{gene}</span>" for gene in asset["attention"].get("genes", [])), unsafe_allow_html=True)
        st.markdown(
            "<div class='pg-callout'>The spotlighted genes are coherent adherence and invasion machinery, including "
            "fimbrial ushers, filamentous hemagglutinin, invasion loci, type-IV pili, and flagellar proteins.</div>",
            unsafe_allow_html=True,
        )

    with tabs[3]:
        animal = asset["attention"]["animal"]
        human = asset["attention"]["human"]
        rows = [
            {
                "test": "within genome",
                "head": "animal",
                "top_attended": pct(animal.get("within_top", 0)),
                "control": pct(animal.get("within_random", 0)),
                "statistic": f"Wilcoxon p={animal.get('within_p', 'n/a')}",
            },
            {
                "test": "between class",
                "head": "animal",
                "top_attended": pct(animal.get("between_top_pathogenic", 0)),
                "control": pct(animal.get("between_top_non_pathogenic", 0)),
                "statistic": f"OR={animal.get('between_or', 'n/a')}, p={animal.get('between_p', 'n/a')}",
            },
            {
                "test": "between class",
                "head": "human",
                "top_attended": "replicates",
                "control": "non-pathogenic genomes",
                "statistic": f"OR={human.get('between_or', 'n/a')}, p={human.get('between_p', 'n/a')}",
            },
            {
                "test": "top-5 ablation",
                "head": "animal",
                "top_attended": pct(animal.get("ablation_flip", 0)),
                "control": "random removal near zero",
                "statistic": f"flip p={animal.get('ablation_p', 'n/a')}",
            },
            {
                "test": "top-5 ablation",
                "head": "human",
                "top_attended": pct(human.get("ablation_flip", 0)),
                "control": "random removal near zero",
                "statistic": f"flip p={human.get('ablation_p', 'n/a')}",
            },
        ]
        render_table(rows, ["test", "head", "top_attended", "control", "statistic"])
        st.markdown(
            "<div class='pg-callout'>The mechanism claim rests on the conjunction: enrichment against an external "
            "virulence-factor database and causal dependence under ablation.</div>",
            unsafe_allow_html=True,
        )

    with tabs[4]:
        rows = asset.get("comparison_rows", [])
        if rows:
            render_table(rows, ["trait", "metric", "best_ours", "run", "prior", "prior_score", "verdict"])
        st.markdown(
            "<div class='pg-callout'>This tab is context for the benchmark, not the core paper claim. The paper's "
            "main contribution is the pooling rule plus mechanistic validation, not a new encoder sweep.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
