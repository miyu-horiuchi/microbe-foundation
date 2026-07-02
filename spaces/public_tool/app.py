from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from statistics import median

import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
EXAMPLE_PATH = ASSETS / "example_predictions.json"
BUNDLE_DIR = ASSETS / "model_bundle"

AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYXBZUO*]", re.IGNORECASE)
DNA_RE = re.compile(r"^[ACGTNRYKMSWBDHV\-\s]+$", re.IGNORECASE)

SAMPLES = {
    "Environmental isolate, short proteome": """>environmental_contig_001 protein_1
MKKLLILTCLVAVALARPKTQETVNVNAGVTGSVSVTIDGDTKVQVTLNDGATVTITR
>environmental_contig_001 protein_2
MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRQLADVVAAG
>environmental_contig_001 protein_3
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPW
""",
    "Marine-like protein FASTA": """>marine_bin protein_1
MNAKQLTAVVAGALAVAGTTAQAAPVSEKTVTIKDGVVNTLQGSSKVTLTIGN
>marine_bin protein_2
MTTQSLVNALAEQKPEVQLTEKAVKQLADENKVLADKIAELQKQLDANLKAE
>marine_bin protein_3
MTPKQAITLALVGLGLVFGGSLTQAAADKNNVVAVTGYGDVGKSTLLNILAG
""",
    "Tiny nucleotide test": """>assembly_fragment_1
ATGAAAAAACTGCTGATTCTGACCTGTCTGGTCGCTGTCGCGCTGGCGCGTCCGAAAACCCAGGAAACCGTGAACGTGAACGCCGGCGTGACCGGCAGCGTGAGCGTGACCATCGACGGCGACACCAAAGTGCAGGTGACC
>assembly_fragment_2
ATGAGCACCAACCCCAAACCGCAGCGTAAAACCAAACGTAACACCAACCGTCGTCCGCAGGACGTGAAATTTCCGGGCGGCGGCCAGATCGTGGGCGGCGTGTATCTGCTGCCGCGTCGTGGCCCGCGTCTGGGCGTGCGT
""",
}


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
  --warn: #a8521a;
  --blue: #3a7d6e;
  --focused-strip: #ede4cd;
  --serif: 'IBM Plex Serif', Georgia, serif;
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--paper) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], footer, #MainMenu { visibility: hidden; }
.stApp { background: var(--paper); }
.block-container { max-width: 100% !important; padding: 34px 48px 5rem !important; }
.main .block-container > div:first-child { padding-top: 0; }
body, p, div, span, li, label, .stMarkdown { font-family: var(--sans) !important; color: var(--ink); }
h1, h2, h3, h4 { font-family: var(--serif) !important; color: var(--ink); letter-spacing: -0.01em; }
code, pre { font-family: var(--mono) !important; }
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] { gap: .75rem; }
.shell { padding: 0; }
.hero {
  border: 1px solid var(--rule);
  padding: 26px 34px 24px;
  margin-bottom: 28px;
  background: var(--paper);
}
.brand-row { display:flex; align-items:baseline; gap:12px; margin-bottom:4px; }
.brand-mark { position:relative; width:18px; height:18px; border-radius:50%; background:var(--accent); top:3px; display:inline-block; }
.brand-mark:before { content:""; position:absolute; inset:4px; border-radius:50%; background:var(--paper); }
.brand-mark:after { content:""; position:absolute; inset:7px; border-radius:50%; background:var(--accent); }
.hero h1 { font:500 22px/1 var(--serif); margin:0; color:var(--ink); }
.hero p { font:400 13.5px/1.5 var(--sans); color:var(--ink-soft); margin:8px 0 0; max-width:700px; }
.kicker { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); letter-spacing: .05em; }
.kicker-up { font-family: var(--mono); font-size: 10px; color: var(--ink-soft); letter-spacing: .08em; text-transform: uppercase; }
.mono-tag { font-family: var(--mono); font-size: 11px; padding: 1px 6px; border: 1px solid var(--accent); color: var(--accent); border-radius: 2px; display:inline-block; margin: 0 6px 6px 0; }
.mode-strip { display:flex; border:1px solid var(--rule); margin: 0 0 28px; }
.mode-pill { flex:1; padding:16px 26px; border-bottom:2px solid transparent; color:var(--ink-faint); font-size:13px; }
.mode-pill.active { background:var(--focused-strip); border-bottom-color:var(--accent); color:var(--ink); }
.lab-card { background:var(--paper); border:1px solid var(--rule); padding:16px 18px; border-radius:2px; transition:border-color 120ms; height:100%; }
.lab-card:hover { border-color:var(--ink); }
.lab-card-featured { background:var(--paper); border:1px solid var(--accent); padding:16px 18px; border-radius:2px; height:100%; }
.metric-num { font-family:var(--serif); font-size:32px; font-weight:500; font-variant-numeric:tabular-nums; line-height:1; color:var(--ink); }
.metric-small { font-family:var(--serif); font-size:22px; font-weight:500; color:var(--ink); }
.verdict-box { padding:14px 16px; background:var(--focused-strip); border:1px solid var(--rule); border-radius:2px; margin-bottom:14px; }
.verdict-kicker { font-family:var(--mono); font-size:11px; color:var(--accent); letter-spacing:.05em; text-transform:uppercase; margin-bottom:4px; }
.verdict-text { font-family:var(--serif); font-size:15px; font-style:italic; color:var(--ink); }
.section-head { font-family:var(--mono); font-size:11px; color:var(--ink-soft); letter-spacing:.08em; text-transform:uppercase; display:flex; align-items:center; gap:10px; margin: 4px 0 12px; }
.section-head .rule { flex:1; height:1px; background:var(--rule); }
.lab-table { width:100%; border-collapse:collapse; font-family:var(--sans); background:var(--paper); border:1px solid var(--rule); }
.lab-table th { background:var(--paper-deep); border-bottom:1px solid var(--rule); padding:8px 12px; text-align:left; font-family:var(--mono); font-size:10px; font-weight:500; color:var(--ink-soft); letter-spacing:.05em; text-transform:uppercase; white-space:nowrap; }
.lab-table td { padding:10px 12px; border-bottom:1px solid var(--rule-soft); font-size:12px; color:var(--ink); vertical-align:middle; }
.lab-table tr:last-child td { border-bottom:none; }
.lab-table tr:hover { background:#ede5cd; }
.stTextInput input, .stTextArea textarea, .stNumberInput input { font-family: var(--sans) !important; border:1px solid var(--rule) !important; border-radius:0 !important; background:rgba(255,255,255,.5) !important; color:var(--ink) !important; }
.stTextInput label, .stTextArea label, .stFileUploader label, .stSelectbox label, .stSlider label, .stCheckbox label { font-family:var(--mono) !important; font-size:11px !important; color:var(--ink-soft) !important; letter-spacing:.05em; text-transform:uppercase; }
[data-testid="stFileUploader"] section { background:rgba(255,255,255,.4) !important; border:1px dashed var(--rule) !important; border-radius:0 !important; padding:12px !important; }
[data-testid="stFileUploaderDropzone"] {
  display: flex !important;
  align-items: center !important;
  gap: 20px !important;
  min-height: 58px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  display: flex !important;
  align-items: center !important;
  min-width: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > span,
[data-testid="stFileUploaderDropzoneInstructions"] p {
  display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div {
  display: flex !important;
  align-items: center !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div:not(:has(small)),
[data-testid="stFileUploaderDropzoneInstructions"] span:not(:has(small)),
[data-testid="stFileUploaderDropzoneInstructions"] p:not(:has(small)) {
  display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
  display: block !important;
  margin: 0 !important;
  font-size: 14px !important;
  color: var(--ink-faint) !important;
  white-space: nowrap !important;
}
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] section [role="button"] {
  min-width: 118px !important;
  width: auto !important;
  height: 38px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex: 0 0 auto !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  color: transparent !important;
  font-size: 0 !important;
  position: relative !important;
}
[data-testid="stFileUploader"] section button::after,
[data-testid="stFileUploader"] section [role="button"]::after {
  content: "Choose file";
  color: var(--ink) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  line-height: 1 !important;
  letter-spacing: .02em !important;
}
[data-testid="stFileUploader"] section button *,
[data-testid="stFileUploader"] section [role="button"] * {
  display: none !important;
  white-space: nowrap !important;
  line-height: 1 !important;
}
[data-testid="stFileUploader"] section small {
  white-space: nowrap !important;
}
.stButton > button, .stDownloadButton > button { font-family:var(--mono) !important; font-size:12px !important; font-weight:400 !important; border-radius:2px !important; border:1px solid var(--rule) !important; background:transparent !important; color:var(--ink) !important; letter-spacing:.02em; padding:6px 12px !important; box-shadow:none !important; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--ink) !important; background:rgba(0,0,0,.04) !important; color:var(--ink) !important; }
.stButton > button[kind="primary"] { background:var(--ink) !important; color:var(--paper) !important; border-color:var(--ink) !important; }
.stButton > button[kind="primary"]:hover { background:var(--accent) !important; border-color:var(--accent) !important; }
[data-baseweb="tab-list"] { background: var(--paper) !important; border-bottom:1px solid var(--rule) !important; padding:18px 0 0 !important; gap:0 !important; }
[data-baseweb="tab"] { font-family:var(--sans) !important; font-size:13px !important; color:var(--ink-faint) !important; padding:12px 18px !important; height:auto !important; background:transparent !important; }
[data-baseweb="tab"] p { color:var(--ink-faint) !important; }
[data-baseweb="tab"][aria-selected="true"] p { color:var(--ink) !important; font-weight:500 !important; }
[data-baseweb="tab-highlight"] { background:var(--ink) !important; height:2px !important; }
[data-baseweb="tab-border"] { display:none !important; }
[data-baseweb="tab-panel"] { padding:18px 0 0 !important; }
[data-testid="stAlert"] { background:var(--paper-deep) !important; border:1px solid var(--rule) !important; border-radius:2px !important; color:var(--ink) !important; }
[data-testid="stAlert"] p { color:var(--ink) !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = "sequence_1"
    seq_parts: list[str] = []
    saw_header = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq_parts:
                records.append((name, clean_sequence("".join(seq_parts))))
            name = line[1:].strip() or f"sequence_{len(records) + 1}"
            seq_parts = []
            saw_header = True
        else:
            seq_parts.append(line)
    if seq_parts:
        records.append((name, clean_sequence("".join(seq_parts))))
    if not saw_header and records:
        records[0] = ("sequence_1", records[0][1])
    return [(n, s) for n, s in records if s]


def clean_sequence(seq: str) -> str:
    return AA_RE.sub("", seq.upper())


def sequence_mode(raw_text: str) -> str:
    compact = "".join(line.strip() for line in raw_text.splitlines() if not line.startswith(">"))
    if compact and DNA_RE.match(compact):
        return "nucleotide"
    return "protein"


def summarize(records: list[tuple[str, str]], mode: str) -> dict:
    lengths = [len(seq) for _, seq in records]
    joined = "".join(seq for _, seq in records)
    total = len(joined)
    aa_counts = {aa: joined.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    hydrophobic = sum(aa_counts[a] for a in "AILMFWYV") / total if total else 0.0
    aromatic = sum(aa_counts[a] for a in "FWY") / total if total else 0.0
    cysteine = aa_counts["C"] / total if total else 0.0
    ambiguous = sum(joined.count(a) for a in "XBZUO*") / total if total else 0.0
    return {
        "mode": mode,
        "records": len(records),
        "total_residues": total,
        "median_length": int(median(lengths)) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "hydrophobic_fraction": hydrophobic,
        "aromatic_fraction": aromatic,
        "cysteine_fraction": cysteine,
        "ambiguous_fraction": ambiguous,
    }


def load_predictor():
    predictor_path = BUNDLE_DIR / "predictor.py"
    if not predictor_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("mf_predictor", predictor_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "predict_fasta", None)


def load_example() -> dict:
    return json.loads(EXAMPLE_PATH.read_text())


def confidence_bar(value: float, color: str = "#3f6b3a") -> str:
    pct = max(0, min(100, int(round(value * 100))))
    return (
        f"<div style='display:flex;align-items:center;gap:8px;min-width:120px;'>"
        f"<div style='flex:1;height:8px;background:rgba(0,0,0,.06);border-radius:2px;position:relative;overflow:hidden;'>"
        f"<div style='position:absolute;inset:0;width:{pct}%;background:{color};border-radius:2px;'></div></div>"
        f"<span style='font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12px;font-weight:500;min-width:36px;text-align:right;color:var(--ink);'>{pct}%</span>"
        f"</div>"
    )


def trait_group(name: str) -> str:
    if name in {"pathogenicity_human", "biosafety_level", "amr_phenotype"}:
        return "safety"
    if name in {"temperature_class", "ph_class", "oxygen_tolerance", "halophily"}:
        return "growth"
    if name in {"gram_stain", "motility", "cell_shape", "sporulation"}:
        return "morphology"
    return "trait"


def verdict_from_result(result: dict, backend_ready: bool) -> str:
    media = result.get("media", [])
    top = media[0]["name"] if media else "the top-ranked medium"
    mode = "Live model" if backend_ready else "Demo model"
    return f"{mode}: prioritize {top}; inspect oxygen, salt, and safety calls before attempting culture."


def render_metric(label: str, value: str, note: str = "") -> str:
    return (
        "<div class='lab-card'>"
        f"<div class='kicker-up'>{label}</div>"
        f"<div class='metric-num'>{value}</div>"
        f"<div class='kicker' style='margin-top:6px;'>{note}</div>"
        "</div>"
    )


def render_trait_cards(result: dict, threshold: float, group_filter: str) -> str:
    cards = []
    for item in result.get("traits", []):
        conf = float(item.get("confidence", 0.0))
        group = trait_group(str(item.get("name", "")))
        if conf < threshold:
            continue
        if group_filter != "all" and group != group_filter:
            continue
        color = "#a8521a" if group == "safety" else "#3a7d6e" if group == "growth" else "#3f6b3a"
        cards.append(
            "<div class='lab-card'>"
            f"<div style='display:flex;justify-content:space-between;gap:12px;align-items:baseline;'>"
            f"<span class='kicker-up'>{item.get('name', '')}</span><span class='mono-tag'>{group}</span></div>"
            f"<div class='metric-small' style='margin:8px 0 10px;color:{color};'>{item.get('prediction', '')}</div>"
            f"{confidence_bar(conf, color)}"
            "</div>"
        )
    if not cards:
        return "<div class='verdict-box'><div class='verdict-text'>No traits match the current filters.</div></div>"
    return "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;'>" + "".join(cards) + "</div>"


def render_media_cards(result: dict) -> str:
    rows = []
    for rank, item in enumerate(result.get("media", []), start=1):
        score = float(item.get("score", 0.0))
        rows.append(
            "<div class='lab-card-featured'>"
            f"<div style='display:flex;justify-content:space-between;gap:12px;align-items:baseline;'>"
            f"<span class='kicker-up'>rank {rank}</span><span class='mono-tag'>screening lead</span></div>"
            f"<div class='metric-small' style='margin:8px 0 10px;color:var(--accent);'>{item.get('name', '')}</div>"
            f"{confidence_bar(score, '#a8521a')}"
            f"<div class='kicker' style='margin-top:10px;line-height:1.45;'>{item.get('rationale', '')}</div>"
            "</div>"
        )
    return "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;'>" + "".join(rows) + "</div>"


def render_trait_table(result: dict, threshold: float) -> None:
    rows = []
    for item in result.get("traits", []):
        conf = float(item.get("confidence", 0.0))
        if conf < threshold:
            continue
        rows.append(
            "<tr>"
            f"<td><code>{item.get('name', '')}</code></td>"
            f"<td>{item.get('prediction', '')}</td>"
            f"<td>{confidence_bar(conf)}</td>"
            "</tr>"
        )
    st.markdown(
        "<table class='lab-table'><tr><th>Trait</th><th>Prediction</th><th>Confidence</th></tr>"
        + "".join(rows)
        + "</table>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="microbe-foundation", page_icon="🦠", layout="wide", initial_sidebar_state="collapsed")
    install_css()

    st.markdown(
        """
<div class="hero">
  <div class="brand-row">
    <div class="brand-mark"></div>
    <h1>microbe-foundation</h1>
    <span style="font:400 11px/1 var(--mono);color:var(--ink-faint);letter-spacing:.04em;">public preview</span>
  </div>
  <p>Genome-conditioned trait and cultivation triage. Paste a protein or genome FASTA,
  pick an example, and review sequence quality, predicted traits, and medium leads in one dashboard.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    predictor = load_predictor()
    backend_ready = predictor is not None

    with st.container():
        st.markdown("<div class='shell'>", unsafe_allow_html=True)
        st.markdown(
            """
<div class="mode-strip">
  <div class="mode-pill active">1. Choose genome</div>
  <div class="mode-pill active">2. Inspect sequence QC</div>
  <div class="mode-pill active">3. Rank traits and media</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([0.38, 0.62], gap="large")
        with left:
            st.markdown("<div class='section-head'>Input panel <span class='rule'></span></div>", unsafe_allow_html=True)
            sample_name = st.selectbox("Try a sample", list(SAMPLES.keys()), index=0)
            uploaded = st.file_uploader("Upload FASTA", type=["fa", "faa", "fasta", "fna", "txt"])
            default_text = SAMPLES[sample_name]
            fasta_text = st.text_area("Paste FASTA", value=default_text, height=260)
            raw = uploaded.getvalue().decode("utf-8", errors="ignore") if uploaded is not None else fasta_text
            threshold = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
            group_filter = st.selectbox("Trait view", ["all", "growth", "morphology", "safety", "trait"], index=0)
            if backend_ready:
                st.markdown("<span class='mono-tag'>live model bundle detected</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='mono-tag'>demo mode: model bundle not attached</span>", unsafe_allow_html=True)

        records = parse_fasta(raw)
        mode = sequence_mode(raw)
        stats = summarize(records, mode) if records else {
            "mode": mode,
            "records": 0,
            "total_residues": 0,
            "median_length": 0,
            "max_length": 0,
            "hydrophobic_fraction": 0.0,
            "aromatic_fraction": 0.0,
            "cysteine_fraction": 0.0,
            "ambiguous_fraction": 0.0,
        }

        if backend_ready and records:
            result = predictor(raw)
            example = False
        else:
            result = load_example()
            example = True

        with right:
            st.markdown("<div class='section-head'>Dashboard readout <span class='rule'></span></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
<div class="verdict-box">
  <div class="verdict-kicker">Triage verdict</div>
  <div class="verdict-text">{verdict_from_result(result, backend_ready)}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(render_metric("records", f"{stats['records']:,}", stats["mode"]), unsafe_allow_html=True)
            c2.markdown(render_metric("residues", f"{stats['total_residues']:,}", "cleaned input"), unsafe_allow_html=True)
            c3.markdown(render_metric("median length", f"{stats['median_length']:,}", f"max {stats['max_length']:,}"), unsafe_allow_html=True)
            c4.markdown(render_metric("ambiguous", f"{stats['ambiguous_fraction']:.1%}", "X/B/Z/U/O/*"), unsafe_allow_html=True)
            st.markdown(
                f"<span class='mono-tag'>hydrophobic {stats['hydrophobic_fraction']:.1%}</span>"
                f"<span class='mono-tag'>aromatic {stats['aromatic_fraction']:.1%}</span>"
                f"<span class='mono-tag'>cysteine {stats['cysteine_fraction']:.1%}</span>",
                unsafe_allow_html=True,
            )
            if example:
                st.info("Predictions below are example outputs. Attach assets/model_bundle/ to enable live inference from the uploaded sequence.")

        tab_overview, tab_media, tab_traits, tab_notes = st.tabs(["Overview", "Medium ranking", "Trait profile", "What this is"])

        with tab_overview:
            st.markdown("<div class='section-head'>Top calls <span class='rule'></span></div>", unsafe_allow_html=True)
            st.markdown(render_trait_cards(result, threshold, group_filter), unsafe_allow_html=True)

        with tab_media:
            st.markdown("<div class='section-head'>Cultivation leads <span class='rule'></span></div>", unsafe_allow_html=True)
            st.markdown(render_media_cards(result), unsafe_allow_html=True)

        with tab_traits:
            st.markdown("<div class='section-head'>Full trait table <span class='rule'></span></div>", unsafe_allow_html=True)
            render_trait_table(result, threshold)

        with tab_notes:
            st.markdown("<div class='section-head'>Deployment status <span class='rule'></span></div>", unsafe_allow_html=True)
            caveats = result.get("caveats", [])
            if caveats:
                st.markdown("".join(f"<span class='mono-tag'>{c}</span>" for c in caveats), unsafe_allow_html=True)
            st.markdown(
                """
<div class="lab-card" style="margin-top:12px;">
  <div class="kicker-up">How to read this dashboard</div>
  <p style="margin:8px 0 0;color:var(--ink-soft);line-height:1.55;">
    This is a screening interface. Medium scores are leads to prioritize, not guaranteed growth recipes.
    The public preview keeps model execution behind an artifact bundle so the UI can be deployed cheaply
    and swapped to live inference without changing the interface.
  </p>
</div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
