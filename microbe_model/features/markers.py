"""Curated Pfam markers for genome-driven phenotype + medium recommendation.

Each entry: Pfam ID -> (column_name, biological role / why it matters).

This file is the verified-correct subset only — every Pfam ID below was checked
against InterPro's DESC field (see scripts/23_verify_markers.py and
data/markers/_verification.tsv) and only those whose actual Pfam family matches
the claimed biology are included.

Future expansion (especially for nitrogen fixation, methanogenesis, sulfate
reduction, methylotrophy, vitamin biosynthesis) should layer on TIGRFAM or KOfam
function-defined markers, which are tighter than Pfam structural domains.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Phenotype markers
# ----------------------------------------------------------------------------

TEMPERATURE_MARKERS: dict[str, tuple[str, str]] = {
    "PF00012": ("Hsp70_DnaK",       "Hsp70/DnaK chaperone — abundance scales with temp"),
    "PF00183": ("Hsp90",            "Hsp90 chaperone"),
    "PF00118": ("Cpn60_GroEL",      "Cpn60/GroEL chaperonin — strong thermophile signal"),
    "PF00011": ("Hsp20",            "small heat-shock protein"),
    "PF00313": ("CSD_cold_shock",   "cold-shock DNA-binding domain — psychrophile marker"),
    "PF02824": ("TGS_thermosome",   "TGS domain (often archaeal thermosome / SecA / ObgE)"),
}

PH_MARKERS: dict[str, tuple[str, str]] = {
    "PF00006": ("ATP_synth_alphabeta",  "F1 ATPase α/β nucleotide-binding — proton motive force"),
    "PF00306": ("ATP_synth_alphabeta_C","F1 ATPase α/β C-terminal"),
    "PF00430": ("ATP_synth_F0_B",       "F0 ATPase B/B' subunit"),
    "PF00999": ("NhaA_Na_H_exch",       "NhaA-style Na+/H+ antiporter — alkaliphile signal"),
    "PF06450": ("NhaB_Na_H_exch",       "NhaB Na+/H+ antiporter"),
    "PF00282": ("Pyridoxal_decarbox",   "pyridoxal-dependent decarboxylase (acid resistance)"),
    "PF01618": ("MotA_TolQ_ExbB",       "proton channel family — flagellar stator / TolQ"),
    "PF03224": ("V_ATPase_subH_N",      "V-ATPase subunit H — acidophile / archaeal"),
}

OXYGEN_MARKERS: dict[str, tuple[str, str]] = {
    "PF00115": ("COX1_aerobic",         "heme-Cu terminal oxidase subunit I"),
    "PF02790": ("COX2_TM_aerobic",      "cytochrome c oxidase II transmembrane"),
    "PF00116": ("COX2_periplasm_aero",  "cytochrome c oxidase II periplasmic"),
    "PF13442": ("Cyt_CBB3_microaero",   "cbb3-type cytochrome c oxidase — microaerophile signal"),
    "PF00355": ("Rieske_2Fe2S",         "Rieske 2Fe-2S — cytochrome bc1 / aerobic resp."),
    "PF00199": ("Catalase",             "H2O2 detox — aerobic defense"),
    "PF00081": ("SOD_FeMn",             "Fe/Mn superoxide dismutase"),
    "PF00080": ("SOD_CuZn",             "Cu/Zn superoxide dismutase"),
    "PF02906": ("FeFe_hyd_anaerobic",   "[FeFe]-hydrogenase — strict-anaerobe marker"),
    "PF00374": ("NiFe_hyd_anaerobic",   "[NiFe]-hydrogenase large subunit"),
    "PF00890": ("FAD_binding_FrdA",     "FAD-binding (fumarate reductase / succinate DH)"),
    "PF00037": ("Fer4_FeS_4Fe4S",       "4Fe-4S ferredoxin — anaerobic energy"),
}

SALT_MARKERS: dict[str, tuple[str, str]] = {
    "PF02702": ("KdpD_osmosensor",      "K+ channel histidine-kinase osmosensor"),
    "PF02386": ("TrkH_K_channel",       "TrkH/H+/K+ cation transport"),
    "PF02028": ("BCCT_compatible",      "BCCT family glycine-betaine/choline transporter"),
    "PF00528": ("BPD_transp_1",         "binding-protein-dependent ABC transporter (broad)"),
    "PF06339": ("EctC_ectoine_synth",   "ectoine synthase — halophile compatible-solute"),
    "PF01036": ("Bact_rhodopsin",       "bacteriorhodopsin family — extreme halophile"),
}

# ----------------------------------------------------------------------------
# Media-component markers — drive recipe choice directly
# ----------------------------------------------------------------------------

VITAMIN_MARKERS: dict[str, tuple[str, str]] = {
    "PF00590": ("TP_methylase_B12",     "tetrapyrrole methylase — B12/heme/F430 biosynthesis"),
    "PF01497": ("Peripla_BP_2",         "periplasmic binding (B12, Fe-siderophore, etc.)"),
    "PF00763": ("THF_DHG_CYH_folate",   "THF dehydrogenase/cyclohydrolase — folate path"),
    "PF02152": ("FolB_folate",          "dihydroneopterin aldolase — folate path"),
    "PF03740": ("PdxJ_pyridoxine",      "pyridoxine biosynthesis PdxJ"),
    "PF00926": ("DHBP_riboflavin",      "DHBP synthase — riboflavin biosynthesis"),
}

NITROGEN_MARKERS: dict[str, tuple[str, str]] = {
    "PF00142": ("NifH_nitrogenase",     "NifH Fe-protein — fixes atmospheric N2"),
    "PF00148": ("NifDK_nitrogenase",    "Nitrogenase MoFe component 1"),
    "PF03460": ("NIR_SIR_ferredoxin",   "nitrite/sulfite reductase ferredoxin half"),
}

CARBON_MARKERS: dict[str, tuple[str, str]] = {
    "PF00016": ("RuBisCO_large_form1",  "RuBisCO large chain — Calvin cycle autotrophy"),
    "PF00101": ("RuBisCO_small_form1",  "RuBisCO small chain (Form I-specific)"),
    "PF00128": ("Alpha_amylase",        "starch utilization"),
    "PF00150": ("Cellulase_GH5",        "GH5 cellulase — plant-polymer carbon source"),
    "PF00553": ("CBM_cellulose",        "cellulose-binding module"),
}

SPECIAL_MARKERS: dict[str, tuple[str, str]] = {
    "PF00384": ("Molybdopterin_OR",     "broad: covers AprA/NarG/FdhF type oxidoreductases"),
    "PF13361": ("UvrD_helicase_C",      "DNA-repair helicase — positive control (in nearly all)"),
}


def all_markers() -> dict[str, tuple[str, str]]:
    merged: dict[str, tuple[str, str]] = {}
    for category in (
        TEMPERATURE_MARKERS,
        PH_MARKERS,
        OXYGEN_MARKERS,
        SALT_MARKERS,
        VITAMIN_MARKERS,
        NITROGEN_MARKERS,
        CARBON_MARKERS,
        SPECIAL_MARKERS,
    ):
        merged.update(category)
    return merged


def category_for(pfam_id: str) -> str:
    for cat_name, cat in (
        ("temperature", TEMPERATURE_MARKERS),
        ("ph", PH_MARKERS),
        ("oxygen", OXYGEN_MARKERS),
        ("salt", SALT_MARKERS),
        ("vitamin", VITAMIN_MARKERS),
        ("nitrogen", NITROGEN_MARKERS),
        ("carbon", CARBON_MARKERS),
        ("special", SPECIAL_MARKERS),
    ):
        if pfam_id in cat:
            return cat_name
    return "other"
