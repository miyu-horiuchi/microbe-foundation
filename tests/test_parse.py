"""Parser extractor invariants on a synthetic record covering all field shapes."""
from __future__ import annotations

import parse_bacdive as p


def test_taxonomy(fixture_record):
    tax = p.x_taxonomy(fixture_record)
    assert tax["family"] == "Enterobacteriaceae"
    assert tax["genus"] == "Escherichia"
    assert tax["species"] == "Escherichia coli"
    assert tax["type_strain"] is True


def test_gram_stain_negative(fixture_record):
    assert p.x_gram_stain(fixture_record) == "negative"


def test_cell_shape_rod(fixture_record):
    """'rod-shaped' must normalize to 'rod', not be dropped."""
    assert p.x_cell_shape(fixture_record) == "rod"


def test_motility_true(fixture_record):
    assert p.x_motility(fixture_record) is True


def test_pigmentation_no(fixture_record):
    assert p.x_pigmentation(fixture_record) is False


def test_oxygen_facultative_anaerobe(fixture_record):
    """Multi-word categorical values must map to the canonical class."""
    assert p.x_oxygen_tolerance(fixture_record) == "facultative_anaerobe"


def test_catalase_positive(fixture_record):
    assert p.x_catalase(fixture_record) is True


def test_cytochrome_oxidase_negative(fixture_record):
    assert p.x_cytochrome_oxidase(fixture_record) is False


def test_halophily_halotolerant(fixture_record):
    """Grows at 0% AND 2% but not 8% → halotolerant by the parser's heuristic."""
    assert p.x_halophily(fixture_record) == "halotolerant"


def test_temperature_class_mesophile(fixture_record):
    """37C must bin to mesophile (20-45)."""
    assert p.x_temperature_class(fixture_record) == "mesophile"


def test_ph_class_neutrophile(fixture_record):
    """pH 7.0 with explicit 'PH range: neutrophile' field."""
    assert p.x_ph_class(fixture_record) == "neutrophile"


def test_cultivation_medium_mediadive_ids(fixture_record):
    """Must extract just the MediaDive IDs from the link URLs."""
    media = p.x_cultivation_medium(fixture_record)
    assert set(media) == {"381", "600"}


def test_carbon_utilization_dict_shape(fixture_record):
    cu = p.x_carbon_utilization(fixture_record)
    assert cu["glucose"] is True
    assert cu["lactose"] is True
    assert cu["xylose"] is False


def test_metabolite_production_yes(fixture_record):
    mp = p.x_metabolite_production(fixture_record)
    assert mp["indole"] is True
    assert mp["acetate"] is True


def test_amr_phenotype_resistant(fixture_record):
    """'is resistant: yes' must map to 'R'."""
    amr = p.x_amr_phenotype(fixture_record)
    assert amr["ampicillin"] == "R"


def test_biosafety_level_bsl2(fixture_record):
    """BSL '2' must map to 'BSL-2' (formatted)."""
    assert p.x_biosafety_level(fixture_record) == "BSL-2"


def test_pathogenicity_human_yes(fixture_record):
    """Freeform 'yes' must extract as True."""
    assert p.x_pathogenicity_human(fixture_record) is True


def test_pathogenicity_animal_none_without_bsl1(fixture_record):
    """Strain is BSL-2 with no animal field → None (no BSL-1 negative derivation)."""
    # Note: BSL-1 derivation happens in parse_record, not in the extractor itself.
    assert p.x_pathogenicity_animal(fixture_record) is None


def test_isolation_source_string(fixture_record):
    assert p.x_isolation_source(fixture_record) == "Human, gut"


def test_country(fixture_record):
    assert p.x_country(fixture_record) == "United States"


def test_fatty_acid_profile_dict(fixture_record):
    """FAME values are floats keyed by fatty acid name."""
    fa = p.x_fatty_acid_profile(fixture_record)
    assert fa["C16:0"] == 35.2
    assert fa["C18:1"] == 28.7
    assert fa["C14:0"] == 6.5


def test_full_parse_no_errors(fixture_record):
    """parse_record must run all 21 extractors without raising."""
    out = p.parse_record(fixture_record)
    assert out["bacdive_id"] == 999999
    assert "_parse_errors" not in out
    # 21 trait keys + 8 metadata keys (taxonomy + bacdive_id)
    assert len(out) >= 21


def test_bsl1_derives_negative_pathogenicity():
    """If a strain is BSL-1 and has no explicit pathogenicity, both derive to False."""
    rec = {
        "_bacdive_id": 1,
        "Name and taxonomic classification": {"species": "X"},
        "Interaction and safety": {
            "risk assessment": {"biosafety level": "1"},
        },
    }
    out = p.parse_record(rec)
    assert out["biosafety_level"] == "BSL-1"
    assert out["pathogenicity_human"] is False
    assert out["pathogenicity_animal"] is False


def test_norm_bool_known_tokens():
    assert p.norm_bool("yes") is True
    assert p.norm_bool("no") is False
    assert p.norm_bool("positive") is True
    assert p.norm_bool("+") is True
    assert p.norm_bool("-") is False
    assert p.norm_bool("maybe") is None
    assert p.norm_bool(None) is None
