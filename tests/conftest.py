"""Shared fixtures: a small synthetic BacDive record mirroring the real v2 shape."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root importable so `import parse_bacdive` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def fixture_record() -> dict:
    """A synthetic BacDive record covering most field shapes the parser handles."""
    return {
        "_bacdive_id": 999999,
        "Name and taxonomic classification": {
            "LPSN": {
                "domain": "Bacteria",
                "phylum": "Pseudomonadota",
                "class": "Gammaproteobacteria",
                "order": "Enterobacterales",
                "family": "Enterobacteriaceae",
                "genus": "Escherichia",
                "species": "Escherichia coli",
            },
            "domain": "Bacteria",
            "phylum": "Proteobacteria",
            "class": "Gammaproteobacteria",
            "order": "Enterobacterales",
            "family": "Enterobacteriaceae",
            "genus": "Escherichia",
            "species": "Escherichia coli",
            "type strain": "yes",
        },
        "Morphology": {
            "cell morphology": {
                "gram stain": "negative",
                "cell shape": "rod-shaped",
                "motility": "yes",
            },
            "pigmentation": [{"production": "no", "name": "none"}],
        },
        "Culture and growth conditions": {
            "culture medium": [
                {
                    "name": "LB",
                    "growth": "yes",
                    "link": "https://mediadive.dsmz.de/medium/381",
                },
                {
                    "name": "M9",
                    "growth": "positive",
                    "link": "https://mediadive.dsmz.de/medium/600",
                },
            ],
            "culture temp": {"growth": "positive", "type": "growth", "temperature": "37"},
            "culture pH": [{"ability": "positive", "type": "growth", "pH": "7.0", "PH range": "neutrophile"}],
        },
        "Physiology and metabolism": {
            "oxygen tolerance": [
                {"oxygen tolerance": "facultative anaerobe"},
            ],
            "enzymes": [
                {"value": "catalase", "activity": "+"},
                {"value": "cytochrome oxidase", "activity": "-"},
            ],
            "halophily": [
                {"salt": "NaCl", "growth": "positive", "concentration": "0 %"},
                {"salt": "NaCl", "growth": "positive", "concentration": "2 %"},
                {"salt": "NaCl", "growth": "no", "concentration": "8 %"},
            ],
            "metabolite utilization": [
                {"metabolite": "glucose", "utilization activity": "+"},
                {"metabolite": "lactose", "utilization activity": "+"},
                {"metabolite": "xylose", "utilization activity": "-"},
            ],
            "metabolite production": [
                {"metabolite": "indole", "production": "yes"},
                {"metabolite": "acetate", "production": "yes"},
            ],
            "antibiotic resistance": [
                {"metabolite": "ampicillin", "is resistant": "yes", "is sensitive": "no"},
            ],
            "fatty acid profile": {
                "fatty acids": [
                    {"fatty acid": "C16:0", "percentage": 35.2, "ECL": 16},
                    {"fatty acid": "C18:1", "percentage": 28.7, "ECL": 18},
                    {"fatty acid": "C14:0", "percentage": 6.5, "ECL": 14},
                ],
            },
        },
        "Interaction and safety": {
            "risk assessment": [
                {"biosafety level": "2", "biosafety level comment": "German classification",
                 "pathogenicity human": "yes"},
            ],
        },
        "Isolation, sampling and environmental information": {
            "isolation": [
                {"sample type": "Human, gut", "country": "United States"},
            ],
        },
        "Sequence information": {
            "Genome sequences": [
                {"INSDC accession": "GCA_000005845", "assembly level": "complete", "score": 100.0,
                 "description": "K-12 substr. MG1655"},
                {"INSDC accession": "GCA_000019385", "assembly level": "contig", "score": 30.0,
                 "description": "an alt assembly"},
            ],
        },
    }
