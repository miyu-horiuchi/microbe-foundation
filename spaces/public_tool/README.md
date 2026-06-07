---
title: microbe-foundation
emoji: 🦠
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predict bacterial traits from genome sequence
---

# microbe-foundation

Public-facing demo for the microbe-foundation predictor.

This Space is designed as the practical counterpart to the predictability-gradient
paper: a user can paste or upload genome/protein FASTA, review sequence quality
signals, and see the model output once the deployable checkpoint bundle is attached.

## Artifact contract

The UI runs without model artifacts, but live prediction is disabled until a bundle is
placed under `assets/model_bundle/`.

Expected production bundle:

```text
assets/model_bundle/
  manifest.json
  trait_schema.json
  vocabularies.json
  checkpoint.pt
  predictor.py
```

`predictor.py` must expose:

```python
def predict_fasta(fasta_text: str) -> dict:
    ...
```

and return:

```json
{
  "traits": [
    {"name": "gram_stain", "prediction": "Gram-negative", "confidence": 0.91}
  ],
  "media": [
    {"name": "R2A medium", "score": 0.42, "rationale": "broad low-nutrient medium"}
  ],
  "caveats": ["Family-level novelty: high uncertainty"]
}
```

Until that bundle is present, the app shows a clearly marked example output and never
claims that uploaded sequences have been predicted.
