# Overfitting vs OOD-gap diagnostic (mean-pool ESM-2)

epochs=40  seeds=[0, 1, 2]  features=esm2_features.npz

Read the gap ACROSS splits: a large train>>test gap on `species` means
overfitting; a small gap on `species` that grows on `family` means an
OOD-coverage limit (the model generalises in-distribution but cannot
extrapolate to unseen taxa).

| split | n | macro train | macro test | gap (train-test) |
|-------|---|-------------|------------|------------------|
| species | 3 | 0.6784 ±0.0012 | 0.6647 ±0.0036 | +0.0137 |
| genus | 3 | 0.6886 ±0.0028 | 0.6583 ±0.0044 | +0.0303 |
| family | 3 | 0.6829 ±0.0042 | 0.6467 ±0.0027 | +0.0361 |
