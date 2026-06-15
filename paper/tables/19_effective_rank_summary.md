# Effective-rank diagnostic (pooling / encoder weights)

Effective rank = exp(entropy of the squared singular-value spectrum).
`pool_eff` is the mean effective rank over the pooler's weight matrices
(640x640 each; `set_transformer` has 12, `attention` a small MLP, `mean` none).
`enc_rank_frac` is effective_rank / min_dim averaged over the two encoder
layers (shapes identical across poolers, so directly comparable).
Spread = max-min of pool_eff across seeds (a proxy for solution
consistency). NOTE: undertrained (sub-40-epoch) checkpoints are excluded
via --min-epochs, since near-random weights have artificially high
effective rank and would masquerade as a collapse.

| cell | n | pool_eff (mean) | pool_eff spread | enc rank-frac (mean) | enc rank-frac range |
|------|---|-----------------|-----------------|----------------------|---------------------|
| attention/species | 3 | 4.098 | 0.312 | 0.156 | 0.155-0.157 |
| attention/genus | 3 | 3.523 | 0.283 | 0.147 | 0.144-0.150 |
| attention/family | 1 | 3.697 | 0.000 | 0.152 | 0.152-0.152 |
| mean/species | 3 | -- | -- | 0.104 | 0.103-0.104 |
| mean/genus | 3 | -- | -- | 0.100 | 0.100-0.100 |
| mean/family | 3 | -- | -- | 0.102 | 0.100-0.105 |
| mean/family (bal) | 3 | -- | -- | 0.093 | 0.091-0.097 |
| set_transformer/species | 3 | 96.6 | 6.4 | 0.384 | 0.382-0.385 |
| set_transformer/genus | 3 | 98.1 | 12.5 | 0.354 | 0.353-0.354 |
| set_transformer/family | 2 | 85.0 | 38.9 | 0.367 | 0.364-0.369 |
| set_transformer/family (bal) | 3 | 87.7 | 19.7 | 0.284 | 0.282-0.288 |

## Per-seed pooler effective rank (set_transformer)

| cell | seed | pool_eff |
|------|------|----------|
| set_transformer/species | 0 | 93.85 |
| set_transformer/species | 1 | 95.86 |
| set_transformer/species | 2 | 100.24 |
| set_transformer/genus | 0 | 92.55 |
| set_transformer/genus | 1 | 105.00 |
| set_transformer/genus | 2 | 96.89 |
| set_transformer/family | 1 | 104.42 |
| set_transformer/family | 2 | 65.54 |
| set_transformer/family (bal) | 0 | 81.37 |
| set_transformer/family (bal) | 1 | 81.07 |
| set_transformer/family (bal) | 2 | 100.77 |
