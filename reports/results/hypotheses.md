# A1 remediation: per-hypothesis effect on FD001

Single seed (42), last-window-per-unit protocol, one change at a time
against the same baseline. `delta` is against H0; negative is better.

The line to beat is the random forest at **RMSE 12.72** (NASA 245).

| key | change | RMSE | delta | NASA | params | best epoch | mins | beats forest |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| H0 | baseline (v0.3-health) | 14.60 | -- | 437 | 738,626 | 10 | 10.7 | no |
| H2 | smaller trunk (32/64 conv, 64 LSTM x1) | 14.02 | -0.57 | 346 | 100,194 | 28 | 3.3 | no |
| H3a | noise sigma 0.01 | 14.60 | +0.00 | 436 | 738,626 | 10 | 8.0 | no |
| H3b | noise sigma 0.03 | 14.60 | -0.00 | 433 | 738,626 | 10 | 7.8 | no |
| H3c | noise sigma 0.05 | 14.58 | -0.01 | 431 | 738,626 | 10 | 7.9 | no |
| H3d | noise 0.03 + channel mask 0.1 + jitter 0.05 | 13.24 | -1.35 | 279 | 738,626 | 16 | 10.1 | no |
| H4a | LayerNorm in the conv block | 17.26 | +2.67 | 642 | 738,626 | 9 | 7.9 | no |
| H4b | no conv normalisation | 13.57 | -1.03 | 299 | 738,242 | 17 | 10.1 | no |
| H5a | lambda = 0 (RUL head only) | 16.11 | +1.52 | 485 | 738,626 | 10 | 8.8 | no |
| H5b | lambda = 0.1 | 16.27 | +1.68 | 470 | 738,626 | 12 | 10.0 | no |
| H5c | lambda = 0.5, failure head detached | 16.11 | +1.52 | 485 | 738,626 | 10 | 9.0 | no |
| H6 | min-max scaling to [-1, 1] | 15.44 | +0.84 | 351 | 738,626 | 16 | 10.7 | no |
| H7 | trend-feature branch (last, slope, delta) | 14.07 | -0.52 | 290 | 753,730 | 9 | 7.5 | no |
| HC | combined: small trunk + no conv norm + full augmentation (3 seeds) | 13.10 +/- 0.32 | -1.50 | 311 | 100,002 | 30 | 3.0 | no |
| HCt | combined + trend branch (3 seeds) | 12.76 +/- 0.16 | -1.84 | 266 | 115,106 | 33 | 3.6 | no |

## Verdict

**No neural variant beat the forest on a multi-seed mean.** The best was HCt (combined + trend branch) at 12.76 +/- 0.16 over 3 seed(s), against the forest's 12.72 +/- 0.16 -- a gap of +0.04, well inside the seed spread of either model.

The stopping rule triggered. The random forest becomes the production hazard model; A1 stays as the deep-learning comparison.

This is parity rather than defeat, and the distinction is worth recording: the gap was 1.88 RMSE when this task began and is now 0.04. The same combination cut the seed spread from +/- 1.11 to +/- 0.16 and the parameter count from 738,626 to 115,106, so what remains is both steadier and six times smaller than what it replaced.
