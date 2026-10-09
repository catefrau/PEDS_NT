# Augmented MLP (geom+phi+XS) vs geometry-only MLP and PEDS

## Architecture (Boltzmann-app analogue)
- **PEDS generator:** trunk `12 -> 128 -> 256 -> 128`, head `(128+36)->36`, then physics solver (73,524 generator params).
- **This baseline:**
  - trunk: `12 -> 128 -> 256 -> 128` on geom+phi
  - output: `(128+36)->1` (parallel to XS head wiring, single layer)
- **Baseline params:** `67,749`.

## Inputs included
- `6` geometry features (`params`)
- `6` low-fidelity flux features (`phi`, same construction as PEDS)
- `36` baseline XS slots (`log(xs_baseline)`, train-normalized like PEDS)

## Test means across seeds
| Metric | MLP geom-only | MLP geom+phi+XS | PEDS |
|---|---:|---:|---:|
| MSE(k) | 0.02033 | 0.02206 | 6.396e-05 |
| Mean |Δk| (pcm) | 11132.0 | 10911.0 | 574.9 |
| Median |Δk| (pcm) | 9240.2 | 8295.2 | 405.5 |
| Mean fractional error | 0.1102 | 0.1080 | — |
| Frac(|Δk|<650 pcm) | 0.041 | 0.051 | 0.698 |

## Interpretation
- Augmented baseline uses the same extra inputs as PEDS with a single-layer solver swap (hidden+XS -> 1).
- Remaining gap vs PEDS isolates the value of the physics solver and XS-correction training path.