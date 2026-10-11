# Diffusion-Only Recheck

- Target/gate: `keff=1.0` with `|Δρ| <= 1500 pcm`
- Input set size: **4** PEDS-feasible geometries
- Diffusion-only runtime: **0.1 s** (0.015 s/case)

## Selection Shift

- PEDS-feasible in input: **4**
- Diffusion-only feasible (same gate): **0**
- PEDS-feasible but NOT diffusion-feasible: **4** (100.0%)
- Top-30 overlap (PEDS rank vs diffusion-only rank): **4/30**
- PEDS top-30 NOT selected by diffusion-only top-30: **0/30** (0.0%)

## Accuracy vs HF

- Rows with HF comparison: **2**
- Mean `|Δρ(PEDS, HF)|`: **45.9 pcm**
- Mean `|Δρ(diffusion-only, HF)|`: **5945.9 pcm**
- Mean `|PPF_PEDS - PPF_HF|`: **2.0871**
- Mean `|PPF_diffonly - PPF_HF|`: **2.2134**

Generated files:
- `diffusion_only_recheck.csv` (full)
- `diffusion_only_recheck_slim.csv` (compact)
- `diffusion_only_summary.json`
- `diffusion_only_summary.md`
