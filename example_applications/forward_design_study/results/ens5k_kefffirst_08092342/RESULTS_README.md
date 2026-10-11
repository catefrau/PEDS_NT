# Results folder guide (`ens5k_kefffirst_08092342`)

This run directory stores the outputs of the canonical flow:
screen (`peds_ensemble_screen.py`) -> HF verify (`openmc_verify_topk.py`) -> report (`build_forward_design_report.py`).

## File roles

- `config_snapshot.json`
  - Exact run configuration (n_LHS, top_k, tolerances, rank mode, seeds).
  - Keep for reproducibility.

- `lhs_peds_all_evals.csv`
  - Raw full table for every valid LHS point evaluated by PEDS ensemble.
  - Large/debug-oriented; optional for archival.

- `lhs_peds_feasible_ranked.csv`
  - Feasible subset after keff tolerance gate, ranked per selected mode.
  - Core screening output; keep.

- `keff_scan_top_candidates.csv`
  - Top-K shortlist exported for OpenMC verification.
  - Core bridge file between screen and HF verification; keep.

- `keff_scan_hf_diagnostics.csv`
  - Detailed OpenMC verification metrics for shortlisted points.
  - Core HF diagnostics; keep.

- `keff_scan_final_summary.csv`
  - Compact HF-ranked summary derived from diagnostics.
  - Core summary for downstream comparisons/plots; keep.

- `lhs_peds_hf_comparison.csv`
  - Joined/augmented PEDS-vs-HF comparison table with error metrics.
  - Core for parity analysis/plots; keep.

- `forward_design_report.md`
  - Human-readable report generated from the verified outputs.
  - Optional; useful for quick communication and archiving.

- `lhs_peds_timing.json`
  - Timing metadata for screen stage.
  - Optional but useful for performance reporting.

- `keff_scan_hf_timing.txt`
  - Timing/log-style summary for HF stage.
  - Optional but useful for performance reporting.

- `ensemble_5k_kefffirst_findings.md`
  - Narrative notes from analysis.
  - Optional; keep if you want human-readable context.

## Minimal keep-set (recommended)

If you want a lean but fully reusable run artifact, keep:
- `config_snapshot.json`
- `lhs_peds_feasible_ranked.csv`
- `keff_scan_top_candidates.csv`
- `keff_scan_hf_diagnostics.csv`
- `keff_scan_final_summary.csv`
- `lhs_peds_hf_comparison.csv`

## Optional cleanup for fewer files

Safe to remove *after* confirming you do not need runtime/provenance details:
- `lhs_peds_all_evals.csv`
- `lhs_peds_timing.json`
- `keff_scan_hf_timing.txt`
- `ensemble_5k_kefffirst_findings.md`
- `forward_design_report.md`
