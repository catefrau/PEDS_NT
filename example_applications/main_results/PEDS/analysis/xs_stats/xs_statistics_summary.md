# XS correction statistics summary

## Scope
- Configurations analyzed: **300**
- XS entries compared: **10800**
- Baseline XS reconstructed with `predict_xs(update_geo(...))` from geometry parameters.
- Correction definition: `delta = final_xs - baseline_xs`, `relative = delta / baseline_xs`.

## Overall correction patterns
- Highest mean absolute relative correction by XS type:

| xs_type | mean_relative_delta_pct | mean_abs_relative_delta_pct | q50_relative_delta_pct | q95_relative_delta_pct |
| --- | --- | --- | --- | --- |
| Ss22 | -9.2140 | 19.0728 | -2.2761 | 26.4621 |
| Ss12 | 2.2330 | 13.4365 | 0.0575 | 39.8059 |
| D2 | 1.5344 | 9.3951 | 1.6770 | 20.8347 |
| Sa2 | 3.7165 | 6.8301 | 1.9801 | 26.5647 |
| D1 | -2.3116 | 6.4027 | -1.7108 | 10.4792 |
| Sa1 | 2.1502 | 4.2837 | 2.3187 | 11.3070 |
| nSf2 | -0.8722 | 0.8857 | 0.0000 | 0.0000 |
| nSf1 | -0.7544 | 0.7546 | 0.0000 | 0.0000 |
| Ss11 | -0.0000 | 0.0000 | -0.0000 | 0.0000 |
| Ss21 | -0.0000 | 0.0000 | -0.0000 | 0.0000 |

- Highest mean absolute relative correction by region and XS type:

| region | xs_type | mean_relative_delta_pct | mean_abs_relative_delta_pct | q50_relative_delta_pct | q95_relative_delta_pct |
| --- | --- | --- | --- | --- | --- |
| absorber | Ss22 | -35.4631 | 39.7062 | -22.7794 | 8.1817 |
| absorber | Ss12 | 20.1654 | 21.5931 | 18.6476 | 50.8641 |
| moderator | Ss12 | -12.8342 | 15.4090 | -13.1759 | 9.4527 |
| absorber | Sa2 | 8.5167 | 15.0994 | 7.2248 | 38.1424 |
| moderator | Ss22 | 9.1043 | 14.8739 | 9.0679 | 35.6930 |
| moderator | D2 | 3.1396 | 10.7653 | 3.3495 | 22.3354 |
| fuel | D2 | 1.4876 | 9.7944 | 0.1638 | 24.2553 |
| absorber | D1 | -7.8415 | 9.1431 | -6.9172 | 5.4455 |
| absorber | D2 | -0.0241 | 7.6256 | 0.7497 | 14.1742 |
| fuel | D1 | 0.7588 | 6.5671 | 0.7041 | 14.0810 |
| absorber | Sa1 | 3.0714 | 5.9883 | 2.1819 | 15.9244 |
| moderator | Sa1 | 0.3057 | 3.7720 | -0.5620 | 8.7409 |

## Gradient clipping from logratio_saturation
- Mean clipped fraction across all epoch/region/xs rows: **0.206%**
- Maximum observed clipped fraction in a single row: **13.400%**
- Rows with any clipping (`frac_at_lower_clip + frac_at_upper_clip > 0`): **14.40%**

- Most clipped channels:

| region | xs_idx | mean_clip_pct | max_clip_pct |
| --- | --- | --- | --- |
| absorber | 8 | 4.4229 | 13.4000 |
| absorber | 3 | 2.1971 | 10.4000 |
| absorber | 2 | 0.2371 | 0.6000 |
| absorber | 7 | 0.2200 | 0.6000 |
| moderator | 7 | 0.1400 | 0.2000 |
| fuel | 1 | 0.0686 | 1.0000 |
| moderator | 2 | 0.0400 | 0.4000 |
| moderator | 8 | 0.0314 | 0.2000 |
| moderator | 3 | 0.0229 | 0.4000 |
| moderator | 0 | 0.0200 | 0.2000 |
| absorber | 1 | 0.0114 | 0.2000 |
| absorber | 0 | 0.0029 | 0.2000 |

- Epoch-level clipping trend snapshot:

| epoch | mean_clip_pct | max_clip_pct |
| --- | --- | --- |
| 1.0000 | 0.0000 | 0.0000 |
| 2.0000 | 0.0000 | 0.0000 |
| 3.0000 | 0.0000 | 0.0000 |
| 4.0000 | 0.0389 | 0.6000 |
| 5.0000 | 0.1167 | 1.6000 |
| 6.0000 | 0.2556 | 4.2000 |
| 7.0000 | 0.3389 | 5.2000 |
| 8.0000 | 0.4278 | 6.4000 |
| 9.0000 | 0.3833 | 6.6000 |
| 10.0000 | 0.4722 | 8.0000 |
