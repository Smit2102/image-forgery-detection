# Phase 6b: gating the mask with the image classifier (validation images)

## Protocol R (uncertain band 0.5 ± 0.24)

|                              |     value |
|:-----------------------------|----------:|
| band                         |    0.24   |
| authentic_any_region_ungated |    0.8915 |
| authentic_any_region_gated   |    0.4217 |
| authentic_mean_area_ungated  |    0.0556 |
| authentic_mean_area_gated    |    0.0365 |
| tampered_shown               |    0.8427 |
| tampered_f1_ungated          |    0.1859 |
| tampered_f1_gated            |    0.1752 |
| n_authentic                  | 1124      |
| n_tampered                   |  769      |

## Protocol B (uncertain band 0.5 ± 0.04)

|                              |     value |
|:-----------------------------|----------:|
| band                         |    0.04   |
| authentic_any_region_ungated |    0.7073 |
| authentic_any_region_gated   |    0.0996 |
| authentic_mean_area_ungated  |    0.0253 |
| authentic_mean_area_gated    |    0.0069 |
| tampered_shown               |    0.762  |
| tampered_f1_ungated          |    0.2611 |
| tampered_f1_gated            |    0.2251 |
| n_authentic                  | 1124      |
| n_tampered                   |  769      |
