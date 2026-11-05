# Local state observation catalog

Seed: `5574668987006270653`. Case: `001_two_turbines_yaw_step_floris.yaml`.
Simulation: 0 to 2000 s in 4 s steps (501 ticks).
Probes: fixed hub-height coordinates above the six turbine bases; the yaw-dependent rotor centers can shift by the 7.07 m overhang.

| Turbine | Coordinate (m) | Effective speed min / mean / max (m/s) | Inferred direction min–max (°) | Max direction difference from forcing (°) | Max distance to OP chain (m) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | (0, -495, 119) | 6.55 / 8.01 / 9.45 | 223.51–274.23 | 0.00 | 7.07 |
| 1 | (990, -495, 119) | 3.51 / 7.33 / 9.10 | 223.51–274.23 | 12.50 | 7.07 |
| 2 | (1980, -495, 119) | 4.57 / 7.27 / 8.97 | 223.51–274.23 | 12.87 | 7.07 |
| 3 | (0, 495, 119) | 6.86 / 8.00 / 9.00 | 223.51–274.23 | 0.00 | 7.07 |
| 4 | (990, 495, 119) | 4.62 / 7.36 / 8.99 | 223.51–274.23 | 12.50 | 7.07 |
| 5 | (1980, 495, 119) | 5.40 / 7.28 / 8.52 | 223.51–274.23 | 17.02 | 7.07 |

## Validation

- 3,006 observations: exactly 501 at each of six points, with the same timestamps as `wind_profile.csv`.
- Every effective speed, ambient speed, inferred direction, and probe-to-chain distance is finite. Speeds are nonnegative and directions are within [0°, 360°).
- Effective speed and inferred direction both vary at every probe over the run.
- Inferred directions differ across probes on 37 of 501 ticks; the farm-wide forcing produces identical or nearly identical directions at the other ticks.
- The direction is inferred from OFF ambient observation-point states. It is not a wake-deflected flow direction. Differences from farm-wide forcing can reflect propagated older states.
- The effective speed is evaluated by the wake solver; ambient speed comes from the nearest observation-point chain. They may differ, including cases where the effective value is higher.

For every tick and point, see `local_wind_samples.csv` or `local_wind_point_N.csv`. The full unperturbed-to-perturbed farm forcing is in `wind_profile.csv`.
