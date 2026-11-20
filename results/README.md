# Benchmark outputs

The checked-in `summary.csv`, `trials.csv`, and `experiment_config.json` capture an illustrative three-trial experiment.

To regenerate images, full joint-angle path CSVs and experiment data locally:

```bash
python run_7dof.py --trials 3 --iterations 1500 --astar-limit 450 --gif
```

This produces:
- `arm_3d_paths.png` — 3D robotic arm, obstacles and end-effector motion for each planner
- `planner_comparison.png` — success, time, joint motion and clearance comparisons
- `arm_motion_3d.gif` — animated seven-joint arm
- `path_rrt_deg.csv`, `path_rrt_connect_deg.csv`, `path_rrt_star_deg.csv` — planning waypoints

The binary example images and GIF were not uploaded through the GitHub connector, but can be generated from the source code. Timing measurements are hardware-dependent and should be reproduced before making quantitative claims.

The original project is attributed in the top-level README; this repository is an independent 3D extension rather than an actual GitHub fork.
