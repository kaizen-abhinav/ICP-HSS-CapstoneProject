# PDE-T-and-OT: End-to-End Project Documentation

Date: 2025-10-17

This report documents the Coconut Harvester Dashboard project from fundamentals to advanced features, connecting it to the five instructional modules and providing a full architecture, data flow, operation guide, telemetry schema, AI reasoning view, and testing strategy.

## 1. Executive overview

- Purpose: Simulate and visualize an autonomous coconut-harvesting mission with realistic field rendering, rover/arm visualization, telemetry, and an AI “Mission Brain.”
- Languages/libraries: Python, NumPy, Matplotlib, SymPy; SciPy used in the .qmd labs; Quarto for docs site.
- Core modules: `main.py` (dashboard & orchestration), `optimizer.py` (mission planner), `dynamics.py` (arm step response), `kinematics.py` (2-link FK + world transform), `thermal.py` (1D heat FTCS + duty cycle).
- Learning mapping: The codebase illustrates numerical PDEs (heat, waves), Laplace transforms, and optimization in a robotics-flavored application.

## 2. Architecture and modules

- `main.py`
  - Field generation: procedural orchard texture with custom colormap and furrow shading; trees (canopy, coconuts) and labeled obstacles.
  - Vehicle rendering: 6-wheel footprints and wheels, chassis polygon, mast, cutter; articulated arm overlay using `kinematics` and response from `dynamics`.
  - Planner integration: `optimizer.find_optimal_plan` builds a mission queue (drive/scan/deploy/cut/cool) under time/energy constraints.
  - Telemetry & UI: Developer-themed multi-panel dashboard using GridSpec; framed panels; monospace blocks for ENVIRONMENT, POWERTRAIN, MANIPULATOR, COMMS & NAV, TASKING; AI panel with state, intent, risk, and rolling thoughts.
  - Simulation loop: per-frame updates of environment, drivetrain/arm states, thermal profile via `thermal.solve_heat_equation`, logging to in-memory records and optional CSV.

- `optimizer.py`
  - Random field generation: obstacle circles and trees with counts.
  - Cost model: time/energy for driving, scanning, arm deployment, cutting, cooling.
  - Feasibility: obstacles block lines-of-sight (segment-circle check); greedy nearest-feasible selection under constraints; returns path, totals, and blocked/skipped trees.

- `dynamics.py`
  - Symbolic second-order ODE (Iθ¨ + cθ˙ + kθ = k·u(t)); solves with SymPy; lambdifies step response; finds 2% settling time from a sampled trace.

- `kinematics.py`
  - Planar 2-link FK for elbow and end-effector; world-frame conversion by translation from rover base.

- `thermal.py`
  - 1D heat equation solved with FTCS; stability ratio r enforced by choosing dt; supports heat input Q for heating or zero for cooling. `find_duty_cycle` sweeps to hit limits.

## 3. Data flow

- Inputs: RNG seed, dynamics-derived times (settling, cut, cool), planner params, constraints.
- Planner → Mission queue: sequence of actions with durations and targets.
- Loop state: sim_time, energy_used, rover pose, arm/joint temps/angles, cutter RPM, environment metrics, obstacle clearances, AI state.
- Thermal coupling: tool temperature profile advanced by `thermal.solve_heat_equation` during CUT/COOL; max temp summarized into telemetry and thermal margin.
- Output: On-screen dashboard; in-memory telemetry list; optional CSV file.

## 4. Telemetry schema (CSV/header)

- time, action, tree, rover_x, rover_y, energy, trees_scanned, harvested_coconuts, max_tool_temp, speed_multiplier
- battery_pct, wind_mps, humidity_pct, ambient_temp, nearest_obstacle_clearance, nav_confidence, planner_phase, distance_to_target, eta_to_target
- cpu_load_pct, loop_latency_ms, packet_loss_pct
- wheel_currents, wheel_torques, wheel_temps, suspension_loads, cutter_rpm
- pack_voltage, pack_current, battery_soc, thermal_margin
- arm_joint_angles, arm_joint_temps, end_effector_force
- ai_mode, ai_intent, ai_risk_assessment

Arrays are serialized as comma-joined strings. Units are SI (temperatures in °C, distances in meters, forces in N).

## 5. UI layout quick tour

- World panel: large left grid; textured field, trees, coconuts, obstacles; rover path trace and pose; arm overlay.
- Tree tracker: table with counts and status; colors reflect state transitions (pending→scanning→arm_ready→harvested/empty/blocked/skipped).
- AI Mission Brain: mode, intent, risk text, next action, and thought buffer.
- Mid-row panels: Mission utilization bars (time/energy), arm step response, cutting tool temperature, systems telemetry block, obstacle clearance chart.
- Bottom row: Real-time telemetry (monospace blocks), mission timeline, summary, planner console.
- Controls: Play/Pause, Skip, Speed slider; keyboard shortcuts (Space/P, N/→, +/- or ↑/↓).

## 6. Mapping to instructional modules (01–05)

- 01 Intro to Python & Scientific Stack (01-intro-python.qmd)
  - Arrays, plotting, simple ODE/PDE stubs; kinematics warm-up tasks (2-link FK) reflect `kinematics.py`.
  - Signal generation and filtering resonate with telemetry time series concepts.

- 02 PDE Numerical (02-pde-numerical.qmd)
  - Upwind advection and central-difference wave equation parallel the simulator’s use of discrete updates.
  - Direct tie-in: `thermal.py` FTCS heat solver and duty cycle; stability via r ≤ 0.5.

- 03 Laplace Basics (03-laplace-basics.qmd)
  - Laplace transforms, frequency response, inverse transforms; step response; bode plots.
  - Direct tie-in: `dynamics.py` step response and settling time used for arm deployment durations.

- 04 Laplace Applications (04-laplace-apps.qmd)
  - RC/RLC responses, piecewise/impulse inputs; modeling of switching events.
  - Conceptual echo: mission phases as piecewise system behavior; AI panel reflects changing regimes.

- 05 Optimization (05-optimization.qmd)
  - Linear programming and transportation problems.
  - Direct tie-in: `optimizer.py` greedy route planner with constraints; could be extended to LP/MIP.

## 7. How to run

- Prereqs
  - Python 3.11+
  - Install deps: `python -m pip install -r requirements.txt`
- Start dashboard: `python main.py`
- Headless and logging: `python main.py --no-show --log-file mission.csv`
- Deterministic runs: `python main.py --seed 123`

## 8. Tests and quality gates

- Tests live under `tests/` and cover:
  - Thermal monotonic cooling and heating energy increase (`test_thermal.py`).
  - Planner respects time/energy and obstacle blocking (`test_optimizer.py`).
  - Dashboard initializes headlessly and writes telemetry (`test_dashboard.py`).
- Running locally:
  - `python -m pytest -q` from workspace root; a test `conftest.py` enforces a non-GUI Matplotlib backend and appends project root to `sys.path` for imports.
- Current status (this run): PASS — 6 tests passed; 1 Matplotlib animation warning; no failures.

## 9. Error handling and edge cases

- No feasible plan: returns early with None figure/animation, empty telemetry, and optional header-only CSV.
- Obstacles absent: obstacle panel hides and risk goes nominal.
- Very small windows: text wrapping and modest font reduction applied, but extreme sizes may still clip.
- Thermal stability: FTCS chooses dt via r safety factor to avoid blow-ups; duty cycle loops have safety caps.

## 10. Extensibility roadmap

- Planner: replace greedy with TSP variants or MILP with obstacle avoidance; incorporate battery/thermal constraints natively.
- Sensing realism: per-tree canopy signatures; probabilistic scan outcomes; terrain slope and traction.
- Controls: PID or MPC for arm/drive with realistic dynamics; link to `dynamics.py` parameters.
- AI panel: plug in a rule-based or RL policy; log and replay decisions; export to JSON.
- Export: richer telemetry schema (JSON/Parquet); replay viewer.

## 11. Troubleshooting

- Import errors when running tests from repo root: ensure `tests/conftest.py` is present; it sets `sys.path` and Matplotlib backend.
- No window shows: ensure you didn’t pass `--no-show`. In CI, this is expected; check CSV output.
- Slow rendering: reduce number of trees/obstacles or lower figure size; turn off interpolation on the terrain image.

## 12. License and credits

- Course book powered by Quarto (`_quarto.yml`) with chapters 01–07.
- Code authored for educational use in B.Tech ECE/Robotics lab contexts.
