# Coconut Harvester Codebase Guide

This guide documents the architecture, data flow, and extension points for the coconut harvester simulation. Use it alongside `README.md` when developing new features or debugging existing modules.

## System Overview

The application models an autonomous harvester traversing a coconut grove while meeting time, energy, and temperature limits. It combines analytical models (`dynamics.py`, `thermal.py`, `kinematics.py`) with a greedy mission planner (`optimizer.py`) and a Matplotlib dashboard (`main.py`). Automated tests under `tests/` exercise safety-critical behaviours.

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐
│ dynamics.py│    │ thermal.py │    │ kinematics │    │ optimizer.py │
└─────┬──────┘    └────┬───────┘    └────┬───────┘    └──────┬───────┘
      │                │                │                   │
      │   parameters   │   duty cycle   │   FK helpers      │
      └──────────────┬─┴────────────────┴─┬─────────────────┘
                     │                    │
                     ▼                    │
                 main.py  ←―― mission plan┘
                     │
                     ▼
                Matplotlib UI
```

## Execution Flow (`main.py`)

1. **Seed & configuration** – `run_dashboard` receives optional `seed`, `show`, `verbose`, and `log_path` arguments.
2. **Subsystem init** – calls into `dynamics.get_arm_dynamics()` and `thermal.find_duty_cycle()` to derive arm settling time plus drill duty cycle. These feed `optimizer.get_default_params()` together with generated obstacles.
3. **Mission planning** – `optimizer.find_optimal_plan()` returns a path, cumulative resources, and lists of `blocked_trees` (obstacle intersections) and `skipped_trees` (constraint violations).
4. **Mission queue build** – the function expands each tree into discrete actions (`DRIVING`, `SCAN_TREE`, `DEPLOYING_ARM`, `CUT_TREE`, `COOL_TOOL`). Tree statuses are initialised (`pending`, `blocked`, `skipped`).
5. **Figure assembly** – Matplotlib `GridSpec` lays out the world view, tree tracker, progress bars, arm response plot, and tool temperature chart. Obstacles render as translucent discs.
6. **Controls & events** – Buttons (`Play`, `Skip >>`) and a speed `Slider` manipulate a shared `controls` dictionary. Keyboard shortcuts mirror these controls.
7. **Animation loop** – `FuncAnimation` calls `update(frame)` which:
   - Applies control state (pause, skip, speed multiplier).
   - Advances simulation time, energy, and temperature via `_process_action`.
   - Updates plots, status text, and tree table.
   - Appends telemetry rows (`time`, `action`, `tree`, position, energy, temperature, speed).
8. **Headless execution** – if `show=False`, `run_dashboard` steps through frames manually, closes the figure, and flushes telemetry.
9. **Return payload** – the function returns handles (`figure`, `animation`, `update`, `controls`, telemetry list, plan, log path) for external consumers/tests.

### Key Data Structures

- **`params` (dict)** – produced by `optimizer.get_default_params`, contains:
  - `start_pos`: tuple (meters)
  - `field`: width, height, seed, `obstacles`
  - `trees`: mapping from name to `{pos, coconuts}`
  - `costs`, `operations`, `dynamics`: scalar parameters used when budgeting actions
- **`plan` (dict)** – `find_optimal_plan` output with path list, resource totals, `blocked_trees`, `skipped_trees`
- **`mission_queue` (list)** – ordered action dictionaries consumed by the animation
- **`telemetry_records` (list)** – each row matches the CSV header defined at top of `run_dashboard`

### Tree Status Colours

| Status     | Colour Hex | Meaning                                             |
|------------|------------|-----------------------------------------------------|
| `pending`  | `#8d6e63`  | Not yet visited                                    |
| `scanning` | `#ffca28`  | Currently scanning                                 |
| `arm_ready`| `#fb8c00`  | Scan complete, ready to deploy arm                 |
| `harvested`| `#2e7d32`  | All coconuts collected                              |
| `empty`    | `#607d8b`  | No coconuts, skipped harvesting                     |
| `blocked`  | `#b71c1c`  | Inside obstacle footprint; planner cannot reach    |
| `skipped`  | `#6d4c41`  | Exceeded constraints; planner intentionally skipped|

## Mission Planning (`optimizer.py`)

### Random Field Generation

- `_generate_obstacles` builds circular obstacles with random radius (2–4 m) and position.
- `_generate_random_trees` scatters trees uniformly while avoiding obstacle radii (`_is_point_blocked`).

### Cost & Constraint Model

- `_estimate_visit_cost` computes drive time/energy + scan + optional deploy/cut/cool contributions for a candidate tree.
- `get_default_params` packages field geometry, tree data, operational constants, and dynamic timings.

### Greedy Planner

- `find_optimal_plan` iterates through reachable trees, choosing the closest feasible target each step.
- Obstacle handling: `_segment_intersects_circle` ensures the rover path from current position to candidate tree does not cross any obstacle.
- Constraint handling: accumulates time/energy; trees exceeding budgets move to `skipped_trees`.
- Returns aggregate metrics plus metadata used by the UI.

## Dynamics (`dynamics.py`)

- Models a damped second-order system (`I*θ̈ + c*θ̇ + k*θ = k u(t)`), solved by SymPy with zero initial conditions.
- Generates high-resolution time series to determine the 2 % settling time.
- Returns `(theta_t_symbolic, theta_func_numeric, settling_time)` used in the dashboard for plotting and timing.

## Thermal Model (`thermal.py`)

- Implements an FTCS finite-difference solver for a 1D rod representing the cutting tool (`solve_heat_equation`).
- `find_duty_cycle` alternates drilling (`Q=50`) and cooling (`Q=0`) phases until hitting `temp_limit` and then cooling to `cool_to_temp`.
- Returns `(drill_time, cool_time, final_profile)`; the first two feed planner dynamics, the profile is used for initial tool temperature.

## Kinematics (`kinematics.py`)

- Provides `forward_kinematics` to locate the elbow and end-effector given link lengths and joint angles (radians).
- `world_coords` translates local arm points into world coordinates using the rover base position.
- Constants `LINK_1_LENGTH`, `LINK_2_LENGTH` match the visualisation’s arm segments.

## Testing (`tests/`)

- `tests/conftest.py` – switches Matplotlib to the non-interactive `'Agg'` backend.
- `tests/test_dashboard.py`
  - Confirms `run_dashboard(show=False)` builds without error and exposes controls/telemetry.
  - Verifies telemetry CSV creation when `log_path` supplied.
- `tests/test_optimizer.py`
  - Ensures plans satisfy time/energy constraints.
  - Exercises obstacle avoidance: trees beyond a barrier should appear in `blocked_trees` and excluded from the path.
- `tests/test_thermal.py`
  - Validates monotonic cooling behaviour and heating response of `solve_heat_equation`.

## Telemetry Specification

CSV header written by `run_dashboard`:

```
time,action,tree,rover_x,rover_y,energy,trees_scanned,harvested_coconuts,max_tool_temp,speed_multiplier
```

- `time` – simulation time (seconds)
- `action` – mission queue state name (`DRIVING`, `SCAN_TREE`, etc., or `MISSION_COMPLETE`)
- `tree` – target identifier when applicable
- `rover_x`, `rover_y` – current position (meters)
- `energy` – cumulative energy usage (joules)
- `trees_scanned` / `harvested_coconuts` – running counters
- `max_tool_temp` – instantaneous peak temp along the tool (°C)
- `speed_multiplier` – animation speed at the time of sampling

## Extending the Codebase

1. **Add new constraints** – modify `_estimate_visit_cost` and `find_optimal_plan`; propagate new metrics through `plan` and update UI text.
2. **Custom scenarios** – add parameters to `get_default_params` (e.g., weather conditions) and surface them via CLI options in `main.py`.
3. **Additional telemetry** – extend `telemetry_headers` and `_process_action` to log new columns; update tests to assert the added data.
4. **Extra visuals** – use `GridSpec` slots in `main.py` or add new figures; ensure headless execution still flushes telemetry.
5. **Tests** – mirror new planner/thermal behaviours with Pytest cases; keep Matplotlib interactions behind the `Agg` backend to avoid GUI dependencies.

## Development Workflow

1. Install dependencies with `python -m pip install -r requirements.txt`.
2. Run unit tests: `python -m pytest`.
3. During UI work, run `python main.py --seed <n>` to reproduce specific layouts.
4. When adding logging or new CLI flags, update both `README.md` and this guide.

With this reference, contributors can understand module responsibilities, the control/telemetry pipeline, and safe ways to evolve the simulator.
