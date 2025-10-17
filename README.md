# Coconut Harvester Dashboard

This project visualizes the mission plan for an autonomous coconut-harvesting rover. The dashboard stitches together the kinematics, dynamics, thermal limits, and mission planner to show the harvester progressing through a grove while respecting time, energy, and tool-temperature constraints.

## Prerequisites

- Python 3.11+ (project developed and tested with 3.12)
- Install core dependencies:

```powershell
python -m pip install -r requirements.txt
```

The `requirements.txt` file lists the packages needed to run the dashboard and execute the automated test suite.

## Running the Dashboard

Launch the visualization with:

```powershell
python main.py
```

Useful CLI options:

- `--seed <int>` – deterministically regenerates the field layout, tree distribution, and obstacle map.
- `--no-show` – runs the full simulation pipeline headlessly (useful in CI) while still producing telemetry.
- `--quiet` – suppresses console logs for batch jobs.
- `--log-file <path>` – streams per-frame telemetry (time, action, energy, pose, temperature, speed multiplier) to a CSV file.

### Interactive Controls

- **Play/Pause:** button in the lower-left or press `Space`/`P`.
- **Skip Action:** `Skip >>` button or press `N`/Right Arrow to jump directly to the next mission step.
- **Speed Slider:** adjust on-screen slider or use `+`/Up Arrow to accelerate and `-`/Down Arrow to slow the animation.

### Obstacles and Tree Status

Random circular obstacles are generated as part of the field layout. Trees that fall inside an obstacle footprint are automatically marked as **blocked** in the tree tracker; trees that violate mission time or energy limits are marked as **skipped**. Both states receive unique colors in the world view and status table.

## Telemetry Output

When `--log-file` is supplied (or when `show=False` inside `run_dashboard`), the simulator writes a CSV file containing the complete mission timeline, including tool temperature peaks and cumulative energy usage. This is handy for regression checks or offline analysis.

## Running Tests

Install dependencies, then execute:

```powershell
python -m pytest
```

The suite verifies the optimizer’s constraint handling, ensures obstacle avoidance behaves as expected, and confirms the dashboard can initialize/run headlessly while capturing telemetry.
