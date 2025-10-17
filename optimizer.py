# optimizer.py
# Mission planner for the coconut harvesting robot.

from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _estimate_visit_cost(current_pos: Tuple[float, float], tree: Dict, params: Dict) -> Dict[str, float]:
    drive_dist = _distance(current_pos, tree['pos'])
    drive_time = drive_dist / params['costs']['drive_speed']
    drive_energy = drive_time * params['costs']['drive_energy']

    scan_time = params['operations']['scan_time']
    scan_energy = scan_time * params['operations']['scan_energy_rate']

    has_coconuts = tree['coconuts'] > 0
    arm_time = params['dynamics']['settling_time'] if has_coconuts else 0.0
    arm_energy = params['costs']['arm_deploy_energy'] if has_coconuts else 0.0

    cut_time = params['dynamics']['cut_time'] if has_coconuts else 0.0
    cut_energy = cut_time * params['operations']['cut_energy_rate'] if has_coconuts else 0.0

    cool_time = params['dynamics']['cool_time'] if has_coconuts else 0.0
    cool_energy = cool_time * params['operations']['cool_energy_rate'] if has_coconuts else 0.0

    total_time = drive_time + scan_time + arm_time + cut_time + cool_time
    total_energy = drive_energy + scan_energy + arm_energy + cut_energy + cool_energy

    return {
        'time': total_time,
        'energy': total_energy,
        'distance': drive_dist
    }


def _segment_intersects_circle(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> bool:
    if radius <= 0.0:
        return False

    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return _distance(p1, center) <= radius

    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return _distance((closest_x, closest_y), center) <= radius


def _is_point_blocked(point: Tuple[float, float], obstacles: List[Dict], margin: float = 0.4) -> bool:
    for obstacle in obstacles:
        if _distance(point, obstacle['center']) <= obstacle['radius'] + margin:
            return True
    return False


def _generate_obstacles(
    num_obstacles: int,
    field_size: Tuple[float, float],
    rng: np.random.Generator,
) -> List[Dict]:
    width, height = field_size
    obstacles: List[Dict] = []

    for idx in range(num_obstacles):
        radius = float(rng.uniform(2.0, 4.0))
        x_low = radius + 0.5
        x_high = max(x_low + 0.1, width - radius - 0.5)
        y_low = radius + 0.5
        y_high = max(y_low + 0.1, height - radius - 0.5)
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))
        obstacles.append(
            {
                'name': f"Obstacle-{idx + 1}",
                'center': (x, y),
                'radius': radius,
            }
        )

    return obstacles


def _generate_random_trees(
    num_trees: int,
    field_size: Tuple[float, float],
    rng: np.random.Generator,
    obstacles: List[Dict],
) -> Dict[str, Dict]:
    width, height = field_size
    trees = {}
    for idx in range(num_trees):
        name = f"Tree-{idx + 1}"
        attempt = 0
        while True:
            x = float(rng.uniform(1.0, width - 1.0))
            y = float(rng.uniform(1.0, height - 1.0))
            if not _is_point_blocked((x, y), obstacles, margin=0.8) or attempt > 25:
                break
            attempt += 1
        coconut_count = int(rng.integers(0, 7))
        trees[name] = {
            'pos': (x, y),
            'coconuts': coconut_count
        }
    return trees


def get_default_params(
    settling_time: float,
    cut_time: float,
    cool_time: float,
    *,
    num_trees: int = 8,
    field_size: Tuple[float, float] = (35.0, 25.0),
    seed: Optional[int] = None,
    num_obstacles: int = 2,
    obstacles: Optional[List[Dict]] = None,
) -> Dict:
    rng = np.random.default_rng(seed)
    obstacle_layout = obstacles if obstacles is not None else _generate_obstacles(num_obstacles, field_size, rng)
    trees = _generate_random_trees(num_trees, field_size, rng, obstacle_layout)

    return {
        'start_pos': (0.0, 0.0),
        'field': {
            'width': field_size[0],
            'height': field_size[1],
            'seed': seed,
            'obstacles': obstacle_layout,
        },
        'obstacles': obstacle_layout,
        'trees': trees,
        'costs': {
            'drive_speed': 1.1,          # m/s
            'drive_energy': 22.0,        # J/s
            'arm_deploy_energy': 420.0   # J per deployment
        },
        'operations': {
            'scan_time': 8.0,             # s
            'scan_energy_rate': 28.0,     # J/s
            'cut_energy_rate': 150.0,     # J/s while cutting
            'cool_energy_rate': 12.0      # J/s while cooling
        },
        'dynamics': {
            'settling_time': settling_time,
            'cut_time': cut_time,
            'cool_time': cool_time
        }
    }


def find_optimal_plan(params: Dict, constraints: Dict[str, float]) -> Dict:
    remaining = set(params['trees'].keys())
    current_pos = params['start_pos']

    plan_path = []
    total_time = 0.0
    total_energy = 0.0
    total_coconuts = 0
    blocked_trees: List[str] = []
    skipped_trees: List[str] = []

    obstacles = params.get('obstacles', [])

    while remaining:
        feasible_candidates = []
        for name in remaining:
            tree_pos = params['trees'][name]['pos']
            is_blocked = any(
                _segment_intersects_circle(current_pos, tree_pos, obstacle['center'], obstacle['radius'])
                for obstacle in obstacles
            )
            if not is_blocked:
                feasible_candidates.append(name)

        if not feasible_candidates:
            blocked_trees.extend(sorted(remaining))
            break

        next_tree = min(feasible_candidates, key=lambda name: _distance(current_pos, params['trees'][name]['pos']))
        tree_info = params['trees'][next_tree]
        visit_cost = _estimate_visit_cost(current_pos, tree_info, params)

        time_after_visit = total_time + visit_cost['time']
        energy_after_visit = total_energy + visit_cost['energy']

        if time_after_visit <= constraints['max_time'] and energy_after_visit <= constraints['max_energy']:
            plan_path.append(next_tree)
            total_time = time_after_visit
            total_energy = energy_after_visit
            total_coconuts += tree_info['coconuts']
            current_pos = tree_info['pos']
        else:
            skipped_trees.append(next_tree)
        remaining.remove(next_tree)

    return {
        'path': plan_path,
        'time': total_time,
        'energy': total_energy,
        'coconuts': total_coconuts,
        'trees_considered': len(params['trees']),
        'blocked_trees': blocked_trees,
        'skipped_trees': skipped_trees,
    }


if __name__ == '__main__':
    print("--- Testing Coconut Field Planner ---")

    settling_time = 4.5
    cut_time = 35.0
    cool_time = 25.0

    mission_params = get_default_params(settling_time, cut_time, cool_time, seed=42)

    mission_constraints = {
        'max_time': 900.0,
        'max_energy': 50000.0
    }

    plan = find_optimal_plan(mission_params, mission_constraints)

    print("\nPlanned tree sequence:", plan['path'])
    print(f"Estimated field time: {plan['time']:.1f}s")
    print(f"Estimated energy: {plan['energy']:.1f}J")
    print(f"Coconuts expected: {plan['coconuts']}")