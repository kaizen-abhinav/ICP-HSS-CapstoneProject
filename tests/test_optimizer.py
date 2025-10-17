import dynamics
import optimizer
import thermal


def test_plan_respects_constraints():
    """Planner output should satisfy time and energy ceilings."""
    _, _, settling_time = dynamics.get_arm_dynamics()
    cut_time, cool_time, _ = thermal.find_duty_cycle(temp_limit=70.0, cool_to_temp=25.0)

    params = optimizer.get_default_params(
        settling_time,
        cut_time,
        cool_time,
        seed=42,
        num_trees=6,
    )
    constraints = {'max_time': 1_200.0, 'max_energy': 65_000.0}

    plan = optimizer.find_optimal_plan(params, constraints)

    assert plan['time'] <= constraints['max_time'] + 1e-6
    assert plan['energy'] <= constraints['max_energy'] + 1e-6
    assert len(plan['path']) <= len(params['trees'])
    assert plan['coconuts'] >= 0


def test_obstacles_block_inaccessible_trees():
    _, _, settling_time = dynamics.get_arm_dynamics()
    cut_time, cool_time, _ = thermal.find_duty_cycle(temp_limit=70.0, cool_to_temp=25.0)

    params = optimizer.get_default_params(
        settling_time,
        cut_time,
        cool_time,
        seed=7,
        num_trees=0,
        obstacles=[],
    )

    obstacle = {'name': 'Barrier', 'center': (15.0, 0.0), 'radius': 5.0}
    params['obstacles'] = [obstacle]
    params['field']['obstacles'] = [obstacle]
    params['trees'] = {
        'Tree-near': {'pos': (4.0, 0.0), 'coconuts': 3},
        'Tree-far': {'pos': (22.0, 0.0), 'coconuts': 4},
    }

    constraints = {'max_time': 1_200.0, 'max_energy': 65_000.0}
    plan = optimizer.find_optimal_plan(params, constraints)

    assert 'Tree-far' in plan.get('blocked_trees', []), 'Obstacle should block the far tree.'
    assert 'Tree-far' not in plan['path']
    assert 'Tree-near' in plan['path']
