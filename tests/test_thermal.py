import numpy as np

import thermal


def test_cooling_monotonic():
    """Cooling with zero heat input should not raise temperature."""
    initial = np.linspace(40.0, 60.0, 21)
    cooled = thermal.solve_heat_equation(duration=5.0, Q=0, initial_temp_profile=initial)
    assert np.all(cooled <= initial + 1e-6)


def test_heating_increases_energy():
    """Positive heat input should raise the average blade temperature."""
    initial = np.full(21, 20.0)
    heated = thermal.solve_heat_equation(duration=5.0, Q=50, initial_temp_profile=initial)
    assert heated.mean() > initial.mean()
