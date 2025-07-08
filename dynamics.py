# dynamics.py
# Module for solving the arm's joint dynamics and finding settling time.

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def get_arm_dynamics():
    """
    Solves the ODE for a single joint and calculates the settling time.
    Returns:
        - The symbolic solution theta(t)
        - A numerical function for plotting
        - The calculated settling time in seconds
    """
    # --- System Modeling ---
    t = sp.Symbol('t', positive=True)
    theta = sp.Function('theta')
    
    # Parameters
    I = 1.0
    c = 4.0
    k = 13.0
    
    # ODE: I*theta'' + c*theta' + k*theta = k * (unit step input)
    ode = I * theta(t).diff(t, 2) + c * theta(t).diff(t) + k * theta(t) - k * sp.Heaviside(t)
    
    # --- Laplace Analysis ---
    # Solve with initial conditions theta(0)=0, theta'(0)=0
    ics = {theta(0): 0, theta(t).diff(t).subs(t, 0): 0}
    solution = sp.dsolve(ode, ics=ics)
    theta_t = solution.rhs

    # --- Constraint Discovery: Settling Time ---
    # Convert symbolic solution to a fast numerical function
    theta_func = sp.lambdify(t, theta_t, 'numpy')
    
    # Generate high-resolution time data to find settling time
    t_vals = np.linspace(0, 10, 2000)
    y_vals = theta_func(t_vals)
    
    # Find settling time (time to get within 2% of final value 1.0)
    settling_time = 0
    final_value = 1.0
    settling_band = 0.02
    
    # Find the last time the system was *outside* the settling band
    outside_band_indices = np.where(np.abs(y_vals - final_value) > settling_band)[0]
    if len(outside_band_indices) > 0:
        last_outside_index = outside_band_indices[-1]
        # The settling time is the time of the next sample
        if last_outside_index + 1 < len(t_vals):
            settling_time = t_vals[last_outside_index + 1]
    
    return theta_t, theta_func, settling_time

# This block allows testing this module independently
if __name__ == '__main__':
    print("--- Testing Dynamics Module ---")
    
    symbolic_sol, numeric_func, settle_time = get_arm_dynamics()
    
    print(f"Symbolic solution theta(t) = {symbolic_sol}")
    print(f"Calculated Settling Time (2%): {settle_time:.2f} seconds")

    # Visualize the result
    t_plot = np.linspace(0, 8, 400)
    y_plot = numeric_func(t_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, y_plot, label='Joint Step Response $\\theta(t)$')
    plt.axhline(1.0, color='gray', linestyle='--', label='Target Angle')
    plt.axhline(1.02, color='r', linestyle=':', label='2% Settling Band')
    plt.axhline(0.98, color='r', linestyle=':')
    plt.axvline(settle_time, color='g', linestyle='--', label=f'Settling Time = {settle_time:.2f}s')
    
    plt.title("Arm Joint Step Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.grid(True)
    plt.legend()
    plt.show()