# thermal.py
# Module for solving the drill's heat equation and finding the duty cycle.

import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(duration, Q, initial_temp_profile):
    """
    Solves the 1D heat equation using FTCS scheme.
    Returns the final temperature profile.
    """
    # Parameters
    L = 0.2           # Length of the drill bit (m)
    alpha = 1e-5      # Thermal diffusivity for steel
    nx = 21           # Number of spatial points
    dx = L / (nx - 1)
    
    # We need to choose dt to satisfy the stability condition
    # r = alpha * dt / dx^2 <= 0.5  => dt <= 0.5 * dx^2 / alpha
    dt = 0.4 * dx**2 / alpha # Use a safety factor of 0.4
    nt = int(duration / dt)
    
    r = alpha * dt / dx**2
    
    u = initial_temp_profile.copy()
    
    for n in range(nt):
        u_old = u.copy()
        for i in range(1, nx - 1):
            u[i] = u_old[i] + r * (u_old[i+1] - 2*u_old[i] + u_old[i-1]) + Q * dt
    
    return u

def find_duty_cycle(temp_limit=80.0, cool_to_temp=20.0):
    """
    Simulates drilling and cooling to find the safe operational duty cycle.
    """
    # --- Drilling Phase ---
    drill_time = 0
    time_step = 0.5 # Simulate in small time steps
    u = np.zeros(21) # Start cold
    
    while np.max(u) < temp_limit:
        u = solve_heat_equation(time_step, Q=50, initial_temp_profile=u)
        drill_time += time_step
        if drill_time > 300: # Safety break to prevent infinite loop
            print("Drill time exceeded safety, breaking.")
            break
            
    # --- Cooling Phase ---
    cool_time = 0
    # u is now the hot profile from the end of drilling
    while np.max(u) > cool_to_temp:
        u = solve_heat_equation(time_step, Q=0, initial_temp_profile=u)
        cool_time += time_step
        if cool_time > 300:
            print("Cool time exceeded safety, breaking.")
            break
            
    return drill_time, cool_time, u

# This block allows testing this module independently
if __name__ == '__main__':
    print("--- Testing Thermal Module ---")
    
    safe_drill_time, required_cool_time, final_temp_profile = find_duty_cycle()
    
    print(f"Maximum continuous drill time: {safe_drill_time:.1f} seconds")
    print(f"Required cooling time: {required_cool_time:.1f} seconds")
    
    # Visualize the final hot temperature profile
    L = 0.2
    x = np.linspace(0, L, 21)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, final_temp_profile, 'r-o')
    plt.axhline(80.0, color='gray', linestyle='--', label='Safety Limit (80 C)')
    plt.title("Temperature Profile of Drill Bit After Max Drilling")
    plt.xlabel("Position along drill (m)")
    plt.ylabel("Temperature (C)")
    plt.grid(True)
    plt.legend()
    plt.show()