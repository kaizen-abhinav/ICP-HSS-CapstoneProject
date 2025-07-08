# main.py
# The main script to run the integrated A.R.E.S. simulation.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt, atan2, ceil

# Import all modules
import kinematics
import dynamics
import thermal
import optimizer

print("--- A.R.E.S. Mission Simulation ---")
print("Initializing all subsystems and calculating optimal plan...")

# --- 1. INITIALIZATION: Get constraints from all modules ---
_, arm_dynamics_func, settling_time = dynamics.get_arm_dynamics()
drill_time, cool_time, _ = thermal.find_duty_cycle()
params = optimizer.get_default_params(settling_time, drill_time, cool_time)
constraints = {'max_time': 900, 'max_energy': 50000}

# Find the best plan
plan = optimizer.find_optimal_plan(params, constraints)
print(f"\nOptimal plan found: {' -> '.join(plan['path'])}")
print("Starting simulation...")

# --- 2. SIMULATION SETUP ---
# Create the state queue based on the plan
mission_queue = []
current_pos = params['start_pos']
if plan['path'][0] != 'IDLE':
    for target_name in plan['path']:
        target_info = params['targets'][target_name]
        dist = sqrt((target_info['pos'][0] - current_pos[0])**2 + (target_info['pos'][1] - current_pos[1])**2)
        drive_duration = dist / params['costs']['drive_speed']
        mission_queue.append({'action': 'DRIVING', 'duration': drive_duration, 'target': target_info['pos'], 'start_pos': current_pos})
        
        mission_queue.append({'action': 'DEPLOYING_ARM', 'duration': params['dynamics']['settling_time']})
        
        num_cycles = ceil(target_info['drill_req'] / params['dynamics']['drill_time'])
        for i in range(num_cycles):
            mission_queue.append({'action': 'DRILLING', 'duration': params['dynamics']['drill_time']})
            if i < num_cycles - 1: # Don't cool after the last drill
                mission_queue.append({'action': 'COOLING', 'duration': params['dynamics']['cool_time']})
        
        current_pos = target_info['pos']

# Simulation state variables
sim_time = 0.0
sim_step = 0.5 # Update every 0.5s
energy_used = 0.0
action_idx = 0
time_in_action = 0.0
rover_pos = np.array(params['start_pos'], dtype=float)

# Thermal simulation variables
drill_temp_profile = np.zeros(21)
drill_x_axis = np.linspace(0, 0.2, 21)

# --- 3. ANIMATION SETUP ---
fig = plt.figure(figsize=(14, 8))
fig.suptitle('A.R.E.S. Mission Dashboard')

# World View
ax_world = fig.add_subplot(2, 2, 1)
ax_world.set_xlim(-2, 15)
ax_world.set_ylim(-2, 10)
ax_world.set_title('World View')
ax_world.set_xlabel('X (m)'); ax_world.set_ylabel('Y (m)')
ax_world.grid(True); ax_world.set_aspect('equal')
rover_marker, = ax_world.plot([], [], 'rs', markersize=10, label='A.R.E.S. Rover') # Start with empty data
target_A_marker, = ax_world.plot(params['targets']['A']['pos'][0], params['targets']['A']['pos'][1], 'bo', markersize=10, label='Target A')
target_B_marker, = ax_world.plot(params['targets']['B']['pos'][0], params['targets']['B']['pos'][1], 'go', markersize=10, label='Target B')
ax_world.legend()

# System Status
ax_status = fig.add_subplot(2, 2, 2)
ax_status.set_title('System Status')
ax_status.axis('off')
time_text = ax_status.text(0.05, 0.8, '', fontsize=12)
energy_text = ax_status.text(0.05, 0.6, '', fontsize=12)
action_text = ax_status.text(0.05, 0.4, '', fontsize=12, fontweight='bold')
plan_text = ax_status.text(0.05, 0.2, f"Plan: {' -> '.join(plan['path'])}", fontsize=10, wrap=True)

# Arm Dynamics
ax_arm = fig.add_subplot(2, 2, 3)
ax_arm.set_xlim(0, settling_time * 1.1)
ax_arm.set_ylim(0, 1.4)
ax_arm.set_title('Arm Joint Dynamics')
ax_arm.set_xlabel('Time in State (s)'); ax_arm.set_ylabel('Angle (rad)')
ax_arm.grid(True)
arm_line, = ax_arm.plot([], [], 'b-')
ax_arm.axhline(1.0, color='gray', linestyle='--')

# Drill Temperature
ax_drill = fig.add_subplot(2, 2, 4)
ax_drill.set_ylim(0, 100)
ax_drill.set_title('Drill Bit Temperature')
ax_drill.set_xlabel('Position along bit (m)'); ax_drill.set_ylabel('Temp (C)')
ax_drill.grid(True)
drill_line, = ax_drill.plot([], [], 'r-o', markersize=3)
ax_drill.axhline(80.0, color='k', linestyle='--', label='Safety Limit')
ax_drill.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])

def update(frame):
    global sim_time, energy_used, action_idx, time_in_action, rover_pos, drill_temp_profile

    if action_idx >= len(mission_queue):
        action_text.set_text('Current Action: MISSION COMPLETE')
        return rover_marker, time_text, energy_text, action_text, arm_line, drill_line # Stop updating
    
    # Update time and action
    sim_time += sim_step
    time_in_action += sim_step
    
    current_state = mission_queue[action_idx]
    action = current_state['action']
    
    # --- State Machine Logic ---
    if action == 'DRIVING':
        energy_used += params['costs']['drive_energy'] * sim_step
        
        # Linear interpolation for rover position
        start_pos = np.array(current_state['start_pos'])
        target_pos = np.array(current_state['target'])
        fraction_done = min(time_in_action / current_state['duration'], 1.0)
        rover_pos = start_pos + (target_pos - start_pos) * fraction_done
    
    elif action == 'DEPLOYING_ARM':
        energy_used += (params['costs']['arm_deploy_energy'] / current_state['duration']) * sim_step
        t_arm = np.linspace(0, time_in_action, 100)
        y_arm = arm_dynamics_func(t_arm)
        arm_line.set_data(t_arm, y_arm)
    
    elif action == 'DRILLING':
        energy_used += params['costs']['drill_energy'] * sim_step
        drill_temp_profile = thermal.solve_heat_equation(sim_step, Q=50, initial_temp_profile=drill_temp_profile)
        drill_line.set_data(drill_x_axis, drill_temp_profile)
        
    elif action == 'COOLING':
        energy_used += params['costs']['cool_idle_energy'] * sim_step
        drill_temp_profile = thermal.solve_heat_equation(sim_step, Q=0, initial_temp_profile=drill_temp_profile)
        drill_line.set_data(drill_x_axis, drill_temp_profile)
        
    # Check if action is complete
    if time_in_action >= current_state['duration']:
        # Finalize state to avoid floating point errors
        if action == 'DRIVING':
            rover_pos = np.array(current_state['target'])
        
        action_idx += 1
        time_in_action = 0
        # Clear plots from previous state
        if action == 'DEPLOYING_ARM':
            arm_line.set_data([], [])

    # Update plots and text
    # ===============================================
    # ERROR FIX IS HERE
    # ===============================================
    rover_marker.set_data([rover_pos[0]], [rover_pos[1]])
    
    time_text.set_text(f'Mission Time: {sim_time:.1f}s / {constraints["max_time"]}s')
    energy_text.set_text(f'Energy Used: {energy_used:.0f}J / {constraints["max_energy"]}J')
    action_text.set_text(f'Current Action: {action}')
    
    return rover_marker, time_text, energy_text, action_text, arm_line, drill_line


# Run the animation
animation_frames = int((constraints['max_time'] / sim_step) + 50) # Add buffer
ani = FuncAnimation(fig, update, frames=animation_frames, interval=50, blit=False, repeat=False)
plt.show()

print("\nSimulation finished.")