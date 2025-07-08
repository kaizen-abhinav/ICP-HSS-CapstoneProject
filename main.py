# main.py
# The main script to run the integrated A.R.E.S. simulation.
# Final version with reliable movement and correct end-state display.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from math import sqrt, atan2, ceil, degrees

# Import all modules
import kinematics
import dynamics
import thermal
import optimizer

print("--- A.R.E.S. Mission Simulation ---")
print("Initializing all subsystems and calculating optimal plan...")

# --- 1. INITIALIZATION ---
_, arm_dynamics_func, settling_time = dynamics.get_arm_dynamics()
drill_time, cool_time, _ = thermal.find_duty_cycle()
params = optimizer.get_default_params(settling_time, drill_time, cool_time)
constraints = {'max_time': 900, 'max_energy': 50000}
plan = optimizer.find_optimal_plan(params, constraints)
print(f"\nOptimal plan found: {' -> '.join(plan['path'])}")
print("Starting simulation...")

# --- 2. SIMULATION SETUP ---
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
            if i < num_cycles - 1:
                mission_queue.append({'action': 'COOLING', 'duration': params['dynamics']['cool_time']})
        current_pos = target_info['pos']

# Simulation state variables
sim_time = 0.0; sim_step = 0.5; energy_used = 0.0
action_idx = 0; time_in_action = 0.0
rover_pos = np.array(params['start_pos'], dtype=float)
rover_angle = 0.0
drill_temp_profile = np.zeros(21)
drill_x_axis = np.linspace(0, 0.2, 21)
is_mission_complete = False

# --- 3. ANIMATION SETUP ---
fig = plt.figure(figsize=(14, 8))
fig.suptitle('A.R.E.S. Mission Dashboard (Final Version)')

# (All subplot setup is identical)
ax_world = fig.add_subplot(2, 2, 1); ax_world.set_xlim(-2, 15); ax_world.set_ylim(-2, 10); ax_world.set_title('World View')
ax_world.set_xlabel('X (m)'); ax_world.set_ylabel('Y (m)'); ax_world.grid(True); ax_world.set_aspect('equal')
target_A_marker, = ax_world.plot(params['targets']['A']['pos'][0], params['targets']['A']['pos'][1], 'bo', markersize=10, label='Target A')
target_B_marker, = ax_world.plot(params['targets']['B']['pos'][0], params['targets']['B']['pos'][1], 'go', markersize=10, label='Target B')
arm_line, = ax_world.plot([], [], 'k-', linewidth=3); ax_world.legend(loc='upper left')
ax_status = fig.add_subplot(2, 2, 2); ax_status.set_title('System Status'); ax_status.axis('off')
time_text = ax_status.text(0.05, 0.8, '', fontsize=12); energy_text = ax_status.text(0.05, 0.6, '', fontsize=12)
action_text = ax_status.text(0.05, 0.4, '', fontsize=12, fontweight='bold')
plan_text = ax_status.text(0.05, 0.2, f"Plan: {' -> '.join(plan['path'])}", fontsize=10, wrap=True)
ax_arm = fig.add_subplot(2, 2, 3); ax_arm.set_xlim(0, settling_time * 1.1); ax_arm.set_ylim(0, 1.4); ax_arm.set_title('Arm Joint Dynamics')
ax_arm.set_xlabel('Time in State (s)'); ax_arm.set_ylabel('Angle (rad)'); ax_arm.grid(True); arm_dynamics_line, = ax_arm.plot([], [], 'b-'); ax_arm.axhline(1.0, color='gray', linestyle='--')
ax_drill = fig.add_subplot(2, 2, 4); ax_drill.set_ylim(0, 100); ax_drill.set_title('Drill Bit Temperature')
ax_drill.set_xlabel('Position along bit (m)'); ax_drill.set_ylabel('Temp (C)'); ax_drill.grid(True); drill_line, = ax_drill.plot([], [], 'r-o', markersize=3)
ax_drill.axhline(80.0, color='k', linestyle='--', label='Safety Limit'); ax_drill.legend(loc='upper right')
plt.tight_layout(rect=[0, 0, 1, 0.96])

rover_artist = None
try:
    rover_img = plt.imread('rover.png')
except FileNotFoundError:
    print("\n\nERROR: 'rover.png' not found."); exit()

def update(frame):
    global sim_time, energy_used, action_idx, time_in_action, rover_pos, rover_angle, drill_temp_profile, rover_artist, is_mission_complete

    # ===============================================
    # NEW LOGIC: Check for completion at the start
    # ===============================================
    if is_mission_complete:
        return # Do nothing if mission is already marked as complete.

    if action_idx >= len(mission_queue):
        action_text.set_text('Current Action: MISSION COMPLETE')
        is_mission_complete = True # Set flag to stop all future updates
        return

    sim_time += sim_step; time_in_action += sim_step
    current_state = mission_queue[action_idx]
    action = current_state['action']
    
    # --- State Machine Logic ---
    if action == 'DRIVING':
        energy_used += params['costs']['drive_energy'] * sim_step
        start_pos = np.array(current_state['start_pos']); target_pos = np.array(current_state['target'])
        direction = target_pos - start_pos
        rover_angle = degrees(atan2(direction[1], direction[0]))
        fraction_done = min(time_in_action / current_state['duration'], 1.0)
        rover_pos = start_pos + direction * fraction_done
        arm_line.set_data([], [])

    elif action == 'DEPLOYING_ARM' or action == 'DRILLING' or action == 'COOLING':
        theta1, theta2 = np.deg2rad(45), np.deg2rad(-60)
        elbow_local, drill_local = kinematics.forward_kinematics(kinematics.LINK_1_LENGTH, kinematics.LINK_2_LENGTH, theta1, theta2)
        elbow_world = kinematics.world_coords(rover_pos, elbow_local); drill_world = kinematics.world_coords(rover_pos, drill_local)
        arm_line.set_data([rover_pos[0], elbow_world[0], drill_world[0]], [rover_pos[1], elbow_world[1], drill_world[1]])
        
        if action == 'DEPLOYING_ARM':
            energy_used += (params['costs']['arm_deploy_energy'] / current_state['duration']) * sim_step
            t_arm = np.linspace(0, time_in_action, 100); y_arm = arm_dynamics_func(t_arm)
            arm_dynamics_line.set_data(t_arm, y_arm)
        elif action == 'DRILLING':
            energy_used += params['costs']['drill_energy'] * sim_step
            drill_temp_profile = thermal.solve_heat_equation(sim_step, Q=50, initial_temp_profile=drill_temp_profile)
            drill_line.set_data(drill_x_axis, drill_temp_profile)
        elif action == 'COOLING':
            energy_used += params['costs']['cool_idle_energy'] * sim_step
            drill_temp_profile = thermal.solve_heat_equation(sim_step, Q=0, initial_temp_profile=drill_temp_profile)
            drill_line.set_data(drill_x_axis, drill_temp_profile)
            
    # State transition
    if time_in_action >= current_state['duration']:
        if action == 'DRIVING': rover_pos = np.array(current_state['target'])
        action_idx += 1; time_in_action = 0
        
    # Robust Redrawing Logic for Rover
    if rover_artist:
        rover_artist.remove()
    imagebox = OffsetImage(rover_img, zoom=0.1)
    rover_artist = AnnotationBbox(imagebox, (rover_pos[0], rover_pos[1]), frameon=False, pad=0)
    ax_world.add_artist(rover_artist)

    # Update text
    time_text.set_text(f'Mission Time: {sim_time:.1f}s / {constraints["max_time"]}s')
    energy_text.set_text(f'Energy Used: {energy_used:.0f}J / {constraints["max_energy"]}J')
    action_text.set_text(f'Current Action: {action}')
    
    return

# Run the animation
animation_frames = int((constraints['max_time'] / sim_step) + 50)
ani = FuncAnimation(fig, update, frames=animation_frames, interval=50, blit=False, repeat=False)
plt.show()

print("\nSimulation finished.")