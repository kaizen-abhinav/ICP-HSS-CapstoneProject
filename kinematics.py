# kinematics.py
# Module for handling rover and arm positions and transformations.

import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
LINK_1_LENGTH = 0.8  # meters
LINK_2_LENGTH = 0.6  # meters

def forward_kinematics(l1, l2, theta1, theta2):
    """
    Calculates the (x, y) position of the arm's end-effector in the rover's frame.
    Angles are in radians.
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    
    # Also return the position of the elbow joint for plotting
    x_elbow = l1 * np.cos(theta1)
    y_elbow = l1 * np.sin(theta1)
    
    return (x_elbow, y_elbow), (x, y)

def world_coords(rover_pos, arm_coords):
    """
    Converts arm coordinates from the rover's frame to the world frame.
    """
    return rover_pos[0] + arm_coords[0], rover_pos[1] + arm_coords[1]

# This block allows testing this module independently
if __name__ == '__main__':
    print("--- Testing Kinematics Module ---")
    
    # Test case
    l1 = LINK_1_LENGTH
    l2 = LINK_2_LENGTH
    # Angles in degrees, then converted to radians
    t1_deg, t2_deg = 45, 30
    theta1_rad = np.deg2rad(t1_deg)
    theta2_rad = np.deg2rad(t2_deg)
    
    rover_position = np.array([2.0, 3.0])
    
    # Calculate kinematics
    elbow_pos_local, drill_pos_local = forward_kinematics(l1, l2, theta1_rad, theta2_rad)
    
    drill_pos_world = world_coords(rover_position, drill_pos_local)

    print(f"Rover Position: {rover_position}")
    print(f"Joint Angles (deg): theta1={t1_deg}, theta2={t2_deg}")
    print(f"Drill Position (relative to rover): {np.round(drill_pos_local, 2)}")
    print(f"Drill Position (world coordinates): {np.round(drill_pos_world, 2)}")

    # Visualize the test case
    plt.figure(figsize=(8, 8))
    plt.plot(rover_position[0], rover_position[1], 'ks', markersize=15, label='Rover Base')
    
    # Plot arm in world coordinates
    elbow_pos_world = world_coords(rover_position, elbow_pos_local)
    plt.plot([rover_position[0], elbow_pos_world[0]], 
             [rover_position[1], elbow_pos_world[1]], 'r-o', linewidth=3, label='Link 1')
    plt.plot([elbow_pos_world[0], drill_pos_world[0]], 
             [elbow_pos_world[1], drill_pos_world[1]], 'b-o', linewidth=3, label='Link 2')
    
    plt.plot(drill_pos_world[0], drill_pos_world[1], 'gX', markersize=12, label='Drill (End-Effector)')
    
    plt.title("Kinematics Test")
    plt.xlabel("World X (m)")
    plt.ylabel("World Y (m)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()