# optimizer.py
# Module for finding the optimal mission plan.

from math import ceil, sqrt

# Define mission parameters as a dictionary for clarity
def get_default_params(settling_time, drill_time, cool_time):
    return {
        'targets': {
            'A': {'pos': (5, 8), 'science': 100, 'drill_req': 60},
            'B': {'pos': (12, 4), 'science': 150, 'drill_req': 90}
        },
        'start_pos': (0, 0),
        'costs': {
            'drive_speed': 1.0,      # m/s
            'drive_energy': 20,      # J/s
            'arm_deploy_energy': 500, # J
            'drill_energy': 100,     # J/s
            'cool_idle_energy': 5,   # J/s
        },
        'dynamics': {
            'settling_time': settling_time,
            'drill_time': drill_time,
            'cool_time': cool_time
        }
    }

def calculate_path_costs(path, params):
    """Calculates total science, time, and energy for a given path."""
    total_science = 0
    total_time = 0
    total_energy = 0
    current_pos = params['start_pos']

    for target_name in path:
        target = params['targets'][target_name]
        
        # --- Driving ---
        dist = sqrt((target['pos'][0] - current_pos[0])**2 + (target['pos'][1] - current_pos[1])**2)
        drive_time = dist / params['costs']['drive_speed']
        total_time += drive_time
        total_energy += drive_time * params['costs']['drive_energy']
        current_pos = target['pos']
        
        # --- Arm Deployment ---
        total_time += params['dynamics']['settling_time']
        total_energy += params['costs']['arm_deploy_energy']
        
        # --- Drilling and Cooling ---
        num_cycles = ceil(target['drill_req'] / params['dynamics']['drill_time'])
        drill_duration = num_cycles * params['dynamics']['drill_time']
        cool_duration = (num_cycles - 1) * params['dynamics']['cool_time'] # Cool between drills
        
        total_time += drill_duration + cool_duration
        total_energy += drill_duration * params['costs']['drill_energy']
        total_energy += cool_duration * params['costs']['cool_idle_energy']
        
        # --- Science ---
        total_science += target['science']
        
    return total_science, total_time, total_energy

def find_optimal_plan(params, constraints):
    """
    Iterates through possible paths and finds the best feasible one.
    This is a direct evaluation method, not a formal LP for simplicity.
    """
    possible_paths = [
        ['A'],
        ['B'],
        ['A', 'B'],
        ['B', 'A']
    ]
    
    best_plan = {'path': ['IDLE'], 'science': 0, 'time': 0, 'energy': 0}
    
    for path in possible_paths:
        science, time, energy = calculate_path_costs(path, params)
        
        # Check if the path is feasible
        if time <= constraints['max_time'] and energy <= constraints['max_energy']:
            # If it's better than the current best, update
            if science > best_plan['science']:
                best_plan = {'path': path, 'science': science, 'time': time, 'energy': energy}

    return best_plan


# This block allows testing this module independently
if __name__ == '__main__':
    print("--- Testing Optimizer Module ---")
    
    # Get parameters from other modules (using default/test values here)
    settling_time = 4.5
    drill_time = 40.0
    cool_time = 30.0
    
    mission_params = get_default_params(settling_time, drill_time, cool_time)
    
    mission_constraints = {
        'max_time': 900,
        'max_energy': 50000
    }
    
    optimal_plan = find_optimal_plan(mission_params, mission_constraints)
    
    print("\n--- Optimal Mission Plan ---")
    print(f"Path: {optimal_plan['path']}")
    print(f"Total Science: {optimal_plan['science']:.0f} points")
    print(f"Estimated Time: {optimal_plan['time']:.1f}s / {mission_constraints['max_time']}s")
    print(f"Estimated Energy: {optimal_plan['energy']:.1f}J / {mission_constraints['max_energy']}J")