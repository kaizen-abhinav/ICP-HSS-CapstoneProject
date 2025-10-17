# main.py
# Coconut harvesting robot simulation with enhanced visualization.

from __future__ import annotations

import argparse
import csv
import sys
from math import atan2, degrees
from pathlib import Path
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, patches, transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

import dynamics
import kinematics
import optimizer
import thermal

plt.style.use('seaborn-v0_8')


def run_dashboard(
    seed: Optional[int] = None,
    *,
    show: bool = True,
    verbose: bool = True,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and run the coconut harvester dashboard."""

    run_seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 1_000_000))

    if verbose:
        print('--- Coconut Field Operations ---')
        print('Initializing subsystems, generating field, and planning harvest route...')

    _, arm_dynamics_func, settling_time = dynamics.get_arm_dynamics()
    cut_time, cool_time, _ = thermal.find_duty_cycle(temp_limit=70.0, cool_to_temp=25.0)

    params = optimizer.get_default_params(settling_time, cut_time, cool_time, seed=run_seed)
    constraints = {'max_time': 1_200.0, 'max_energy': 65_000.0}
    log_file_path = Path(log_path).expanduser() if log_path else None
    telemetry_headers = [
        'time',
        'action',
        'tree',
        'rover_x',
        'rover_y',
        'energy',
        'trees_scanned',
        'harvested_coconuts',
        'max_tool_temp',
        'speed_multiplier',
        'battery_pct',
        'wind_mps',
        'humidity_pct',
        'ambient_temp',
        'nearest_obstacle_clearance',
        'nav_confidence',
        'planner_phase',
        'distance_to_target',
        'eta_to_target',
        'cpu_load_pct',
        'loop_latency_ms',
        'packet_loss_pct',
        'wheel_currents',
        'wheel_torques',
    'wheel_temps',
    'suspension_loads',
        'cutter_rpm',
    'pack_voltage',
    'pack_current',
    'battery_soc',
    'thermal_margin',
        'arm_joint_angles',
    'arm_joint_temps',
    'end_effector_force',
        'ai_mode',
        'ai_intent',
        'ai_risk_assessment',
    ]
    plan = optimizer.find_optimal_plan(params, constraints)

    if not plan['path']:
        if verbose:
            print('\nNo feasible harvesting plan found under current constraints. Aborting mission.')
        if log_file_path is not None:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with log_file_path.open('w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(telemetry_headers)
        if show:
            plt.close('all')
        return {
            'figure': None,
            'animation': None,
            'plan': plan,
            'seed': run_seed,
            'update': None,
            'controls': None,
            'telemetry': [],
            'log_file': str(log_file_path) if log_file_path is not None else None,
        }

    field_info = params['field']
    field_width = float(field_info['width'])
    field_height = float(field_info['height'])
    obstacles = list(field_info.get('obstacles', params.get('obstacles', [])))

    tree_status: Dict[str, str] = {name: 'pending' for name in params['trees']}
    for blocked_tree in plan.get('blocked_trees', []):
        if blocked_tree in tree_status:
            tree_status[blocked_tree] = 'blocked'
    for skipped_tree in plan.get('skipped_trees', []):
        if skipped_tree in tree_status and tree_status[skipped_tree] == 'pending':
            tree_status[skipped_tree] = 'skipped'

    base_sim_step = 0.5
    env_rng = np.random.default_rng(run_seed + 17)

    rover_pos = np.array(params['start_pos'], dtype=float)
    rover_angle = 0.0
    mission_queue: list[Dict[str, Any]] = []
    travel_speed = max(params['costs']['drive_speed'], 0.1)
    current_pos = rover_pos.copy()

    for tree_name in plan['path']:
        tree_info = params['trees'][tree_name]
        target_vec = np.array(tree_info['pos'], dtype=float)
        drive_distance = float(np.linalg.norm(target_vec - current_pos))
        drive_duration = drive_distance / travel_speed if drive_distance > 1e-6 else 0.0
        mission_queue.append(
            {
                'action': 'DRIVING',
                'duration': drive_duration,
                'start_pos': tuple(current_pos),
                'target': tuple(target_vec),
                'tree': tree_name,
            }
        )
        mission_queue.append(
            {
                'action': 'SCAN_TREE',
                'duration': params['operations']['scan_time'],
                'tree': tree_name,
            }
        )
        if tree_info['coconuts'] > 0:
            mission_queue.append(
                {
                    'action': 'DEPLOYING_ARM',
                    'duration': params['dynamics']['settling_time'],
                    'tree': tree_name,
                }
            )
            mission_queue.append(
                {
                    'action': 'CUT_TREE',
                    'duration': params['dynamics']['cut_time'],
                    'tree': tree_name,
                }
            )
            mission_queue.append(
                {
                    'action': 'COOL_TOOL',
                    'duration': params['dynamics']['cool_time'],
                    'tree': tree_name,
                }
            )
        current_pos = target_vec

    telemetry_records: list[Dict[str, Any]] = []
    telemetry_written = False
    is_mission_complete = False
    action_idx = 0
    time_in_action = 0.0
    sim_time = 0.0
    energy_used = 0.0
    trees_scanned = 0
    harvested_coconuts = 0
    total_drive_distance = 0.0
    drive_time_accum = 0.0
    path_positions = [rover_pos.copy()]
    tool_x_axis = np.linspace(0.0, 0.6, 25)
    tool_temp_profile = np.full_like(tool_x_axis, 28.0)
    max_tool_temp_seen = float(np.max(tool_temp_profile))
    latest_clearance = float('inf')
    action_counter: Counter[str] = Counter()
    controls = {'is_paused': False, 'skip': False, 'speed': 1.0}

    environment_state = {
        'battery': 100.0,
        'wind_speed': float(env_rng.uniform(2.0, 6.0)),
        'wind_direction': float(env_rng.uniform(0.0, 360.0)),
        'humidity': float(env_rng.uniform(45.0, 80.0)),
        'ambient_temp': float(env_rng.uniform(26.0, 34.0)),
        'tool_temp': max_tool_temp_seen,
    }

    planner_context = {
        'phase': 'BOOT',
        'target_tree': '',
        'distance_remaining': 0.0,
        'eta_remaining': 0.0,
        'nav_confidence': 0.97,
        'risk_index': 0.05,
    }
    system_metrics = {
        'cpu_load': float(env_rng.uniform(18.0, 28.0)),
        'loop_latency': float(env_rng.uniform(22.0, 40.0)),
        'packet_loss': 0.2,
    }
    planner_log: Deque[str] = deque(maxlen=12)

    drivetrain_state = {
        'wheel_currents': np.zeros(6),
        'wheel_torques': np.zeros(6),
        'wheel_temps': np.full(6, 42.0),
        'suspension_loads': np.zeros(3),
        'cutter_rpm': 0.0,
        'pack_voltage': 52.0,
        'pack_current': 0.0,
        'soc': 100.0,
        'thermal_margin': 35.0,
    }
    arm_state = {
        'joint_angles': np.zeros(6),
        'joint_temps': np.full(6, 38.0),
        'end_effector_force': 0.0,
    }
    joint_pose_library = {
        'stowed': np.array([0.15, -0.42, 0.25, 0.0, 0.0, 0.0]),
        'scan': np.array([0.55, -0.85, 0.6, 0.1, -0.05, 0.0]),
        'deploy': np.array([0.95, -1.12, 0.88, 0.25, -0.18, 0.05]),
        'cut': np.array([1.18, -1.28, 1.05, 0.32, -0.24, 0.12]),
    }
    ai_state = {
        'mode': 'Boot',
        'intent': 'Initializing systems',
        'next_action': 'Assess field layout',
        'risk': 'Nominal',
        'thoughts': deque(maxlen=8),
    }

    wheel_profile = np.array([1.00, 0.94, 0.98, 1.03, 1.07, 1.02])
    torque_profile = np.array([1.00, 0.97, 1.05, 1.02, 0.98, 1.04])
    suspension_profile = np.array([1.00, 0.92, 1.08])

    if obstacles:
        initial_clearances = [
            max(0.0, np.linalg.norm(rover_pos - np.array(obs['center'])) - obs['radius'])
            for obs in obstacles
        ]
        if initial_clearances:
            latest_clearance = min(initial_clearances)

    def _generate_orchard_texture(width: float, height: float) -> np.ndarray:
        cols = 180
        rows = 140
        x = np.linspace(0.0, width, cols)
        y = np.linspace(0.0, height, rows)
        xx, yy = np.meshgrid(x, y)
        furrow_frequency = 2.8
        furrow_base = 0.45 + 0.35 * np.sin((yy / height) * np.pi * furrow_frequency + env_rng.uniform(0.0, 2.0))
        irrigation = 0.08 * np.sin((xx / width) * np.pi * 7.5 + env_rng.uniform(0.0, 2.0))
        canopy_shadow = 0.2 * np.cos((xx / width) * np.pi * 5.3 + (yy / height) * np.pi * 3.1)
        green_bloom = 0.12 * np.exp(-((xx - width * 0.5) ** 2 + (yy - height * 0.5) ** 2) / (0.18 * width * height))
        texture = furrow_base + irrigation + canopy_shadow + green_bloom
        texture += env_rng.normal(0.0, 0.04, size=(rows, cols))
        texture -= texture.min()
        texture /= max(np.ptp(texture), 1e-6)
        return texture

    terrain_map = _generate_orchard_texture(field_width, field_height)

    fig = plt.figure(figsize=(18.2, 10.6))
    fig.patch.set_facecolor('#0f1117')
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.16)

    def _add_panel_frame(ax: plt.Axes, pad: float = 0.02) -> None:
        frame = patches.FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            boxstyle=f"round,pad={pad}",
            linewidth=1.1,
            edgecolor='#2c313c',
            facecolor='none',
            zorder=12,
        )
        frame.set_clip_on(False)
        ax.add_patch(frame)
    gs = gridspec.GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.42, 0.88, 1.22],
    width_ratios=[8.0, 0.75, 0.75, 0.95, 1.1, 1.2],
        hspace=0.54,
        wspace=0.38,
    )
    ax_world = fig.add_subplot(gs[0, 0:4])

    ax_world.set_xlim(-3, field_width + 3)
    ax_world.set_ylim(-3, field_height + 3)
    ax_world.set_title('Field Telemetry View', loc='left', fontsize=12, color='#f5f5f5', fontfamily='Consolas')
    ax_world.set_xlabel('X (m)', color='#e0e0e0', fontfamily='Consolas')
    ax_world.set_ylabel('Y (m)', color='#e0e0e0', fontfamily='Consolas')
    ax_world.set_aspect('equal')
    ax_world.set_facecolor('#1b1f29')
    ax_world.grid(True, color='#2c313c', linestyle=':', linewidth=0.8)
    ax_world.tick_params(colors='#c5d0e3')
    for spine in ax_world.spines.values():
        spine.set_color('#39404d')
    orchard_cmap = LinearSegmentedColormap.from_list(
        'orchard',
        ['#151012', '#241712', '#3a2218', '#51321b', '#4c5537', '#3c7044'],
    )
    terrain_img = ax_world.imshow(
        terrain_map,
        extent=(0, field_width, 0, field_height),
        origin='lower',
        cmap=orchard_cmap,
        interpolation='lanczos',
        alpha=0.82,
        zorder=0,
    )
    field_outline = patches.Rectangle(
        (0, 0),
        field_width,
        field_height,
        linewidth=2.0,
        edgecolor='#2e7d32',
        facecolor='none',
        linestyle='--',
    )
    ax_world.add_patch(field_outline)

    obstacle_patches = []
    for obstacle in obstacles:
        patch = patches.Circle(
            obstacle['center'],
            obstacle['radius'],
            facecolor='#90a4ae',
            edgecolor='#455a64',
            linewidth=1.3,
            alpha=0.35,
        )
        ax_world.add_patch(patch)
        label = obstacle.get('name', 'Obstacle')
        ax_world.text(
            obstacle['center'][0],
            obstacle['center'][1] + obstacle['radius'] + 0.4,
            label,
            fontsize=8,
            color='#aab2c8',
            ha='center',
            fontfamily='Consolas',
        )
        obstacle_patches.append(patch)

    arm_line, = ax_world.plot([], [], color='#4a148c', linewidth=3, alpha=0.9)
    path_line, = ax_world.plot([], [], color='#1565c0', linewidth=2, linestyle='--', alpha=0.6)

    ax_tree_table = fig.add_subplot(gs[0, 4])
    ax_tree_table.set_title('Tree Tracker', fontsize=11, loc='left', pad=8, color='#f5f5f5', fontfamily='Consolas')
    ax_tree_table.axis('off')
    _add_panel_frame(ax_tree_table)

    ax_ai_panel = fig.add_subplot(gs[0, 5])
    ax_ai_panel.set_facecolor('#151922')
    ax_ai_panel.set_title('AI Mission Brain', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_ai_panel.axis('off')
    _add_panel_frame(ax_ai_panel)
    ai_text = ax_ai_panel.text(
        0.04,
        0.94,
        '',
        fontsize=9.5,
        color='#e0e0e0',
        va='top',
        fontfamily='Consolas',
        transform=ax_ai_panel.transAxes,
        wrap=True,
    )

    def _risk_to_text(value: float) -> str:
        if value < 0.35:
            return 'Nominal'
        if value < 0.8:
            return 'Watch'
        if value < 1.2:
            return 'Caution'
        return 'Critical'

    def _update_ai_panel(thought: Optional[str] = None) -> None:
        if thought:
            ai_state['thoughts'].appendleft(thought)
        ai_state['risk'] = _risk_to_text(planner_context['risk_index'])
        header_lines = [
            f"Mode: {ai_state['mode']}",
            f"Intent: {ai_state['intent']}",
            f"Risk: {ai_state['risk']} ({planner_context['risk_index']:.2f})",
            f"Next: {ai_state['next_action']}",
        ]
        if ai_state['thoughts']:
            header_lines.append('')
            header_lines.append('Recent Thoughts:')
            for entry in ai_state['thoughts']:
                header_lines.append(f"  - {entry}")
        ai_text.set_text('\n'.join(header_lines))

    def _describe_action(action_name: str, tree: str) -> str:
        action_map = {
            'DRIVING': f"Navigate toward {tree}",
            'SCAN_TREE': f"Analyze canopy at {tree}",
            'DEPLOYING_ARM': f"Align arm for {tree}",
            'CUT_TREE': f"Harvest coconuts from {tree}",
            'COOL_TOOL': 'Restore blade temperature',
        }
        return action_map.get(action_name, 'Monitor systems')

    def _next_action_label(current_idx: int) -> str:
        upcoming = mission_queue[current_idx + 1:] if current_idx + 1 < len(mission_queue) else []
        if not upcoming:
            return 'Await completion'
        next_item = upcoming[0]
        return f"{next_item['action']}({next_item.get('tree', 'Field')})"

    def _update_ai_context(action_name: str, tree: str) -> None:
        ai_state['mode'] = action_name.replace('_', ' ').title()
        ai_state['intent'] = _describe_action(action_name, tree or 'field')
        ai_state['next_action'] = _next_action_label(action_idx)
        clearance_text = f"{latest_clearance:.1f} m" if np.isfinite(latest_clearance) else 'clear'
        if action_name == 'DRIVING':
            new_thought = f"Plotting smooth approach; clearance {clearance_text}."
        elif action_name == 'SCAN_TREE':
            new_thought = f"Scanning canopy signature for {tree}."
        elif action_name == 'DEPLOYING_ARM':
            new_thought = f"Positioning joints for stable reach on {tree}."
        elif action_name == 'CUT_TREE':
            new_thought = f"Executing cut; monitoring cutter temp {max_tool_temp_seen:.1f} °C."
        elif action_name == 'COOL_TOOL':
            new_thought = "Cooling blade to safe margins."
        else:
            new_thought = "Maintaining situational awareness."
        ai_state['risk'] = _risk_to_text(planner_context['risk_index'])
        _update_ai_panel(new_thought)

    for row in np.arange(1.25, field_height, 4.8):
        ax_world.axhspan(row, min(row + 1.4, field_height), color='#1e2a23', alpha=0.18, zorder=0.35)

    orchard_rng = np.random.default_rng(run_seed + 101)
    tree_markers: Dict[str, Any] = {}
    canopy_highlights = []
    for name, info in params['trees'].items():
        tree_info = params['trees'][name]
        canopy_radius = 1.2 + 0.08 * max(tree_info['coconuts'], 1)
        canopy_patch = patches.Circle(
            info['pos'],
            radius=canopy_radius,
            facecolor='#2f6f45',
            edgecolor='#1d3f2c',
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
        )
        ax_world.add_patch(canopy_patch)
        trunk_patch = patches.Circle(
            info['pos'],
            radius=0.22,
            facecolor='#5d4037',
            edgecolor='#3e2723',
            linewidth=1.0,
            zorder=4,
        )
        ax_world.add_patch(trunk_patch)
        coconut_count = max(tree_info['coconuts'], 0)
        if coconut_count > 0:
            coconut_angles = orchard_rng.uniform(0.0, 2 * np.pi, size=coconut_count)
            coconut_radii = orchard_rng.uniform(0.35, max(0.4, canopy_radius - 0.25), size=coconut_count)
            coconut_points = np.stack(
                (
                    info['pos'][0] + coconut_radii * np.cos(coconut_angles),
                    info['pos'][1] + coconut_radii * np.sin(coconut_angles),
                ),
                axis=1,
            )
            ax_world.scatter(
                coconut_points[:, 0],
                coconut_points[:, 1],
                s=18,
                c='#c49b4e',
                edgecolors='#8d6e41',
                linewidths=0.6,
                alpha=0.9,
                zorder=5,
            )
        leaf_angles = orchard_rng.uniform(0.0, 2 * np.pi, size=12)
        leaf_radii = orchard_rng.uniform(0.18, max(0.28, canopy_radius - 0.4), size=12)
        leaf_points = np.stack(
            (
                info['pos'][0] + leaf_radii * np.cos(leaf_angles),
                info['pos'][1] + leaf_radii * np.sin(leaf_angles),
            ),
            axis=1,
        )
        canopy_highlights.append((leaf_points, canopy_radius))
        ax_world.text(
            info['pos'][0] - 0.9,
            info['pos'][1] - canopy_radius - 0.4,
            name,
            fontsize=8,
            color='#d0d7ff',
            weight='bold',
            fontfamily='Consolas',
            zorder=6,
        )
        tree_markers[name] = canopy_patch

    for points, radius in canopy_highlights:
        ax_world.scatter(
            points[:, 0],
            points[:, 1],
            s=22,
            c='#6cbf6f',
            alpha=0.28,
            linewidths=0.0,
            marker='o',
            zorder=4.4,
        )

    tree_names_sorted = sorted(params['trees'].keys())
    tree_table_data = [
        [name, params['trees'][name]['coconuts'], tree_status[name]] for name in tree_names_sorted
    ]
    tree_table = ax_tree_table.table(
        cellText=tree_table_data,
        colLabels=['Tree', 'Coconuts', 'Status'],
        cellLoc='left',
        colLoc='left',
        loc='center',
    )
    tree_table.auto_set_font_size(False)
    tree_table.set_fontsize(9)
    tree_table.scale(1.1, 1.3)
    for cell in tree_table.get_celld().values():
        cell.set_facecolor('#151922')
        cell.set_edgecolor('#2c313c')
        cell.get_text().set_color('#e5e9f0')
        cell.get_text().set_fontfamily('Consolas')

    ax_progress = fig.add_subplot(gs[1, 0])
    ax_progress.set_facecolor('#151922')
    ax_progress.set_title('Mission Utilization', loc='left', fontsize=12, color='#f5f5f5', fontfamily='Consolas')
    ax_progress.set_xlim(0, 1.0)
    ax_progress.set_ylim(-0.2, 1.2)
    ax_progress.set_yticks([0.75, 0.25])
    ax_progress.set_yticklabels(['Energy', 'Time'], fontfamily='Consolas', color='#c5d0e3')
    ax_progress.set_xlabel('Fraction of Allocation', color='#c5d0e3', fontfamily='Consolas')
    ax_progress.tick_params(axis='x', colors='#c5d0e3')
    ax_progress.grid(True, axis='x', linestyle=':', color='#324050')
    ax_progress.axvline(1.0, color='#62727b', linestyle='--', linewidth=1.0)
    _add_panel_frame(ax_progress, pad=0.015)

    time_bar = ax_progress.barh(0.25, width=0.0, height=0.35, color='#29b6f6', alpha=0.9)[0]
    energy_bar = ax_progress.barh(0.75, width=0.0, height=0.35, color='#ef6c00', alpha=0.9)[0]

    ax_arm = fig.add_subplot(gs[1, 1])
    ax_arm.set_facecolor('#151922')
    ax_arm.set_xlim(0, max(settling_time * 1.15, 1.0))
    ax_arm.set_ylim(0, 1.4)
    ax_arm.set_title('Arm Joint Response', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_arm.set_xlabel('Time in State (s)', color='#c5d0e3', fontfamily='Consolas')
    ax_arm.set_ylabel('Angle (rad)', color='#c5d0e3', fontfamily='Consolas')
    ax_arm.tick_params(colors='#c5d0e3')
    ax_arm.grid(True, linestyle=':', linewidth=0.8, color='#324050')
    ax_arm.axhline(1.0, color='#62727b', linestyle='--', linewidth=1.0)
    arm_dynamics_line, = ax_arm.plot([], [], color='#5e35b1', linewidth=2.2)
    _add_panel_frame(ax_arm, pad=0.015)

    ax_tool = fig.add_subplot(gs[1, 2])
    ax_tool.set_facecolor('#151922')
    ax_tool.set_ylim(0, 100)
    ax_tool.set_title('Cutting Tool Temperature', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_tool.set_xlabel('Position along blade (m)', color='#c5d0e3', fontfamily='Consolas')
    ax_tool.set_ylabel('Temp (C)', color='#c5d0e3', fontfamily='Consolas')
    ax_tool.tick_params(colors='#c5d0e3')
    ax_tool.grid(True, linestyle=':', linewidth=0.8, color='#324050')
    ax_tool.axhline(70.0, color='#ff7043', linestyle='--', linewidth=1.2, label='Safety Limit')
    tool_line, = ax_tool.plot([], [], 'o-', color='#c62828', markersize=4)
    ax_tool.legend(loc='upper right', fontsize=9)
    _add_panel_frame(ax_tool, pad=0.015)

    ax_diagnostics = fig.add_subplot(gs[1, 3:5])
    ax_diagnostics.set_facecolor('#151922')
    ax_diagnostics.set_title('Systems Telemetry', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_diagnostics.axis('off')
    diagnostic_text = ax_diagnostics.text(
        0.02,
        0.92,
        '',
        fontsize=10,
        color='#e0e0e0',
        va='top',
        fontfamily='Consolas',
        transform=ax_diagnostics.transAxes,
        wrap=True,
    )
    _add_panel_frame(ax_diagnostics)

    ax_obstacle_panel = fig.add_subplot(gs[1, 5])
    ax_obstacle_panel.set_facecolor('#151922')
    ax_obstacle_panel.set_title('Obstacle Clearance', loc='center', fontsize=11, color='#f5f5f5', fontfamily='Consolas', pad=16)
    ax_obstacle_panel.set_xlabel('Clearance (m)', color='#c5d0e3', fontfamily='Consolas')
    ax_obstacle_panel.tick_params(colors='#c5d0e3')
    for spine in ax_obstacle_panel.spines.values():
        spine.set_color('#324050')
    obstacle_empty_text = None
    obstacle_bars = []
    obstacle_names = [obs.get('name', f"Obstacle-{idx}") for idx, obs in enumerate(obstacles, start=1)]
    if obstacle_names:
        initial_clearances = []
        for obs in obstacles:
            center = np.array(obs['center'])
            clearance = max(0.0, np.linalg.norm(rover_pos - center) - obs['radius'])
            initial_clearances.append(clearance)
        obstacle_bars = ax_obstacle_panel.barh(obstacle_names, initial_clearances, color='#64b5f6')
        ax_obstacle_panel.invert_yaxis()
        ax_obstacle_panel.set_xlim(0.0, max(initial_clearances + [5.0]) + 2.0)
        latest_clearance = min(initial_clearances) if initial_clearances else float('inf')
    else:
        ax_obstacle_panel.axis('off')
        obstacle_empty_text = ax_obstacle_panel.text(
            0.1,
            0.5,
            'No obstacles detected',
            fontsize=10,
            color='#90a4ae',
            transform=ax_obstacle_panel.transAxes,
        )
    _add_panel_frame(ax_obstacle_panel)

    ax_realtime = fig.add_subplot(gs[2, 0:3])
    ax_realtime.set_facecolor('#151922')
    ax_realtime.set_title('Real-Time Telemetry', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_realtime.axis('off')
    realtime_text = ax_realtime.text(
        0.02,
        0.92,
        '',
        fontsize=9.5,
        color='#e0e0e0',
        va='top',
        fontfamily='Consolas',
        transform=ax_realtime.transAxes,
        wrap=True,
    )
    _add_panel_frame(ax_realtime)
    ax_mission_panel = fig.add_subplot(gs[2, 3])
    ax_mission_panel.set_facecolor('#151922')
    ax_mission_panel.set_title('Mission Timeline', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_mission_panel.axis('off')
    mission_text = ax_mission_panel.text(
        0.05,
        0.92,
        '',
        fontsize=9.5,
        color='#e0e0e0',
        va='top',
        fontfamily='Consolas',
        transform=ax_mission_panel.transAxes,
        wrap=True,
    )
    _add_panel_frame(ax_mission_panel)

    ax_summary_panel = fig.add_subplot(gs[2, 4])
    ax_summary_panel.set_facecolor('#151922')
    ax_summary_panel.set_title('Mission Summary', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_summary_panel.axis('off')
    summary_text = ax_summary_panel.text(
        0.05,
        0.92,
        '',
        fontsize=9.5,
        color='#e0e0e0',
        va='top',
        fontfamily='Consolas',
        transform=ax_summary_panel.transAxes,
        wrap=True,
    )
    _add_panel_frame(ax_summary_panel)

    ax_log_panel = fig.add_subplot(gs[2, 5])
    ax_log_panel.set_facecolor('#151922')
    ax_log_panel.set_title('Planner Console', loc='left', fontsize=11, color='#f5f5f5', fontfamily='Consolas')
    ax_log_panel.axis('off')
    log_text = ax_log_panel.text(
        0.02,
        0.92,
        '',
        fontsize=9.5,
        color='#e0e0e0',
        va='top',
        fontfamily='Consolas',
        transform=ax_log_panel.transAxes,
        wrap=True,
    )
    _add_panel_frame(ax_log_panel)

    action_header = fig.text(0.32, 0.955, '', fontsize=12, fontweight='bold', color='#f5f5f5', fontfamily='Consolas')
    status_header = fig.text(0.32, 0.925, '', fontsize=11, color='#c5d0e3', fontfamily='Consolas')
    fig.text(
        0.05,
        0.035,
        (
            f"Plan: {len(plan['path'])}/{plan['trees_considered']} trees"
            f"  |  Est. Coconuts: {plan['coconuts']}"
            f"  |  Blocked: {len(plan.get('blocked_trees', []))}"
            f"  |  Seed: {run_seed}"
        ),
        fontsize=9,
        color='#8fa3c1',
        fontfamily='Consolas',
    )

    chassis_template = np.array(
        [
            [-1.2, -0.75],
            [1.1, -0.75],
            [1.35, -0.2],
            [1.35, 0.2],
            [1.1, 0.75],
            [-1.2, 0.75],
        ],
        dtype=float,
    )
    chassis_patch = patches.Polygon(
        chassis_template + rover_pos,
        closed=True,
        facecolor='#455a64',
        edgecolor='#b0bec5',
        linewidth=1.4,
        alpha=0.95,
        zorder=4,
    )
    ax_world.add_patch(chassis_patch)

    mast_patch = patches.Circle(
        (rover_pos[0] - 0.6, rover_pos[1]),
        radius=0.22,
        facecolor='#90a4ae',
        edgecolor='#cfd8dc',
        linewidth=1.0,
        zorder=5,
    )
    ax_world.add_patch(mast_patch)

    wheel_offsets = np.array(
        [
            (-0.95, 0.75),
            (-0.25, 0.75),
            (0.45, 0.75),
            (-0.95, -0.75),
            (-0.25, -0.75),
            (0.45, -0.75),
        ]
    )
    footprint_size = (0.62, 0.26)
    footprint_patches: List[patches.Rectangle] = []
    for offset in wheel_offsets:
        track = patches.Rectangle(
            (
                rover_pos[0] + offset[0] - footprint_size[0] / 2,
                rover_pos[1] + offset[1] - footprint_size[1] / 2,
            ),
            footprint_size[0],
            footprint_size[1],
            angle=rover_angle,
            facecolor='#1c242c',
            edgecolor='#54606b',
            linewidth=0.8,
            alpha=0.45,
            zorder=3.4,
        )
        ax_world.add_patch(track)
        footprint_patches.append(track)
    wheel_patches: List[patches.Circle] = []
    for offset in wheel_offsets:
        wheel = patches.Circle(
            (rover_pos[0] + offset[0], rover_pos[1] + offset[1]),
            radius=0.24,
            facecolor='#263238',
            edgecolor='#90a4ae',
            linewidth=1.1,
            zorder=5,
        )
        ax_world.add_patch(wheel)
        wheel_patches.append(wheel)

    harvester_cutter = patches.Circle(
        (rover_pos[0] + 1.1, rover_pos[1]),
        radius=0.26,
        facecolor='#d84315',
        edgecolor='#ffe0b2',
        linewidth=1.2,
        zorder=6,
    )
    ax_world.add_patch(harvester_cutter)

    joint_nodes = [
        patches.Circle(rover_pos, radius=0.16, facecolor='#4db6ac', edgecolor='#004d40', linewidth=1.2, alpha=0.85, zorder=6.1),
        patches.Circle(rover_pos, radius=0.13, facecolor='#7e57c2', edgecolor='#311b92', linewidth=1.1, alpha=0.9, zorder=6.2),
        patches.Circle(rover_pos, radius=0.12, facecolor='#ff7043', edgecolor='#bf360c', linewidth=1.1, alpha=0.95, zorder=6.3),
    ]
    for idx, node in enumerate(joint_nodes):
        node.set_visible(idx == 0)
        ax_world.add_patch(node)

    arm_pose = {'elbow': None, 'tool': None}

    path_line.set_data([rover_pos[0]], [rover_pos[1]])
    tool_line.set_data(tool_x_axis, tool_temp_profile)

    play_ax = fig.add_axes([0.06, 0.07, 0.095, 0.05])
    skip_ax = fig.add_axes([0.18, 0.07, 0.095, 0.05])
    speed_ax = fig.add_axes([0.32, 0.079, 0.25, 0.033])

    play_button = Button(play_ax, 'Pause')
    skip_button = Button(skip_ax, 'Skip >>')
    speed_slider = Slider(speed_ax, 'Speed', 0.2, 3.0, valinit=1.0, valstep=0.1)

    fig.text(
    0.06,
    0.12,
        'Controls: Space/P toggle pause, N jumps to next action, +/- adjust speed',
        fontsize=8,
        color='#9fb4cc',
        fontfamily='Consolas',
    )

    ani_handle: Dict[str, Optional[FuncAnimation]] = {'instance': None}

    def _toggle_play(event=None) -> None:
        controls['is_paused'] = not controls['is_paused']
        play_button.label.set_text('Play' if controls['is_paused'] else 'Pause')
        animation = ani_handle['instance']
        if animation is not None:
            if controls['is_paused']:
                animation.event_source.stop()
            else:
                animation.event_source.start()
        fig.canvas.draw_idle()

    def _skip_event(event=None) -> None:
        controls['skip'] = True

    def _on_speed_change(val: float) -> None:
        controls['speed'] = max(val, 0.1)

    play_button.on_clicked(_toggle_play)
    skip_button.on_clicked(_skip_event)
    speed_slider.on_changed(_on_speed_change)

    def _on_key(event) -> None:
        if event.key in {' ', 'p'}:
            _toggle_play()
        elif event.key in {'n', 'right'}:
            _skip_event()
        elif event.key in {'+', 'up'}:
            proposed = min(speed_slider.val + 0.1, speed_slider.valmax)
            speed_slider.set_val(round(proposed, 2))
        elif event.key in {'-', 'down'}:
            proposed = max(speed_slider.val - 0.1, speed_slider.valmin)
            speed_slider.set_val(round(proposed, 2))

    fig.canvas.mpl_connect('key_press_event', _on_key)

    def _status_color(status: str) -> str:
        return {
            'pending': '#b0bec5',
            'scanning': '#ffd54f',
            'arm_ready': '#ffa726',
            'harvested': '#66bb6a',
            'empty': '#90a4ae',
            'blocked': '#ef5350',
            'skipped': '#9575cd',
        }.get(status, '#b0bec5')

    def _refresh_tree_appearance(tree_name: str) -> None:
        marker = tree_markers.get(tree_name)
        if marker is not None:
            marker.set_facecolor(_status_color(tree_status[tree_name]))
            marker.set_edgecolor('#cfd8dc')

    def _update_tree_table() -> None:
        for row_idx, name in enumerate(tree_names_sorted, start=1):
            tree_info = params['trees'][name]
            status = tree_status[name]
            tree_table[(row_idx, 1)].get_text().set_text(str(tree_info['coconuts']))
            tree_table[(row_idx, 2)].get_text().set_text(status)
            row_color = _status_color(status)
            for col_idx in range(3):
                cell = tree_table[(row_idx, col_idx)]
                cell.set_facecolor('#151922')
                if col_idx < 2:
                    cell.get_text().set_color('#e5e9f0')
                else:
                    cell.get_text().set_color(row_color)

    def _apply_arm_overlay() -> None:
        joint_nodes[0].center = (rover_pos[0], rover_pos[1])
        joint_nodes[0].set_visible(True)

        elbow_pos = arm_pose.get('elbow')
        tool_pos = arm_pose.get('tool')

        if elbow_pos is None:
            joint_nodes[1].set_visible(False)
        else:
            joint_nodes[1].set_visible(True)
            joint_nodes[1].center = (elbow_pos[0], elbow_pos[1])

        if tool_pos is None:
            joint_nodes[2].set_visible(False)
        else:
            joint_nodes[2].set_visible(True)
            joint_nodes[2].center = (tool_pos[0], tool_pos[1])

    def _update_harvester_pose() -> None:
        rot = np.deg2rad(rover_angle)
        rot_matrix = np.array(
            [
                [np.cos(rot), -np.sin(rot)],
                [np.sin(rot), np.cos(rot)],
            ]
        )
        chassis_points = chassis_template @ rot_matrix.T + rover_pos
        chassis_patch.set_xy(chassis_points)

        mast_center = rover_pos + rot_matrix @ np.array([-0.6, 0.0])
        mast_patch.center = (mast_center[0], mast_center[1])

        for offset, track in zip(wheel_offsets, footprint_patches):
            center = rover_pos + rot_matrix @ offset
            track.set_xy((center[0] - footprint_size[0] / 2, center[1] - footprint_size[1] / 2))
            track.set_transform(
                transforms.Affine2D().rotate_deg_around(center[0], center[1], rover_angle) + ax_world.transData
            )

        for offset, wheel in zip(wheel_offsets, wheel_patches):
            wheel_center = rover_pos + rot_matrix @ offset
            wheel.center = (wheel_center[0], wheel_center[1])

        cutter_offset = rot_matrix @ np.array([1.35, 0.0])
        harvester_cutter.center = (
            rover_pos[0] + cutter_offset[0],
            rover_pos[1] + cutter_offset[1],
        )
        _apply_arm_overlay()

    _update_harvester_pose()

    def _direction_to_cardinal(angle_deg: float) -> str:
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        normalized = angle_deg % 360.0
        idx = int((normalized / 45.0) + 0.5) % len(directions)
        return directions[idx]

    def _update_environment(elapsed_dt: float) -> None:
        nonlocal system_metrics, planner_context
        if elapsed_dt <= 0:
            return
        environment_state['wind_speed'] = float(
            np.clip(
                environment_state['wind_speed'] + env_rng.normal(0.0, 0.2) * np.sqrt(elapsed_dt),
                0.5,
                12.0,
            )
        )
        environment_state['wind_direction'] = float(
            (environment_state['wind_direction'] + env_rng.normal(0.0, 6.0) * np.sqrt(elapsed_dt)) % 360.0
        )
        environment_state['humidity'] = float(
            np.clip(environment_state['humidity'] + env_rng.normal(0.0, 0.4), 30.0, 95.0)
        )
        environment_state['ambient_temp'] = float(
            np.clip(environment_state['ambient_temp'] + env_rng.normal(0.0, 0.15), 24.0, 38.0)
        )
        environment_state['battery'] = max(
            0.0,
            100.0 * (1.0 - (energy_used / max(constraints['max_energy'], 1.0))),
        )
        environment_state['tool_temp'] = float(np.max(tool_temp_profile))
        system_metrics['cpu_load'] = float(
            np.clip(system_metrics['cpu_load'] + env_rng.normal(0.0, 1.4), 12.0, 94.0)
        )
        system_metrics['loop_latency'] = float(
            np.clip(system_metrics['loop_latency'] + env_rng.normal(0.0, 2.1), 12.0, 85.0)
        )
        system_metrics['packet_loss'] = float(
            np.clip(system_metrics['packet_loss'] + env_rng.normal(0.0, 0.2), 0.0, 6.0)
        )
        clearance_penalty = 0.25 if latest_clearance < 2.5 else 0.08 if latest_clearance < 6.0 else 0.03
        planner_context['nav_confidence'] = float(
            np.clip(0.98 - clearance_penalty + env_rng.normal(0.0, 0.01), 0.55, 0.99)
        )

    def _update_real_time_panel() -> None:
        wind_cardinal = _direction_to_cardinal(environment_state['wind_direction'])
        nearest_value = latest_clearance if np.isfinite(latest_clearance) else float('nan')
        nearest_display = f"{nearest_value:.1f}" if np.isfinite(nearest_value) else '--'

        def fmt_line(label: str, value: str) -> str:
            return f"  {label:<17}: {value}\n"

        currents_left = ' '.join(f"{val:4.1f}" for val in drivetrain_state['wheel_currents'][:3])
        currents_right = ' '.join(f"{val:4.1f}" for val in drivetrain_state['wheel_currents'][3:])
        torques_left = ' '.join(f"{val:4.0f}" for val in drivetrain_state['wheel_torques'][:3])
        torques_right = ' '.join(f"{val:4.0f}" for val in drivetrain_state['wheel_torques'][3:])
        temps_left = ' '.join(f"{val:4.1f}" for val in drivetrain_state['wheel_temps'][:3])
        temps_right = ' '.join(f"{val:4.1f}" for val in drivetrain_state['wheel_temps'][3:])
        suspension = ' / '.join(f"{val:.1f}" for val in drivetrain_state['suspension_loads']) + ' kN'
        arm_angles = ' '.join(f"{val:.2f}" for val in arm_state['joint_angles'][:3])
        arm_temps = ' '.join(f"{val:4.1f}" for val in arm_state['joint_temps'][:3])

        env_block = (
            "ENVIRONMENT\n"
            + fmt_line('Battery', f"{environment_state['battery']:.1f}%")
            + fmt_line('Pack Voltage', f"{drivetrain_state['pack_voltage']:.1f} V")
            + fmt_line('Pack Current', f"{drivetrain_state['pack_current']:.1f} A")
            + fmt_line('SOC', f"{drivetrain_state['soc']:.1f}%")
            + fmt_line('Thermal Margin', f"{drivetrain_state['thermal_margin']:.1f} °C")
            + fmt_line('Ambient', f"{environment_state['ambient_temp']:.1f} °C")
            + fmt_line('Humidity', f"{environment_state['humidity']:.1f}%")
            + fmt_line('Wind', f"{environment_state['wind_speed']:.1f} m/s {wind_cardinal}")
        )

        drive_block = (
            "POWERTRAIN\n"
            + fmt_line('Wheel Currents L', currents_left)
            + fmt_line('Wheel Currents R', currents_right)
            + fmt_line('Wheel Torques L', torques_left)
            + fmt_line('Wheel Torques R', torques_right)
            + fmt_line('Wheel Temps L', temps_left)
            + fmt_line('Wheel Temps R', temps_right)
            + fmt_line('Suspension', suspension)
            + fmt_line('Cutter RPM', f"{drivetrain_state['cutter_rpm']:.0f}")
        )

        arm_block = (
            "MANIPULATOR\n"
            + fmt_line('Joint Angles', arm_angles + ' rad')
            + fmt_line('Joint Temps', arm_temps + ' °C')
            + fmt_line('Tool Temp', f"{max_tool_temp_seen:.1f} °C")
            + fmt_line('End Effector', f"{arm_state['end_effector_force']:.1f} N")
        )

        comms_block = (
            "COMMS & NAV\n"
            + fmt_line('Latency', f"{system_metrics['loop_latency']:.0f} ms")
            + fmt_line('Packet Loss', f"{system_metrics['packet_loss']:.1f}%")
            + fmt_line('Nav Confidence', f"{planner_context['nav_confidence'] * 100:.0f}%")
            + fmt_line('Nearest Obs', f"{nearest_display} m")
            + fmt_line('Risk Index', f"{planner_context['risk_index']:.2f}")
        )

        task_block = (
            "TASKING\n"
            + fmt_line('Objective', planner_context['target_tree'] or 'Field sweep')
            + fmt_line('ETA', f"{planner_context['eta_remaining']:.1f} s")
            + fmt_line('Distance', f"{planner_context['distance_remaining']:.1f} m")
            + fmt_line('Speed Cmd', f"{controls['speed']:.1f}x")
        )

        realtime_text.set_text(
            env_block
            + '\n'
            + drive_block
            + '\n'
            + arm_block
            + '\n'
            + comms_block
            + '\n'
            + task_block
        )

    def _update_stats_panel() -> None:
        average_speed = total_drive_distance / max(sim_time, 1e-6)
        energy_fraction = energy_used / max(constraints['max_energy'], 1.0)
        actions_summary = ', '.join(f"{name}:{count}" for name, count in action_counter.most_common())
        mission_text.set_text(
            (
                f"Timeline\n"
                f"  Elapsed: {sim_time:.1f} s\n"
                f"  Phase: {planner_context['phase']}\n"
                f"  Drive Time: {drive_time_accum:.1f} s\n\n"
                f"Performance\n"
                f"  Distance: {total_drive_distance:.1f} m\n"
                f"  Avg Speed: {average_speed:.2f} m/s\n"
                f"  Energy Used: {energy_used:.0f} J ({min(energy_fraction, 1.0):.0%})\n"
                f"  Cutter Duty: {drivetrain_state['cutter_rpm']:.0f} rpm\n"
                f"  End Effector Force: {arm_state['end_effector_force']:.1f} N"
            )
        )
        status_counts = Counter(tree_status.values())
        summary_text.set_text(
            (
                f"Trees\n"
                f"  Pending: {status_counts.get('pending', 0)}\n"
                f"  Scanning: {status_counts.get('scanning', 0)}\n"
                f"  Arm Ready: {status_counts.get('arm_ready', 0)}\n"
                f"  Harvested: {status_counts.get('harvested', 0)}\n"
                f"  Empty: {status_counts.get('empty', 0)}\n"
                f"  Blocked: {status_counts.get('blocked', 0)}\n"
                f"  Skipped: {status_counts.get('skipped', 0)}\n\n"
                f"Action Counts\n  {actions_summary or 'n/a'}"
            )
        )

    def _update_diagnostics_panel() -> None:
        diagnostic_text.set_text(
            (
                f"Phase: {planner_context['phase']}   Focus: {planner_context['target_tree'] or 'Field'}\n"
                f"CPU Load: {system_metrics['cpu_load']:.0f}%   Loop Latency: {system_metrics['loop_latency']:.0f} ms\n"
                f"Packet Loss: {system_metrics['packet_loss']:.1f}%   Nav Confidence: {planner_context['nav_confidence'] * 100:.0f}%\n"
                f"Risk Index: {planner_context['risk_index']:.2f}   Battery: {environment_state['battery']:.1f}%"
            )
        )

    def _refresh_planner_console() -> None:
        if planner_log:
            log_text.set_text('\n'.join(planner_log))
        else:
            log_text.set_text('> awaiting mission events')

    def _push_planner_log(message: str) -> None:
        planner_log.appendleft(message)
        _refresh_planner_console()

    def _update_obstacle_panel() -> None:
        nonlocal latest_clearance, planner_context
        if not obstacle_bars:
            latest_clearance = float('inf')
            planner_context['risk_index'] = 0.0
            return
        clearances = []
        for bar, obs in zip(obstacle_bars, obstacles):
            center = np.array(obs['center'])
            clearance = max(0.0, np.linalg.norm(rover_pos - center) - obs['radius'])
            clearances.append(clearance)
            bar.set_width(clearance)
            if clearance < 2.0:
                bar.set_color('#ef5350')
            elif clearance < 5.0:
                bar.set_color('#ffca28')
            else:
                bar.set_color('#64b5f6')
        if clearances:
            latest_clearance = min(clearances)
            upper = max(clearances)
            ax_obstacle_panel.set_xlim(0.0, max(upper, 1.0) + 2.0)
            planner_context['risk_index'] = float(np.clip(1.4 / max(latest_clearance, 0.6), 0.0, 4.0))
        else:
            planner_context['risk_index'] = 0.0

    _update_tree_table()
    _update_obstacle_panel()
    _update_real_time_panel()
    _update_stats_panel()
    _update_diagnostics_panel()
    _refresh_planner_console()
    _push_planner_log(f"[{sim_time:6.1f}s] Systems boot complete. Awaiting first action...")
    _update_ai_panel('Boot sequence nominal; monitoring field state.')

    def _flush_telemetry() -> None:
        nonlocal telemetry_written
        if telemetry_written or log_file_path is None:
            return
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with log_file_path.open('w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=telemetry_headers)
            writer.writeheader()
            if telemetry_records:
                writer.writerows(telemetry_records)
        telemetry_written = True

    def _complete_mission() -> None:
        nonlocal is_mission_complete
        if is_mission_complete:
            return
        action_header.set_text('Mission Complete')
        status_header.set_text(f'Total Coconuts Collected: {harvested_coconuts}')
        is_mission_complete = True
        planner_context['phase'] = 'COMPLETE'
        planner_context['target_tree'] = ''
        planner_context['distance_remaining'] = 0.0
        planner_context['eta_remaining'] = 0.0
        _update_obstacle_panel()
        _update_real_time_panel()
        _update_stats_panel()
        _update_diagnostics_panel()
        _update_ai_panel('Mission complete; awaiting next directive.')
        _push_planner_log(f"[{sim_time:6.1f}s] Mission complete. Harvested {harvested_coconuts} coconuts.")
        if not telemetry_records or telemetry_records[-1]['action'] != 'MISSION_COMPLETE':
            telemetry_records.append(
                {
                    'time': round(sim_time, 3),
                    'action': 'MISSION_COMPLETE',
                    'tree': '',
                    'rover_x': float(rover_pos[0]),
                    'rover_y': float(rover_pos[1]),
                    'energy': float(energy_used),
                    'trees_scanned': trees_scanned,
                    'harvested_coconuts': harvested_coconuts,
                    'max_tool_temp': float(np.max(tool_temp_profile)),
                    'speed_multiplier': round(controls['speed'], 2),
                    'battery_pct': environment_state['battery'],
                    'wind_mps': environment_state['wind_speed'],
                    'humidity_pct': environment_state['humidity'],
                    'ambient_temp': environment_state['ambient_temp'],
                    'nearest_obstacle_clearance': latest_clearance,
                    'nav_confidence': planner_context['nav_confidence'],
                    'planner_phase': planner_context['phase'],
                    'distance_to_target': planner_context['distance_remaining'],
                    'eta_to_target': planner_context['eta_remaining'],
                    'cpu_load_pct': system_metrics['cpu_load'],
                    'loop_latency_ms': system_metrics['loop_latency'],
                    'packet_loss_pct': system_metrics['packet_loss'],
                    'wheel_currents': ','.join(f"{val:.2f}" for val in drivetrain_state['wheel_currents']),
                    'wheel_torques': ','.join(f"{val:.2f}" for val in drivetrain_state['wheel_torques']),
                    'wheel_temps': ','.join(f"{val:.2f}" for val in drivetrain_state['wheel_temps']),
                    'suspension_loads': ','.join(f"{val:.2f}" for val in drivetrain_state['suspension_loads']),
                    'cutter_rpm': drivetrain_state['cutter_rpm'],
                    'pack_voltage': drivetrain_state['pack_voltage'],
                    'pack_current': drivetrain_state['pack_current'],
                    'battery_soc': drivetrain_state['soc'],
                    'thermal_margin': drivetrain_state['thermal_margin'],
                    'arm_joint_angles': ','.join(f"{val:.3f}" for val in arm_state['joint_angles']),
                    'arm_joint_temps': ','.join(f"{val:.2f}" for val in arm_state['joint_temps']),
                    'end_effector_force': arm_state['end_effector_force'],
                    'ai_mode': ai_state['mode'],
                    'ai_intent': ai_state['intent'],
                    'ai_risk_assessment': ai_state['risk'],
                }
            )
        _flush_telemetry()

    def _process_action(speed_multiplier: float, skip_requested: bool) -> None:
        nonlocal sim_time, energy_used, action_idx, time_in_action
        nonlocal rover_pos, rover_angle, tool_temp_profile, trees_scanned, harvested_coconuts
        nonlocal path_positions, total_drive_distance, drive_time_accum, max_tool_temp_seen
        nonlocal action_counter, planner_context, system_metrics

        if is_mission_complete:
            _flush_telemetry()
            return

        if action_idx >= len(mission_queue):
            _complete_mission()
            return

        state = mission_queue[action_idx]
        action = state['action']
        duration = state['duration']
        previous_time_in_action = time_in_action
        remaining = max(duration - previous_time_in_action, 0.0) if duration > 0 else 0.0
        base_dt = base_sim_step * speed_multiplier

        if previous_time_in_action == 0.0:
            descriptor = state.get('tree', 'Field')
            _push_planner_log(f"[{sim_time:6.1f}s] -> {action} ({descriptor})")
            _update_ai_context(action, descriptor)

        planner_context['phase'] = action
        planner_context['target_tree'] = state.get('tree', '')

        if duration > 0:
            if skip_requested:
                dt = remaining
            else:
                dt = base_dt if remaining > base_dt else remaining
            if remaining <= 1e-6 and dt <= 1e-8:
                dt = 0.0
        else:
            dt = base_dt

        if dt < 0.0:
            dt = 0.0

        sim_time += dt
        if duration > 0:
            time_in_action = min(previous_time_in_action + dt, duration)
            elapsed_dt = time_in_action - previous_time_in_action
        else:
            time_in_action = previous_time_in_action + dt
            elapsed_dt = dt

        elapsed_dt = max(elapsed_dt, 0.0)
        remaining_after = max(duration - time_in_action, 0.0) if duration > 0 else 0.0
        planner_context['eta_remaining'] = remaining_after
        planner_context['distance_remaining'] = 0.0

        load_factor = {
            'DRIVING': 1.0,
            'SCAN_TREE': 0.45,
            'DEPLOYING_ARM': 0.7,
            'CUT_TREE': 0.9,
            'COOL_TOOL': 0.3,
        }.get(action, 0.25)

        base_current = 18.0 + 20.0 * load_factor
        base_torque = 80.0 + 180.0 * load_factor
        base_temp = 37.0 + 32.0 * load_factor
        base_suspension = 4.0 + 6.5 * load_factor

        drivetrain_state['wheel_currents'] = np.clip(
            base_current * wheel_profile + env_rng.normal(0.0, 1.2, size=6),
            6.0,
            72.0,
        )
        drivetrain_state['wheel_torques'] = np.clip(
            base_torque * torque_profile + env_rng.normal(0.0, 8.0, size=6),
            0.0,
            380.0,
        )
        wheel_target = base_temp
        drivetrain_state['wheel_temps'] = np.clip(
            drivetrain_state['wheel_temps'] + 0.35 * (wheel_target - drivetrain_state['wheel_temps']) + env_rng.normal(0.0, 0.45, size=6),
            34.0,
            90.0,
        )
        suspension_target = base_suspension * suspension_profile
        drivetrain_state['suspension_loads'] = np.clip(
            drivetrain_state['suspension_loads'] + 0.45 * (suspension_target - drivetrain_state['suspension_loads']) + env_rng.normal(0.0, 0.18, size=3),
            3.0,
            12.5,
        )

        target_current_draw = 8.0 + 24.0 * load_factor
        drivetrain_state['pack_current'] = float(
            np.clip(
                drivetrain_state['pack_current'] + 0.35 * (target_current_draw - drivetrain_state['pack_current']) + env_rng.normal(0.0, 0.8),
                -5.0,
                48.0,
            )
        )
        drivetrain_state['pack_voltage'] = float(
            np.clip(
                52.0 - 0.003 * energy_used - 0.4 * load_factor + env_rng.normal(0.0, 0.05),
                46.5,
                52.0,
            )
        )
        drivetrain_state['soc'] = max(5.0, 100.0 - (energy_used / max(constraints['max_energy'], 1.0)) * 100.0)
        drivetrain_state['thermal_margin'] = float(
            np.clip(36.0 - max_tool_temp_seen * 0.11 + env_rng.normal(0.0, 0.3), 4.0, 34.0)
        )

        pose_map = {
            'DRIVING': 'stowed',
            'SCAN_TREE': 'scan',
            'DEPLOYING_ARM': 'deploy',
            'CUT_TREE': 'cut',
            'COOL_TOOL': 'stowed',
        }
        pose_key = pose_map.get(action, 'stowed')
        if pose_key == 'stowed':
            target_angles = joint_pose_library['stowed']
        elif pose_key == 'scan':
            target_angles = joint_pose_library['scan']
        elif pose_key == 'deploy':
            target_angles = joint_pose_library['deploy']
        elif pose_key == 'cut':
            target_angles = joint_pose_library['cut']
        else:
            target_angles = joint_pose_library['stowed']
        arm_state['joint_angles'] = np.clip(
            target_angles + env_rng.normal(0.0, 0.03, size=6),
            -2.2,
            2.2,
        )
        arm_temp_target = 36.0 + 24.0 * load_factor
        arm_state['joint_temps'] = np.clip(
            arm_state['joint_temps'] + 0.4 * (arm_temp_target - arm_state['joint_temps']) + env_rng.normal(0.0, 0.25, size=6),
            34.0,
            85.0,
        )
        if action in {'CUT_TREE', 'DEPLOYING_ARM'}:
            arm_state['end_effector_force'] = float(np.clip(20.0 + env_rng.normal(0.0, 4.0), 0.0, 140.0))
        elif action == 'SCAN_TREE':
            arm_state['end_effector_force'] = float(np.clip(6.0 + env_rng.normal(0.0, 1.2), 0.0, 40.0))
        else:
            arm_state['end_effector_force'] = float(np.clip(env_rng.normal(0.0, 0.8), 0.0, 25.0))

        if action == 'CUT_TREE':
            target_rpm = 2850.0
        elif action == 'DEPLOYING_ARM':
            target_rpm = 1200.0
        else:
            target_rpm = 0.0
        drivetrain_state['cutter_rpm'] = float(
            np.clip(
                drivetrain_state['cutter_rpm'] + 0.45 * (target_rpm - drivetrain_state['cutter_rpm']) + env_rng.normal(0.0, 45.0),
                0.0,
                3200.0,
            )
        )

        if action == 'DRIVING':
            energy_used += params['costs']['drive_energy'] * elapsed_dt
            start_pos = np.array(state['start_pos'])
            target_pos = np.array(state['target'])
            direction = target_pos - start_pos
            distance = np.linalg.norm(direction)
            distance_progress = 0.0
            if distance > 1e-6 and duration > 0:
                rover_angle = degrees(atan2(direction[1], direction[0]))
                fraction_done = min(time_in_action / duration, 1.0)
                rover_pos = start_pos + direction * fraction_done
                progress_fraction = elapsed_dt / max(duration, 1e-6)
                distance_progress = min(progress_fraction, 1.0) * distance
            else:
                rover_pos = target_pos
            total_drive_distance += distance_progress
            drive_time_accum += elapsed_dt
            arm_line.set_data([], [])
            arm_dynamics_line.set_data([], [])
            current_trail = np.vstack([path_positions, rover_pos])
            path_line.set_data(current_trail[:, 0], current_trail[:, 1])
            planner_context['distance_remaining'] = float(
                max(0.0, np.linalg.norm(target_pos - rover_pos))
            )

        elif action == 'SCAN_TREE':
            tree_name = state['tree']
            if tree_status[tree_name] == 'pending':
                tree_status[tree_name] = 'scanning'
                _refresh_tree_appearance(tree_name)
            energy_used += params['operations']['scan_energy_rate'] * elapsed_dt
            arm_line.set_data([], [])
            arm_dynamics_line.set_data([], [])

        elif action == 'DEPLOYING_ARM':
            if duration > 0:
                energy_used += (
                    params['costs']['arm_deploy_energy'] / max(duration, base_sim_step)
                ) * elapsed_dt
            theta1, theta2 = np.deg2rad(62), np.deg2rad(-48)
            elbow_local, tool_local = kinematics.forward_kinematics(
                kinematics.LINK_1_LENGTH,
                kinematics.LINK_2_LENGTH,
                theta1,
                theta2,
            )
            elbow_world = kinematics.world_coords(rover_pos, elbow_local)
            tool_world = kinematics.world_coords(rover_pos, tool_local)
            arm_line.set_data(
                [rover_pos[0], elbow_world[0], tool_world[0]],
                [rover_pos[1], elbow_world[1], tool_world[1]],
            )
            arm_pose['elbow'] = elbow_world
            arm_pose['tool'] = tool_world
            t_arm = np.linspace(0, max(time_in_action, base_sim_step), 120)
            arm_dynamics_line.set_data(t_arm, arm_dynamics_func(t_arm))

        elif action == 'CUT_TREE':
            energy_used += params['operations']['cut_energy_rate'] * elapsed_dt
            theta1, theta2 = np.deg2rad(58), np.deg2rad(-35)
            elbow_local, tool_local = kinematics.forward_kinematics(
                kinematics.LINK_1_LENGTH,
                kinematics.LINK_2_LENGTH,
                theta1,
                theta2,
            )
            elbow_world = kinematics.world_coords(rover_pos, elbow_local)
            tool_world = kinematics.world_coords(rover_pos, tool_local)
            arm_line.set_data(
                [rover_pos[0], elbow_world[0], tool_world[0]],
                [rover_pos[1], elbow_world[1], tool_world[1]],
            )
            arm_pose['elbow'] = elbow_world
            arm_pose['tool'] = tool_world
            if elapsed_dt > 0:
                tool_temp_profile = thermal.solve_heat_equation(
                    elapsed_dt,
                    Q=60,
                    initial_temp_profile=tool_temp_profile,
                )
            tool_line.set_data(tool_x_axis, tool_temp_profile)

        elif action == 'COOL_TOOL':
            energy_used += params['operations']['cool_energy_rate'] * elapsed_dt
            if elapsed_dt > 0:
                tool_temp_profile = thermal.solve_heat_equation(
                    elapsed_dt,
                    Q=0,
                    initial_temp_profile=tool_temp_profile,
                )
            tool_line.set_data(tool_x_axis, tool_temp_profile)
            arm_line.set_data([], [])
            arm_pose['elbow'] = None
            arm_pose['tool'] = None
            arm_dynamics_line.set_data([], [])

        else:
            arm_line.set_data([], [])
            arm_pose['elbow'] = None
            arm_pose['tool'] = None
            arm_dynamics_line.set_data([], [])

        max_tool_temp_seen = max(max_tool_temp_seen, float(np.max(tool_temp_profile)))
        _update_environment(elapsed_dt)

        action_complete = duration == 0 or (duration > 0 and time_in_action >= duration - 1e-6)

        if action_complete:
            if action == 'DRIVING':
                rover_pos = np.array(state['target'])
                path_positions.append(rover_pos.copy())
                path_array = np.array(path_positions)
                path_line.set_data(path_array[:, 0], path_array[:, 1])
            elif action == 'SCAN_TREE':
                tree_name = state['tree']
                tree_info = params['trees'][tree_name]
                trees_scanned += 1
                tree_status[tree_name] = 'arm_ready' if tree_info['coconuts'] > 0 else 'empty'
                _refresh_tree_appearance(tree_name)
            elif action == 'DEPLOYING_ARM':
                tree_name = state['tree']
                tree_status[tree_name] = 'arm_ready'
                _refresh_tree_appearance(tree_name)
            elif action == 'CUT_TREE':
                tree_name = state['tree']
                harvested_coconuts += params['trees'][tree_name]['coconuts']
                params['trees'][tree_name]['coconuts'] = 0
                tree_status[tree_name] = 'harvested'
                _refresh_tree_appearance(tree_name)
            elif action == 'COOL_TOOL':
                tree_name = state['tree']
                tree_status[tree_name] = 'harvested'
                _refresh_tree_appearance(tree_name)

            _push_planner_log(f"[{sim_time:6.1f}s] [done] {action} ({state.get('tree', 'Field')})")
            action_idx += 1
            time_in_action = 0.0
            action_counter[action] += 1
            planner_context['distance_remaining'] = 0.0
            planner_context['eta_remaining'] = 0.0
            ai_state['next_action'] = _next_action_label(action_idx)

            if action_idx >= len(mission_queue):
                _complete_mission()

        _update_harvester_pose()
        _update_tree_table()
        _update_obstacle_panel()
        _update_real_time_panel()
        _update_stats_panel()
        _update_diagnostics_panel()
        _update_ai_panel()

        time_fraction = min(sim_time / constraints['max_time'], 1.0)
        energy_fraction = min(energy_used / constraints['max_energy'], 1.0)
        time_bar.set_width(time_fraction)
        energy_bar.set_width(energy_fraction)

        if not is_mission_complete and action_idx < len(mission_queue):
            next_state = mission_queue[action_idx]
            header_action = next_state['action']
            header_target = next_state.get('tree', 'Field')
        else:
            header_action = action
            header_target = state.get('tree', 'Field')

        if is_mission_complete:
            _flush_telemetry()
        else:
            action_header.set_text(f'Action> {header_action} :: {header_target}')
            status_header.set_text(
                (
                    f"t={sim_time:.1f}s | E={energy_used:.0f}J | trees={trees_scanned} | "
                    f"coconuts={harvested_coconuts} | nav={planner_context['nav_confidence'] * 100:.0f}%"
                )
            )

        telemetry_records.append(
            {
                'time': round(sim_time, 3),
                'action': action,
                'tree': state.get('tree', ''),
                'rover_x': float(rover_pos[0]),
                'rover_y': float(rover_pos[1]),
                'energy': float(energy_used),
                'trees_scanned': trees_scanned,
                'harvested_coconuts': harvested_coconuts,
                'max_tool_temp': float(np.max(tool_temp_profile)),
                'speed_multiplier': round(speed_multiplier, 2),
                'battery_pct': environment_state['battery'],
                'wind_mps': environment_state['wind_speed'],
                'humidity_pct': environment_state['humidity'],
                'ambient_temp': environment_state['ambient_temp'],
                'nearest_obstacle_clearance': latest_clearance,
                'nav_confidence': planner_context['nav_confidence'],
                'planner_phase': planner_context['phase'],
                'distance_to_target': planner_context['distance_remaining'],
                'eta_to_target': planner_context['eta_remaining'],
                'cpu_load_pct': system_metrics['cpu_load'],
                'loop_latency_ms': system_metrics['loop_latency'],
                'packet_loss_pct': system_metrics['packet_loss'],
                'wheel_currents': ','.join(f"{val:.2f}" for val in drivetrain_state['wheel_currents']),
                'wheel_torques': ','.join(f"{val:.2f}" for val in drivetrain_state['wheel_torques']),
                'wheel_temps': ','.join(f"{val:.2f}" for val in drivetrain_state['wheel_temps']),
                'suspension_loads': ','.join(f"{val:.2f}" for val in drivetrain_state['suspension_loads']),
                'cutter_rpm': drivetrain_state['cutter_rpm'],
                'pack_voltage': drivetrain_state['pack_voltage'],
                'pack_current': drivetrain_state['pack_current'],
                'battery_soc': drivetrain_state['soc'],
                'thermal_margin': drivetrain_state['thermal_margin'],
                'arm_joint_angles': ','.join(f"{val:.3f}" for val in arm_state['joint_angles']),
                'arm_joint_temps': ','.join(f"{val:.2f}" for val in arm_state['joint_temps']),
                'end_effector_force': arm_state['end_effector_force'],
                'ai_mode': ai_state['mode'],
                'ai_intent': ai_state['intent'],
                'ai_risk_assessment': ai_state['risk'],
            }
        )

    def update(frame: int) -> None:
        nonlocal controls
        if is_mission_complete:
            _flush_telemetry()
            return
        if controls['is_paused']:
            return
        speed_multiplier = max(controls['speed'], 0.1)
        skip_now = controls['skip']
        controls['skip'] = False
        _process_action(speed_multiplier, skip_now)

    animation_frames = int((constraints['max_time'] / base_sim_step) + 120)
    ani = FuncAnimation(fig, update, frames=animation_frames, interval=50, blit=False, repeat=False)
    ani_handle['instance'] = ani

    if show:
        try:
            plt.show()
        finally:
            _flush_telemetry()
    else:
        controls['is_paused'] = False
        ani.event_source.stop()
        for frame in range(animation_frames):
            if is_mission_complete:
                break
            update(frame)
        _flush_telemetry()
        plt.close(fig)

    if verbose:
        print('\nField operation complete.')

    return {
        'figure': fig,
        'animation': ani,
        'plan': plan,
        'seed': run_seed,
        'update': update,
        'controls': controls,
        'telemetry': telemetry_records,
        'log_file': str(log_file_path) if log_file_path is not None else None,
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the coconut harvester mission dashboard.')
    parser.add_argument('--seed', type=int, default=None, help='Optional integer seed for deterministic tree placement.')
    parser.add_argument('--no-show', action='store_true', help='Run the simulation without opening the Matplotlib window.')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output for non-interactive runs.')
    parser.add_argument('--log-file', type=str, default=None, help='Write mission telemetry to the provided CSV path.')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run_dashboard(
        seed=args.seed,
        show=not args.no_show,
        verbose=not args.quiet,
        log_path=args.log_file,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
