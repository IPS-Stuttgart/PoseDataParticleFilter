import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import quaternion
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyrecest
from pyrecest.filters import *
from pyrecest.distributions import *
import requests
import io
import gdown

# input_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\KIT_Quaternions\3\912_3_01_poses_quaternions.npz"


file_id = "1uDQf4jswrMZ_M6Z-RdhlpM0ybMS99gnw"  # File Id of Motion Clip on Google Drive (.npz file)

def process_sequence(file_id):
    buffer = io.BytesIO()
    gdown.download(id=file_id, output=buffer, quiet=False)
    buffer.seek(0)
    data = np.load(buffer, allow_pickle=True)
    print(data.files)
    poses = data['poses_quat']  
    trans= data['trans']
    betas= data['betas']
    gender= data['gender']
    dmpls= data['dmpls']
    mocap_framerate= data['mocap_framerate']
    return trans, betas, gender, dmpls, mocap_framerate, poses 

trans, betas, gender, dmpls, mocap_framerate, poses = process_sequence(file_id)

# data = np.load(input_path, allow_pickle=True)

joint_idx = 16  # joint used below for diagnostic plots
num_frames = len(poses)
num_joints = poses.shape[1]
n_particles = 1000
process_noise_std = 0.01
high_measurement_kappa = 500.0
low_measurement_kappa = 50.0
jump_threshold_std_multiplier = 3.0


def normalize_quat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.array(q, copy=True)
    q *= np.where(q[..., -1:] < 0, -1.0, 1.0)
    return q


def amass_to_filter_quat(q_amass):
    """Convert AMASS quaternion format (w, x, y, z) to PyRecEst format (x, y, z, w)."""
    q = np.array([q_amass[1], q_amass[2], q_amass[3], q_amass[0]])
    return normalize_quat(q)


def initialize_particles(q, n_particles, noise_std):
    particles = q + np.random.randn(n_particles, 4) * noise_std
    return normalize_quat(particles)


def quat_geodesic_distance(q1, q2):
    """Angular distance between two quaternions in radians."""
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)  # abs handles double cover
    return 2 * np.arccos(dot)


def compute_observed_jump_statistics(poses):
    observed_jumps = np.zeros((num_frames - 1, num_joints))

    for frame_idx in range(1, num_frames):
        for current_joint_idx in range(num_joints):
            q_prev = amass_to_filter_quat(poses[frame_idx - 1, current_joint_idx, :])
            q_curr = amass_to_filter_quat(poses[frame_idx, current_joint_idx, :])
            observed_jumps[frame_idx - 1, current_joint_idx] = quat_geodesic_distance(q_prev, q_curr)

    mean_jumps = np.mean(observed_jumps, axis=0)
    std_jumps = np.std(observed_jumps, axis=0)
    thresholds = mean_jumps + jump_threshold_std_multiplier * std_jumps

    return observed_jumps, mean_jumps, std_jumps, thresholds


def adaptive_measurement_kappa(current_jump, mean_jump, threshold):
    if current_jump <= mean_jump:
        return high_measurement_kappa

    if current_jump >= threshold or np.isclose(threshold, mean_jump):
        return low_measurement_kappa

    jump_ratio = (current_jump - mean_jump) / (threshold - mean_jump)
    return high_measurement_kappa - jump_ratio * (high_measurement_kappa - low_measurement_kappa)


observed_jump_magnitudes, mean_jump_per_joint, std_jump_per_joint, jump_thresholds = (
    compute_observed_jump_statistics(poses)
)


# One independent particle filter is maintained for each joint orientation.
particle_filters = [
    HyperhemisphericalParticleFilter(n_particles=n_particles, dim=3)
    for _ in range(num_joints)
]

# Store estimated orientations for every frame and every joint in (x, y, z, w) format.
estimates = np.zeros((num_frames, num_joints, 4))
measurement_kappas = np.full((num_frames, num_joints), high_measurement_kappa)

for current_joint_idx, pf in enumerate(particle_filters):
    q0 = amass_to_filter_quat(poses[0, current_joint_idx, :])
    particles = initialize_particles(q0, n_particles, process_noise_std)
    pf.set_state(HyperhemisphericalDiracDistribution(particles))
    estimates[0, current_joint_idx, :] = normalize_quat(pf.filter_state.mean())

for frame_idx in range(1, num_frames):
    for current_joint_idx, pf in enumerate(particle_filters):
        # STEP 1 - PREDICT (random walk with noise)
        particles = pf.filter_state.d
        particles = particles + np.random.randn(len(particles), 4) * process_noise_std
        particles = normalize_quat(particles)
        pf.filter_state.d = particles

        # STEP 2 - OBSERVE current joint quaternion
        q_obs = amass_to_filter_quat(poses[frame_idx, current_joint_idx, :])

        # STEP 3 - UPDATE (reweight particles against observation)
        current_jump = observed_jump_magnitudes[frame_idx - 1, current_joint_idx]
        measurement_kappa = adaptive_measurement_kappa(
            current_jump,
            mean_jump_per_joint[current_joint_idx],
            jump_thresholds[current_joint_idx],
        )
        measurement_kappas[frame_idx, current_joint_idx] = measurement_kappa
        meas_noise = HyperhemisphericalWatsonDistribution(q_obs, kappa=measurement_kappa)
        pf.update_nonlinear_using_likelihood(meas_noise.pdf)

        # STEP 4 - GET ESTIMATE
        estimates[frame_idx, current_joint_idx, :] = normalize_quat(pf.filter_state.mean())

# Optional copy in AMASS/numpy-quaternion order (w, x, y, z), useful when saving
# estimates alongside the original AMASS pose data.
estimates_amass_order = estimates[:, :, [3, 0, 1, 2]]

#Calculate differnce in observed ad estimated orientations for every frame

# Calculate jump magnitude between consecutive observed frames
jump_magnitudes = []
for frame_idx in range(1, num_frames):
    q1 = estimates[frame_idx - 1, joint_idx, :]  # previous frame
    q2 = estimates[frame_idx, joint_idx, :]      # current frame
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    
    dist = quat_geodesic_distance(q1, q2)
    jump_magnitudes.append(dist)

jump_magnitudes = np.array(jump_magnitudes)

# # Adaptive threshold: mean + 3*std
mean_jump = np.mean(jump_magnitudes)
std_jump  = np.std(jump_magnitudes)
threshold = mean_jump + 3 * std_jump

# fig, ax = plt.subplots(figsize=(14, 5))
# frames = np.arange(1, num_frames)

# # --- split into normal / anomaly series for cleaner legend ---
# anomaly_mask = jump_magnitudes > threshold
# normal_mask  = ~anomaly_mask

# # main jump line
# ax.plot(frames, jump_magnitudes, color='#378ADD', linewidth=1.2,
#         alpha=0.85, zorder=2, label='jump magnitude')

# # shade under the line
# ax.fill_between(frames, jump_magnitudes, alpha=0.08, color='#378ADD', zorder=1)

# # threshold + mean lines
# ax.axhline(threshold, color='#E24B4A', linewidth=1.4, linestyle='--',
#            zorder=3, label=f'threshold  μ+3σ  ({threshold:.3f} rad)')
# ax.axhline(mean_jump,  color='#888780', linewidth=1.0, linestyle=':',
#            zorder=3, label=f'mean  ({mean_jump:.3f} rad)')

# # anomaly scatter
# ax.scatter(frames[anomaly_mask], jump_magnitudes[anomaly_mask],
#            color='#E24B4A', s=55, zorder=5, label=f'anomaly  (n={anomaly_mask.sum()})')

# # vertical drop-lines from anomaly dots to x-axis (optional, aids reading)
# for f, v in zip(frames[anomaly_mask], jump_magnitudes[anomaly_mask]):
#     ax.vlines(f, 0, v, color='#E24B4A', linewidth=0.6, alpha=0.35, zorder=4)

# # --- shaded band: mean ± 1σ ---
# ax.axhspan(mean_jump - std_jump, mean_jump + std_jump,
#            color='#888780', alpha=0.07, zorder=0, label='±1σ band')

# # labels & formatting
# ax.set_xlabel('Frame index', fontsize=11)
# ax.set_ylabel('Geodesic distance (rad)', fontsize=11)
# ax.set_title(f'Orientation jump magnitudes — joint {joint_idx}', fontsize=13, fontweight='normal')
# ax.set_xlim(frames[0], frames[-1])
# ax.set_ylim(bottom=0)
# ax.grid(True, linewidth=0.4, alpha=0.5, linestyle='--')
# ax.spines[['top', 'right']].set_visible(False)
# ax.legend(fontsize=9, framealpha=0.85, loc='upper right')

# # annotate anomaly frame indices
# for f, v in zip(frames[anomaly_mask], jump_magnitudes[anomaly_mask]):
#     ax.annotate(f'f{f}', xy=(f, v), xytext=(4, 6),
#                 textcoords='offset points', fontsize=8,
#                 color='#E24B4A', fontweight='bold')

# plt.tight_layout()
# plt.savefig(f'jump_magnitudes_joint{joint_idx}.png', dpi=150, bbox_inches='tight')
# plt.show()

#2. Sudden jumps compared to estimated orientations
# Distance between filter estimate and observation each frame
filter_residuals = []
for frame_idx in range(1, num_frames):
    q_est = estimates[frame_idx, joint_idx, :]
    
    q_obs_raw = poses[frame_idx, joint_idx, :]
    q_obs = np.array([q_obs_raw[1], q_obs_raw[2], q_obs_raw[3], q_obs_raw[0]])
    if q_obs[-1] < 0: q_obs = -q_obs
    
    dist = quat_geodesic_distance(q_est, q_obs)
    filter_residuals.append(dist)

filter_residuals = np.array(filter_residuals)

# Calculate residuals: estimate vs observation for every frame
residuals = np.zeros(num_frames)

for frame_idx in range(1, num_frames):
    # filter estimate
    q_est = estimates[frame_idx, joint_idx, :]
    
    # observed quaternion
    q_obs_raw = poses[frame_idx, joint_idx, :]
    q_obs = np.array([q_obs_raw[1], q_obs_raw[2], q_obs_raw[3], q_obs_raw[0]])
    if q_obs[-1] < 0:
        q_obs = -q_obs
    
    residuals[frame_idx] = quat_geodesic_distance(q_est, q_obs)

# Threshold
# mean_res = np.mean(residuals[1:])
# std_res  = np.std(residuals[1:])
# threshold = mean_res + 3 * std_res

# # Incorrect frames
# incorrect_frames = np.where(residuals > threshold)[0]

# print(f"Mean residual : {np.degrees(mean_res):.2f} degrees")
# print(f"Std residual  : {np.degrees(std_res):.2f} degrees")  
# print(f"Threshold     : {np.degrees(threshold):.2f} degrees")
# print(f"Incorrect frames: {incorrect_frames}")
# print(f"Residuals at incorrect frames (degrees):")
# for f in incorrect_frames:
#     print(f"  frame {f:4d}: {np.degrees(residuals[f]):.2f}°")


# plt.figure(figsize=(12, 4))
# plt.plot(np.degrees(residuals), label='filter residual', color='steelblue')
# plt.axhline(np.degrees(threshold), color='red', linestyle='--', label=f'threshold ({np.degrees(threshold):.1f}°)')
# plt.scatter(incorrect_frames, np.degrees(residuals[incorrect_frames]), 
#             color='red', zorder=5, s=50, label=f'incorrect ({len(incorrect_frames)} frames)')
# plt.xlabel('Frame')
# plt.ylabel('Angular error (degrees)')
# plt.title('Incorrect observed orientations — Joint 0')
# plt.legend()
# plt.tight_layout()
# plt.show()


fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.subplots_adjust(hspace=0.08)  # tight gap since x-axis is shared

frames = np.arange(1, num_frames)

# ── shared helpers ────────────────────────────────────────────────────────────
def plot_jump_panel(ax, data, label_y, title, color='#378ADD'):
    mean_v = np.mean(data)
    std_v  = np.std(data)
    thresh = mean_v + 3 * std_v
    mask   = data > thresh

    ax.plot(frames, data, color=color, linewidth=1.2, alpha=0.85, zorder=2,
            label=label_y)
    ax.fill_between(frames, data, alpha=0.08, color=color, zorder=1)

    ax.axhline(thresh,       color='#E24B4A', linewidth=1.4, linestyle='--', zorder=3,
               label=f'threshold  μ+3σ  ({thresh:.3f} rad)')
    ax.axhline(mean_v,       color='#888780', linewidth=1.0, linestyle=':',  zorder=3,
               label=f'mean  ({mean_v:.3f} rad)')
    ax.axhspan(mean_v - std_v, mean_v + std_v,
               color='#888780', alpha=0.07, zorder=0, label='±1σ band')

    ax.scatter(frames[mask], data[mask],
               color='#E24B4A', s=55, zorder=5,
               label=f'anomaly  (n={mask.sum()})')
    for f, v in zip(frames[mask], data[mask]):
        ax.vlines(f, 0, v, color='#E24B4A', linewidth=0.6, alpha=0.35, zorder=4)
        ax.annotate(f'f{f}', xy=(f, v), xytext=(4, 6),
                    textcoords='offset points', fontsize=8,
                    color='#E24B4A', fontweight='bold')

    ax.set_ylabel('Geodesic distance (rad)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='normal', pad=6)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.5, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8.5, framealpha=0.85, loc='upper right')

    return thresh, mask          # caller can use if needed

# ── panel 1 : consecutive-frame jumps ────────────────────────────────────────
plot_jump_panel(
    axes[0], jump_magnitudes,
    label_y='jump magnitude',
    title=f'Consecutive-frame orientation jumps — joint {joint_idx}',
)

# ── panel 2 : filter residuals (estimate vs observation) ─────────────────────
plot_jump_panel(
    axes[1], filter_residuals,
    label_y='filter residual',
    title=f'Filter residual (estimate vs observation) — joint {joint_idx}',
    color='#1D9E75',             # teal to distinguish from panel 1
)

axes[1].set_xlabel('Frame index', fontsize=10)
axes[0].set_xlim(frames[0], frames[-1])   # shared x propagates automatically

plt.suptitle(f'Joint {joint_idx} — orientation diagnostics', fontsize=13,
             y=1.01, fontweight='normal')

plt.tight_layout()
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / f'orientation_diagnostics_joint{joint_idx}.png', dpi=150,
            bbox_inches='tight')
plt.show()
