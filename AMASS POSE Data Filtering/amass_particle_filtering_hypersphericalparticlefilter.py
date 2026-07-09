import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
# import quaternion
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

input_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\KIT_Quaternions\3\912_3_01_poses_quaternions.npz"


file_id = "1F-XL8Tf59lbakNkMhEPjvkr3wT4O0MEW"  # File Id of Motion Clip on Google Drive (.npz file)

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
    print(len(poses))
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
MAX_SAFE_KAPPA = 500.0



class OnlineKappaEstimator:
    """
    Maintains a per-joint running estimate of mean jump and std,
    then recomputes kappa bounds each frame via EMA.
    """
    def __init__(self, n_joints, alpha=0.02, 
                 initial_mean=None, initial_std=None):
        """
        alpha: EMA smoothing factor. 
               ~0.02 → memory of ~50 frames (slow adaptation, stable).
               ~0.1  → memory of ~10 frames (fast adaptation, responsive).
        """
        self.alpha = alpha
        self.ema_mean = initial_mean.copy() if initial_mean is not None \
                        else np.full(n_joints, 0.1)
        self.ema_var  = (initial_std ** 2).copy() if initial_std is not None \
                        else np.full(n_joints, 0.01)

    # def update(self, jumps):
    #     """jumps: shape (n_joints,) — geodesic distances for current frame."""
    #     self.ema_mean += self.alpha * (jumps - self.ema_mean)
    #     self.ema_var  += self.alpha * (
    #         (jumps - self.ema_mean) ** 2 - self.ema_var
    #     )

    def update(self, jumps):
        """jumps: shape (n_joints,) — geodesic distances for current frame."""
        delta_pre  = jumps - self.ema_mean                        # residual w.r.t. OLD mean
        
        self.ema_mean += self.alpha * delta_pre                   # update mean
        
        delta_post = jumps - self.ema_mean                        # residual w.r.t. NEW mean
        
        # Welford-style: cross product of pre/post deltas gives unbiased variance update
        self.ema_var = (1.0 - self.alpha) * self.ema_var + \
                    self.alpha * delta_pre * delta_post        # ← correct
    


    def kappa_bounds(self, std_multiplier=3.0):
        ema_std   = np.sqrt(np.maximum(self.ema_var, 1e-8))
        threshold = self.ema_mean + std_multiplier * ema_std

        high_kappa = np.clip(1.0 / (2.0 * self.ema_mean ** 2), 10.0, 500.0)  # ← 500 not 2000
        low_kappa  = np.clip(1.0 / (2.0 * threshold ** 2),      1.0, 100.0)

        return high_kappa, low_kappa, threshold
    






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

def compute_kappa_bounds(mean_jump_per_joint, std_jump_per_joint, jump_thresholds):
    """
    high_kappa: tight trust — derived from typical (mean) jump size.
    low_kappa:  loose trust — derived from the anomaly threshold.
    Clipped to sane physical limits to avoid numerical issues.
    """
    # At normal motion: spread ≈ mean_jump → kappa_high ≈ 1/(2*mean²)
    high_kappa = np.clip(
        1.0 / (2.0 * mean_jump_per_joint ** 2 + 1e-8),
        10.0, 2000.0
    )
    # At anomalous motion: spread ≈ threshold → kappa_low ≈ 1/(2*threshold²)
    low_kappa = np.clip(
        1.0 / (2.0 * jump_thresholds ** 2 + 1e-8),
        1.0, 200.0
    )
    return high_kappa, low_kappa



# def adaptive_measurement_kappa(current_jump, mean_jump, threshold):
#     if current_jump <= mean_jump:
#         return high_measurement_kappa

#     if current_jump >= threshold or np.isclose(threshold, mean_jump):
#         return low_measurement_kappa

#     jump_ratio = (current_jump - mean_jump) / (threshold - mean_jump)
#     return high_measurement_kappa - jump_ratio * (high_measurement_kappa - low_measurement_kappa)


observed_jump_magnitudes, mean_jump_per_joint, std_jump_per_joint, jump_thresholds = (
    compute_observed_jump_statistics(poses)
)

high_kappa_per_joint, low_kappa_per_joint = compute_kappa_bounds(
    mean_jump_per_joint, std_jump_per_joint, jump_thresholds
)



# EMA estimator — one object, tracks ALL joints simultaneously
# warm-started so frame 1 already has sensible bounds
kappa_estimator = OnlineKappaEstimator(
    n_joints=num_joints,
    alpha=0.02,
    initial_mean=mean_jump_per_joint,
    initial_std=std_jump_per_joint,
)




def adaptive_measurement_kappa(current_jump, mean_jump, threshold,
                                   high_kappa, low_kappa, smoothness=3.0):
    """
    Smooth exponential blend between high_kappa and low_kappa.
    All inputs are scalars (called per-joint inside the loop).
    smoothness: controls how sharply kappa drops as jump approaches threshold.
    """
    jump_ratio = np.clip(
        (current_jump - mean_jump) / (threshold - mean_jump + 1e-9),
        0.0, 1.0
    )
    # Exponential decay: stays near high_kappa until ratio rises, then drops fast
    blend = 1.0 - np.exp(-smoothness * (1.0 - jump_ratio))
    # Remap so blend=0 → high_kappa, blend=1 → low_kappa
    weight = np.exp(-smoothness * jump_ratio)
    return high_kappa * weight + low_kappa * (1.0 - weight)





def precompute_kappa_table(poses, num_frames, num_joints,
                            std_multiplier=3.0, smoothness=3.0):
    """
    PASS 1: Single loop over all frames and joints.
    Returns kappa_table shape (num_frames, num_joints) —
    exact per-joint per-frame kappa, no EMA approximation.
    """

    # ── Step 1: compute all geodesic jumps ──────────────────────────────────
    jump_magnitudes = np.zeros((num_frames - 1, num_joints))

    for frame_idx in range(1, num_frames):
        for joint_idx in range(num_joints):
            q_prev = amass_to_filter_quat(poses[frame_idx - 1, joint_idx, :])
            q_curr = amass_to_filter_quat(poses[frame_idx,     joint_idx, :])
            jump_magnitudes[frame_idx - 1, joint_idx] = quat_geodesic_distance(q_prev, q_curr)

    # ── Step 2: per-joint statistics (axis=0 → over frames) ─────────────────
    mean_jump = np.mean(jump_magnitudes, axis=0)   # shape (n_joints,)
    std_jump  = np.std(jump_magnitudes,  axis=0)   # shape (n_joints,)
    threshold = mean_jump + std_multiplier * std_jump  # shape (n_joints,)

    # ── Step 3: per-joint kappa bounds from statistics ───────────────────────
    high_kappa = np.clip(1.0 / (2.0 * mean_jump ** 2 + 1e-8), 10.0, 500.0)
    low_kappa  = np.clip(1.0 / (2.0 * threshold ** 2  + 1e-8),  1.0, 100.0)

    # ── Step 4: per-frame per-joint kappa via smooth blend ───────────────────
    # jump_magnitudes shape: (num_frames-1, num_joints)
    # mean_jump, threshold shape: (num_joints,) → broadcast over frames

    jump_ratio = np.clip(
        (jump_magnitudes - mean_jump) / (threshold - mean_jump + 1e-9),
        0.0, 1.0
    )                                               # shape (num_frames-1, num_joints)

    weight = np.exp(-smoothness * jump_ratio)       # shape (num_frames-1, num_joints)

    kappa_table = np.clip(
        high_kappa * weight + low_kappa * (1.0 - weight),
        1.0, 500.0
    )                                               # shape (num_frames-1, num_joints)

    return kappa_table, jump_magnitudes, mean_jump, std_jump, threshold


kappa_table, observed_jump_magnitudes, mean_jump_per_joint, \
    std_jump_per_joint, jump_thresholds = precompute_kappa_table(
        poses, num_frames, num_joints
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
    # Called ONCE per frame — returns shape (n_joints,) arrays
    # EMA state at this point reflects all frames seen so far
    # hk, lk, thresh_online = kappa_estimator.kappa_bounds()

    current_frame_jumps = observed_jump_magnitudes[frame_idx - 1]  


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
        # measurement_kappa = adaptive_measurement_kappa(
        #     current_jump,
        #     mean_jump_per_joint[current_joint_idx],
        #     jump_thresholds[current_joint_idx],
        # )
        # measurement_kappa = adaptive_measurement_kappa(
        #     current_jump = current_frame_jumps[current_joint_idx],
        #     mean_jump    = kappa_estimator.ema_mean[current_joint_idx],
        #     threshold    = thresh_online[current_joint_idx],
        #     high_kappa   = hk[current_joint_idx],
        #     low_kappa    = lk[current_joint_idx],
        # )
        measurement_kappa = float(kappa_table[frame_idx - 1, current_joint_idx])
        measurement_kappas[frame_idx, current_joint_idx] = measurement_kappa
        # measurement_kappa = float(np.clip(measurement_kappa, 1.0, MAX_SAFE_KAPPA))
        # measurement_kappas[frame_idx, current_joint_idx] = measurement_kappa
        meas_noise = HyperhemisphericalWatsonDistribution(q_obs, kappa=measurement_kappa)
        pf.update_nonlinear_using_likelihood(meas_noise.pdf)

        # STEP 4 - GET ESTIMATE
        estimates[frame_idx, current_joint_idx, :] = normalize_quat(pf.filter_state.mean())
    # EMA update — AFTER all joints processed for this frame
    # uses current_frame_jumps shape (n_joints,) — updates all joints at once
    # kappa_estimator.update(current_frame_jumps)

# Optional copy in AMASS/numpy-quaternion order (w, x, y, z), useful when saving
# estimates alongside the original AMASS pose data.
estimates_amass_order = estimates[:, :, [3, 0, 1, 2]]

#Calculate differnce in observed ad estimated orientations for every frame

# Calculate jump magnitude between consecutive observed frames
jump_magnitudes = []
# for frame_idx in range(1, num_frames):
#     q1 = observed_jump_magnitudes [frame_idx - 1, joint_idx, :]  # previous frame
#     q2 = observed_jump_magnitudes [frame_idx, joint_idx, :]      # current frame
#     q1 = normalize_quat(q1)
#     q2 = normalize_quat(q2)
    
#     dist = quat_geodesic_distance(q1, q2)
#     jump_magnitudes.append(dist)

# jump_magnitudes = np.array(jump_magnitudes)

jump_magnitudes = observed_jump_magnitudes[:, joint_idx]

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
