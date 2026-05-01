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

input_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\KIT_Quaternions\3\912_3_01_poses_quaternions.npz"


file_id = "1uDQf4jswrMZ_M6Z-RdhlpM0ybMS99gnw"  # replace with yours


# response = requests.get(url)
# data = np.load(io.BytesIO(response.content))


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

# pf = HyperhemisphericalParticleFilter(n_particles=1000, dim=3)
pf= HyperhemisphericalParticleFilter(n_particles= 1000, dim =3)
# print(data.files)

# poses= data['poses_quat']
joint_idx= 16
# print(poses[0][0])

q_amass= poses [0, joint_idx, :]

q = np.array([  q_amass[1], q_amass[2], q_amass[3], q_amass[0]])
# print(q_amass)

if q[-1] < 0:
    q = -q
# print(q)
num_frames= len(poses)

#Particle Filter Initialization
n_particles= 1000
noise = np.random.randn(n_particles, 4)* 0.01
particles = q+ noise

particles= particles/ np.linalg.norm(particles, axis = 1, keepdims= True)


particles[particles[:, -1]<0]*= -1

new_state= HyperhemisphericalDiracDistribution(particles)

pf.set_state(new_state)

kappa = 100.0  

# initial mean direction (just needs to be a valid quaternion to init)
mean_dir = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion
noise_dist = HyperhemisphericalWatsonDistribution(mean_dir, kappa)

estimates = np.zeros((num_frames, poses.shape[1], 4))
estimates[0, joint_idx, :] = pf.filter_state.mean()

for frame_idx in range(1, num_frames):
    # STEP 1 - PREDICT 
    # pf.predict_identity(noise_dist) (random walk with noise)
    particles = pf.filter_state.d  # current particles
    # print(frame_idx)
    noise = np.random.randn(len(particles), 4) * 0.01
    particles = particles + noise
    particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
    particles[particles[:, -1] < 0] *= -1
    pf.filter_state.d = particles
    
    # STEP 2 - OBSERVE get observed quaternion
    q_amass = poses[frame_idx, joint_idx, :]  # (w, x, y, z)
    q_obs = np.array([q_amass[1], q_amass[2], q_amass[3], q_amass[0]])  # (x,y,z,w)
    if q_obs[-1] < 0:
        q_obs = -q_obs
    
    # STEP 3 - UPDATE (reweight particles against observation)
    meas_noise = HyperhemisphericalWatsonDistribution(q_obs, kappa=500.0)
    # pf.update_identity(meas_noise, q_obs)
    pf.update_nonlinear_using_likelihood(meas_noise.pdf)
    
    # STEP 4 - GET ESTIMATE
    estimate = pf.filter_state.mean()  # shape (4,P)
    estimates[frame_idx, joint_idx, :] = estimate



#Calculate differnce in observed ad estimated orientations for every frame



# print(estimates.shape)
# for frame in range(len(poses)):
#     print(poses[frame, 0, :])
#     print(estimates[frame, 0, :])
# for frame in range(len(estimates)):
#     print(estimates[frame, 0, :])
# print(estimates[24])



# Calculate sudden jumps in observed orientations
#1. Sudden jumps using only observed quaternions
def quat_geodesic_distance(q1, q2):
    """Angular distance between two quaternions in radians"""
    dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)  # abs handles double cover
    return 2 * np.arccos(dot)

# Calculate jump magnitude between consecutive observed frames
jump_magnitudes = []
for frame_idx in range(1, num_frames):
    q1 = estimates[frame_idx - 1, joint_idx, :]  # previous frame
    q2 = estimates[frame_idx, joint_idx, :]      # current frame
    
    # convert to pyrecest format (x,y,z,w)
    q1 = np.array([q1[1], q1[2], q1[3], q1[0]])
    q2 = np.array([q2[1], q2[2], q2[3], q2[0]])
    
    # enforce upper hemisphere
    if q1[-1] < 0: q1 = -q1
    if q2[-1] < 0: q2 = -q2
    
    dist = quat_geodesic_distance(q1, q2)
    jump_magnitudes.append(dist)

jump_magnitudes = np.array(jump_magnitudes)

# # Adaptive threshold: mean + 3*std
mean_jump = np.mean(jump_magnitudes)
std_jump  = np.std(jump_magnitudes)
threshold = mean_jump + 3 * std_jump

fig, ax = plt.subplots(figsize=(14, 5))
frames = np.arange(1, num_frames)

# --- split into normal / anomaly series for cleaner legend ---
anomaly_mask = jump_magnitudes > threshold
normal_mask  = ~anomaly_mask

# main jump line
ax.plot(frames, jump_magnitudes, color='#378ADD', linewidth=1.2,
        alpha=0.85, zorder=2, label='jump magnitude')

# shade under the line
ax.fill_between(frames, jump_magnitudes, alpha=0.08, color='#378ADD', zorder=1)

# threshold + mean lines
ax.axhline(threshold, color='#E24B4A', linewidth=1.4, linestyle='--',
           zorder=3, label=f'threshold  μ+3σ  ({threshold:.3f} rad)')
ax.axhline(mean_jump,  color='#888780', linewidth=1.0, linestyle=':',
           zorder=3, label=f'mean  ({mean_jump:.3f} rad)')

# anomaly scatter
ax.scatter(frames[anomaly_mask], jump_magnitudes[anomaly_mask],
           color='#E24B4A', s=55, zorder=5, label=f'anomaly  (n={anomaly_mask.sum()})')

# vertical drop-lines from anomaly dots to x-axis (optional, aids reading)
for f, v in zip(frames[anomaly_mask], jump_magnitudes[anomaly_mask]):
    ax.vlines(f, 0, v, color='#E24B4A', linewidth=0.6, alpha=0.35, zorder=4)

# --- shaded band: mean ± 1σ ---
ax.axhspan(mean_jump - std_jump, mean_jump + std_jump,
           color='#888780', alpha=0.07, zorder=0, label='±1σ band')

# labels & formatting
ax.set_xlabel('Frame index', fontsize=11)
ax.set_ylabel('Geodesic distance (rad)', fontsize=11)
ax.set_title(f'Orientation jump magnitudes — joint {joint_idx}', fontsize=13, fontweight='normal')
ax.set_xlim(frames[0], frames[-1])
ax.set_ylim(bottom=0)
ax.grid(True, linewidth=0.4, alpha=0.5, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=9, framealpha=0.85, loc='upper right')

# annotate anomaly frame indices
for f, v in zip(frames[anomaly_mask], jump_magnitudes[anomaly_mask]):
    ax.annotate(f'f{f}', xy=(f, v), xytext=(4, 6),
                textcoords='offset points', fontsize=8,
                color='#E24B4A', fontweight='bold')

plt.tight_layout()
plt.savefig(f'jump_magnitudes_joint{joint_idx}.png', dpi=150, bbox_inches='tight')
plt.show()

# anomaly_frames = np.where(jump_magnitudes > threshold)[0] + 1  # +1 because diff starts at frame 1

# print(f"Mean jump:  {np.degrees(mean_jump):.2f} degrees")
# print(f"Std jump:   {np.degrees(std_jump):.2f} degrees")
# print(f"Threshold:  {np.degrees(threshold):.2f} degrees")
# print(f"Anomalies at frames: {anomaly_frames}")
# print(f"Jump magnitudes at anomalies: {np.degrees(jump_magnitudes[anomaly_frames-1])}")


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


#Plotting differneces in observed and estimated quternion orientations

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# ax1.plot(np.degrees(jump_magnitudes), label='observed jumps')
# ax1.axhline(np.degrees(threshold), color='r', linestyle='--', label='threshold')
# ax1.scatter(anomaly_frames-1, np.degrees(jump_magnitudes[anomaly_frames-1]), 
#             color='red', zorder=5, label='anomalies')
# ax1.set_ylabel('degrees')
# ax1.set_title('Consecutive frame jump magnitude')
# ax1.legend()

# ax2.plot(np.degrees(filter_residuals), label='filter residual', color='orange')
# ax2.set_ylabel('degrees')
# ax2.set_title('Filter estimate vs observation')
# ax2.legend()

# plt.tight_layout()
# plt.show()


def quat_geodesic_distance(q1, q2):
    """Angular distance between two quaternions in radians"""
    dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)
    return 2 * np.arccos(dot)

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
mean_res = np.mean(residuals[1:])
std_res  = np.std(residuals[1:])
threshold = mean_res + 3 * std_res

# Incorrect frames
incorrect_frames = np.where(residuals > threshold)[0]

print(f"Mean residual : {np.degrees(mean_res):.2f} degrees")
print(f"Std residual  : {np.degrees(std_res):.2f} degrees")  
print(f"Threshold     : {np.degrees(threshold):.2f} degrees")
print(f"Incorrect frames: {incorrect_frames}")
print(f"Residuals at incorrect frames (degrees):")
for f in incorrect_frames:
    print(f"  frame {f:4d}: {np.degrees(residuals[f]):.2f}°")


# frames = np.arange(1, num_frames)
# fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# axes[0].plot(frames, np.degrees(observed_jumps), color='steelblue')
# axes[0].set_ylabel('degrees')
# axes[0].set_title(f'Observed jumps — joint {joint_idx}')

# axes[1].plot(frames, np.degrees(filtered_jumps), color='green')
# axes[1].set_ylabel('degrees')
# axes[1].set_title(f'Filtered jumps — joint {joint_idx}')

plt.figure(figsize=(12, 4))
plt.plot(np.degrees(residuals), label='filter residual', color='steelblue')
plt.axhline(np.degrees(threshold), color='red', linestyle='--', label=f'threshold ({np.degrees(threshold):.1f}°)')
plt.scatter(incorrect_frames, np.degrees(residuals[incorrect_frames]), 
            color='red', zorder=5, s=50, label=f'incorrect ({len(incorrect_frames)} frames)')
plt.xlabel('Frame')
plt.ylabel('Angular error (degrees)')
plt.title('Incorrect observed orientations — Joint 0')
plt.legend()
plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from pyrecest.filters import *
# from pyrecest.distributions import *

# input_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\KIT_Quaternions\3\912_3_05_poses_quaternions.npz"

# data = np.load(input_path, allow_pickle=True)
# poses = data['poses_quat']

# joint_idx  = 16
# num_frames = len(poses)
# n_particles = 1000

# # ── Particle Filter Initialization ──────────────────────────────
# pf = HyperhemisphericalParticleFilter(n_particles=n_particles, dim=3)

# q_amass = poses[0, joint_idx, :]
# q = np.array([q_amass[1], q_amass[2], q_amass[3], q_amass[0]])  # x,y,z,w
# if q[-1] < 0:
#     q = -q

# noise     = np.random.randn(n_particles, 4) * 0.01
# particles = q + noise
# particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
# particles[particles[:, -1] < 0] *= -1

# pf.set_state(HyperhemisphericalDiracDistribution(particles))

# estimates = np.zeros((num_frames, poses.shape[1], 4))
# estimates[0, joint_idx, :] = pf.filter_state.mean()  # frame 0

# # ── Filter Loop ──────────────────────────────────────────────────
# for frame_idx in range(1, num_frames):

#     # PREDICT
#     particles = pf.filter_state.d.copy()
#     noise     = np.random.randn(n_particles, 4) * 0.01
#     particles = particles + noise
#     particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
#     particles[particles[:, -1] < 0] *= -1
#     pf.filter_state.d = particles

#     # OBSERVE
#     q_amass = poses[frame_idx, joint_idx, :]
#     q_obs   = np.array([q_amass[1], q_amass[2], q_amass[3], q_amass[0]])  # x,y,z,w
#     if q_obs[-1] < 0:
#         q_obs = -q_obs

#     # UPDATE
#     meas_noise = HyperhemisphericalWatsonDistribution(q_obs, kappa=500.0)
#     pf.update_nonlinear_using_likelihood(meas_noise.pdf)

#     # ESTIMATE
#     estimates[frame_idx, joint_idx, :] = pf.filter_state.mean()

# # ── Jump Detection ───────────────────────────────────────────────
# def quat_geodesic_distance(q1, q2):
#     dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)
#     return 2 * np.arccos(dot)

# observed_jumps = []
# filtered_jumps = []
# jump_gaps      = []

# for frame_idx in range(1, num_frames):
#     # observed consecutive jump
#     q_obs_prev = poses[frame_idx-1, joint_idx, [1,2,3,0]]
#     q_obs_curr = poses[frame_idx,   joint_idx, [1,2,3,0]]
#     if q_obs_prev[-1] < 0: q_obs_prev = -q_obs_prev
#     if q_obs_curr[-1] < 0: q_obs_curr = -q_obs_curr
#     obs_jump = quat_geodesic_distance(q_obs_prev, q_obs_curr)

#     # filtered consecutive jump
#     q_est_prev = estimates[frame_idx-1, joint_idx, :]
#     q_est_curr = estimates[frame_idx,   joint_idx, :]
#     filt_jump  = quat_geodesic_distance(q_est_prev, q_est_curr)

#     # gap = anomaly signal
#     gap = np.abs(obs_jump - filt_jump)

#     observed_jumps.append(obs_jump)
#     filtered_jumps.append(filt_jump)
#     jump_gaps.append(gap)

# observed_jumps = np.array(observed_jumps)
# filtered_jumps = np.array(filtered_jumps)
# jump_gaps      = np.array(jump_gaps)

# # Threshold
# mean_gap  = np.mean(jump_gaps)
# std_gap   = np.std(jump_gaps)
# threshold = mean_gap + 3 * std_gap

# anomaly_frames = np.where(jump_gaps > threshold)[0] + 1

# print(f"Joint {joint_idx} — {len(anomaly_frames)} anomalies detected")
# print(f"Threshold: {np.degrees(threshold):.2f} degrees")
# for f in anomaly_frames:
#     print(f"  frame {f:4d}: "
#           f"obs={np.degrees(observed_jumps[f-1]):.2f}°  "
#           f"filt={np.degrees(filtered_jumps[f-1]):.2f}°  "
#           f"gap={np.degrees(jump_gaps[f-1]):.2f}°")

# # ── Plot ─────────────────────────────────────────────────────────
# frames = np.arange(1, num_frames)
# fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# axes[0].plot(frames, np.degrees(observed_jumps), color='steelblue')
# axes[0].set_ylabel('degrees')
# axes[0].set_title(f'Observed jumps — joint {joint_idx}')

# axes[1].plot(frames, np.degrees(filtered_jumps), color='green')
# axes[1].set_ylabel('degrees')
# axes[1].set_title(f'Filtered jumps — joint {joint_idx}')

# axes[2].plot(frames, np.degrees(jump_gaps), color='orange')
# axes[2].axhline(np.degrees(threshold), color='red',
#                 linestyle='--', label=f'threshold ({np.degrees(threshold):.1f}°)')
# axes[2].scatter(anomaly_frames, np.degrees(jump_gaps[anomaly_frames-1]),
#                 color='red', zorder=5, s=50,
#                 label=f'anomalies ({len(anomaly_frames)})')
# axes[2].set_ylabel('degrees')
# axes[2].set_xlabel('Frame')
# axes[2].set_title('Gap (anomaly signal)')
# axes[2].legend()

# plt.suptitle(f'Joint {joint_idx} Jump Detection', fontsize=13)
# plt.tight_layout()
# plt.show()