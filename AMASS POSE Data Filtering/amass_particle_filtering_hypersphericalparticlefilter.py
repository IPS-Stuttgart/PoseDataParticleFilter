import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import quaternion
import os
import matplotlib
import matplotlib.pyplot as plt
import pyrecest
from pyrecest.filters import *
from pyrecest.distributions import *

input_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\KIT_Quaternions\3\912_3_01_poses_quaternions.npz"

data = np.load(input_path, allow_pickle=True)

# pf = HyperhemisphericalParticleFilter(n_particles=1000, dim=3)
pf= HyperhemisphericalParticleFilter(n_particles= 1000, dim =3)
print(data.files)

poses= data['poses_quat']
# print(poses[0][0])

q_amass= poses [0, 0, :]

q = np.array([q_amass[3], q_amass[0], q_amass[1], q_amass[2]])
# print(q_amass)

if q[-1] < 0:
    q = -q

# print(q)
num_frames= 1965

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
estimates[0, 0, :] = pf.filter_state.mean()

for frame_idx in range(1, num_frames):
    # STEP 1 - PREDICT (random walk with noise)
    # pf.predict_identity(noise_dist)
    particles = pf.filter_state.d  # current particles
    noise = np.random.randn(len(particles), 4) * 0.01
    particles = particles + noise
    particles = particles / np.linalg.norm(particles, axis=1, keepdims=True)
    particles[particles[:, -1] < 0] *= -1
    pf.filter_state.d = particles
    
    # STEP 2 - get observed quaternion
    q_amass = poses[frame_idx, 0, :]  # (w, x, y, z)
    q_obs = np.array([q_amass[1], q_amass[2], q_amass[3], q_amass[0]])  # (x,y,z,w)
    if q_obs[-1] < 0:
        q_obs = -q_obs
    
    # STEP 3 - UPDATE (reweight particles against observation)
    meas_noise = HyperhemisphericalWatsonDistribution(q_obs, kappa=500.0)
    # pf.update_identity(meas_noise, q_obs)
    pf.update_nonlinear_using_likelihood(meas_noise.pdf)
    
    # STEP 4 - GET ESTIMATE
    estimate = pf.filter_state.mean()  # shape (4,)
    estimates[frame_idx, 0, :] = estimate


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
    q1 = poses[frame_idx - 1, 0, :]  # previous frame
    q2 = poses[frame_idx, 0, :]      # current frame
    
    # convert to pyrecest format (x,y,z,w)
    q1 = np.array([q1[1], q1[2], q1[3], q1[0]])
    q2 = np.array([q2[1], q2[2], q2[3], q2[0]])
    
    # enforce upper hemisphere
    if q1[-1] < 0: q1 = -q1
    if q2[-1] < 0: q2 = -q2
    
    dist = quat_geodesic_distance(q1, q2)
    jump_magnitudes.append(dist)

jump_magnitudes = np.array(jump_magnitudes)

# Adaptive threshold: mean + 3*std
mean_jump = np.mean(jump_magnitudes)
std_jump  = np.std(jump_magnitudes)
threshold = mean_jump + 3 * std_jump

anomaly_frames = np.where(jump_magnitudes > threshold)[0] + 1  # +1 because diff starts at frame 1

print(f"Mean jump:  {np.degrees(mean_jump):.2f} degrees")
print(f"Std jump:   {np.degrees(std_jump):.2f} degrees")
print(f"Threshold:  {np.degrees(threshold):.2f} degrees")
print(f"Anomalies at frames: {anomaly_frames}")
print(f"Jump magnitudes at anomalies: {np.degrees(jump_magnitudes[anomaly_frames-1])}")


#2. Sudden jumps compared to estimated orientations
# Distance between filter estimate and observation each frame
filter_residuals = []
for frame_idx in range(1, num_frames):
    q_est = estimates[frame_idx, 0, :]
    
    q_obs_raw = poses[frame_idx, 0, :]
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
    q_est = estimates[frame_idx, 0, :]
    
    # observed quaternion
    q_obs_raw = poses[frame_idx, 0, :]
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