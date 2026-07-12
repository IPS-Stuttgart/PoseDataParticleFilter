import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from skel.kin_skel import pose_param_names, pose_limits

# ============================================================
# Reference: standard SMPL body joint indices (0-21), for
# interpreting which joint index = which body part below.
# SMPL+H extends this with hand joints from index 22 onward.
# ============================================================
SMPL_JOINT_NAMES = {
    0: 'pelvis', 1: 'left_hip', 2: 'right_hip', 3: 'spine1',
    4: 'left_knee', 5: 'right_knee', 6: 'spine2', 7: 'left_ankle',
    8: 'right_ankle', 9: 'spine3', 10: 'left_foot', 11: 'right_foot',
    12: 'neck', 13: 'left_collar', 14: 'right_collar', 15: 'head',
    16: 'left_shoulder', 17: 'right_shoulder', 18: 'left_elbow',
    19: 'right_elbow', 20: 'left_wrist', 21: 'right_wrist',
}

# ============================================================
# PART 0: Load data
# ============================================================

amass_npz_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\Subject_1_F_1_poses.npz"   # 'poses' key, axis-angle
amass_data = np.load(amass_npz_path, allow_pickle=True)
poses_aa = amass_data['poses']                 # (num_frames, 156) flat axis-angle
num_frames = poses_aa.shape[0]
num_joints = poses_aa.shape[1] // 3            # 52 for SMPL+H
poses_aa = poses_aa.reshape(num_frames, num_joints, 3)

skel_pkl_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\SKEL\SKEL\output\Subject_1_F_1_poses\Subject_1_F_1_poses_skel.pkl"
with open(skel_pkl_path, "rb") as f:
    skel_data = pickle.load(f)
skel_poses = skel_data['poses']                # (num_frames_skel, 46)

n_frames_common = min(num_frames, skel_poses.shape[0])
if num_frames != skel_poses.shape[0]:
    print(f"WARNING: AMASS frames ({num_frames}) != SKEL frames "
          f"({skel_poses.shape[0]}); using the smaller common range.")

# ============================================================
# PART 1: Raw geodesic-jump thresholding, every joint
# ============================================================

def normalize_quat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.array(q, copy=True)
    q *= np.where(q[..., -1:] < 0, -1.0, 1.0)
    return q

def quat_geodesic_distance(q1, q2):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)  # abs handles double cover
    return 2 * np.arccos(dot)

def axis_angle_to_quat(aa):
    return R.from_rotvec(aa).as_quat()  # (x, y, z, w)

def compute_all_joint_jumps(poses_aa):
    n_frames, n_joints, _ = poses_aa.shape
    jump_magnitudes = np.zeros((n_frames - 1, n_joints))
    for frame_idx in range(1, n_frames):
        for j in range(n_joints):
            q_prev = axis_angle_to_quat(poses_aa[frame_idx - 1, j])
            q_curr = axis_angle_to_quat(poses_aa[frame_idx, j])
            jump_magnitudes[frame_idx - 1, j] = quat_geodesic_distance(q_prev, q_curr)
    mean_jump = np.mean(jump_magnitudes, axis=0)
    std_jump = np.std(jump_magnitudes, axis=0)
    thresholds = mean_jump + 3.0 * std_jump
    return jump_magnitudes, thresholds

print("Computing per-joint jump statistics across all frames/joints...")
jump_magnitudes, jump_thresholds = compute_all_joint_jumps(poses_aa)

flagged_by_joint = {}
for j in range(num_joints):
    exceed = np.where(jump_magnitudes[:, j] > jump_thresholds[j])[0]
    if len(exceed) > 0:
        flagged_by_joint[j] = (exceed + 1).tolist()  # jump i -> frame i+1

flagged_frames_all = sorted(set(
    f for frames in flagged_by_joint.values() for f in frames
))
print(f"Frames flagged on at least one joint: {len(flagged_frames_all)}")

# ============================================================
# PART 2: SKEL anatomical limit check, every frame
# ============================================================

pose_limits = dict(pose_limits)
for name, bounds in list(pose_limits.items()):
    if name.endswith('_r'):
        mirrored = name[:-2] + '_l'
        if mirrored not in pose_limits:
            pose_limits[mirrored] = bounds  # best-effort mirror, not guaranteed correct

def check_all_joint_limit_violations(skel_poses, n_frames):
    violations = {}
    for f in range(n_frames):
        frame_violations = []
        for i, name in enumerate(pose_param_names):
            if name not in pose_limits:
                continue
            val = skel_poses[f, i]
            lo, hi = pose_limits[name]
            true_lo, true_hi = min(lo, hi), max(lo, hi)
            if val < true_lo or val > true_hi:
                frame_violations.append((name, float(val), [true_lo, true_hi]))
        if frame_violations:
            violations[f] = frame_violations
    return violations

print("Checking SKEL joint limits across all frames...")
skel_violations = check_all_joint_limit_violations(skel_poses, n_frames_common)
print(f"Frames with at least one SKEL limit violation: {len(skel_violations)}")

# ============================================================
# PART 3: Cross-reference
# ============================================================

frames_flagged_by_both = sorted(set(flagged_frames_all) & set(skel_violations.keys()))

print("\n" + "=" * 70)
print(f"Total frames:                             {n_frames_common}")
print(f"Flagged by raw jump detection:            {len(flagged_frames_all)}")
print(f"Flagged by SKEL joint limits:              {len(skel_violations)}")
print(f"Flagged by BOTH (high confidence):         {len(frames_flagged_by_both)}")
print("=" * 70)

print("\n--- High-confidence frames (both methods agree) ---")
for f in frames_flagged_by_both:
    joints_here = [j for j, frames in flagged_by_joint.items() if f in frames]
    joint_names = [SMPL_JOINT_NAMES.get(j, f'joint_{j}') for j in joints_here]
    print(f"\nFrame {f}:")
    print(f"  Raw-jump flagged SMPL joints: {joint_names}")
    print(f"  SKEL limit violations:")
    for name, val, limits in skel_violations[f]:
        print(f"    {name}: {val:.3f} rad, limit {limits}")

print("\n--- All raw-jump-flagged frames, by joint ---")
for j, frames in sorted(flagged_by_joint.items()):
    jn = SMPL_JOINT_NAMES.get(j, f'joint_{j}')
    print(f"  [{j}] {jn}: {len(frames)} frames -> {frames[:10]}{'...' if len(frames) > 10 else ''}")

print("\n--- All SKEL limit-violation frames ---")
for f, vlist in sorted(skel_violations.items()):
    print(f"  Frame {f}: {[v[0] for v in vlist]}")