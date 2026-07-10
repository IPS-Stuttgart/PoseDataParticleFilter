import pickle
import numpy as np
from skel.kin_skel import pose_param_names, pose_limits

# Mirror the right-shoulder limit onto the left, since the repo doesn't define it
pose_limits = dict(pose_limits)  # copy so we don't mutate the imported dict
if 'shoulder_r_y' in pose_limits and 'shoulder_l_y' not in pose_limits:
    pose_limits['shoulder_l_y'] = pose_limits['shoulder_r_y']

with open("output/clip_axis_angle/clip_axis_angle_skel.pkl", "rb") as f:
    skel_data = pickle.load(f)

skel_poses = skel_data['poses']   # shape (num_frames, 46)

def check_joint_limit_violations(skel_poses, flagged_frames):
    """
    Returns dict: frame_idx -> list of (param_name, value, [min,max]) violations
    Only checks params that have a defined limit in pose_limits.
    """
    violations = {}
    for f in flagged_frames:
        frame_violations = []
        for i, name in enumerate(pose_param_names):
            if name not in pose_limits:
                continue  # unchecked DOF (e.g. hips, most shoulder axes)
            val = skel_poses[f, i]
            lo, hi = pose_limits[name]
            if val < lo or val > hi:
                frame_violations.append((name, val, [lo, hi]))
        if frame_violations:
            violations[f] = frame_violations
    return violations

# flagged_frames = your HHPF-detected outlier frame indices for this clip
violations = check_joint_limit_violations(skel_poses, flagged_frames)

for f, vlist in violations.items():
    print(f"Frame {f}:")
    for name, val, limits in vlist:
        print(f"  {name}: {val:.3f} rad, limit {limits}")