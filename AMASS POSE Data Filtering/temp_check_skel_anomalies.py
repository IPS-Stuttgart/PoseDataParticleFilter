import pickle
import numpy as np
from skel.kin_skel import pose_param_names, pose_limits
# skel_pkl_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\SKEL\SKEL\data\skel\skel_male.pkl"

skel_pkl_path = r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\SKEL\SKEL\output\Subject_1_F_1_poses\Subject_1_F_1_poses_skel.pkl"
with open(skel_pkl_path, "rb") as f:
    skel_data = pickle.load(f)
skel_poses = skel_data['poses']  # (num_frames, 46)
num_frames = skel_poses.shape[0]



def fitted_pose_reference_ranges(raw_limits):
    """Return ranges in the signs used by fitted SKEL pose vectors.

    ``kin_skel.pose_limits`` stores the right scapula-elevation range with
    the opposite sign.  SKEL applies the right parameter directly, but
    negates the left parameter in ``left_scapula``.  Therefore a symmetric
    pose has positive right elevation and negative left elevation.
    """
    ranges = {name: tuple(sorted(bounds)) for name, bounds in raw_limits.items()}
    ranges.update({
        'scapula_elevation_r': (0.1, 0.4),
        'scapula_elevation_l': (-0.4, -0.1),
    })
    return ranges


# These are diagnostic reference ranges.  The SKEL fitter does not enforce
# them as hard constraints during optimization, so an out-of-range value is
# a fit-quality signal, not proof that the source AMASS frame is invalid.
pose_limits = fitted_pose_reference_ranges(pose_limits)

print(f"{'Parameter':<25} {'ViolationRate':>13} {'DataMin':>10} {'DataMax':>10} {'LimitLo':>10} {'LimitHi':>10} {'Overlap?':>9}")
print("-" * 95)


for i, name in enumerate(pose_param_names):
    if name not in pose_limits:
        continue
    lo, hi = pose_limits[name]
    vals = skel_poses[:, i]
    violations = (vals < lo) | (vals > hi)
    rate = violations.mean()
    overlaps = (vals.min() <= hi) and (vals.max() >= lo)
    print(f"{name:<25} {rate*100:>12.1f}% {vals.min():>10.3f} {vals.max():>10.3f} {lo:>10.3f} {hi:>10.3f} {str(overlaps):>9}")
