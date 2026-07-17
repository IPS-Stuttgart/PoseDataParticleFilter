"""
Ground-penetration anomaly detector for AMASS SMPL+H pose sequences.

Uses forward kinematics (via the plain SMPL body model) to get each joint's
3D position in world space, estimates the floor height from the lowest
points the feet reach over the sequence, then flags any frame/joint
combination whose vertical position sinks below the floor by more than a
tolerance.

This is a THIRD, independent anomaly signal to sit alongside HHPF and the
SKEL joint-limit check: it catches errors that are physically implausible
(a joint below the ground) rather than statistically unlikely (HHPF) or
anatomically extreme (SKEL limits). Useful as another leg in the
"convergence across independent signals" argument for the paper.

NOTE ON THE UP-AXIS
--------------------
SMPL / SMPL+H canonical space is Y-up by convention, and AMASS generally
preserves that. This script prints per-axis coordinate ranges before doing
anything else so you can confirm UP_AXIS=1 is correct for this file. A
standing/walking sequence should show a ~1.5-2.0 m range on the true "up"
axis; if that's not axis 1, switch UP_AXIS to 2 and re-run.

WHY PLAIN SMPL, NOT SMPL+H
---------------------------
SMPL and SMPL+H share an identical kinematic tree from the pelvis through
the wrists (joints 0-21). SMPL+H then branches into full MANO hand joints;
plain SMPL instead has two simple terminal "hand" joints (22, 23). Because
leg/foot joint positions only depend on their ancestors in the chain -
never on the hand pose - zero-padding the SMPL+H body pose (63 values, 21
joints) with 6 zeros to fill SMPL's body_pose (69 values, 23 joints) gives
correct leg/foot positions without needing the full SMPL+H hand model.
This lets you reuse the plain SMPL model files you already extracted for
the SKEL pipeline (SMPL_zip_contents), rather than downloading a separate
SMPL+H model.

NOT YET TEST-RUN: this was written against your file layout and the
smplx 0.1.28 API but has not been executed against your actual data or
model files in this session - double check the SMPL_MODEL_DIR path and
folder layout smplx expects (typically model_path/smpl/SMPL_<GENDER>.pkl)
before trusting the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import smplx

# ---------------------------------------------------------------------------
# Configuration - edit these paths for your machine
# ---------------------------------------------------------------------------
AMASS_NPZ_PATH = Path(
    r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\Subject_1_F_1_poses.npz"
)
SMPL_MODEL_DIR = Path(
    r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\SKEL\SKEL\data\SMPL_zip_contents"
)
OUTPUT_DIR = Path("outputs") / "ground_penetration"

UP_AXIS = 1                       # 1 = Y-up (SMPL default). Try 2 if the diagnostic looks wrong.
FLOOR_PERCENTILE = 1.0            # low percentile of per-frame foot minima used as the floor
PENETRATION_TOLERANCE_M = 0.02    # 2 cm slack for mocap/model noise

# Standard SMPL 24-joint names, for reporting.
SMPL_JOINT_NAMES = {
    0: "pelvis", 1: "left_hip", 2: "right_hip", 3: "spine1",
    4: "left_knee", 5: "right_knee", 6: "spine2", 7: "left_ankle",
    8: "right_ankle", 9: "spine3", 10: "left_foot", 11: "right_foot",
    12: "neck", 13: "left_collar", 14: "right_collar", 15: "head",
    16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow",
    19: "right_elbow", 20: "left_wrist", 21: "right_wrist",
    22: "left_hand", 23: "right_hand",
}
FOOT_JOINTS = [7, 8, 10, 11]  # ankles + feet, used to estimate floor height


def load_amass_sequence(path: Path):
    data = np.load(path, allow_pickle=True)
    poses = data["poses"].astype(np.float32)     # (T, 156) SMPL+H axis-angle
    trans = data["trans"].astype(np.float32)     # (T, 3)
    betas = data["betas"].astype(np.float32)     # (16,) or (10,)
    gender = str(data["gender"])
    return poses, trans, betas, gender


def smplh_pose_to_smpl_body_pose(poses_smplh: np.ndarray) -> np.ndarray:
    """Zero-pad the SMPL+H body segment (21 joints) with two placeholder
    hand joints so it matches SMPL's 69-value body_pose (23 joints).
    Leg/foot joint positions are unaffected by this padding."""
    body_63 = poses_smplh[:, 3:66]
    hand_placeholder = np.zeros((poses_smplh.shape[0], 6), dtype=poses_smplh.dtype)
    return np.concatenate([body_63, hand_placeholder], axis=1)


def run_forward_kinematics(poses, trans, betas, gender):
    T = poses.shape[0]
    model = smplx.create(
        model_path=str(SMPL_MODEL_DIR),
        model_type="smpl",
        gender=gender if gender in ("male", "female") else "neutral",
        num_betas=betas.shape[0],
        batch_size=T,
    )

    global_orient = torch.tensor(poses[:, 0:3], dtype=torch.float32)
    body_pose = torch.tensor(smplh_pose_to_smpl_body_pose(poses), dtype=torch.float32)
    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(T, -1)
    transl_t = torch.tensor(trans, dtype=torch.float32)

    with torch.no_grad():
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_t,
            transl=transl_t,
        )
    return output.joints[:, :24, :].cpu().numpy()  # (T, 24, 3)


def diagnose_up_axis(joints: np.ndarray) -> None:
    print("Per-axis coordinate ranges across all joints/frames (sanity check):")
    for axis in range(3):
        lo = joints[..., axis].min()
        hi = joints[..., axis].max()
        print(f"  axis {axis}: [{lo:.3f}, {hi:.3f}]  range={hi - lo:.3f} m")
    print(
        f"Using UP_AXIS = {UP_AXIS}. A standing/walking sequence should show "
        "a ~1.5-2.0 m range on the up axis; confirm before trusting flags.\n"
    )


def estimate_floor_height(joints: np.ndarray) -> float:
    foot_heights = joints[:, FOOT_JOINTS, UP_AXIS]     # (T, 4)
    per_frame_min = foot_heights.min(axis=1)           # (T,)
    return float(np.percentile(per_frame_min, FLOOR_PERCENTILE))


def flag_ground_penetration(joints: np.ndarray, floor_height: float):
    heights = joints[..., UP_AXIS] - floor_height       # (T, 24)
    violation_mask = heights < -PENETRATION_TOLERANCE_M
    return heights, violation_mask


def summarize(heights, violation_mask, floor_height):
    T, J = violation_mask.shape
    flagged_frames = np.flatnonzero(violation_mask.any(axis=1))
    print(f"Estimated floor height: {floor_height:.4f} m (UP_AXIS={UP_AXIS})")
    print(f"Frames with at least one joint below ground: {len(flagged_frames)}/{T}\n")
    for joint_idx in range(J):
        frames = np.flatnonzero(violation_mask[:, joint_idx])
        if len(frames) == 0:
            continue
        depths_cm = -heights[frames, joint_idx] * 100.0
        name = SMPL_JOINT_NAMES.get(joint_idx, f"joint_{joint_idx}")
        print(
            f"  [{joint_idx:2d}] {name:<14} {len(frames):4d} frames | "
            f"max penetration {depths_cm.max():.1f} cm | frames -> "
            f"{frames[:10].tolist()}{'...' if len(frames) > 10 else ''}"
        )
    return flagged_frames


def main() -> None:
    poses, trans, betas, gender = load_amass_sequence(AMASS_NPZ_PATH)
    joints = run_forward_kinematics(poses, trans, betas, gender)

    diagnose_up_axis(joints)
    floor_height = estimate_floor_height(joints)
    heights, violation_mask = flag_ground_penetration(joints, floor_height)
    flagged_frames = summarize(heights, violation_mask, floor_height)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "ground_penetration_flags.npz",
        joints=joints,
        heights=heights,
        violation_mask=violation_mask,
        floor_height=floor_height,
        flagged_frames=flagged_frames,
    )
    print(f"\nSaved: {OUTPUT_DIR / 'ground_penetration_flags.npz'}")


if __name__ == "__main__":
    main()
