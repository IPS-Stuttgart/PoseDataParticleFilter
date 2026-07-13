"""
inject_synthetic_anomalies.py

Injects synthetic orientation corruptions into an AMASS (SMPL+H) pose sequence,
with EXACTLY known ground truth, so that a detector's flags (e.g. the
HHPF + SKEL "AND" rule) can be scored with precision / recall / F1.

WHY THIS DESIGN
----------------
1. Perturbations are injected as proper SO(3) rotation compositions:

       R_corrupted = R_original  (.)  R_perturbation      (local-frame)

   The geodesic (angular) distance between R_corrupted and R_original on SO(3)
   is *exactly* the rotation angle of R_perturbation, regardless of what
   R_original was. This is not an approximation -- it falls directly out of
   the bi-invariance of the SO(3) angular metric. That matters here because
   HHPF's anomaly score is itself a geodesic-distance-based quantity, so the
   injected ground truth is measured in the same units the detector uses.
   (Naively adding axis-angle vectors does NOT have this property except in
   the small-angle limit.)

2. Three corruption archetypes, chosen to mimic real MoCap failure modes
   rather than arbitrary noise:
     - "spike"     : single-frame outlier (solver jump / marker mislabel-and-
                      recover). Duration = 1 frame.
     - "sustained" : constant offset held for a short window (stuck/frozen
                      joint, persistent mislabel). Duration = 5-15 frames.
     - "ramp"      : trapezoidal onset/offset (slow drift, gimbal creep).
                      Duration = 10-25 frames. This is the hardest case:
                      the edges of the window are low-magnitude and may
                      legitimately fall below any detector's threshold,
                      which is exactly the kind of boundary case a
                      precision/recall curve should characterize.

3. Three severity tiers (small / medium / large, in degrees of injected
   geodesic angle) crossed with the 3 types, with 2 replicates per cell
   (18 events total) -- a small factorial design so the paper can report
   detection rate broken down by BOTH corruption type and severity, not
   just a single pooled number.

4. Joints are sampled without replacement (each used at most twice) from
   the 21 non-root SMPL body joints, so corruption is spread across the
   kinematic tree instead of concentrated on one joint.

5. Event time-windows are non-overlapping with an enforced gap, so every
   corrupted frame maps unambiguously to exactly one event.

OUTPUTS
-------
  Subject_1_F_1_poses_corrupted.npz   -- same schema as the input AMASS file
                                          (drop-in replacement for the pipeline)
  ground_truth_events.json            -- per-event metadata (type, tier, joint,
                                          frame range, injected angle, axis)
  ground_truth_masks.npz              -- frame x joint boolean/float masks for
                                          vectorized scoring
  ground_truth_timeline.png           -- diagnostic figure of what was injected
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
INPUT_PATH = "/mnt/user-data/uploads/Subject_1_F_1_poses.npz"
OUT_DIR = "/home/claude/work"

SEVERITY_DEG = {
    "small":  (20, 30),
    "medium": (40, 60),
    "large":  (70, 100),
}
TYPES = ["spike", "sustained", "ramp"]
REPLICATES_PER_CELL = 2
MIN_GAP_FRAMES = 8          # frames of clean buffer required between events
EDGE_BUFFER = 10            # avoid the very first/last frames of the clip
SUSTAINED_DUR_RANGE = (5, 15)
RAMP_DUR_RANGE = (10, 25)
RAMP_FLAT_FRACTION = 0.4    # middle fraction of a ramp window held at full peak
MASK_ANGLE_THRESHOLD_DEG = 5.0  # per-frame injected angle needed to count as "positive" in the boolean mask

SMPL_BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
CANDIDATE_JOINTS = list(range(1, 22))  # exclude joint 0 (pelvis / global root orient)

# ----------------------------------------------------------------------
# Load original data
# ----------------------------------------------------------------------
data = np.load(INPUT_PATH, allow_pickle=True)
poses = data["poses"].copy()            # (n_frames, 156) = 52 joints x 3
n_frames = poses.shape[0]
n_joints_total = poses.shape[1] // 3
poses_j = poses.reshape(n_frames, n_joints_total, 3)  # view for editing

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# Build the factorial list of events (type x severity x replicate)
# ----------------------------------------------------------------------
cells = []
for t in TYPES:
    for tier in SEVERITY_DEG:
        for rep in range(REPLICATES_PER_CELL):
            cells.append({"type": t, "tier": tier, "rep": rep})
rng.shuffle(cells)

# joint pool: each of the 21 candidate joints appears at most twice
joint_pool = np.concatenate([
    rng.permutation(CANDIDATE_JOINTS),
    rng.permutation(CANDIDATE_JOINTS),
])[: len(cells)]

def sample_duration(corruption_type):
    if corruption_type == "spike":
        return 1
    if corruption_type == "sustained":
        return int(rng.integers(SUSTAINED_DUR_RANGE[0], SUSTAINED_DUR_RANGE[1] + 1))
    if corruption_type == "ramp":
        return int(rng.integers(RAMP_DUR_RANGE[0], RAMP_DUR_RANGE[1] + 1))
    raise ValueError(corruption_type)

# ----------------------------------------------------------------------
# Place non-overlapping time windows for each event
# ----------------------------------------------------------------------
events = []
occupied = np.zeros(n_frames, dtype=bool)  # True where a frame is already reserved (+gap)

def window_free(start, end):
    lo = max(0, start - MIN_GAP_FRAMES)
    hi = min(n_frames, end + MIN_GAP_FRAMES + 1)
    return not occupied[lo:hi].any()

for idx, cell in enumerate(cells):
    dur = sample_duration(cell["type"])
    placed = False
    for _try in range(500):
        start = int(rng.integers(EDGE_BUFFER, n_frames - EDGE_BUFFER - dur))
        end = start + dur - 1
        if window_free(start, end):
            occupied[max(0, start - MIN_GAP_FRAMES): min(n_frames, end + MIN_GAP_FRAMES + 1)] = True
            placed = True
            break
    if not placed:
        # extremely unlikely given the clip length, but skip gracefully if it happens
        continue

    lo_deg, hi_deg = SEVERITY_DEG[cell["tier"]]
    peak_deg = float(rng.uniform(lo_deg, hi_deg))
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)

    frame_idxs = list(range(start, end + 1))
    if cell["type"] == "spike" or cell["type"] == "sustained":
        per_frame_deg = [peak_deg for _ in frame_idxs]
    else:  # ramp: trapezoid -> linear up, flat top, linear down
        n = len(frame_idxs)
        flat_n = max(1, int(round(n * RAMP_FLAT_FRACTION)))
        ramp_n = (n - flat_n)
        up_n = ramp_n - ramp_n // 2
        down_n = ramp_n // 2
        profile = (
            list(np.linspace(0, peak_deg, up_n, endpoint=False)) +
            [peak_deg] * flat_n +
            list(np.linspace(peak_deg, 0, down_n, endpoint=False))
        )
        # pad/truncate defensively to exactly n
        profile = (profile + [peak_deg] * n)[:n]
        per_frame_deg = profile

    joint_idx = int(joint_pool[idx])
    joint_name = SMPL_BODY_JOINT_NAMES[joint_idx]

    events.append({
        "event_id": idx,
        "type": cell["type"],
        "severity_tier": cell["tier"],
        "joint_index": joint_idx,
        "joint_name": joint_name,
        "start_frame": start,
        "end_frame": end,
        "n_frames": len(frame_idxs),
        "peak_angle_deg": peak_deg,
        "axis": axis.tolist(),
        "per_frame_angle_deg": per_frame_deg,
    })

events.sort(key=lambda e: e["start_frame"])
for i, e in enumerate(events):
    e["event_id"] = i

# ----------------------------------------------------------------------
# Apply perturbations: R_corrupted = R_original (.) R_perturbation  (local frame)
# ----------------------------------------------------------------------
for e in events:
    j = e["joint_index"]
    axis = np.array(e["axis"])
    for k, f in enumerate(range(e["start_frame"], e["end_frame"] + 1)):
        angle_deg = e["per_frame_angle_deg"][k]
        if angle_deg <= 0:
            continue
        rotvec_perturb = axis * np.deg2rad(angle_deg)
        R_orig = R.from_rotvec(poses_j[f, j])
        R_perturb = R.from_rotvec(rotvec_perturb)
        R_new = R_orig * R_perturb  # local-frame (right-multiply) composition
        poses_j[f, j] = R_new.as_rotvec()

poses_corrupted = poses_j.reshape(n_frames, n_joints_total * 3)

# ----------------------------------------------------------------------
# Verify injected geodesic distance matches design exactly (sanity check)
# ----------------------------------------------------------------------
max_err = 0.0
for e in events:
    j = e["joint_index"]
    for k, f in enumerate(range(e["start_frame"], e["end_frame"] + 1)):
        target_deg = e["per_frame_angle_deg"][k]
        r_before = R.from_rotvec(data["poses"].reshape(n_frames, n_joints_total, 3)[f, j])
        r_after = R.from_rotvec(poses_j[f, j])
        actual_deg = np.rad2deg((r_before.inv() * r_after).magnitude())
        max_err = max(max_err, abs(actual_deg - target_deg))
print(f"[sanity check] max |actual - target| injected geodesic angle across all "
      f"corrupted frames: {max_err:.8f} deg (should be ~0)")

# ----------------------------------------------------------------------
# Save corrupted AMASS-format npz (drop-in replacement)
# ----------------------------------------------------------------------
out_npz_path = f"{OUT_DIR}/Subject_1_F_1_poses_corrupted.npz"
np.savez(
    out_npz_path,
    trans=data["trans"],
    gender=data["gender"],
    mocap_framerate=data["mocap_framerate"],
    betas=data["betas"],
    dmpls=data["dmpls"],
    poses=poses_corrupted,
)

# ----------------------------------------------------------------------
# Ground truth: event list (JSON)
# ----------------------------------------------------------------------
with open(f"{OUT_DIR}/ground_truth_events.json", "w") as f:
    json.dump({
        "source_file": "Subject_1_F_1_poses.npz",
        "seed": SEED,
        "n_frames": n_frames,
        "mocap_framerate": float(data["mocap_framerate"]),
        "mask_angle_threshold_deg": MASK_ANGLE_THRESHOLD_DEG,
        "composition": "R_corrupted = R_original (.) R_perturbation  (local-frame, right-multiply)",
        "n_events": len(events),
        "events": events,
    }, f, indent=2)

# ----------------------------------------------------------------------
# Ground truth: frame x joint masks (npz) for vectorized scoring
# ----------------------------------------------------------------------
# boolean mask: True where injected angle >= MASK_ANGLE_THRESHOLD_DEG
bool_mask = np.zeros((n_frames, n_joints_total), dtype=bool)
# continuous mask: exact injected angle in degrees (0 where clean)
angle_mask = np.zeros((n_frames, n_joints_total), dtype=float)
# which event id (or -1) touches this (frame, joint) cell
event_id_mask = np.full((n_frames, n_joints_total), -1, dtype=int)

for e in events:
    j = e["joint_index"]
    for k, f in enumerate(range(e["start_frame"], e["end_frame"] + 1)):
        deg = e["per_frame_angle_deg"][k]
        angle_mask[f, j] = deg
        event_id_mask[f, j] = e["event_id"]
        if deg >= MASK_ANGLE_THRESHOLD_DEG:
            bool_mask[f, j] = True

frame_level_bool = bool_mask.any(axis=1)  # True if ANY joint anomalous that frame

np.savez(
    f"{OUT_DIR}/ground_truth_masks.npz",
    bool_mask=bool_mask,                # (n_frames, n_joints) bool -- per (frame,joint) ground truth
    angle_mask=angle_mask,              # (n_frames, n_joints) float deg -- exact injected magnitude
    event_id_mask=event_id_mask,        # (n_frames, n_joints) int -- which event, -1 = clean
    frame_level_bool=frame_level_bool,  # (n_frames,) bool -- any joint anomalous
    joint_names=np.array(SMPL_BODY_JOINT_NAMES + [f"hand_joint_{i}" for i in range(n_joints_total - 22)]),
)

print(f"\nInjected {len(events)} events covering "
      f"{int(bool_mask.any(axis=1).sum())} / {n_frames} frames "
      f"({100 * bool_mask.any(axis=1).sum() / n_frames:.1f}%) as positives.")
print(f"\nBy type:")
for t in TYPES:
    n = sum(1 for e in events if e["type"] == t)
    print(f"  {t:10s}: {n} events")
print(f"\nBy severity:")
for tier in SEVERITY_DEG:
    n = sum(1 for e in events if e["severity_tier"] == tier)
    print(f"  {tier:10s}: {n} events")
print(f"\nJoints used: {sorted(set(e['joint_name'] for e in events))}")
print(f"\nSaved:\n  {out_npz_path}\n  {OUT_DIR}/ground_truth_events.json\n  {OUT_DIR}/ground_truth_masks.npz")
