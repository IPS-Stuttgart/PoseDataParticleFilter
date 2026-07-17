"""
detect_ground_penetration.py

Detect AMASS/SMPL+H frames in which part of the body mesh penetrates below
the (estimated) floor plane -- a different failure mode from the temporal
geodesic-jump and SKEL joint-limit checks already in this repo
(identifing_incorrect_frames.py, check_flagged_frames_against_pose_limits.py,
amass_particle_filtering_hypersphericalparticlefilter.py), all of which only
ever look at *local* per-joint rotation values.

WHY "ORIENTATION" ALONE CANNOT ANSWER THIS
-------------------------------------------
The axis-angle values in AMASS's `poses` array are *local* joint rotations
relative to each joint's parent in the kinematic tree. Whether a joint ends
up above or below the floor is a property of its position in the *world*
frame, which only exists after:

  1. forward kinematics through the whole kinematic chain (parent-to-child
     rotations), using the body SHAPE (`betas`) for bone lengths, and
  2. adding the per-frame global translation (`trans`).

Two frames can have an identical local ankle rotation and still put the
ankle at completely different world heights, depending on the hip/pelvis
pose and `trans`. So this is answerable, but it needs the full SMPL+H
forward pass (shape + pose + translation), not just the pose array -- it is
a genuinely different, complementary signal to what check_flagged_frames_
against_pose_limits.py and the HHPF scripts already compute.

METHOD
------
  1. Run the official SMPL+H forward kinematics (via `smplx`, the reference
     implementation of the SMPL model family: Choutas et al.,
     https://github.com/vchoutas/smplx -- the same package used in AMASS's
     own visualization tutorials) on every frame, using the standard AMASS
     poses[:,156] -> {global_orient, body_pose, left_hand_pose,
     right_hand_pose} decomposition.
  2. This yields, per frame, the 6890 world-space mesh vertices (already
     including shape and translation -- unlike the raw `poses` array).
  3. AMASS's world frame is Y-up (see References) but the floor is not
     necessarily at Y=0, and can vary slightly per file/sub-dataset, so it
     is estimated from the data itself: a low percentile of the sequence's
     own per-frame lowest-vertex height, on the assumption that the
     subject's feet legitimately reach the true floor somewhere in the
     clip (true for essentially all locomotion/standing AMASS clips).
     Percentile (not the bare minimum) is used so that a handful of
     anomalous frames cannot single-handedly bias the very baseline used
     to detect them.
  4. Per-frame penetration depth = how far the single lowest mesh vertex
     dips below that floor estimate. Frames whose depth exceeds a
     configurable tolerance are flagged and clustered into contiguous
     "events" (the same clustering idea evaluate_detection.py in this repo
     already uses for scoring).

This is a genuine, actively-studied artifact category for AMASS/SMPL
kinematic motion, not something invented for this script -- see References.

LIMITATIONS -- READ BEFORE TRUSTING FLAGGED FRAMES
---------------------------------------------------
  * Assumes one locally flat floor per file. Stairs/ramps/multi-level scenes
    will produce false positives.
  * The floor estimate needs the feet to legitimately touch the ground
    somewhere in the clip. Clips that are airborne or seated throughout will
    not calibrate correctly -- check `diagnostics` in the summary CSV.
  * Robust to a *minority* of contaminated frames, not to a sequence that is
    mostly broken.
  * Pure height-thresholding is a known-imperfect baseline for contact
    detection in the literature (Mourot et al. 2022 report ~57-60% accuracy
    for naive height/velocity thresholds vs. ~90%+ for a learned
    classifier) -- treat this as a solid, cheap first-pass filter and
    starting point for precision/recall analysis (pair it with
    inject_synthetic_anomalies.py / evaluate_detection.py in this repo, the
    same way the HHPF detector is validated), not a final contact
    classifier. A velocity-based refinement is sketched at the bottom.

USAGE
-----
    # Validate the detection/statistics logic on synthetic data -- no
    # AMASS file, no smplx, no torch, no SMPL+H model files needed:
    python detect_ground_penetration.py --self-test

    # Real run (needs torch + smplx + licensed SMPL+H model files, see
    # build_smplh_model() docstring below for where to get them):
    python detect_ground_penetration.py \\
        --input "C:/path/to/Subject_1_F_1_poses.npz" \\
        --model-root "C:/path/to/body_models"

    python detect_ground_penetration.py --input "C:/path/to/KIT" \\
        --model-root "C:/path/to/body_models" --pattern "*_poses.npz"

REFERENCES
----------
  [1] Mahmood, N., Ghorbani, N., Troje, N.F., Pons-Moll, G., Black, M.J.
      "AMASS: Archive of Motion Capture as Surface Shapes." ICCV 2019.
  [2] Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.
      "SMPL: A Skinned Multi-Person Linear Model." ACM TOG (SIGGRAPH Asia)
      2015.
  [3] Romero, J., Tzionas, D., Black, M.J. "Embodied Hands: Modeling and
      Capturing Hands and Bodies Together." ACM TOG (SIGGRAPH Asia) 2017.
      (MANO; combined with SMPL to form SMPL+H, the body model AMASS uses.)
  [4] Choutas, V. et al. `smplx` -- reference PyTorch implementation of
      SMPL / SMPL+H / SMPL-X. https://github.com/vchoutas/smplx
  [5] Rempe, D., Guibas, L., Hertzmann, A., Russell, B., Villegas, R.,
      Yang, J. "Contact and Human Dynamics from Monocular Video." ECCV
      2020. (Joint-level foot/ground contact from heuristics and learned
      models; reports heuristic-vs-learned contact-classification accuracy.)
  [6] Rempe, D., Birdal, T., Hertzmann, A., Yang, J., Sridhar, S.,
      Guibas, L. "HuMoR: 3D Human Motion Model for Robust Pose
      Estimation." ICCV 2021. (Explicitly uses SMPL+H "since it is used by
      AMASS"; models person-ground contact from joint/vertex positions.)
  [7] Mourot, L., Hoyet, L., Le Clerc, F., Hellier, P. "UnderPressure: Deep
      Learning for Foot Contact Detection, Ground Reaction Force
      Estimation and Footskate Cleanup." Computer Graphics Forum
      (SCA) 41(8), 2022. arXiv:2208.04598. (Quantifies how much a learned
      classifier beats naive height/velocity thresholding -- the honesty
      check for this script's approach.)
  [8] Luo, Z., Yuan, Y., et al. "Perpetual Humanoid Control for Real-time
      Simulated Avatars." ICCV 2023. (Ground penetration / floating as
      named, addressed artifacts when imitating AMASS motion.)
  [9] Subsequent physics-based-control papers building on PHC-filtered
      AMASS subsets (2025-2026) that report "ground penetration",
      "self-penetration", "foot sliding" and "floating" as standard
      per-clip filtering/evaluation metrics for kinematic AMASS-derived
      motion, confirming this remains an active, currently-used practice.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Optional heavy deps. Only needed for the real SMPL+H forward-kinematics
# path -- NOT needed for --self-test, which exercises only the detection
# logic (floor estimate + thresholding + event clustering) on synthetic
# vertex-height data, so the statistics can be validated even before torch
# / smplx / the licensed model files are set up in this environment.
# ----------------------------------------------------------------------
try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

try:
    import smplx
    _HAVE_SMPLX = True
except ImportError:
    _HAVE_SMPLX = False


DEFAULT_OUTPUT_DIR = Path("outputs") / "ground_penetration"

# Lower-body joints in the standard 0-21 SMPL/SMPL+H body-joint layout --
# same numbering as SMPL_JOINT_NAMES in
# check_flagged_frames_against_pose_limits.py elsewhere in this repo.
LOWER_BODY_JOINTS = {
    0: "pelvis", 1: "left_hip", 2: "right_hip", 4: "left_knee", 5: "right_knee",
    7: "left_ankle", 8: "right_ankle", 10: "left_foot", 11: "right_foot",
}


# ========================================================================
# PART 0 -- AMASS I/O
# ========================================================================

def decode_gender(raw) -> str:
    g = raw.item() if hasattr(raw, "item") else raw
    if isinstance(g, bytes):
        g = g.decode("utf-8")
    g = str(g).strip().lower()
    if g not in ("male", "female", "neutral"):
        raise ValueError(f"Unrecognized AMASS gender value: {raw!r}")
    return g


def load_amass_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {
        "poses": np.asarray(data["poses"], dtype=np.float64),
        "trans": np.asarray(data["trans"], dtype=np.float64),
        "betas": np.asarray(data["betas"], dtype=np.float64),
        "gender": decode_gender(data["gender"]),
    }
    if "mocap_framerate" in data.files:
        out["mocap_framerate"] = float(data["mocap_framerate"])
    return out


def decompose_amass_pose(poses: np.ndarray):
    """Split AMASS SMPL+H (T, 156) axis-angle poses into model inputs.

    Layout: [0:3] global orient: [3:66] body pose (21 joints x 3):
    [66:111] left hand (15 joints x 3): [111:156] right hand (15 joints x 3).
    See References [1]/[3] and the AMASS repo (github.com/nghorbani/amass).
    """
    if poses.shape[-1] != 156:
        raise ValueError(
            f"Expected AMASS SMPL+H poses with last dim 156, got {poses.shape[-1]}."
        )
    global_orient = poses[:, 0:3]
    body_pose = poses[:, 3:66]
    left_hand_pose = poses[:, 66:111]
    right_hand_pose = poses[:, 111:156]
    return global_orient, body_pose, left_hand_pose, right_hand_pose


# ========================================================================
# PART 1 -- SMPL+H forward kinematics (the part that needs torch/smplx)
# ========================================================================

def build_smplh_model(model_root: Path, gender: str, num_betas: int = 16, batch_size: int = 1):
    """Build the official SMPL+H model via the `smplx` reference implementation.

    Requires ``pip install torch smplx --break-system-packages`` plus the
    licensed SMPL+H model files, which cannot be redistributed here.
    Download the "Extended SMPL+H model" (the AMASS-compatible one, with
    16 betas) from https://mano.is.tue.mpg.de/ (free registration
    required -- this is the exact download AMASS's own tutorials point
    users to). Arrange --model-root so it contains e.g.
    model_root/smplh/{MALE,FEMALE,NEUTRAL}/model.npz (or SMPLH_MALE.pkl
    etc., depending on which release you downloaded / how you merged in
    MANO hand shapes via smplx's tools/merge_smplh_mano.py).
    """
    if not _HAVE_TORCH or not _HAVE_SMPLX:
        raise RuntimeError(
            "This step needs `torch` and `smplx` installed and the "
            "licensed SMPL+H model files under --model-root. Run with "
            "--self-test to validate the detection logic without either."
        )
    return smplx.create(
        model_path=str(model_root),
        model_type="smplh",
        gender=gender,
        num_betas=num_betas,
        use_pca=False,        # AMASS stores full 45-DoF axis-angle hand poses, not PCA coefficients
        flat_hand_mean=True,  # AMASS hand poses are relative to a flat hand, not MANO's relaxed mean pose
        batch_size=batch_size,
    )


def run_forward_kinematics_streaming(
    model,
    poses: np.ndarray,
    betas: np.ndarray,
    trans: np.ndarray,
    num_betas: int,
    up_axis: int,
    up_sign: float,
    chunk_size: int = 256,
    device: str = "cpu",
):
    """Batched SMPL+H forward pass, memory-bounded for long sequences.

    Rather than keeping the full (T, 6890, 3) vertex array in memory (which
    gets large for long clips), this keeps only the one scalar per frame
    that ground-penetration detection actually needs: the signed height
    (``up_sign * coordinate_along_up_axis``) of the single lowest mesh
    vertex. Joint positions (much smaller: (T, ~52-73, 3)) are kept in
    full, for the lower-body interpretability breakdown.

    Returns
    -------
    min_signed_height : (T,) float64
    joints             : (T, J, 3) float64 (joints[:, :22] match the
                          0-21 SMPL_JOINT_NAMES layout used elsewhere in
                          this repo)
    """
    if not _HAVE_TORCH:
        raise RuntimeError("torch is required for run_forward_kinematics_streaming().")

    global_orient, body_pose, left_hand, right_hand = decompose_amass_pose(poses)
    n_frames = poses.shape[0]
    betas_t = torch.as_tensor(betas[:num_betas], dtype=torch.float32, device=device)

    min_signed_height = np.empty(n_frames, dtype=np.float64)
    joint_chunks = []
    model = model.to(device)
    with torch.no_grad():
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            b = end - start
            out = model(
                global_orient=torch.as_tensor(global_orient[start:end], dtype=torch.float32, device=device),
                body_pose=torch.as_tensor(body_pose[start:end], dtype=torch.float32, device=device),
                left_hand_pose=torch.as_tensor(left_hand[start:end], dtype=torch.float32, device=device),
                right_hand_pose=torch.as_tensor(right_hand[start:end], dtype=torch.float32, device=device),
                betas=betas_t.unsqueeze(0).expand(b, -1),
                transl=torch.as_tensor(trans[start:end], dtype=torch.float32, device=device),
                return_verts=True,
            )
            verts = out.vertices.detach().cpu().numpy()          # (b, 6890, 3)
            signed = up_sign * verts[:, :, up_axis]                # (b, 6890)
            min_signed_height[start:end] = signed.min(axis=1)
            joint_chunks.append(out.joints.detach().cpu().numpy())

    joints = np.concatenate(joint_chunks, axis=0).astype(np.float64)
    return min_signed_height, joints


# ========================================================================
# PART 2 -- floor estimation + penetration detection
# (pure numpy; independent of how the per-frame minimum height was
#  obtained, so this part is fully unit-testable without smplx/torch --
#  see --self-test / self_test() below)
# ========================================================================

def per_frame_min_height(points: np.ndarray, up_axis: int, up_sign: float) -> np.ndarray:
    """points: (T, N, 3) -> (T,) signed height of the lowest point per frame.

    Multiplying by up_sign *before* taking the min (rather than after) is
    what makes this correct for both sign conventions: if up_sign=-1, the
    "lowest in the true up direction" point is the one with the *largest*
    raw coordinate, and flipping sign before min() handles that
    automatically.
    """
    return (up_sign * points[:, :, up_axis]).min(axis=1)


def estimate_floor_height(min_height_per_frame: np.ndarray, floor_percentile: float = 5.0) -> dict:
    floor_est = float(np.percentile(min_height_per_frame, floor_percentile))
    absolute_min = float(np.min(min_height_per_frame))
    nearest_frame = int(np.argmin(np.abs(min_height_per_frame - floor_est)))
    return {
        "floor_height_m": floor_est,
        "absolute_min_height_m": absolute_min,
        "gap_floor_to_absolute_min_m": floor_est - absolute_min,
        "example_calibration_frame": nearest_frame,
        "floor_percentile_used": floor_percentile,
    }


def cluster_into_events(flag_bool: np.ndarray, gap_tolerance: int = 1) -> list:
    """Merge flagged frames separated by <= gap_tolerance clean frames into
    contiguous (start, end) inclusive events -- same clustering idea as
    evaluate_detection.py's cluster_into_events elsewhere in this repo."""
    idxs = np.flatnonzero(flag_bool)
    if len(idxs) == 0:
        return []
    events = []
    start = prev = int(idxs[0])
    for i in idxs[1:]:
        i = int(i)
        if i - prev <= gap_tolerance + 1:
            prev = i
        else:
            events.append((start, prev))
            start = prev = i
    events.append((start, prev))
    return events


def diagnose_up_axis(points: np.ndarray) -> dict:
    """Heuristic sanity check (NOT authoritative -- a diagnostic aid only).

    The true vertical axis should be the one where the per-frame minimum
    position is *most stable* over time: feet repeatedly return to close
    to the same floor height. The two horizontal axes drift with
    locomotion and have much larger spread. Disagreement with --up-axis is
    worth a second look, not an automatic override.
    """
    spreads = [float(np.std(points[:, :, axis].min(axis=1))) for axis in range(3)]
    return {"per_axis_std_of_frame_min": spreads, "heuristic_guessed_up_axis": int(np.argmin(spreads))}


@dataclass
class PenetrationResult:
    floor_height: float
    depths: np.ndarray            # (T,) penetration depth in meters, 0 where not penetrating
    flagged: np.ndarray           # (T,) bool
    events: list                  # list of (start, end) inclusive frame-index tuples
    diagnostics: dict


def detect_ground_penetration(
    min_height_per_frame: np.ndarray,
    floor_percentile: float = 5.0,
    penetration_threshold_m: float = 0.02,
    gap_tolerance: int = 2,
) -> PenetrationResult:
    floor_info = estimate_floor_height(min_height_per_frame, floor_percentile)
    floor_height = floor_info["floor_height_m"]
    depths = np.clip(floor_height - min_height_per_frame, a_min=0.0, a_max=None)
    flagged = depths > penetration_threshold_m
    events = cluster_into_events(flagged, gap_tolerance=gap_tolerance)
    return PenetrationResult(
        floor_height=floor_height, depths=depths, flagged=flagged,
        events=events, diagnostics=floor_info,
    )


def lower_body_breakdown(joints: np.ndarray, floor_height: float, up_axis: int, up_sign: float,
                          frame_idxs: Iterable[int]) -> dict:
    """For each given frame, which lower-body joints sit below the floor."""
    out = {}
    for f in frame_idxs:
        f = int(f)
        below = []
        for j, name in LOWER_BODY_JOINTS.items():
            if j >= joints.shape[1]:
                continue
            h = up_sign * joints[f, j, up_axis]
            if h < floor_height:
                below.append(name)
        out[f] = below
    return out


# ========================================================================
# PART 3 -- reporting
# ========================================================================

def write_reports(result: PenetrationResult, joints: Optional[np.ndarray], up_axis: int, up_sign: float,
                   output_dir: Path, source_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / f"{source_name}_ground_penetration_frames.csv"
    summary_path = output_dir / f"{source_name}_ground_penetration_summary.csv"

    flagged_idxs = np.flatnonzero(result.flagged)
    breakdown = (
        lower_body_breakdown(joints, result.floor_height, up_axis, up_sign, flagged_idxs)
        if joints is not None else {}
    )
    event_id_of = {}
    for eid, (s, e) in enumerate(result.events):
        for f in range(s, e + 1):
            event_id_of[f] = eid

    with detailed_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "penetration_depth_m", "event_id", "joints_below_floor"])
        for f in flagged_idxs:
            f = int(f)
            writer.writerow([f, f"{result.depths[f]:.5f}", event_id_of.get(f, -1),
                              ";".join(breakdown.get(f, []))])

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["source_file", source_name])
        writer.writerow(["frames_flagged", int(result.flagged.sum())])
        writer.writerow(["n_events", len(result.events)])
        max_depth = float(result.depths.max()) if result.depths.size else 0.0
        writer.writerow(["max_penetration_depth_m", f"{max_depth:.5f}"])
        for k, v in result.diagnostics.items():
            writer.writerow([k, v])
        for i, (s, e) in enumerate(result.events):
            writer.writerow([f"event_{i}_frames", f"{s}-{e}"])

    return detailed_path, summary_path


def plot_penetration(min_height_per_frame: np.ndarray, result: PenetrationResult,
                      output_dir: Path, source_name: str) -> Path:
    import matplotlib.pyplot as plt

    frames = np.arange(len(min_height_per_frame))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frames, min_height_per_frame, color="#378ADD", linewidth=1.2,
            alpha=0.9, zorder=2, label="lowest body point (signed height)")
    ax.fill_between(frames, min_height_per_frame, result.floor_height, alpha=0.06, color="#378ADD", zorder=1)
    ax.axhline(result.floor_height, color="#2E7D32", linestyle=":", linewidth=1.4, zorder=3,
               label=f"estimated floor ({result.floor_height:.3f} m)")
    mask = result.flagged
    ax.scatter(frames[mask], min_height_per_frame[mask], color="#E24B4A", s=32, zorder=5,
               label=f"flagged ({int(mask.sum())} frames, {len(result.events)} event(s))")
    ax.set_xlabel("Frame index", fontsize=10)
    ax.set_ylabel("Signed height (m)", fontsize=10)
    ax.set_title(f"Ground-penetration diagnostics -- {source_name}", fontsize=12)
    ax.grid(True, linewidth=0.4, alpha=0.5, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, framealpha=0.85, loc="best")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{source_name}_ground_penetration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ========================================================================
# PART 4 -- top-level per-file pipeline
# ========================================================================

def analyze_amass_file(
    npz_path: Path,
    model_root: Optional[Path],
    up_axis: int = 1,
    up_sign: float = 1.0,
    floor_percentile: float = 5.0,
    penetration_threshold_m: float = 0.02,
    gap_tolerance: int = 2,
    num_betas: int = 16,
    chunk_size: int = 256,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    plot: bool = True,
) -> PenetrationResult:
    data = load_amass_npz(npz_path)
    if model_root is None:
        raise ValueError("--model-root is required outside of --self-test.")

    model = build_smplh_model(model_root, data["gender"], num_betas=num_betas)
    min_h, joints = run_forward_kinematics_streaming(
        model, data["poses"], data["betas"], data["trans"],
        num_betas=num_betas, up_axis=up_axis, up_sign=up_sign, chunk_size=chunk_size,
    )

    axis_diag = diagnose_up_axis(joints)
    if axis_diag["heuristic_guessed_up_axis"] != up_axis:
        print(
            f"[warning] up-axis heuristic guesses axis {axis_diag['heuristic_guessed_up_axis']} "
            f"but axis {up_axis} is configured (per-axis std of per-frame minimum: "
            f"{axis_diag['per_axis_std_of_frame_min']}). Sanity-check --up-axis/--up-sign "
            f"before trusting flagged frames on this file."
        )

    result = detect_ground_penetration(
        min_h, floor_percentile=floor_percentile,
        penetration_threshold_m=penetration_threshold_m, gap_tolerance=gap_tolerance,
    )

    source_name = npz_path.stem
    detailed_path, summary_path = write_reports(result, joints, up_axis, up_sign, output_dir, source_name)
    print(f"[{source_name}] floor={result.floor_height:.4f} m | "
          f"{int(result.flagged.sum())} flagged frame(s) in {len(result.events)} event(s)")
    print(f"  detailed: {detailed_path}")
    print(f"  summary : {summary_path}")

    if plot:
        plot_path = plot_penetration(min_h, result, output_dir, source_name)
        print(f"  plot    : {plot_path}")

    return result


# ========================================================================
# PART 5 -- self-test on synthetic ground truth
# (mirrors the inject_synthetic_anomalies.py / evaluate_detection.py
#  pattern already used in this repo for the HHPF detector, but scoped to
#  just the detection logic -- no smplx/torch/model files needed)
# ========================================================================

def _make_synthetic_sequence(n_frames: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_floor = 0.0
    t = np.arange(n_frames)
    swing = 0.06 * np.abs(np.sin(2 * np.pi * t / 24.0))     # 0-6 cm gait swing ABOVE the floor (normal, not anomalous)
    noise = rng.normal(0, 0.004, size=n_frames)              # 4 mm contact jitter
    min_height = true_floor + swing + noise

    inj_start, inj_end = 150, 161                            # deliberate penetration event: 5 cm below floor
    min_height[inj_start:inj_end + 1] -= 0.05

    min_height[300] -= 0.005                                 # sub-threshold dip: should NOT be flagged at 2 cm

    return min_height, true_floor, (inj_start, inj_end)


def self_test(penetration_threshold_m: float = 0.02, floor_percentile: float = 5.0,
              gap_tolerance: int = 2, verbose: bool = True) -> bool:
    min_height, true_floor, injected_event = _make_synthetic_sequence()
    result = detect_ground_penetration(
        min_height, floor_percentile=floor_percentile,
        penetration_threshold_m=penetration_threshold_m, gap_tolerance=gap_tolerance,
    )

    floor_ok = abs(result.floor_height - true_floor) < 0.01
    events = result.events
    hit = any(s <= injected_event[1] and injected_event[0] <= e for s, e in events)
    no_spurious = len(events) == 1
    passed = floor_ok and hit and no_spurious

    if verbose:
        print("=== self-test: synthetic ground-truth ground-penetration ===")
        print(f"  true floor height        : {true_floor:.4f} m")
        print(f"  estimated floor height   : {result.floor_height:.4f} m "
              f"({'OK' if floor_ok else 'FAIL'}, within 1 cm)")
        print(f"  injected penetration event: frames {injected_event}")
        print(f"  detected events          : {events}")
        print(f"  injected event recovered : {'YES' if hit else 'NO'}")
        print(f"  no spurious extra events : {'YES' if no_spurious else 'NO'} (found {len(events)} total)")
        print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ========================================================================
# CLI
# ========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect AMASS SMPL+H frames where the body mesh penetrates the floor.",
    )
    parser.add_argument("--input", type=Path, default=None, help="AMASS *_poses.npz file or a directory of them.")
    parser.add_argument("--pattern", default="*_poses.npz", help="Glob used when --input is a directory.")
    parser.add_argument("--model-root", type=Path, default=None,
                         help="Folder with the licensed SMPL+H model files (see build_smplh_model() docstring).")
    parser.add_argument("--up-axis", type=int, default=1, choices=[0, 1, 2],
                         help="0=X, 1=Y, 2=Z. Default 1 (Y-up), the SMPL/AMASS convention.")
    parser.add_argument("--up-sign", type=float, default=1.0, choices=[1.0, -1.0])
    parser.add_argument("--floor-percentile", type=float, default=5.0)
    parser.add_argument("--penetration-threshold-cm", type=float, default=2.0)
    parser.add_argument("--gap-tolerance", type=int, default=2)
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                         help="Validate the detection logic on synthetic data; needs no AMASS "
                              "file, no smplx/torch, no model files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        ok = self_test(
            penetration_threshold_m=args.penetration_threshold_cm / 100.0,
            floor_percentile=args.floor_percentile,
            gap_tolerance=args.gap_tolerance,
        )
        raise SystemExit(0 if ok else 1)

    if args.input is None:
        raise SystemExit("--input is required (or pass --self-test to validate the logic alone).")

    files = [args.input] if args.input.is_file() else sorted(args.input.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched under {args.input}")

    for f in files:
        analyze_amass_file(
            f,
            model_root=args.model_root,
            up_axis=args.up_axis,
            up_sign=args.up_sign,
            floor_percentile=args.floor_percentile,
            penetration_threshold_m=args.penetration_threshold_cm / 100.0,
            gap_tolerance=args.gap_tolerance,
            num_betas=args.num_betas,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir,
            plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
