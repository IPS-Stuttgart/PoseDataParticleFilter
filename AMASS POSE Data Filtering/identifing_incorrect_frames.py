"""
Find suspicious / incorrect frames in AMASS SMPL+H axis-angle pose files.

The AMASS KIT SMPL+H files store ``poses`` as rotation vectors:

    poses[:, 0:3]    -> root/global orientation
    poses[:, 3:66]   -> body joints
    poses[:, 66:156] -> hand joints

That is 52 rotations in standard AMASS SMPL+H files. Some exports count the
body/root joints differently, so this script also accepts any pose length that
is divisible by 3 and reports the detected joint count.

The cited Akhter & Black CVPR 2015 paper learns pose-conditioned joint-angle
limits from mocap data. Their learned prior is not included in this repository,
so this script implements practical axis-angle checks in the same spirit:

1. representation validity: finite values and canonical axis-angle magnitude;
2. broad per-group joint-angle limits for body and hands;
3. temporal discontinuities using SO(3) geodesic jumps per joint.

Run examples:

    python "AMASS POSE Data Filtering/identifing_incorrect_frames.py"
    python "AMASS POSE Data Filtering/identifing_incorrect_frames.py" --input "C:/path/to/KIT"
    python "AMASS POSE Data Filtering/identifing_incorrect_frames.py" --input file_poses.npz
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_INPUT = Path(
    r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\KIT"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "incorrect_frames_axis_angle"

POSE_KEY_CANDIDATES = ("poses", "pose", "poses_aa", "axis_angle", "axis_angle_joints")
STANDARD_SMPLH_JOINTS = 52


@dataclass(frozen=True)
class FrameIssue:
    file_path: Path
    frame: int
    previous_frame: int | None
    joint: int | None
    joint_group: str
    check: str
    value_deg: float | None
    threshold_deg: float | None
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find suspicious frames in AMASS SMPL+H axis-angle .npz pose files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input .npz file or directory containing AMASS .npz files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_poses.npz",
        help="File pattern used when --input is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for detailed and summary CSV reports.",
    )
    parser.add_argument(
        "--pose-key",
        default=None,
        help="Pose array key inside each .npz. Defaults to auto-detecting 'poses'.",
    )
    parser.add_argument(
        "--canonical-limit-deg",
        type=float,
        default=180.0,
        help="Raw axis-angle norm above this value is flagged.",
    )
    parser.add_argument(
        "--body-limit-deg",
        type=float,
        default=140.0,
        help="Broad body joint rotation limit, excluding root.",
    )
    parser.add_argument(
        "--hand-limit-deg",
        type=float,
        default=180.0,
        help="Broad hand joint rotation limit.",
    )
    parser.add_argument(
        "--temporal-min-jump-deg",
        type=float,
        default=45.0,
        help="Minimum absolute per-frame geodesic jump needed before a temporal flag.",
    )
    parser.add_argument(
        "--temporal-mad-multiplier",
        type=float,
        default=8.0,
        help="Robust median/MAD multiplier for per-joint temporal jump thresholds.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for quick tests.",
    )
    return parser.parse_args()


def iter_npz_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(input_path.rglob(pattern))


def find_pose_key(data: np.lib.npyio.NpzFile, requested_key: str | None) -> str:
    if requested_key is not None:
        if requested_key not in data.files:
            raise KeyError(f"Requested pose key '{requested_key}' not found.")
        return requested_key

    for key in POSE_KEY_CANDIDATES:
        if key in data.files:
            return key

    raise KeyError(
        f"No axis-angle pose key found. Available keys: {', '.join(data.files)}"
    )


def as_axis_angle_frames(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 2:
        if poses.shape[1] % 3 != 0:
            raise ValueError(f"Pose width {poses.shape[1]} is not divisible by 3.")
        return poses.reshape(poses.shape[0], poses.shape[1] // 3, 3)

    if poses.ndim == 3 and poses.shape[-1] == 3:
        return poses

    raise ValueError(
        f"Expected poses shaped [frames, joints*3] or [frames, joints, 3], got {poses.shape}."
    )


def joint_group(joint: int, num_joints: int) -> str:
    if joint == 0:
        return "root"

    if num_joints >= STANDARD_SMPLH_JOINTS:
        # AMASS SMPL+H: 0 root, 1..21 body, 22..51 hands.
        return "body" if joint <= 21 else "hand"

    # Fallback for non-standard exports: keep the first ~40% as body.
    body_cutoff = max(1, int(round(num_joints * 0.4)))
    return "body" if joint <= body_cutoff else "hand"


def robust_temporal_thresholds(
    jumps_deg: np.ndarray,
    min_jump_deg: float,
    mad_multiplier: float,
) -> np.ndarray:
    median = np.median(jumps_deg, axis=0)
    mad = np.median(np.abs(jumps_deg - median), axis=0)
    robust_sigma = 1.4826 * mad
    adaptive = median + mad_multiplier * robust_sigma
    return np.maximum(adaptive, min_jump_deg)


def geodesic_jumps_deg(axis_angle: np.ndarray) -> np.ndarray:
    num_frames, num_joints, _ = axis_angle.shape
    if num_frames < 2:
        return np.empty((0, num_joints), dtype=np.float64)

    previous = R.from_rotvec(axis_angle[:-1].reshape(-1, 3))
    current = R.from_rotvec(axis_angle[1:].reshape(-1, 3))
    jumps = (previous.inv() * current).magnitude()
    return np.degrees(jumps.reshape(num_frames - 1, num_joints))


def add_scalar_frame_issues(
    issues: list[FrameIssue],
    file_path: Path,
    frame_mask: np.ndarray,
    check: str,
    message: str,
) -> None:
    for frame in np.flatnonzero(frame_mask):
        issues.append(
            FrameIssue(
                file_path=file_path,
                frame=int(frame),
                previous_frame=None,
                joint=None,
                joint_group="all",
                check=check,
                value_deg=None,
                threshold_deg=None,
                message=message,
            )
        )


def analyze_file(
    file_path: Path,
    pose_key: str | None,
    canonical_limit_deg: float,
    body_limit_deg: float,
    hand_limit_deg: float,
    temporal_min_jump_deg: float,
    temporal_mad_multiplier: float,
) -> tuple[list[FrameIssue], int, int]:
    issues: list[FrameIssue] = []

    with np.load(file_path, allow_pickle=True) as data:
        key = find_pose_key(data, pose_key)
        axis_angle = as_axis_angle_frames(data[key])

    num_frames, num_joints, _ = axis_angle.shape
    if num_joints not in (52, 53):
        issues.append(
            FrameIssue(
                file_path=file_path,
                frame=-1,
                previous_frame=None,
                joint=None,
                joint_group="all",
                check="joint_count_warning",
                value_deg=None,
                threshold_deg=None,
                message=f"Detected {num_joints} joints; expected AMASS SMPL+H 52 or user-described 53.",
            )
        )

    finite_joint = np.isfinite(axis_angle).all(axis=-1)
    add_scalar_frame_issues(
        issues,
        file_path,
        ~finite_joint.all(axis=1),
        "non_finite",
        "Frame contains NaN or infinite axis-angle values.",
    )

    raw_angles_deg = np.degrees(np.linalg.norm(axis_angle, axis=-1))
    raw_limit_mask = raw_angles_deg > canonical_limit_deg
    raw_limit_mask[:, 0] = False  # Root/global orientation can wrap during turns.
    for frame, joint in np.argwhere(raw_limit_mask):
        issues.append(
            FrameIssue(
                file_path=file_path,
                frame=int(frame),
                previous_frame=None,
                joint=int(joint),
                joint_group=joint_group(int(joint), num_joints),
                check="axis_angle_norm",
                value_deg=float(raw_angles_deg[frame, joint]),
                threshold_deg=canonical_limit_deg,
                message="Raw axis-angle vector norm exceeds the canonical rotation-vector limit.",
            )
        )

    group_limits = {"root": math.inf, "body": body_limit_deg, "hand": hand_limit_deg}
    for joint in range(num_joints):
        group = joint_group(joint, num_joints)
        limit = group_limits[group]
        if math.isinf(limit):
            continue
        mask = raw_angles_deg[:, joint] > limit
        for frame in np.flatnonzero(mask):
            issues.append(
                FrameIssue(
                    file_path=file_path,
                    frame=int(frame),
                    previous_frame=None,
                    joint=joint,
                    joint_group=group,
                    check=f"{group}_angle_limit",
                    value_deg=float(raw_angles_deg[frame, joint]),
                    threshold_deg=limit,
                    message=f"{group} joint axis-angle magnitude exceeds configured limit.",
                )
            )

    if finite_joint.all() and num_frames >= 2:
        jumps_deg = geodesic_jumps_deg(axis_angle)
        thresholds = robust_temporal_thresholds(
            jumps_deg,
            min_jump_deg=temporal_min_jump_deg,
            mad_multiplier=temporal_mad_multiplier,
        )
        temporal_mask = jumps_deg > thresholds
        for jump_idx, joint in np.argwhere(temporal_mask):
            issues.append(
                FrameIssue(
                    file_path=file_path,
                    frame=int(jump_idx + 1),
                    previous_frame=int(jump_idx),
                    joint=int(joint),
                    joint_group=joint_group(int(joint), num_joints),
                    check="temporal_geodesic_jump",
                    value_deg=float(jumps_deg[jump_idx, joint]),
                    threshold_deg=float(thresholds[joint]),
                    message="Current frame has a sudden SO(3) geodesic jump from the previous frame.",
                )
            )

    return issues, num_frames, num_joints


def write_detailed_report(path: Path, issues: Iterable[FrameIssue]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "frame",
                "previous_frame",
                "joint",
                "joint_group",
                "check",
                "value_deg",
                "threshold_deg",
                "message",
            ],
        )
        writer.writeheader()
        for issue in issues:
            writer.writerow(
                {
                    "file": str(issue.file_path),
                    "frame": issue.frame,
                    "previous_frame": "" if issue.previous_frame is None else issue.previous_frame,
                    "joint": "" if issue.joint is None else issue.joint,
                    "joint_group": issue.joint_group,
                    "check": issue.check,
                    "value_deg": "" if issue.value_deg is None else f"{issue.value_deg:.6f}",
                    "threshold_deg": ""
                    if issue.threshold_deg is None
                    else f"{issue.threshold_deg:.6f}",
                    "message": issue.message,
                }
            )


def write_summary_report(
    path: Path,
    per_file: list[tuple[Path, int, int, list[FrameIssue]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "frames",
                "joints",
                "incorrect_frame_count",
                "issue_count",
                "incorrect_frames",
            ],
        )
        writer.writeheader()
        for file_path, num_frames, num_joints, issues in per_file:
            frames = sorted({issue.frame for issue in issues if issue.frame >= 0})
            writer.writerow(
                {
                    "file": str(file_path),
                    "frames": num_frames,
                    "joints": num_joints,
                    "incorrect_frame_count": len(frames),
                    "issue_count": len(issues),
                    "incorrect_frames": ";".join(str(frame) for frame in frames),
                }
            )


def main() -> None:
    args = parse_args()
    files = iter_npz_files(args.input, args.pattern)
    if args.max_files is not None:
        files = files[: args.max_files]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output_dir / "incorrect_frames_detailed.csv"
    summary_path = args.output_dir / "incorrect_frames_summary.csv"

    all_issues: list[FrameIssue] = []
    per_file: list[tuple[Path, int, int, list[FrameIssue]]] = []

    print(f"Scanning {len(files)} .npz file(s) from: {args.input}")
    for index, file_path in enumerate(files, start=1):
        try:
            file_issues, num_frames, num_joints = analyze_file(
                file_path=file_path,
                pose_key=args.pose_key,
                canonical_limit_deg=args.canonical_limit_deg,
                body_limit_deg=args.body_limit_deg,
                hand_limit_deg=args.hand_limit_deg,
                temporal_min_jump_deg=args.temporal_min_jump_deg,
                temporal_mad_multiplier=args.temporal_mad_multiplier,
            )
        except Exception as exc:  # Keep long scans moving while recording failures.
            file_issues = [
                FrameIssue(
                    file_path=file_path,
                    frame=-1,
                    previous_frame=None,
                    joint=None,
                    joint_group="all",
                    check="file_error",
                    value_deg=None,
                    threshold_deg=None,
                    message=str(exc),
                )
            ]
            num_frames = 0
            num_joints = 0

        all_issues.extend(file_issues)
        per_file.append((file_path, num_frames, num_joints, file_issues))

        if index == 1 or index % 100 == 0 or index == len(files):
            flagged_files = sum(1 for _, _, _, issues in per_file if issues)
            print(f"  processed {index}/{len(files)} files | flagged files: {flagged_files}")

    write_detailed_report(detailed_path, all_issues)
    write_summary_report(summary_path, per_file)

    incorrect_files = sum(1 for _, _, _, issues in per_file if issues)
    incorrect_frames = sum(
        len({issue.frame for issue in issues if issue.frame >= 0})
        for _, _, _, issues in per_file
    )

    print()
    print(f"Done. Files with issues: {incorrect_files}/{len(files)}")
    print(f"Total incorrect/suspicious frames: {incorrect_frames}")
    print(f"Detailed report: {detailed_path.resolve()}")
    print(f"Summary report : {summary_path.resolve()}")


if __name__ == "__main__":
    main()
