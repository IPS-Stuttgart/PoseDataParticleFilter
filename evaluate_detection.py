"""
evaluate_detection.py

Scores a detector's output (e.g. your HHPF + SKEL "AND" rule) against the
synthetic ground truth produced by inject_synthetic_anomalies.py.

HOW TO WIRE IN YOUR REAL DETECTOR
----------------------------------
Run your HHPF + SKEL pipeline on `Subject_1_F_1_poses_corrupted.npz` exactly
as you would on real data, collect its flagged (frame, joint) pairs, and pass
them to `evaluate(...)` in ANY of these formats:

  1) boolean array, shape (n_frames, n_joints)   -- joint-level flags
  2) boolean array, shape (n_frames,)             -- frame-level only (any joint)
  3) iterable of (frame_idx, joint_idx) tuples    -- sparse joint-level flags
  4) iterable of frame_idx ints                   -- sparse frame-level flags

Then just replace `mock_detector(...)` below with your real output and run:

    python evaluate_detection.py

THREE GRANULARITIES REPORTED
-----------------------------
- frame-level   : did we flag AT LEAST ONE anomalous joint in a frame that
                   truly had one? (coarsest, easiest to satisfy)
- joint-level   : did we flag the EXACT (frame, joint) cell that was
                   corrupted? (strictest, penalizes flagging the wrong joint
                   in an otherwise-correct frame)
- event-level   : cluster consecutive flagged frames into "detected events"
                   and check overlap against the true event windows. This is
                   the number that matches how a human reviewer -- or your
                   paper -- would actually describe results ("we caught 14
                   of 17 injected anomalies"), and it's robust to a detector
                   that fires on every frame of a long event rather than
                   just one.

Also breaks recall down by corruption type (spike/sustained/ramp) and
severity tier (small/medium/large) -- this is the table you want for the
paper: it shows where the method's sensitivity boundary actually is,
rather than a single pooled number that hides it.
"""

import json
import numpy as np

GT_MASK_PATH = "/home/claude/work/ground_truth_masks.npz"
GT_EVENTS_PATH = "/home/claude/work/ground_truth_events.json"


# ----------------------------------------------------------------------
# Normalize whatever format the detector's output comes in
# ----------------------------------------------------------------------
def _to_joint_bool_array(detected, n_frames, n_joints):
    """Returns (joint_level_array_or_None, frame_level_array)."""
    if isinstance(detected, np.ndarray) and detected.dtype == bool:
        if detected.ndim == 2:
            assert detected.shape == (n_frames, n_joints), \
                f"expected shape ({n_frames},{n_joints}), got {detected.shape}"
            return detected, detected.any(axis=1)
        elif detected.ndim == 1:
            assert detected.shape[0] == n_frames
            return None, detected
        else:
            raise ValueError("unsupported array ndim")

    # iterable of tuples or ints
    detected = list(detected)
    if len(detected) == 0:
        return np.zeros((n_frames, n_joints), dtype=bool), np.zeros(n_frames, dtype=bool)
    if isinstance(detected[0], (tuple, list)) and len(detected[0]) == 2:
        arr = np.zeros((n_frames, n_joints), dtype=bool)
        for f, j in detected:
            arr[f, j] = True
        return arr, arr.any(axis=1)
    else:
        arr = np.zeros(n_frames, dtype=bool)
        for f in detected:
            arr[f] = True
        return None, arr


# ----------------------------------------------------------------------
# Basic P/R/F1 from boolean arrays
# ----------------------------------------------------------------------
def _prf1(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
          else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


# ----------------------------------------------------------------------
# Cluster flagged frames into contiguous "detected events"
# ----------------------------------------------------------------------
def cluster_into_events(flagged_frame_bool, gap_tolerance=1):
    """gap_tolerance: merge flagged frames separated by <= this many clean frames."""
    idxs = np.where(flagged_frame_bool)[0]
    if len(idxs) == 0:
        return []
    events = []
    start = prev = idxs[0]
    for i in idxs[1:]:
        if i - prev <= gap_tolerance + 1:
            prev = i
        else:
            events.append((start, prev))
            start = prev = i
    events.append((start, prev))
    return events  # list of (start_frame, end_frame) inclusive


def _windows_overlap(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def event_level_score(true_events, detected_events):
    """
    Event counted as TP (recall) if it overlaps >=1 detected event.
    Detected event counted as TP (precision) if it overlaps >=1 true event.
    (Standard range-overlap matching used in time-series anomaly benchmarks.)
    """
    true_hit = [any(_windows_overlap(ts, te, ds, de) for ds, de in detected_events)
                for ts, te in true_events]
    det_hit = [any(_windows_overlap(ds, de, ts, te) for ts, te in true_events)
               for ds, de in detected_events]

    tp_recall = sum(true_hit)
    fn = len(true_events) - tp_recall
    tp_precision = sum(det_hit)
    fp = len(detected_events) - tp_precision

    recall = tp_recall / len(true_events) if true_events else float("nan")
    precision = tp_precision / len(detected_events) if detected_events else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
          else float("nan"))
    return {
        "n_true_events": len(true_events), "n_detected_events": len(detected_events),
        "true_events_recalled": tp_recall, "true_events_missed": fn,
        "detected_events_matched": tp_precision, "detected_events_spurious": fp,
        "precision": precision, "recall": recall, "f1": f1,
        "true_event_hit_flags": true_hit,
    }


# ----------------------------------------------------------------------
# Main evaluate() entry point
# ----------------------------------------------------------------------
def evaluate(detected, gap_tolerance=1, verbose=True):
    gt = np.load(GT_MASK_PATH, allow_pickle=True)
    bool_mask = gt["bool_mask"]              # (n_frames, n_joints)
    frame_level_true = gt["frame_level_bool"]  # (n_frames,)
    n_frames, n_joints = bool_mask.shape

    with open(GT_EVENTS_PATH) as f:
        gt_meta = json.load(f)
    events_meta = gt_meta["events"]
    true_windows = [(e["start_frame"], e["end_frame"]) for e in events_meta]

    joint_pred, frame_pred = _to_joint_bool_array(detected, n_frames, n_joints)

    results = {}
    results["frame_level"] = _prf1(frame_level_true, frame_pred)
    if joint_pred is not None:
        results["joint_level"] = _prf1(bool_mask, joint_pred)
    else:
        results["joint_level"] = None

    detected_windows = cluster_into_events(frame_pred, gap_tolerance=gap_tolerance)
    ev_score = event_level_score(true_windows, detected_windows)
    results["event_level"] = ev_score

    # breakdown by type / severity using which true events were recalled
    breakdown = {"by_type": {}, "by_severity": {}}
    for e, hit in zip(events_meta, ev_score["true_event_hit_flags"]):
        breakdown["by_type"].setdefault(e["type"], []).append(hit)
        breakdown["by_severity"].setdefault(e["severity_tier"], []).append(hit)
    for key in breakdown:
        breakdown[key] = {
            k: {"recalled": int(sum(v)), "total": len(v), "recall": sum(v) / len(v)}
            for k, v in breakdown[key].items()
        }
    results["breakdown"] = breakdown

    if verbose:
        _print_report(results)
    return results


def _print_report(results):
    print("=" * 70)
    print("FRAME-LEVEL  (any joint anomalous in frame)")
    fl = results["frame_level"]
    print(f"  TP={fl['tp']:4d}  FP={fl['fp']:4d}  FN={fl['fn']:4d}  TN={fl['tn']:4d}")
    print(f"  precision={fl['precision']:.3f}  recall={fl['recall']:.3f}  f1={fl['f1']:.3f}")

    if results["joint_level"] is not None:
        print("\nJOINT-LEVEL  (exact frame,joint cell match)")
        jl = results["joint_level"]
        print(f"  TP={jl['tp']:4d}  FP={jl['fp']:4d}  FN={jl['fn']:4d}  TN={jl['tn']:4d}")
        print(f"  precision={jl['precision']:.3f}  recall={jl['recall']:.3f}  f1={jl['f1']:.3f}")
    else:
        print("\nJOINT-LEVEL  -- skipped (no per-joint detections supplied)")

    print("\nEVENT-LEVEL  (clustered flagged frames vs true event windows)  <-- headline metric")
    el = results["event_level"]
    print(f"  true events            : {el['n_true_events']}")
    print(f"  detected events        : {el['n_detected_events']}")
    print(f"  true events recalled   : {el['true_events_recalled']}  (missed: {el['true_events_missed']})")
    print(f"  detected events correct: {el['detected_events_matched']}  (spurious: {el['detected_events_spurious']})")
    print(f"  precision={el['precision']:.3f}  recall={el['recall']:.3f}  f1={el['f1']:.3f}")

    print("\nRECALL BY CORRUPTION TYPE")
    for k, v in results["breakdown"]["by_type"].items():
        print(f"  {k:10s}: {v['recalled']}/{v['total']}  (recall={v['recall']:.2f})")
    print("\nRECALL BY SEVERITY TIER")
    for k, v in results["breakdown"]["by_severity"].items():
        print(f"  {k:10s}: {v['recalled']}/{v['total']}  (recall={v['recall']:.2f})")
    print("=" * 70)


# ----------------------------------------------------------------------
# Demo with a MOCK detector so the script is runnable end-to-end out of
# the box. REPLACE mock_detector's output with your real HHPF+SKEL AND-rule
# flags before reporting numbers.
# ----------------------------------------------------------------------
def mock_detector(seed=7, hit_rate=0.7, false_positive_rate=0.01):
    """
    Toy stand-in: flags a random subset of true corrupted (frame,joint) cells
    (simulating imperfect recall) plus a sprinkling of random false positives.
    ONLY for demonstrating evaluate()'s I/O -- not a real detector.
    """
    gt = np.load(GT_MASK_PATH, allow_pickle=True)
    bool_mask = gt["bool_mask"]
    rng = np.random.default_rng(seed)
    pred = np.zeros_like(bool_mask)
    true_idx = np.argwhere(bool_mask)
    keep = rng.random(len(true_idx)) < hit_rate
    for (f, j) in true_idx[keep]:
        pred[f, j] = True
    n_frames, n_joints = bool_mask.shape
    n_fp = int(false_positive_rate * n_frames * n_joints)
    for _ in range(n_fp):
        f = rng.integers(0, n_frames)
        j = rng.integers(0, n_joints)
        pred[f, j] = True
    return pred


if __name__ == "__main__":
    print("Demo run using a MOCK detector (NOT your real pipeline) to show "
          "the report format:\n")
    detected = mock_detector()
    evaluate(detected)
