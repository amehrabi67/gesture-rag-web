from __future__ import annotations

import csv
import json
import math
import shutil
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABEL_CSV = ROOT / "assets" / "models" / "gesture_labels.csv"
BASE_POINT_HISTORY_CSV = ROOT / "assets" / "source_model" / "base_point_history.csv"
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"

HISTORY_LENGTH = 16
PRED_EVERY_N_FRAMES = 3
SMOOTH_WIN = 9
MIN_SEG_SEC = 0.35
MERGE_GAP_SEC = 0.20
MAX_FRAME_WIDTH = 640
UNKNOWN_LABEL_ID = -1
UNKNOWN_LABEL = "Unknown"
MIN_DISTANCE_THRESHOLD = 0.25
THRESHOLD_MULTIPLIER = 2.5


def load_labels() -> list[str]:
    return [line.strip() for line in LABEL_CSV.read_text(encoding="utf-8").splitlines() if line.strip()]


def project_dir(project_id: str) -> Path:
    return UPLOADS_DIR / "projects" / project_id


def _clean_labels(labels: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for label in labels:
        name = " ".join(str(label).strip().split())
        key = name.lower()
        if name and key not in seen:
            cleaned.append(name)
            seen.add(key)
    return cleaned


def _label_id_map(selected_gestures: list[str]) -> dict[str, int]:
    base_labels = load_labels()
    used_ids: set[int] = set()
    mapping: dict[str, int] = {}
    next_custom_id = 1000
    for label in selected_gestures:
        if label in base_labels:
            gid = base_labels.index(label)
        else:
            while next_custom_id in used_ids:
                next_custom_id += 1
            gid = next_custom_id
            next_custom_id += 1
        mapping[label] = gid
        used_ids.add(gid)
    return mapping


def _project_label_lookup(project: dict[str, Any]) -> dict[int, str]:
    if "id_to_label" in project:
        return {int(k): str(v) for k, v in project["id_to_label"].items()}
    return {int(gid): label for gid, label in zip(project["selected_ids"], project["selected_gestures"])}


def _project_gesture_id(project: dict[str, Any], label: str) -> int:
    if "label_to_id" in project and label in project["label_to_id"]:
        return int(project["label_to_id"][label])
    if label in project.get("selected_gestures", []):
        return int(project["selected_ids"][project["selected_gestures"].index(label)])
    raise ValueError(f"{label} is not selected for this project.")


def _ensure_project_gesture(project: dict[str, Any], label: str) -> int:
    label = _clean_labels([label])[0] if _clean_labels([label]) else ""
    if not label:
        raise ValueError("Gesture name is required.")
    if label in project.get("selected_gestures", []):
        return _project_gesture_id(project, label)

    used_ids = {int(gid) for gid in project.get("selected_ids", [])}
    base_labels = load_labels()
    if label in base_labels and base_labels.index(label) not in used_ids:
        gid = base_labels.index(label)
    else:
        gid = max([999, *used_ids]) + 1

    project.setdefault("selected_gestures", []).append(label)
    project.setdefault("selected_ids", []).append(gid)
    project.setdefault("label_to_id", {})[label] = gid
    project.setdefault("id_to_label", {})[str(gid)] = label
    project.setdefault("training_counts", {})[label] = 0
    return gid


def create_project(selected_gestures: list[str]) -> dict[str, Any]:
    selected_gestures = _clean_labels(selected_gestures)
    if not selected_gestures:
        raise ValueError("Add at least one gesture name.")
    label_to_id = _label_id_map(selected_gestures)

    pid = uuid.uuid4().hex[:12]
    pdir = project_dir(pid)
    (pdir / "training_videos").mkdir(parents=True, exist_ok=True)
    (pdir / "test_videos").mkdir(parents=True, exist_ok=True)
    (pdir / "outputs").mkdir(parents=True, exist_ok=True)
    metadata = {
        "project_id": pid,
        "selected_gestures": selected_gestures,
        "selected_ids": [label_to_id[g] for g in selected_gestures],
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
        "training_counts": {g: 0 for g in selected_gestures},
    }
    (pdir / "project.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def read_project(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id) / "project.json"
    if not path.exists():
        raise FileNotFoundError(f"Project not found: {project_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_project(project: dict[str, Any]) -> None:
    (project_dir(project["project_id"]) / "project.json").write_text(
        json.dumps(project, indent=2), encoding="utf-8"
    )


def normalize_seq(seq_xy: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(seq_xy, np.float32)
    center = pts.mean(axis=0, keepdims=True)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
        scale = 1.0
    return (pts / scale).reshape(-1).astype(np.float32)


def _lazy_cv_imports():
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:  # pragma: no cover - depends on runtime packages
        raise RuntimeError(
            "OpenCV and MediaPipe are required for video processing. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return cv2, mp


def extract_samples_from_video(video_path: Path) -> list[np.ndarray]:
    cv2, mp = _lazy_cv_imports()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    mp_hands = mp.solutions.hands
    samples: list[np.ndarray] = []
    history: deque[tuple[float, float]] = deque(maxlen=HISTORY_LENGTH)
    frame_i = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / w
                frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)))
                h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark[8]
                history.append((lm.x * w, lm.y * h))
                if len(history) == HISTORY_LENGTH and frame_i % 2 == 0:
                    samples.append(normalize_seq(history))
            frame_i += 1

    cap.release()
    return samples


def add_training_video(project_id: str, gesture_label: str, video_path: Path) -> dict[str, Any]:
    project = read_project(project_id)
    gid = _ensure_project_gesture(project, gesture_label)
    gesture_label = _project_label_lookup(project)[gid]

    pdir = project_dir(project_id)
    safe_name = f"{uuid.uuid4().hex}_{video_path.name}"
    stored_video = pdir / "training_videos" / safe_name
    shutil.copy2(video_path, stored_video)

    samples = extract_samples_from_video(stored_video)
    if not samples:
        raise RuntimeError("No hand trajectory samples were extracted. Try brighter video and keep the hand visible.")

    train_csv = pdir / "training_point_history.csv"
    with train_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sample in samples:
            writer.writerow([gid, *sample.tolist()])

    counts = project.setdefault("training_counts", {g: 0 for g in project["selected_gestures"]})
    counts[gesture_label] = counts.get(gesture_label, 0) + len(samples)
    write_project(project)
    return {
        "saved_video": stored_video.name,
        "samples_added": len(samples),
        "gesture_label": gesture_label,
        "selected_gestures": project["selected_gestures"],
        "training_counts": counts,
    }


def _load_point_history_rows(path: Path, allowed_ids: set[int]) -> tuple[list[list[float]], list[int]]:
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    if not path.exists():
        return x_rows, y_rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            label = int(row[0])
            if label in allowed_ids:
                y_rows.append(label)
                x_rows.append([float(v) for v in row[1:]])
    return x_rows, y_rows


def train_project_model(project_id: str, include_base: bool = True) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.neighbors import KNeighborsClassifier
    except Exception as exc:  # pragma: no cover - depends on runtime packages
        raise RuntimeError(
            "scikit-learn and joblib are required for KNN training. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    project = read_project(project_id)
    allowed_ids = set(project["selected_ids"])
    label_lookup = _project_label_lookup(project)
    pdir = project_dir(project_id)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []

    if include_base:
        x_base, y_base = _load_point_history_rows(BASE_POINT_HISTORY_CSV, allowed_ids)
        x_rows.extend(x_base)
        y_rows.extend(y_base)

    x_project, y_project = _load_point_history_rows(pdir / "training_point_history.csv", allowed_ids)
    x_rows.extend(x_project)
    y_rows.extend(y_project)

    label_counts = Counter(y_rows)
    missing = [label_lookup.get(gid, str(gid)) for gid in allowed_ids if label_counts.get(gid, 0) == 0]
    if missing:
        raise RuntimeError("Need at least one extracted sample for: " + ", ".join(missing))
    if len(label_counts) < 2:
        raise RuntimeError("Train at least two different gestures before running the model.")
    k = min(7, max(1, int(math.sqrt(len(y_rows)))))
    x_arr = np.asarray(x_rows, np.float32)
    y_arr = np.asarray(y_rows, np.int32)
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(x_arr, y_arr)
    distance_threshold = _estimate_distance_threshold(x_arr)
    model_path = pdir / "point_history_knn.pkl"
    joblib.dump(clf, model_path)

    project["model_path"] = str(model_path)
    project["model_summary"] = {
        "n_samples": len(y_rows),
        "n_neighbors": k,
        "distance_threshold": distance_threshold,
        "label_counts": {label_lookup.get(gid, str(gid)): int(label_counts[gid]) for gid in sorted(label_counts)},
        "included_base_csv": include_base,
    }
    write_project(project)
    return project["model_summary"]


@dataclass
class FramePrediction:
    frame: int
    time_sec: float
    label_id: int
    label: str
    confidence: float | None
    distance: float | None


def _estimate_distance_threshold(x_arr: np.ndarray) -> float:
    if len(x_arr) < 2:
        return MIN_DISTANCE_THRESHOLD
    diffs = x_arr[:, None, :] - x_arr[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if not len(finite):
        return MIN_DISTANCE_THRESHOLD
    threshold = float(np.percentile(finite, 95) * THRESHOLD_MULTIPLIER)
    return round(max(MIN_DISTANCE_THRESHOLD, threshold), 4)


def _predict_confidence(clf: Any, vec: np.ndarray, distance_threshold: float | None) -> tuple[int, float | None, float | None]:
    distance: float | None = None
    if distance_threshold is not None and hasattr(clf, "kneighbors"):
        distances, _ = clf.kneighbors([vec], n_neighbors=1)
        distance = float(distances[0][0])
        if distance > distance_threshold:
            confidence = round(1.0 / (1.0 + distance), 4)
            return UNKNOWN_LABEL_ID, confidence, round(distance, 4)

    pred = int(clf.predict([vec])[0])
    conf: float | None = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([vec])[0]
        conf = float(np.max(probs))
    return pred, conf, round(distance, 4) if distance is not None else None


def infer_video(project_id: str, video_path: Path) -> dict[str, Any]:
    try:
        import joblib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("joblib is required for inference. Install requirements.txt") from exc

    cv2, mp = _lazy_cv_imports()
    project = read_project(project_id)
    model_path = Path(project.get("model_path", ""))
    if not model_path.exists():
        raise RuntimeError("No trained KNN model found. Add training clips and train first.")

    label_lookup = _project_label_lookup(project)
    model_summary = project.get("model_summary") or {}
    distance_threshold = model_summary.get("distance_threshold")
    clf = joblib.load(model_path)

    pdir = project_dir(project_id)
    safe_name = f"{uuid.uuid4().hex}_{video_path.name}"
    stored_video = pdir / "test_videos" / safe_name
    shutil.copy2(video_path, stored_video)

    cap = cv2.VideoCapture(str(stored_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_predictions: list[FramePrediction] = []
    history: deque[tuple[float, float]] = deque(maxlen=HISTORY_LENGTH)
    smooth_q: deque[int] = deque(maxlen=SMOOTH_WIN)
    frame_i = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / w
                frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)))
                h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark[8]
                history.append((lm.x * w, lm.y * h))
                if len(history) == HISTORY_LENGTH and frame_i % PRED_EVERY_N_FRAMES == 0:
                    pred, conf, distance = _predict_confidence(clf, normalize_seq(history), distance_threshold)
                    smooth_q.append(pred)
                    voted = Counter(smooth_q).most_common(1)[0][0]
                    label = UNKNOWN_LABEL if voted == UNKNOWN_LABEL_ID else label_lookup.get(voted, str(voted))
                    frame_predictions.append(
                        FramePrediction(frame_i, frame_i / fps, voted, label, conf, distance)
                    )
            frame_i += 1
    cap.release()

    segments = _predictions_to_segments(frame_predictions, fps)
    result = {
        "project_id": project_id,
        "video_file": str(stored_video),
        "fps": fps,
        "width": width,
        "height": height,
        "frames_seen": frame_i,
        "segments": segments,
        "unknown_rejection": {
            "enabled": distance_threshold is not None,
            "distance_threshold": distance_threshold,
        },
    }
    out_path = pdir / "outputs" / f"{Path(safe_name).stem}_gesture_segments.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["segments_path"] = str(out_path)
    return result


def _predictions_to_segments(preds: list[FramePrediction], fps: float) -> list[dict[str, Any]]:
    if not preds:
        return []

    raw: list[dict[str, Any]] = []
    current = {
        "label_id": preds[0].label_id,
        "label": preds[0].label,
        "start_frame": preds[0].frame,
        "end_frame": preds[0].frame,
        "confidences": [],
    }
    for pred in preds:
        if pred.label_id != current["label_id"]:
            raw.append(current)
            current = {
                "label_id": pred.label_id,
                "label": pred.label,
                "start_frame": pred.frame,
                "end_frame": pred.frame,
                "confidences": [],
            }
        current["end_frame"] = pred.frame
        if pred.confidence is not None:
            current["confidences"].append(pred.confidence)
    raw.append(current)

    kept = []
    for seg in raw:
        if seg["label_id"] == UNKNOWN_LABEL_ID:
            continue
        start = seg["start_frame"] / fps
        end = seg["end_frame"] / fps
        dur = max(0.0, end - start)
        if dur >= MIN_SEG_SEC:
            kept.append(
                {
                    "label_id": seg["label_id"],
                    "label": seg["label"],
                    "start_time_sec": round(start, 3),
                    "end_time_sec": round(end, 3),
                    "duration_sec": round(dur, 3),
                    "avg_confidence": (
                        round(float(np.mean(seg["confidences"])), 4) if seg["confidences"] else None
                    ),
                }
            )

    merged: list[dict[str, Any]] = []
    for seg in kept:
        if (
            merged
            and merged[-1]["label"] == seg["label"]
            and seg["start_time_sec"] - merged[-1]["end_time_sec"] <= MERGE_GAP_SEC
        ):
            merged[-1]["end_time_sec"] = seg["end_time_sec"]
            merged[-1]["duration_sec"] = round(merged[-1]["end_time_sec"] - merged[-1]["start_time_sec"], 3)
            if seg["avg_confidence"] is not None and merged[-1]["avg_confidence"] is not None:
                merged[-1]["avg_confidence"] = round(
                    (merged[-1]["avg_confidence"] + seg["avg_confidence"]) / 2, 4
                )
        else:
            merged.append(seg)
    return merged
