from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "widerface"
IMG_ROOT = DATA_ROOT / "WIDER_val" / "images"
ANN_PATH = DATA_ROOT / "wider_face_split" / "wider_face_val_bbx_gt.txt"
OUT_DIR = ROOT / "figures" / "ch1_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_IMAGES = int(os.getenv("SEC116_MAX_IMAGES", "500"))


def load_annotations(path: Path) -> List[Tuple[str, List[Tuple[int, int, int, int]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    records: List[Tuple[str, List[Tuple[int, int, int, int]]]] = []
    while idx < len(lines):
        rel_path = lines[idx].strip()
        idx += 1
        n = int(lines[idx].strip())
        idx += 1
        boxes: List[Tuple[int, int, int, int]] = []
        for _ in range(n):
            parts = lines[idx].split()
            idx += 1
            if len(parts) < 10:
                continue
            x, y, w, h = map(int, parts[:4])
            invalid = int(parts[7])
            if w > 0 and h > 0 and invalid == 0:
                boxes.append((x, y, w, h))
        records.append((rel_path, boxes))
    return records


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    gts: List[Tuple[int, int, int, int]],
    dets: List[Tuple[int, int, int, int]],
    thr: float = 0.3,
) -> Tuple[int, int, int]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gts):
        for di, d in enumerate(dets):
            ov = iou(g, d)
            if ov >= thr:
                pairs.append((ov, gi, di))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g, used_d = set(), set()
    for _, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
    tp = len(used_g)
    fn = len(gts) - tp
    fp = len(dets) - len(used_d)
    return tp, fp, fn


def evaluate_config(
    records: List[Tuple[str, List[Tuple[int, int, int, int]]]],
    detector: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    iou_thr: float = 0.3,
) -> Dict[str, float]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_det = 0
    n_images = 0

    for rel_path, gts in records:
        if not gts:
            continue
        if n_images >= MAX_IMAGES:
            break
        img = cv2.imread(str(IMG_ROOT / rel_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw = detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        dets = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in raw]
        tp, fp, fn = greedy_match(gts, dets, thr=iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gts)
        total_det += len(dets)
        n_images += 1

    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    recall = (total_tp / total_gt) if total_gt else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "images": n_images,
        "gt": total_gt,
        "det": total_det,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    records = load_annotations(ANN_PATH)
    detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )

    # 11.6.1 style: vary detector operating point and report (recall, false detections).
    roc_rows: List[Dict[str, float]] = []
    for min_neighbors in [0, 1, 2, 3, 4, 5, 6]:
        stats = evaluate_config(
            records,
            detector,
            scale_factor=1.1,
            min_neighbors=min_neighbors,
            min_size=20,
            iou_thr=0.3,
        )
        roc_rows.append(
            {
                "scaleFactor": 1.1,
                "minNeighbors": min_neighbors,
                "minSize": 20,
                **stats,
            }
        )

    # 11.6.2 style: compare subwindow size.
    subwindow_rows: List[Dict[str, float]] = []
    for min_size in [16, 20, 24, 28, 32]:
        stats = evaluate_config(
            records,
            detector,
            scale_factor=1.1,
            min_neighbors=3,
            min_size=min_size,
            iou_thr=0.3,
        )
        subwindow_rows.append(
            {
                "scaleFactor": 1.1,
                "minNeighbors": 3,
                "minSize": min_size,
                **stats,
            }
        )

    roc_csv = OUT_DIR / "sec116_roc_like_points.csv"
    sub_csv = OUT_DIR / "sec116_subwindow_size_compare.csv"
    save_csv(
        roc_csv,
        roc_rows,
        [
            "scaleFactor",
            "minNeighbors",
            "minSize",
            "images",
            "gt",
            "det",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )
    save_csv(
        sub_csv,
        subwindow_rows,
        [
            "scaleFactor",
            "minNeighbors",
            "minSize",
            "images",
            "gt",
            "det",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )

    summary = {
        "dataset": "WIDER FACE validation",
        "evaluation_images_with_gt": MAX_IMAGES,
        "opencv_version": cv2.__version__,
        "model": "haarcascade_frontalface_default.xml",
        "iou_match_threshold": 0.3,
        "roc_like_points_csv": str(roc_csv.relative_to(ROOT)).replace("\\", "/"),
        "subwindow_size_csv": str(sub_csv.relative_to(ROOT)).replace("\\", "/"),
        "best_f1_point": max(roc_rows, key=lambda r: r["f1"]),
        "best_recall_point": max(roc_rows, key=lambda r: r["recall"]),
        "subwindow_best_f1": max(subwindow_rows, key=lambda r: r["f1"]),
    }
    summary_path = OUT_DIR / "sec116_reproduce_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", roc_csv)
    print("Saved:", sub_csv)
    print("Saved:", summary_path)
    print("Best F1 point:", summary["best_f1_point"])
    print("Best recall point:", summary["best_recall_point"])
    print("Best subwindow by F1:", summary["subwindow_best_f1"])


if __name__ == "__main__":
    main()
