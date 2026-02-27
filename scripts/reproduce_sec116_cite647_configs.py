from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "widerface"
IMG_ROOT = DATA_ROOT / "WIDER_val" / "images"
ANN_PATH = DATA_ROOT / "wider_face_split" / "wider_face_val_bbx_gt.txt"
OUT_DIR = ROOT / "figures" / "ch1_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_IMAGES = int(os.getenv("SEC116_MAX_IMAGES", "200"))


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
    if inter <= 0:
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


def nms(boxes: List[Tuple[int, int, int, int]], thr: float = 0.35) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idx = area.argsort()
    keep = []
    while len(idx) > 0:
        i = idx[-1]
        keep.append(i)
        idx = idx[:-1]
        if len(idx) == 0:
            break
        xx1 = x1[idx].clip(min=x1[i])
        yy1 = y1[idx].clip(min=y1[i])
        xx2 = x2[idx].clip(max=x2[i])
        yy2 = y2[idx].clip(max=y2[i])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        union = area[i] + area[idx] - inter
        iou_vals = inter / union
        idx = idx[iou_vals <= thr]
    return [tuple(map(int, boxes[i])) for i in keep]


def detect_with_config(gray, cfg: Dict[str, object]) -> List[Tuple[int, int, int, int]]:
    scale_factor = float(cfg["scaleFactor"])
    min_neighbors = int(cfg["minNeighbors"])
    min_size = int(cfg["minSize"])
    mirror = bool(cfg.get("mirror", False))
    nms_thr = float(cfg.get("nms", 0.35))

    boxes: List[Tuple[int, int, int, int]] = []
    width = gray.shape[1]
    for cascade_name in cfg["cascades"]:
        det = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / cascade_name))
        raw = det.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        boxes.extend((int(x), int(y), int(w), int(h)) for (x, y, w, h) in raw)
        if mirror:
            raw_flip = det.detectMultiScale(
                cv2.flip(gray, 1),
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_size, min_size),
            )
            for x, y, w, h in raw_flip:
                boxes.append((width - int(x) - int(w), int(y), int(w), int(h)))
    return nms(boxes, thr=nms_thr)


def evaluate_config(
    records: List[Tuple[str, List[Tuple[int, int, int, int]]]],
    cfg: Dict[str, object],
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
        dets = detect_with_config(gray, cfg)
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


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    records = load_annotations(ANN_PATH)

    # These are practical OpenCV mappings from the 3 cited detector families in Fig. 11.16.
    # They are not the original training configs of [6], [17], [47].
    cfgs = [
        {
            "id": "cite6_twopoint_like",
            "citation": "[6] Brubaker et al. (IJCV 2008)",
            "cascades": ["haarcascade_frontalface_default.xml"],
            "scaleFactor": 1.05,  # smaller scanning rescaling factor
            "minNeighbors": 2,
            "minSize": 20,
            "mirror": False,
            "nms": 0.35,
        },
        {
            "id": "cite17_floatboost_multiview_like",
            "citation": "[17] Li and Zhang (TPAMI 2004)",
            "cascades": [
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_profileface.xml",
            ],
            "scaleFactor": 1.08,
            "minNeighbors": 3,
            "minSize": 20,
            "mirror": True,
            "nms": 0.35,
        },
        {
            "id": "cite47_asymboost_fda_like",
            "citation": "[47] Wu et al. (TPAMI 2008)",
            "cascades": ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"],
            "scaleFactor": 1.05,
            "minNeighbors": 1,  # looser operating point for higher recall
            "minSize": 20,
            "mirror": False,
            "nms": 0.35,
        },
    ]

    rows: List[Dict[str, object]] = []
    for cfg in cfgs:
        stats = evaluate_config(records, cfg, iou_thr=0.3)
        row = {**cfg, **stats}
        row["cascades"] = ";".join(cfg["cascades"])
        rows.append(row)
        print(cfg["id"], "->", f"det={stats['det']} tp={stats['tp']} fp={stats['fp']} f1={stats['f1']:.4f}")

    out_csv = OUT_DIR / "sec116_cite647_configs.csv"
    save_csv(
        out_csv,
        rows,
        [
            "id",
            "citation",
            "cascades",
            "scaleFactor",
            "minNeighbors",
            "minSize",
            "mirror",
            "nms",
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
        "dataset": "WIDER FACE validation subset",
        "images_with_gt": MAX_IMAGES,
        "opencv_version": cv2.__version__,
        "iou_match_threshold": 0.3,
        "csv": str(out_csv.relative_to(ROOT)).replace("\\", "/"),
        "best_f1": max(rows, key=lambda r: float(r["f1"]))["id"],
        "note": (
            "Hyperparameters are citation-guided approximations for OpenCV inference. "
            "Exact ROC in Fig. 11.16 cannot be matched without original MIT-CMU protocol "
            "and original trained cascades from each method."
        ),
    }
    out_json = OUT_DIR / "sec116_cite647_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
