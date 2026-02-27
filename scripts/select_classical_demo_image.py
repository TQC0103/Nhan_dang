from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "widerface"
IMG_ROOT = DATA_ROOT / "WIDER_val" / "images"
ANN_PATH = DATA_ROOT / "wider_face_split" / "wider_face_val_bbx_gt.txt"
OUT_DIR = ROOT / "figures" / "ch1_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
            if w <= 0 or h <= 0 or invalid == 1:
                continue
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
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gts):
        for di, d in enumerate(dets):
            ov = iou(g, d)
            if ov >= thr:
                pairs.append((ov, gi, di))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g, used_d = set(), set()
    matches: List[Tuple[int, int]] = []
    for _, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        matches.append((gi, di))
    fn = [gi for gi in range(len(gts)) if gi not in used_g]
    fp = [di for di in range(len(dets)) if di not in used_d]
    return matches, fn, fp


def pick_best_case(records: List[Tuple[str, List[Tuple[int, int, int, int]]]]) -> dict:
    detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    candidates = []
    report_rows = []
    for rel_path, gts in records:
        if len(gts) < 3:
            continue
        img_path = IMG_ROOT / rel_path
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets_raw = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
        )
        dets = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in dets_raw]
        matches, fn_idx, fp_idx = greedy_match(gts, dets, thr=0.3)
        tp = len(matches)
        fn = len(fn_idx)
        fp = len(fp_idx)
        report_rows.append((rel_path, len(gts), len(dets), tp, fp, fn))
        if tp >= 1 and fp >= 1 and fn >= 1:
            # Prefer images that are readable in report: moderate crowd, mixed outcomes.
            score = (
                min(tp, 4) * 2.0
                + min(fp, 4) * 1.2
                + min(fn, 4) * 1.2
                - abs(len(gts) - 8) * 0.25
            )
            candidates.append(
                {
                    "score": score,
                    "rel_path": rel_path,
                    "img_path": str(img_path),
                    "gt_count": len(gts),
                    "det_count": len(dets),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "gts": gts,
                    "dets": dets,
                    "matches": matches,
                    "fn_idx": fn_idx,
                    "fp_idx": fp_idx,
                }
            )

    csv_path = DATA_ROOT / "viola_widerface_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "gt_count", "det_count", "tp", "fp", "fn"])
        writer.writerows(report_rows)

    if not candidates:
        raise RuntimeError("Khong tim thay anh nao co du TP + FP + FN.")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


def draw_case(case: dict) -> None:
    img = cv2.imread(case["img_path"])
    assert img is not None
    vis = img.copy()

    matched_det = {di for _, di in case["matches"]}
    matched_gt = {gi for gi, _ in case["matches"]}

    # TP detections: green
    for di in matched_det:
        x, y, w, h = case["dets"][di]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (40, 220, 40), 2)

    # FP detections: red
    for di in case["fp_idx"]:
        x, y, w, h = case["dets"][di]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (30, 30, 230), 2)

    # FN ground truths: yellow
    for gi in case["fn_idx"]:
        x, y, w, h = case["gts"][gi]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 255), 2)

    text = (
        f"WIDER val | TP={case['tp']} FP={case['fp']} FN={case['fn']} "
        f"| GT={case['gt_count']} DET={case['det_count']}"
    )
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 700), 42), (0, 0, 0), -1)
    cv2.putText(vis, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    out_main = OUT_DIR / "wider_classical_mixed_case.jpg"
    out_raw = OUT_DIR / "wider_classical_mixed_case_raw.jpg"
    cv2.imwrite(str(out_main), vis)
    cv2.imwrite(str(out_raw), img)

    meta = {
        "dataset": "WIDER FACE validation",
        "selected_image": case["rel_path"],
        "tp": case["tp"],
        "fp": case["fp"],
        "fn": case["fn"],
        "gt_count": case["gt_count"],
        "det_count": case["det_count"],
        "legend": {
            "green_box": "true positive detection",
            "red_box": "false positive detection",
            "yellow_box": "false negative ground-truth face",
        },
        "params": {
            "detector": "OpenCV haarcascade_frontalface_default",
            "scaleFactor": 1.1,
            "minNeighbors": 3,
            "minSize": [20, 20],
            "iou_match_threshold": 0.3,
        },
    }
    (OUT_DIR / "wider_classical_mixed_case.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Selected:", case["rel_path"])
    print(f"TP={case['tp']} FP={case['fp']} FN={case['fn']}")
    print("Saved:", out_main)
    print("Saved:", out_raw)


def main() -> None:
    records = load_annotations(ANN_PATH)
    case = pick_best_case(records)
    draw_case(case)


if __name__ == "__main__":
    main()
