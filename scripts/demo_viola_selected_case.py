from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "widerface"
IMG_ROOT = DATA_ROOT / "WIDER_val" / "images"
ANN_PATH = DATA_ROOT / "wider_face_split" / "wider_face_val_bbx_gt.txt"
CASE_META = ROOT / "figures" / "ch1_classical" / "wider_classical_mixed_case.json"
OUT_DIR = ROOT / "figures" / "ch1_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_utf8_text(
    img_bgr,
    text: str,
    origin: Tuple[int, int],
    size: int = 22,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(size)
    draw.text(origin, text, font=font, fill=(color[2], color[1], color[0]))
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:, :] = out


def load_gt_for_image(target_rel: str) -> List[Tuple[int, int, int, int]]:
    lines = ANN_PATH.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        rel_path = lines[idx].strip()
        idx += 1
        n = int(lines[idx].strip())
        idx += 1
        boxes: List[Tuple[int, int, int, int]] = []
        for _ in range(n):
            parts = lines[idx].split()
            idx += 1
            x, y, w, h = map(int, parts[:4])
            invalid = int(parts[7])
            if w > 0 and h > 0 and invalid == 0:
                boxes.append((x, y, w, h))
        if rel_path == target_rel:
            return boxes
    raise RuntimeError(f"Không tìm thấy nhãn cho ảnh: {target_rel}")


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
    pairs = []
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


def put_label(img, text: str) -> None:
    cv2.rectangle(img, (8, 8), (min(img.shape[1] - 8, 560), 40), (0, 0, 0), -1)
    _draw_utf8_text(img, text, (14, 12), size=22, color=(255, 255, 255))


def main() -> None:
    case_meta = json.loads(CASE_META.read_text(encoding="utf-8"))
    rel_path = case_meta["selected_image"]
    img_path = IMG_ROOT / rel_path
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Không đọc được ảnh: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )

    # Approximate Step 1 output: very loose detections as dense candidates.
    dense_raw = detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=0, minSize=(20, 20)
    )
    dense = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in dense_raw]

    # Step 2+3 practical output with grouping.
    final_raw = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    final = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in final_raw]

    gts = load_gt_for_image(rel_path)
    matches, fn_idx, fp_idx = greedy_match(gts, final, thr=0.3)
    tp = len(matches)
    fn = len(fn_idx)
    fp = len(fp_idx)

    vis_dense = img.copy()
    for x, y, w, h in dense:
        cv2.rectangle(vis_dense, (x, y), (x + w, y + h), (255, 180, 0), 1)
    put_label(vis_dense, f"Bước 1 mô phỏng: ứng viên dày | số khung={len(dense)}")

    vis_final = img.copy()
    for x, y, w, h in final:
        cv2.rectangle(vis_final, (x, y), (x + w, y + h), (40, 220, 40), 2)
    put_label(vis_final, f"Bước 2+3: sau cascade + gom nhóm | số khung={len(final)}")

    vis_eval = img.copy()
    matched_det = {di for _, di in matches}
    for di in matched_det:
        x, y, w, h = final[di]
        cv2.rectangle(vis_eval, (x, y), (x + w, y + h), (40, 220, 40), 2)  # TP
    for di in fp_idx:
        x, y, w, h = final[di]
        cv2.rectangle(vis_eval, (x, y), (x + w, y + h), (30, 30, 230), 2)  # FP
    for gi in fn_idx:
        x, y, w, h = gts[gi]
        cv2.rectangle(vis_eval, (x, y), (x + w, y + h), (0, 220, 255), 2)  # FN
    put_label(vis_eval, f"Đánh giá: TP={tp} FP={fp} FN={fn} | GT={len(gts)} DET={len(final)}")

    cv2.imwrite(str(OUT_DIR / "demo_step1_dense_candidates.jpg"), vis_dense)
    cv2.imwrite(str(OUT_DIR / "demo_step2_step3_final.jpg"), vis_final)
    cv2.imwrite(str(OUT_DIR / "demo_tp_fp_fn_overlay.jpg"), vis_eval)

    summary = {
        "image": rel_path,
        "step1_candidate_count": len(dense),
        "final_detection_count": len(final),
        "gt_count": len(gts),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "params": {
            "step1_like": {"scaleFactor": 1.05, "minNeighbors": 0, "minSize": [20, 20]},
            "final": {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": [20, 20]},
            "iou_match_threshold": 0.3,
        },
    }
    (OUT_DIR / "demo_selected_case_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Selected image:", rel_path)
    print("Step1-like candidates:", len(dense))
    print(f"Final: TP={tp}, FP={fp}, FN={fn}, GT={len(gts)}, DET={len(final)}")
    print("Saved:", OUT_DIR / "demo_step1_dense_candidates.jpg")
    print("Saved:", OUT_DIR / "demo_step2_step3_final.jpg")
    print("Saved:", OUT_DIR / "demo_tp_fp_fn_overlay.jpg")
    print("Saved:", OUT_DIR / "demo_selected_case_summary.json")


if __name__ == "__main__":
    main()
