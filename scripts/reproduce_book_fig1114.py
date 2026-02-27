from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BOOK_PAGE = ROOT / "resources" / "book_pages" / "page-20.png"
OUT_DIR = ROOT / "figures" / "ch1_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed crop for full Fig. 11.14 on page-20.png (3x3 collage).
CROP = {"x": 146, "y": 148, "w": 933, "h": 710}


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


def put_label(img, text: str) -> None:
    cv2.rectangle(img, (8, 8), (min(img.shape[1] - 8, 860), 40), (0, 0, 0), -1)
    _draw_utf8_text(img, text, (14, 12), size=22, color=(255, 255, 255))


def draw_boxes(
    img,
    dets: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int],
    text: str,
) -> "cv2.Mat":
    out = img.copy()
    for x, y, w, h in dets:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    put_label(out, text)
    return out


def nms(
    boxes: List[Tuple[int, int, int, int]],
    thr: float = 0.35,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idx = area.argsort()
    pick = []
    while len(idx) > 0:
        i = idx[-1]
        pick.append(i)
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
        iou = inter / union
        idx = idx[iou <= thr]
    return [tuple(map(int, boxes[i])) for i in pick]


def detect(
    gray,
    cascade: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> List[Tuple[int, int, int, int]]:
    raw = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in raw]


def _find_two_splits(
    profile: np.ndarray,
    orth_len: int,
    low_ratio: float,
    min_run: int,
    fallback_a: int,
    fallback_b: int,
) -> Tuple[int, int]:
    """Find two white-gutter split positions from a 1D non-white projection."""
    thr = int(orth_len * low_ratio)
    low = profile < thr
    runs: List[Tuple[int, int]] = []
    start = -1
    for i, v in enumerate(low):
        if v and start < 0:
            start = i
        if (not v) and start >= 0:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = -1
    if start >= 0 and len(low) - start >= min_run:
        runs.append((start, len(low) - 1))

    # Remove runs too close to borders.
    margin = max(10, len(low) // 20)
    runs = [(a, b) for (a, b) in runs if a > margin and b < len(low) - margin]
    if len(runs) < 2:
        return fallback_a, fallback_b

    # Pick two widest runs then convert to centers.
    runs.sort(key=lambda r: (r[1] - r[0]), reverse=True)
    best = sorted(runs[:2], key=lambda r: r[0])
    s1 = (best[0][0] + best[0][1]) // 2
    s2 = (best[1][0] + best[1][1]) // 2
    return int(s1), int(s2)


def find_panels(fig_bgr) -> List[Tuple[int, int, int, int]]:
    """Find 3x3 collage panels by locating white gutters along X/Y axes."""
    gray = cv2.cvtColor(fig_bgr, cv2.COLOR_BGR2GRAY)
    non_white = (gray < 245).astype(np.uint8)
    h, w = gray.shape[:2]
    col_profile = non_white.sum(axis=0)
    row_profile = non_white.sum(axis=1)

    sx1, sx2 = _find_two_splits(
        col_profile,
        orth_len=h,
        low_ratio=0.12,
        min_run=6,
        fallback_a=int(w * 0.20),
        fallback_b=int(w * 0.60),
    )
    sy1, sy2 = _find_two_splits(
        row_profile,
        orth_len=w,
        low_ratio=0.12,
        min_run=6,
        fallback_a=int(h * 0.34),
        fallback_b=int(h * 0.67),
    )

    xs = [0, sx1, sx2, w]
    ys = [0, sy1, sy2, h]
    panels: List[Tuple[int, int, int, int]] = []
    # Trim each cell to avoid white gutters.
    pad = 4
    for ry in range(3):
        for cx in range(3):
            x1 = xs[cx] + pad
            x2 = xs[cx + 1] - pad
            y1 = ys[ry] + pad
            y2 = ys[ry + 1] - pad
            if x2 > x1 and y2 > y1:
                panels.append((x1, y1, x2 - x1, y2 - y1))
    return panels


def detect_per_panel(
    gray,
    panels: List[Tuple[int, int, int, int]],
    cascade: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    max_area_ratio: float = 0.30,
    min_ar: float = 0.55,
    max_ar: float = 1.70,
) -> List[Tuple[int, int, int, int]]:
    dets: List[Tuple[int, int, int, int]] = []
    for px, py, pw, ph in panels:
        roi = gray[py : py + ph, px : px + pw]
        raw = cascade.detectMultiScale(
            roi,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        for x, y, w, h in raw:
            area_ratio = (float(w) * float(h)) / float(pw * ph)
            ar = float(w) / float(h)
            if area_ratio > max_area_ratio:
                continue
            if ar < min_ar or ar > max_ar:
                continue
            dets.append((px + int(x), py + int(y), int(w), int(h)))
    return dets


def main() -> None:
    page = cv2.imread(str(BOOK_PAGE))
    if page is None:
        raise RuntimeError(f"Không đọc được ảnh trang sách: {BOOK_PAGE}")

    x, y, w, h = CROP["x"], CROP["y"], CROP["w"], CROP["h"]
    fig = page[y : y + h, x : x + w].copy()
    gray = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)
    panels = find_panels(fig)
    if len(panels) != 9:
        print(f"Warning: expected 9 panels, found {len(panels)}")

    cascade_fd = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    cascade_fa2 = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml")
    )

    # Config A: selected from hyperparameter sweep for closest match to book style.
    cfg_a = {
        "scaleFactor": 1.05,
        "minNeighbors": 1,
        "minSize": 18,
        "maxAreaRatio": 0.20,
        "cascadeSet": "frontal_default+frontal_alt2",
    }
    det_a = []
    det_a += detect_per_panel(
        gray,
        panels,
        cascade_fd,
        scale_factor=cfg_a["scaleFactor"],
        min_neighbors=cfg_a["minNeighbors"],
        min_size=cfg_a["minSize"],
        max_area_ratio=cfg_a["maxAreaRatio"],
    )
    det_a += detect_per_panel(
        gray,
        panels,
        cascade_fa2,
        scale_factor=cfg_a["scaleFactor"],
        min_neighbors=cfg_a["minNeighbors"],
        min_size=cfg_a["minSize"],
        max_area_ratio=cfg_a["maxAreaRatio"],
    )
    det_a = nms(det_a, thr=0.35)

    # Config B: higher-recall variant from sweep for side-by-side comparison.
    cfg_b = {
        "scaleFactor": 1.07,
        "minNeighbors": 1,
        "minSize": 18,
        "maxAreaRatio": 0.20,
        "cascadeSet": "frontal_default+frontal_alt2",
    }
    det_b = []
    det_b += detect_per_panel(
        gray,
        panels,
        cascade_fd,
        scale_factor=cfg_b["scaleFactor"],
        min_neighbors=cfg_b["minNeighbors"],
        min_size=cfg_b["minSize"],
        max_area_ratio=cfg_b["maxAreaRatio"],
    )
    det_b += detect_per_panel(
        gray,
        panels,
        cascade_fa2,
        scale_factor=cfg_b["scaleFactor"],
        min_neighbors=cfg_b["minNeighbors"],
        min_size=cfg_b["minSize"],
        max_area_ratio=cfg_b["maxAreaRatio"],
    )
    det_b = nms(det_b, thr=0.35)

    fig_book = fig.copy()
    put_label(fig_book, "Hình sách Fig. 11.14: kết quả detector trong chương")
    fig_a = draw_boxes(
        fig,
        det_a,
        color=(40, 220, 40),
        text=f"Tái lập A (cấu hình nhóm): DET={len(det_a)}",
    )
    fig_b = draw_boxes(
        fig,
        det_b,
        color=(0, 220, 255),
        text=f"Tái lập B (ưu tiên độ phủ): DET={len(det_b)}",
    )

    cv2.imwrite(str(OUT_DIR / "book_fig1114_from_book.jpg"), fig_book)
    cv2.imwrite(str(OUT_DIR / "book_fig1114_reproduce_cfgA.jpg"), fig_a)
    cv2.imwrite(str(OUT_DIR / "book_fig1114_reproduce_cfgB.jpg"), fig_b)

    summary: Dict[str, object] = {
        "source_page": str(BOOK_PAGE.relative_to(ROOT)).replace("\\", "/"),
        "crop": CROP,
        "opencv_version": cv2.__version__,
        "model": "haarcascade_frontalface_default.xml",
        "mode": "per_panel_detection",
        "num_panels": len(panels),
        "panels": panels,
        "config_A": {**cfg_a, "detections": len(det_a)},
        "config_B": {**cfg_b, "detections": len(det_b)},
    }
    (OUT_DIR / "book_fig1114_reproduce_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Saved:", OUT_DIR / "book_fig1114_from_book.jpg")
    print("Saved:", OUT_DIR / "book_fig1114_reproduce_cfgA.jpg")
    print("Saved:", OUT_DIR / "book_fig1114_reproduce_cfgB.jpg")
    print("Saved:", OUT_DIR / "book_fig1114_reproduce_summary.json")
    print("Detections A:", len(det_a))
    print("Detections B:", len(det_b))


if __name__ == "__main__":
    main()
