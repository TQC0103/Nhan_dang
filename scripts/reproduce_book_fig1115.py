from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BOOK_PAGE = ROOT / "resources" / "book_pages" / "page-21.png"
OUT_DIR = ROOT / "figures" / "ch1_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed crop for full Fig. 11.15 on page-21.png.
CROP = {"x": 140, "y": 150, "w": 940, "h": 1030}


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
    cv2.rectangle(img, (8, 8), (min(img.shape[1] - 8, 760), 42), (0, 0, 0), -1)
    _draw_utf8_text(img, text, (14, 12), size=22, color=(255, 255, 255))


def nms(boxes: List[Tuple[int, int, int, int]], thr: float = 0.35) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    import numpy as np

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


def run_multiview_like(
    gray,
    cascades: List[Tuple[cv2.CascadeClassifier, bool]],
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    width = gray.shape[1]
    for det, mirror_for_pose in cascades:
        raw = det.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        boxes.extend((int(x), int(y), int(w), int(h)) for (x, y, w, h) in raw)
        # Mirror pass is only for profile detectors to cover left/right views.
        if mirror_for_pose:
            raw_flip = det.detectMultiScale(
                cv2.flip(gray, 1),
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_size, min_size),
            )
            for x, y, w, h in raw_flip:
                boxes.append((width - int(x) - int(w), int(y), int(w), int(h)))
    return nms(boxes, thr=0.35)


def draw_boxes(
    img,
    dets: List[Tuple[int, int, int, int]],
    text: str,
) -> "cv2.Mat":
    out = img.copy()
    for x, y, w, h in dets:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 255), 2)
    put_label(out, text)
    return out


def main() -> None:
    page = cv2.imread(str(BOOK_PAGE))
    if page is None:
        raise RuntimeError(f"Không đọc được ảnh trang sách: {BOOK_PAGE}")
    x, y, w, h = CROP["x"], CROP["y"], CROP["w"], CROP["h"]
    fig = page[y : y + h, x : x + w].copy()
    gray = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)

    haar_root = Path(cv2.data.haarcascades)
    c_frontal_default = cv2.CascadeClassifier(str(haar_root / "haarcascade_frontalface_default.xml"))
    c_profile = cv2.CascadeClassifier(str(haar_root / "haarcascade_profileface.xml"))

    # Selected from grid sweep for closest overall match to book density and box scale.
    cfg = {
        "scaleFactor": 1.08,
        "minNeighbors": 3,
        "minSize": 20,
        "cascades": [
            "haarcascade_frontalface_default.xml",
            "haarcascade_profileface.xml",
        ],
    }
    dets = run_multiview_like(
        gray,
        cascades=[(c_frontal_default, False), (c_profile, True)],
        scale_factor=cfg["scaleFactor"],
        min_neighbors=cfg["minNeighbors"],
        min_size=cfg["minSize"],
    )

    book = fig.copy()
    put_label(book, "Hình sách Fig. 11.15: phát hiện khuôn mặt đa góc nhìn")
    rep = draw_boxes(
        fig,
        dets,
        text=f"Tái lập gần đúng: DET={len(dets)} | sf={cfg['scaleFactor']} mn={cfg['minNeighbors']}",
    )

    cv2.imwrite(str(OUT_DIR / "book_fig1115_from_book.jpg"), book)
    cv2.imwrite(str(OUT_DIR / "book_fig1115_reproduce_best.jpg"), rep)

    summary: Dict[str, object] = {
        "source_page": str(BOOK_PAGE.relative_to(ROOT)).replace("\\", "/"),
        "crop": CROP,
        "opencv_version": cv2.__version__,
        "selected_config": {**cfg, "detections": len(dets)},
        "note": "Closest OpenCV approximation; book figure likely produced by dedicated multiview cascade (FloatBoost/extended Haar), not OpenCV pretrained frontal-only setup.",
    }
    (OUT_DIR / "book_fig1115_reproduce_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Saved:", OUT_DIR / "book_fig1115_from_book.jpg")
    print("Saved:", OUT_DIR / "book_fig1115_reproduce_best.jpg")
    print("Saved:", OUT_DIR / "book_fig1115_reproduce_summary.json")
    print("Detections:", len(dets))


if __name__ == "__main__":
    main()
