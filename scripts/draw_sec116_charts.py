from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "figures" / "ch1_classical"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _font(size: int):
    for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_xy_plot(
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    points: List[Tuple[float, float, str]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    w, h = 1200, 780
    left, right, top, bottom = 120, 70, 100, 120
    plot_w = w - left - right
    plot_h = h - top - bottom
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(34)
    f_axis = _font(24)
    f_tick = _font(18)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        xx = left + int((x - x_min) / (x_max - x_min + 1e-9) * plot_w)
        yy = top + plot_h - int((y - y_min) / (y_max - y_min + 1e-9) * plot_h)
        return xx, yy

    d.text((left, 35), title, fill="black", font=f_title)
    d.rectangle([left, top, left + plot_w, top + plot_h], outline="black", width=2)

    # Grid and ticks
    for i in range(6):
        x = left + int(i * plot_w / 5)
        y = top + int(i * plot_h / 5)
        d.line([(x, top), (x, top + plot_h)], fill=(225, 225, 225), width=1)
        d.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225), width=1)
        xv = x_min + (x_max - x_min) * i / 5
        yv = y_max - (y_max - y_min) * i / 5
        d.text((x - 20, top + plot_h + 12), f"{xv:.0f}", fill="black", font=f_tick)
        d.text((left - 80, y - 10), f"{yv:.2f}", fill="black", font=f_tick)

    d.text((left + plot_w // 2 - 70, h - 65), x_label, fill="black", font=f_axis)
    d.text((20, top + plot_h // 2), y_label, fill="black", font=f_axis)

    # Draw line
    ordered = sorted(points, key=lambda p: p[0])
    px_pts = [to_px(p[0], p[1]) for p in ordered]
    if len(px_pts) >= 2:
        d.line(px_pts, fill=(30, 100, 220), width=3)

    # Draw points + labels
    for x, y, tag in ordered:
        px, py = to_px(x, y)
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(220, 50, 50), outline=(220, 50, 50))
        d.text((px + 8, py - 18), tag, fill="black", font=f_tick)

    img.save(out_path)


def draw_subwindow_plot(out_path: Path, rows: List[Dict[str, str]]) -> None:
    w, h = 1200, 780
    left, right, top, bottom = 120, 70, 100, 120
    plot_w = w - left - right
    plot_h = h - top - bottom
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(34)
    f_axis = _font(24)
    f_tick = _font(18)

    sizes = [int(r["minSize"]) for r in rows]
    recalls = [float(r["recall"]) for r in rows]
    precisions = [float(r["precision"]) for r in rows]
    x_min, x_max = min(sizes), max(sizes)
    y_min, y_max = 0.0, max(max(recalls), max(precisions)) * 1.1

    def to_px(x: float, y: float) -> Tuple[int, int]:
        xx = left + int((x - x_min) / (x_max - x_min + 1e-9) * plot_w)
        yy = top + plot_h - int((y - y_min) / (y_max - y_min + 1e-9) * plot_h)
        return xx, yy

    d.text((left, 35), "So sánh kích thước cửa sổ con", fill="black", font=f_title)
    d.rectangle([left, top, left + plot_w, top + plot_h], outline="black", width=2)

    for i in range(6):
        y = top + int(i * plot_h / 5)
        d.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225), width=1)
        yv = y_max - (y_max - y_min) * i / 5
        d.text((left - 80, y - 10), f"{yv:.2f}", fill="black", font=f_tick)
    for s in sizes:
        x, _ = to_px(s, 0)
        d.line([(x, top), (x, top + plot_h)], fill=(235, 235, 235), width=1)
        d.text((x - 20, top + plot_h + 12), str(s), fill="black", font=f_tick)

    d.text((left + plot_w // 2 - 110, h - 65), "minSize", fill="black", font=f_axis)
    d.text((20, top + plot_h // 2), "Tỷ lệ", fill="black", font=f_axis)

    rec_pts = [to_px(s, r) for s, r in zip(sizes, recalls)]
    pre_pts = [to_px(s, p) for s, p in zip(sizes, precisions)]
    d.line(rec_pts, fill=(30, 120, 30), width=3)
    d.line(pre_pts, fill=(220, 90, 30), width=3)
    for p in rec_pts:
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(30, 120, 30))
    for p in pre_pts:
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(220, 90, 30))

    # Legend
    lx, ly = left + 20, top + 20
    d.line([(lx, ly), (lx + 45, ly)], fill=(30, 120, 30), width=3)
    d.text((lx + 55, ly - 10), "Recall", fill="black", font=f_tick)
    d.line([(lx, ly + 35), (lx + 45, ly + 35)], fill=(220, 90, 30), width=3)
    d.text((lx + 55, ly + 25), "Precision", fill="black", font=f_tick)

    img.save(out_path)


def main() -> None:
    roc_rows = read_rows(IN_DIR / "sec116_roc_like_points.csv")
    sub_rows = read_rows(IN_DIR / "sec116_subwindow_size_compare.csv")

    roc_points = [
        (float(r["fp"]), float(r["recall"]), f"mn={int(float(r['minNeighbors']))}")
        for r in roc_rows
    ]
    x_vals = [p[0] for p in roc_points]
    draw_xy_plot(
        IN_DIR / "sec116_roc_like_curve.png",
        "ROC-like cho Viola-Jones trên WIDER subset",
        "Số báo động giả sau hậu xử lý",
        "Tỷ lệ phát hiện",
        roc_points,
        x_min=min(x_vals),
        x_max=max(x_vals),
        y_min=0.0,
        y_max=max(p[1] for p in roc_points) * 1.1,
    )
    draw_subwindow_plot(IN_DIR / "sec116_subwindow_compare_curve.png", sub_rows)
    print("Saved:", IN_DIR / "sec116_roc_like_curve.png")
    print("Saved:", IN_DIR / "sec116_subwindow_compare_curve.png")


if __name__ == "__main__":
    main()
