from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.reproduce_book_fig1114 as fig1114_mod
import scripts.reproduce_book_fig1115 as fig1115_mod

OUT_DIR = ROOT / "figures" / "ch1_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _draw_boxes(
    img,
    dets: List[Tuple[int, int, int, int]],
    title: str,
    color: Tuple[int, int, int] = (0, 220, 255),
):
    out = img.copy()
    for x, y, w, h in dets:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    fig1114_mod.put_label(out, title)
    return out


def _detect_1114(
    gray,
    panels: List[Tuple[int, int, int, int]],
    cascade_fd: cv2.CascadeClassifier,
    cascade_fa2: cv2.CascadeClassifier,
    use_fa2: bool,
    sf: float,
    mn: int,
    ms: int,
    mar: float,
) -> List[Tuple[int, int, int, int]]:
    dets = fig1114_mod.detect_per_panel(
        gray,
        panels,
        cascade_fd,
        scale_factor=sf,
        min_neighbors=mn,
        min_size=ms,
        max_area_ratio=mar,
    )
    if use_fa2:
        dets += fig1114_mod.detect_per_panel(
            gray,
            panels,
            cascade_fa2,
            scale_factor=sf,
            min_neighbors=mn,
            min_size=ms,
            max_area_ratio=mar,
        )
    return fig1114_mod.nms(dets, thr=0.35)


def _score_1114(
    dets: List[Tuple[int, int, int, int]],
    panels: List[Tuple[int, int, int, int]],
    target_det: int = 27,
) -> Dict[str, float]:
    cnt = [0] * len(panels)
    oversize = 0
    total_area_ratio = 0.0
    for x, y, w, h in dets:
        area = float(w * h)
        for i, (px, py, pw, ph) in enumerate(panels):
            if px <= x < px + pw and py <= y < py + ph:
                cnt[i] += 1
                ar = area / float(pw * ph)
                total_area_ratio += ar
                if ar > 0.20:
                    oversize += 1
                break
    panel_hits = sum(1 for c in cnt if c > 0)
    mean_area_ratio = (total_area_ratio / len(dets)) if dets else 0.0
    score = abs(len(dets) - target_det) + 2.0 * (9 - panel_hits) + 1.5 * oversize
    return {
        "det": float(len(dets)),
        "panel_hits": float(panel_hits),
        "oversize": float(oversize),
        "mean_area_ratio": mean_area_ratio,
        "score": score,
    }


def sweep_fig1114() -> Dict[str, object]:
    page = cv2.imread(str(fig1114_mod.BOOK_PAGE))
    x, y, w, h = (
        fig1114_mod.CROP["x"],
        fig1114_mod.CROP["y"],
        fig1114_mod.CROP["w"],
        fig1114_mod.CROP["h"],
    )
    fig = page[y : y + h, x : x + w].copy()
    gray = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)
    panels = fig1114_mod.find_panels(fig)

    haar = Path(cv2.data.haarcascades)
    cascades = {
        "fd": cv2.CascadeClassifier(str(haar / "haarcascade_frontalface_default.xml")),
        "fa2": cv2.CascadeClassifier(str(haar / "haarcascade_frontalface_alt2.xml")),
    }

    rows: List[Dict[str, object]] = []
    for use_fa2 in [False, True]:
        set_name = "fd+fa2" if use_fa2 else "fd"
        for sf in [1.03, 1.05, 1.07, 1.08, 1.10]:
            for mn in [1, 2, 3, 4]:
                for ms in [14, 16, 18, 20, 24]:
                    for mar in [0.20, 0.25, 0.30]:
                        dets = _detect_1114(
                            gray,
                            panels,
                            cascade_fd=cascades["fd"],
                            cascade_fa2=cascades["fa2"],
                            use_fa2=use_fa2,
                            sf=sf,
                            mn=mn,
                            ms=ms,
                            mar=mar,
                        )
                        m = _score_1114(dets, panels, target_det=27)
                        rows.append(
                            {
                                "cascade_set": set_name,
                                "scaleFactor": sf,
                                "minNeighbors": mn,
                                "minSize": ms,
                                "maxAreaRatio": mar,
                                **m,
                            }
                        )

    rows.sort(key=lambda r: float(r["score"]))
    top = rows[:5]

    # Save best image candidates for inspection.
    for i, r in enumerate(top[:3], start=1):
        dets = _detect_1114(
            gray,
            panels,
            cascade_fd=cascades["fd"],
            cascade_fa2=cascades["fa2"],
            use_fa2=(r["cascade_set"] == "fd+fa2"),
            sf=float(r["scaleFactor"]),
            mn=int(r["minNeighbors"]),
            ms=int(r["minSize"]),
            mar=float(r["maxAreaRatio"]),
        )
        vis = _draw_boxes(
            fig,
            dets,
            title=(
                f"Fig11.14 top{i} | set={r['cascade_set']} sf={r['scaleFactor']}"
                f" mn={r['minNeighbors']} ms={r['minSize']} | DET={int(r['det'])}"
            ),
        )
        cv2.imwrite(str(OUT_DIR / f"book_fig1114_sweep_top{i}.jpg"), vis)

    csv_path = OUT_DIR / "book_fig1114_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cascade_set",
                "scaleFactor",
                "minNeighbors",
                "minSize",
                "maxAreaRatio",
                "det",
                "panel_hits",
                "oversize",
                "mean_area_ratio",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": rows, "top": top, "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/")}


def _score_1115(
    dets: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    has_profile: bool,
    target_det: int = 20,
) -> Dict[str, float]:
    total = float(max(1, len(dets)))
    area_ratios = [float(w * h) / float(img_w * img_h) for (_, _, w, h) in dets]
    oversize = sum(1 for a in area_ratios if a > 0.02)
    mean_area_ratio = sum(area_ratios) / total
    score = abs(len(dets) - target_det) + 2.0 * oversize
    # Fig. 11.15 is a multiview example, so profile-aware configuration is preferred.
    if not has_profile:
        score += 0.8
    return {
        "det": float(len(dets)),
        "oversize": float(oversize),
        "mean_area_ratio": mean_area_ratio,
        "score": score,
    }


def _detect_1115(gray, cfg: Dict[str, object]) -> List[Tuple[int, int, int, int]]:
    haar_root = Path(cv2.data.haarcascades)
    cas = [
        (
            cv2.CascadeClassifier(str(haar_root / c)),
            "profileface" in c,
        )
        for c in cfg["cascades"]
    ]
    return fig1115_mod.run_multiview_like(
        gray,
        cascades=cas,
        scale_factor=float(cfg["scaleFactor"]),
        min_neighbors=int(cfg["minNeighbors"]),
        min_size=int(cfg["minSize"]),
    )


def sweep_fig1115() -> Dict[str, object]:
    page = cv2.imread(str(fig1115_mod.BOOK_PAGE))
    x, y, w, h = (
        fig1115_mod.CROP["x"],
        fig1115_mod.CROP["y"],
        fig1115_mod.CROP["w"],
        fig1115_mod.CROP["h"],
    )
    fig = page[y : y + h, x : x + w].copy()
    gray = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)

    cascade_sets = [
        ["haarcascade_frontalface_default.xml"],
        ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"],
        [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        ],
    ]
    rows: List[Dict[str, object]] = []
    for cset in cascade_sets:
        for sf in [1.05, 1.07, 1.08, 1.10]:
            for mn in [1, 2, 3, 4]:
                for ms in [16, 18, 20, 24]:
                    cfg = {
                        "cascades": cset,
                        "scaleFactor": sf,
                        "minNeighbors": mn,
                        "minSize": ms,
                    }
                    dets = _detect_1115(gray, cfg)
                    m = _score_1115(
                        dets,
                        w,
                        h,
                        has_profile=any("profileface" in c for c in cset),
                        target_det=20,
                    )
                    rows.append(
                        {
                            "cascade_set": "+".join(
                                c.replace("haarcascade_", "").replace(".xml", "") for c in cset
                            ),
                            "scaleFactor": sf,
                            "minNeighbors": mn,
                            "minSize": ms,
                            **m,
                        }
                    )

    rows.sort(key=lambda r: float(r["score"]))
    top = rows[:5]
    for i, r in enumerate(top[:3], start=1):
        cset = []
        for token in str(r["cascade_set"]).split("+"):
            cset.append("haarcascade_" + token + ".xml")
        cfg = {
            "cascades": cset,
            "scaleFactor": float(r["scaleFactor"]),
            "minNeighbors": int(r["minNeighbors"]),
            "minSize": int(r["minSize"]),
        }
        dets = _detect_1115(gray, cfg)
        vis = _draw_boxes(
            fig,
            dets,
            title=(
                f"Fig11.15 top{i} | set={r['cascade_set']} sf={r['scaleFactor']}"
                f" mn={r['minNeighbors']} ms={r['minSize']} | DET={int(r['det'])}"
            ),
        )
        cv2.imwrite(str(OUT_DIR / f"book_fig1115_sweep_top{i}.jpg"), vis)

    csv_path = OUT_DIR / "book_fig1115_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cascade_set",
                "scaleFactor",
                "minNeighbors",
                "minSize",
                "det",
                "oversize",
                "mean_area_ratio",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": rows, "top": top, "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/")}


def main() -> None:
    r14 = sweep_fig1114()
    r15 = sweep_fig1115()
    summary = {
        "fig1114": {
            "csv": r14["csv"],
            "top3": r14["top"][:3],
            "selected_for_cfgA": r14["top"][0],
            "selected_for_cfgB": r14["top"][1],
        },
        "fig1115": {
            "csv": r15["csv"],
            "top3": r15["top"][:3],
            "selected_for_best": r15["top"][0],
        },
    }
    out = OUT_DIR / "book_reproduce_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", OUT_DIR / "book_fig1114_sweep.csv")
    print("Saved:", OUT_DIR / "book_fig1115_sweep.csv")
    print("Saved:", out)
    print("Top Fig11.14:", summary["fig1114"]["top3"][0])
    print("Top Fig11.15:", summary["fig1115"]["top3"][0])


if __name__ == "__main__":
    main()
