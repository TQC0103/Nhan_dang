import argparse
import json
import math
import os
import os.path as osp
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_run_paths(experiment_root: str, run_name: str) -> Dict[str, str]:
    run_root = osp.join(experiment_root, run_name)
    return {
        'root': run_root,
        'metrics_baseline': osp.join(run_root, 'metrics', 'baseline_results_summary.json'),
        'metrics_improved': osp.join(run_root, 'metrics', 'asr_jsar_results_summary.json'),
        'face_size': osp.join(run_root, 'face_size_analysis', 'face_size_distribution_analysis.json'),
        'jsar': osp.join(run_root, 'jsar_assignment', 'jsar_assignment_summary.json'),
        'hard_subset': osp.join(run_root, 'hard_subset_analysis', 'hard_subset_analysis.json'),
        'scale_hist': osp.join(run_root, 'scale_prob_history', 'scale_prob_history_summary.json'),
    }


def compute_scale_stats(scale_candidates: List[float], scale_probs: List[float]) -> Dict[str, float]:
    weighted_inv_scale = float(sum((p / s) for s, p in zip(scale_candidates, scale_probs)))
    weighted_scale = float(sum((p * s) for s, p in zip(scale_candidates, scale_probs)))
    zoom_in_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s < 1.0))
    zoom_out_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s > 1.0))
    extreme_zoom_in_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s <= 0.6))
    extreme_zoom_out_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s >= 2.0))
    return {
        'expected_inverse_scale': weighted_inv_scale,
        'expected_scale': weighted_scale,
        'zoom_in_mass': zoom_in_mass,
        'zoom_out_mass': zoom_out_mass,
        'extreme_zoom_in_mass': extreme_zoom_in_mass,
        'extreme_zoom_out_mass': extreme_zoom_out_mass,
        'min_scale': float(min(scale_candidates)),
        'max_scale': float(max(scale_candidates)),
        'max_magnification': float(max(1.0 / s for s in scale_candidates)),
        'min_magnification': float(min(1.0 / s for s in scale_candidates)),
    }


def collect_run_summary(run_name: str, paths: Dict[str, str]) -> Dict:
    baseline_metrics = load_json(paths['metrics_baseline'])
    improved_metrics = load_json(paths['metrics_improved'])
    face_size = load_json(paths['face_size'])
    jsar = load_json(paths['jsar'])
    hard_subset = load_json(paths['hard_subset']) if osp.exists(paths['hard_subset']) else None
    scale_hist = load_json(paths['scale_hist'])

    baseline_scale_stats = compute_scale_stats(
        face_size['metadata']['baseline_scale_candidates'],
        face_size['metadata']['baseline_scale_probs'])
    improved_scale_stats = compute_scale_stats(
        face_size['metadata']['improved_scale_candidates'],
        face_size['metadata']['improved_scale_probs'])

    return {
        'name': run_name,
        'baseline_metrics': baseline_metrics,
        'improved_metrics': improved_metrics,
        'face_size': face_size,
        'jsar': jsar,
        'hard_subset': hard_subset,
        'scale_hist': scale_hist,
        'baseline_scale_stats': baseline_scale_stats,
        'improved_scale_stats': improved_scale_stats,
    }


def plot_metric_bars(runs: List[Dict], out_path: str):
    metrics = ['easy_AP', 'medium_AP', 'hard_AP', 'mAP']
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.5))

    series = [
        ('Default baseline', runs[0]['baseline_metrics'], -1.5 * width, '#4c78a8'),
        ('Default ASR+JSAR', runs[0]['improved_metrics'], -0.5 * width, '#f58518'),
        ('Paper SR12 baseline', runs[1]['baseline_metrics'], 0.5 * width, '#54a24b'),
        ('Paper SR12 ASR+JSAR', runs[1]['improved_metrics'], 1.5 * width, '#e45756'),
    ]
    for label, source, offset, color in series:
        vals = [source[m] for m in metrics]
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.65, 0.93)
    ax.set_ylabel('AP')
    ax.set_title('WIDERFace metrics across scale sets')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_asr_delta(runs: List[Dict], out_path: str):
    metrics = ['easy_AP', 'medium_AP', 'hard_AP', 'mAP']
    labels = ['easy', 'medium', 'hard', 'mAP']
    default_delta = [runs[0]['improved_metrics'][m] - runs[0]['baseline_metrics'][m] for m in metrics]
    paper_delta = [runs[1]['improved_metrics'][m] - runs[1]['baseline_metrics'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, default_delta, width, label='Default scale set', color='#4c78a8')
    bars2 = ax.bar(x + width / 2, paper_delta, width, label='Paper SR12 scale set', color='#e45756')
    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.001 if val >= 0 else -0.001),
                    f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('ASR+JSAR - Baseline')
    ax.set_title('Metric deltas from ASR+JSAR')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scale_pool_geometry(runs: List[Dict], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, run in enumerate(runs):
        meta = run['face_size']['metadata']
        base_candidates = np.array(meta['baseline_scale_candidates'], dtype=float)
        base_probs = np.array(meta['baseline_scale_probs'], dtype=float)
        imp_candidates = np.array(meta['improved_scale_candidates'], dtype=float)
        imp_probs = np.array(meta['improved_scale_probs'], dtype=float)

        ax = axes[0, col]
        ax.plot(base_candidates, 1.0 / base_candidates, marker='o', label='candidate magnification 1/scale')
        ax.set_title(f"{run['name']}: candidate pool geometry")
        ax.set_xlabel('scale candidate')
        ax.set_ylabel('relative face magnification')
        ax.grid(True, linestyle='--', alpha=0.25)

        ax2 = axes[1, col]
        width = 0.035 if len(base_candidates) <= 10 else 0.03
        ax2.bar(base_candidates - width / 2, base_probs, width, label='baseline uniform', color='#b0b0b0')
        ax2.bar(imp_candidates + width / 2, imp_probs, width, label='ASR mean prob', color='#4c78a8')
        ax2.set_title(f"{run['name']}: scale probability profile")
        ax2.set_xlabel('scale candidate')
        ax2.set_ylabel('probability')
        ax2.grid(True, linestyle='--', alpha=0.25)
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_distribution_pressure(runs: List[Dict], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    run_labels = ['Default baseline', 'Default ASR+JSAR', 'Paper baseline', 'Paper ASR+JSAR']

    tiny_ratios = [
        runs[0]['face_size']['baseline_summary']['bin_ratios'][0],
        runs[0]['face_size']['improved_summary']['bin_ratios'][0],
        runs[1]['face_size']['baseline_summary']['bin_ratios'][0],
        runs[1]['face_size']['improved_summary']['bin_ratios'][0],
    ]
    p50s = [
        runs[0]['face_size']['baseline_summary']['p50'],
        runs[0]['face_size']['improved_summary']['p50'],
        runs[1]['face_size']['baseline_summary']['p50'],
        runs[1]['face_size']['improved_summary']['p50'],
    ]
    promote_tiny = [
        runs[0]['face_size']['baseline_summary']['promote_tiny_to_ge16_ratio'],
        runs[0]['face_size']['improved_summary']['promote_tiny_to_ge16_ratio'],
        runs[1]['face_size']['baseline_summary']['promote_tiny_to_ge16_ratio'],
        runs[1]['face_size']['improved_summary']['promote_tiny_to_ge16_ratio'],
    ]
    inv_scale = [
        runs[0]['baseline_scale_stats']['expected_inverse_scale'],
        runs[0]['improved_scale_stats']['expected_inverse_scale'],
        runs[1]['baseline_scale_stats']['expected_inverse_scale'],
        runs[1]['improved_scale_stats']['expected_inverse_scale'],
    ]

    def bar_plot(ax, values, title, ylabel, color):
        x = np.arange(len(run_labels))
        bars = ax.bar(x, values, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=20, ha='right')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8)

    bar_plot(axes[0, 0], tiny_ratios, 'Tiny-face ratio after SR simulation', 'ratio', '#4c78a8')
    bar_plot(axes[0, 1], p50s, 'Median face size after SR simulation', 'pixels', '#f58518')
    bar_plot(axes[1, 0], promote_tiny, 'Tiny -> >=16 px promotion ratio', 'ratio', '#54a24b')
    bar_plot(axes[1, 1], inv_scale, 'Expected magnification E[1/scale]', 'relative factor', '#e45756')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_jsar_tiny_supervision(runs: List[Dict], out_path: str):
    labels = ['Default', 'Paper SR12']
    before = [runs[0]['jsar']['final_before_per_gt'][0], runs[1]['jsar']['final_before_per_gt'][0]]
    after = [runs[0]['jsar']['final_after_per_gt'][0], runs[1]['jsar']['final_after_per_gt'][0]]
    boost = [runs[0]['jsar']['final_boost_ratio'][0], runs[1]['jsar']['final_boost_ratio'][0]]

    x = np.arange(len(labels))
    width = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    bars1 = axes[0].bar(x - width / 2, before, width, label='before JSAR', color='#b0b0b0')
    bars2 = axes[0].bar(x + width / 2, after, width, label='after JSAR', color='#4c78a8')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('positives per tiny GT')
    axes[0].set_title('Tiny-face supervision density')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.25)
    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, val, f'{val:.2f}',
                         ha='center', va='bottom', fontsize=8)

    bars3 = axes[1].bar(x, boost, width=0.42, color='#e45756')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('after / before')
    axes[1].set_title('Tiny-face JSAR boost ratio')
    axes[1].grid(axis='y', linestyle='--', alpha=0.25)
    for bar in bars3:
        val = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f'{val:.3f}',
                     ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hard_size_gain(run: Dict, out_path: str, title: str):
    if not run.get('hard_subset'):
        return
    bins = run['hard_subset']['size_bins']
    labels = [item['size_bin'] for item in bins]
    deltas = [item['recall_proxy_delta'] for item in bins]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(x, deltas, color=['#e45756' if d < 0 else '#54a24b' for d in deltas])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('recall proxy delta')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.005 if val >= 0 else -0.005),
                f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hard_size_gain_compare(runs: List[Dict], out_path: str):
    if not all(run.get('hard_subset') for run in runs):
        return
    labels = [item['size_bin'] for item in runs[0]['hard_subset']['size_bins']]
    default_deltas = [item['recall_proxy_delta'] for item in runs[0]['hard_subset']['size_bins']]
    paper_deltas = [item['recall_proxy_delta'] for item in runs[1]['hard_subset']['size_bins']]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars1 = ax.bar(x - width / 2, default_deltas, width, label='Default scale set', color='#4c78a8')
    bars2 = ax.bar(x + width / 2, paper_deltas, width, label='Paper SR12 scale set', color='#e45756')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('recall proxy delta')
    ax.set_title('Hard-subset recall gain by face-size bin')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.005 if val >= 0 else -0.005),
                    f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_box(ax, x, y, w, h, text, facecolor='#f8f8f8', edgecolor='#444', fontsize=11):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.02',
                           linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=14,
                            linewidth=1.4, color='#333')
    ax.add_patch(arrow)


def plot_asr_schematic(out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.9), gridspec_kw={'width_ratios': [1.25, 1.1, 1.0]})

    def draw_scene(ax, x, y, w, h, crop_w, face_r, title, note, color):
        outer = Rectangle((x, y), w, h, facecolor='#fcfcfc', edgecolor='#444', linewidth=1.2)
        ax.add_patch(outer)
        face_center = (x + w * 0.55, y + h * 0.50)
        ax.add_patch(Circle(face_center, face_r, facecolor='#333', edgecolor='none'))
        crop = Rectangle((face_center[0] - crop_w / 2, face_center[1] - crop_w / 2),
                         crop_w, crop_w, fill=False, edgecolor=color, linewidth=2.0)
        ax.add_patch(crop)
        ax.text(x + w / 2, y + h + 0.03, title, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x + w / 2, y - 0.045, note, ha='center', va='top', fontsize=9, color=color)

        draw_arrow(ax, x + w + 0.015, y + h / 2, x + w + 0.09, y + h / 2)
        resized = Rectangle((x + w + 0.10, y), 0.16, h, facecolor='#ffffff', edgecolor='#444', linewidth=1.2)
        ax.add_patch(resized)
        ax.add_patch(Circle((x + w + 0.18, face_center[1]), face_r * (0.16 / crop_w) * 1.4,
                            facecolor='#333', edgecolor='none'))
        ax.text(x + w + 0.18, y - 0.045, 'sau đổi cỡ về 640×640', ha='center', va='top', fontsize=8.5)

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Bước 1. Tỉ lệ cắt quyết định kích thước mặt sau tăng cường dữ liệu', fontsize=12, pad=10)
    draw_scene(ax, 0.03, 0.60, 0.18, 0.22, crop_w=0.10, face_r=0.020,
               title='Tỉ lệ nhỏ', note='vùng cắt hẹp hơn → mặt lớn hơn', color='#4c78a8')
    draw_scene(ax, 0.03, 0.33, 0.18, 0.22, crop_w=0.18, face_r=0.020,
               title='Tỉ lệ trung bình', note='mức thay đổi vừa phải', color='#72b7b2')
    draw_scene(ax, 0.03, 0.06, 0.18, 0.22, crop_w=0.26, face_r=0.020,
               title='Tỉ lệ lớn', note='vùng cắt rộng hơn → mặt nhỏ hơn', color='#e45756')
    ax.text(0.58, 0.92,
            'Trong SCRFD, tỉ lệ cắt không biểu diễn trực tiếp\n"khuôn mặt lớn" hay "khuôn mặt nhỏ".\n'
            'Nó điều khiển vùng ảnh được cắt ra trước khi đổi cỡ.',
            ha='center', va='top', fontsize=10)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Bước 2. Dùng phản hồi huấn luyện để cập nhật xác suất lấy mẫu', fontsize=12, pad=10)
    draw_box(ax, 0.05, 0.73, 0.24, 0.14, 'Huấn luyện 1 epoch\nvới phân phối hiện tại', '#eef4fb', fontsize=10)
    draw_box(ax, 0.38, 0.73, 0.24, 0.14, 'Thống kê theo nhóm kích thước:\n'
                                        'số mặt, số anchor dương,\n'
                                        'lỗi phân loại, lỗi hồi quy', '#fdf4e7', fontsize=9.5)
    draw_box(ax, 0.71, 0.73, 0.24, 0.14, 'Tính độ khó:\nnhóm nào đang thiếu giám sát\nvà còn lỗi cao', '#eef8e8', fontsize=9.5)
    draw_arrow(ax, 0.29, 0.80, 0.38, 0.80)
    draw_arrow(ax, 0.62, 0.80, 0.71, 0.80)

    draw_box(ax, 0.18, 0.46, 0.26, 0.14, 'Chiếu độ khó sang từng\n'
                                         'tỉ lệ cắt ứng viên', '#fbeaea', fontsize=10)
    draw_box(ax, 0.56, 0.46, 0.26, 0.14, 'Tăng xác suất ở những tỉ lệ cắt\n'
                                         'giúp mô hình nhìn thấy nhiều\n'
                                         'mẫu khó hữu ích hơn', '#eef4fb', fontsize=10)
    draw_arrow(ax, 0.78, 0.73, 0.31, 0.60)
    draw_arrow(ax, 0.44, 0.53, 0.56, 0.53)

    bins = ['rất nhỏ', 'nhỏ', 'vừa', 'lớn']
    hard = np.array([0.95, 0.72, 0.34, 0.15])
    inset = ax.inset_axes([0.05, 0.08, 0.34, 0.22])
    inset.bar(np.arange(4), hard, color=['#e45756', '#f58518', '#72b7b2', '#b0b0b0'])
    inset.set_xticks(np.arange(4))
    inset.set_xticklabels(bins, fontsize=8)
    inset.set_yticks([])
    inset.set_title('độ khó theo nhóm kích thước', fontsize=8.5)
    inset.spines[['top', 'right', 'left']].set_visible(False)

    scale_candidates = np.array([0.35, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=float)
    uniform = np.full(scale_candidates.shape[0], 1.0 / scale_candidates.shape[0])
    adapted = np.array([0.047, 0.059, 0.085, 0.112, 0.130, 0.120, 0.118, 0.112, 0.109, 0.108])
    inset2 = ax.inset_axes([0.55, 0.07, 0.37, 0.24])
    inset2.plot(scale_candidates, uniform, color='#9d9da1', linewidth=1.8, label='ban đầu')
    inset2.plot(scale_candidates, adapted, color='#4c78a8', linewidth=2.2, label='sau cập nhật')
    inset2.axvline(1.0, color='#666', linestyle='--', linewidth=1)
    inset2.set_xticks([0.35, 0.8, 1.0, 1.6, 2.0])
    inset2.set_yticks([])
    inset2.set_title('xác suất lấy mẫu theo tỉ lệ cắt', fontsize=8.5)
    inset2.legend(fontsize=7.5, frameon=False, loc='upper left')
    inset2.spines[['top', 'right', 'left']].set_visible(False)

    ax = axes[2]
    raw = np.array([0.46, 0.27, 0.21, 0.06])
    improved = np.array([0.52, 0.23, 0.19, 0.06])
    xx = np.arange(4)
    width = 0.34
    ax.bar(xx - width / 2, raw, width, label='mô hình gốc', color='#cfd8e3', edgecolor='#51647b')
    ax.bar(xx + width / 2, improved, width, label='có phân phối thích nghi', color='#4c78a8')
    ax.set_xticks(xx)
    ax.set_xticklabels(['rất nhỏ', 'nhỏ', 'vừa', 'lớn'])
    ax.set_ylabel('tỉ lệ trong dữ liệu huấn luyện')
    ax.set_title('Bước 3. Phân phối huấn luyện dịch về phía nhiều mặt nhỏ hơn')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    ax.annotate('nhiều mẫu rất nhỏ hơn', xy=(0.18, improved[0]), xytext=(1.15, 0.57),
                arrowprops=dict(arrowstyle='->', color='#e45756', linewidth=1.6),
                fontsize=10, color='#e45756')
    ax.text(0.5, -0.20,
            'Cốt lõi của cơ chế này là: nếu mô hình đang học kém ở nhóm khuôn mặt rất nhỏ,\n'
            'xác suất lấy các tỉ lệ cắt tạo ra nhiều tình huống đó sẽ tăng lên ở epoch sau.',
            transform=ax.transAxes, ha='center', va='top', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_jsar_schematic(out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(16.3, 5.9), gridspec_kw={'width_ratios': [1.0, 1.0, 1.05]})

    def draw_assignment_panel(ax, title, subtitle, threshold_radius, center_radius, added_indices=None):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        anchors = [(0.22, 0.74), (0.50, 0.74), (0.78, 0.74),
                   (0.22, 0.50), (0.50, 0.50), (0.78, 0.50),
                   (0.22, 0.26), (0.50, 0.26), (0.78, 0.26)]

        gt = Rectangle((0.44, 0.44), 0.12, 0.12, linewidth=2, edgecolor='#333', facecolor='none')
        ax.add_patch(gt)
        ax.add_patch(Circle((0.50, 0.50), center_radius, facecolor='#fbeaea', edgecolor='none', alpha=0.70))
        ax.add_patch(Circle((0.50, 0.50), threshold_radius, facecolor='none', edgecolor='#e45756',
                            linewidth=1.8, linestyle='--'))

        base_indices = {4}
        added_indices = set(added_indices or [])
        for idx, (x, y) in enumerate(anchors):
            ax.add_patch(Circle((x, y), 0.030, facecolor='#d9d9d9', edgecolor='#777', linewidth=1.1))
            if idx in base_indices:
                ax.add_patch(Circle((x, y), 0.036, facecolor='#4c78a8', edgecolor='white', linewidth=1.6))
            elif idx in added_indices:
                ax.add_patch(Circle((x, y), 0.036, facecolor='#f58518', edgecolor='white', linewidth=1.6))

        ax.text(0.50, 0.94, title, ha='center', va='top', fontsize=12, fontweight='bold')
        ax.text(0.50, 0.885, subtitle, ha='center', va='top', fontsize=9.5, color='#555')
        ax.text(0.50, 0.14, 'hình chữ nhật đen: khuôn mặt rất nhỏ', ha='center', va='center', fontsize=9)
        ax.text(0.50, 0.08, 'vòng đỏ đứt: điều kiện IoU\nvùng hồng: điều kiện gần tâm',
                ha='center', va='center', fontsize=8.7)

        legend_y = 0.985
        ax.add_patch(Circle((0.18, legend_y), 0.015, facecolor='#4c78a8', edgecolor='white', linewidth=1))
        ax.text(0.22, legend_y, 'anchor dương ban đầu', va='center', fontsize=8.5)
        ax.add_patch(Circle((0.64, legend_y), 0.015, facecolor='#f58518', edgecolor='white', linewidth=1))
        ax.text(0.68, legend_y, 'anchor được thêm', va='center', fontsize=8.5)

    draw_assignment_panel(
        axes[0],
        'Bước 1. Gán nhãn ban đầu',
        'Khuôn mặt rất nhỏ thường chỉ nhận rất ít anchor dương',
        threshold_radius=0.15,
        center_radius=0.09,
        added_indices=[],
    )
    axes[0].text(0.50, 0.84, 'Ví dụ: chỉ có 1 anchor dương',
                 ha='center', va='top', fontsize=10, color='#4c78a8')

    draw_assignment_panel(
        axes[1],
        'Bước 2. Nới điều kiện cho đối tượng nhỏ',
        'Hạ ngưỡng IoU và nới vùng gần tâm cho mặt rất nhỏ',
        threshold_radius=0.24,
        center_radius=0.18,
        added_indices=[1, 3, 5, 7],
    )
    axes[1].text(0.50, 0.84, 'Nhiều anchor lân cận được giữ lại hơn',
                 ha='center', va='top', fontsize=10, color='#e45756')

    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Bước 3. Bổ sung anchor nếu vẫn còn thiếu', fontsize=12, pad=10)
    draw_box(ax, 0.08, 0.73, 0.84, 0.14,
             'Nếu số anchor dương của khuôn mặt rất nhỏ vẫn nhỏ hơn ngưỡng yêu cầu,\n'
             'hệ thống sẽ tìm thêm một số anchor nền lân cận.',
             '#eef4fb', fontsize=10)
    draw_box(ax, 0.08, 0.49, 0.84, 0.14,
             'Các anchor ứng viên được chấm điểm theo hai yếu tố:\n'
             'độ chồng lắp với khuôn mặt và khoảng cách tới khuôn mặt.',
             '#fdf4e7', fontsize=10)
    draw_box(ax, 0.08, 0.16, 0.84, 0.16,
             'Chọn thêm một vài anchor có điểm cao nhất vào tập dương.\n'
             'Nhờ đó, khuôn mặt rất nhỏ vẫn nhận đủ tín hiệu giám sát để học.',
             '#eef8e8', fontsize=10)
    draw_arrow(ax, 0.50, 0.73, 0.50, 0.63)
    draw_arrow(ax, 0.50, 0.49, 0.50, 0.39)

    score_x = np.array([0.16, 0.32, 0.48, 0.64, 0.80])
    score_h = [0.32, 0.48, 0.70, 0.57, 0.42]
    for idx, (x, h) in enumerate(zip(score_x, score_h)):
        color = '#f58518' if idx < 3 else '#d9d9d9'
        ax.add_patch(Rectangle((x - 0.05, 0.36), 0.10, h * 0.18, facecolor=color, edgecolor='#666'))
        ax.text(x, 0.34, f'a{idx+1}', ha='center', va='top', fontsize=8.5)
    ax.text(0.50, 0.31, 'ví dụ xếp hạng anchor ứng viên theo điểm số', ha='center', va='top', fontsize=9)
    ax.text(0.50, 0.05,
            'Tóm lại, cơ chế này làm tăng số anchor dương đúng ở những khuôn mặt rất nhỏ,\n'
            'nơi quy tắc gán nhãn ban đầu của SCRFD thường còn quá khắt khe.',
            ha='center', va='bottom', fontsize=9.5, wrap=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary_json(runs: List[Dict], out_path: str):
    payload = {
        'default_source_scale_set': {
            'baseline_metrics': runs[0]['baseline_metrics'],
            'improved_metrics': runs[0]['improved_metrics'],
            'baseline_scale_stats': runs[0]['baseline_scale_stats'],
            'improved_scale_stats': runs[0]['improved_scale_stats'],
            'face_size_baseline': runs[0]['face_size']['baseline_summary'],
            'face_size_improved': runs[0]['face_size']['improved_summary'],
            'jsar': runs[0]['jsar'],
        },
        'paper_sr12_scale_set': {
            'baseline_metrics': runs[1]['baseline_metrics'],
            'improved_metrics': runs[1]['improved_metrics'],
            'baseline_scale_stats': runs[1]['baseline_scale_stats'],
            'improved_scale_stats': runs[1]['improved_scale_stats'],
            'face_size_baseline': runs[1]['face_size']['baseline_summary'],
            'face_size_improved': runs[1]['face_size']['improved_summary'],
            'jsar': runs[1]['jsar'],
        },
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build report assets that compare default SCRFD scale set and paper SR12 scale set.')
    parser.add_argument(
        '--experiment-root',
        required=True,
        help='Root directory that contains default_source_scale_set and paper_sr12_scale_set.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Directory for generated report assets. Defaults to <experiment_root>/report_assets.')
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_root = osp.abspath(args.experiment_root)
    out_dir = osp.abspath(args.out_dir or osp.join(experiment_root, 'report_assets'))
    ensure_dir(out_dir)

    default_run = collect_run_summary('default_source_scale_set', get_run_paths(experiment_root, 'default_source_scale_set'))
    paper_run = collect_run_summary('paper_sr12_scale_set', get_run_paths(experiment_root, 'paper_sr12_scale_set'))
    runs = [default_run, paper_run]

    plot_metric_bars(runs, osp.join(out_dir, 'metrics_by_scale_set.png'))
    plot_asr_delta(runs, osp.join(out_dir, 'asr_delta_by_scale_set.png'))
    plot_scale_pool_geometry(runs, osp.join(out_dir, 'scale_pool_geometry.png'))
    plot_distribution_pressure(runs, osp.join(out_dir, 'distribution_pressure.png'))
    plot_jsar_tiny_supervision(runs, osp.join(out_dir, 'jsar_tiny_supervision.png'))
    plot_hard_size_gain(default_run, osp.join(out_dir, 'default_hard_size_gain.png'),
                        'Default scale set: hard-subset recall gain by face-size bin')
    plot_hard_size_gain(paper_run, osp.join(out_dir, 'paper_hard_size_gain.png'),
                        'Paper SR12 scale set: hard-subset recall gain by face-size bin')
    plot_hard_size_gain_compare(runs, osp.join(out_dir, 'hard_size_gain_by_scale_set.png'))
    plot_asr_schematic(osp.join(out_dir, 'asr_schematic.png'))
    plot_jsar_schematic(osp.join(out_dir, 'jsar_schematic.png'))
    write_summary_json(runs, osp.join(out_dir, 'report_summary.json'))

    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    safe_message = f'Wrote report assets to {out_dir}'.encode(encoding, errors='backslashreplace').decode(encoding)
    print(safe_message)


if __name__ == '__main__':
    main()
