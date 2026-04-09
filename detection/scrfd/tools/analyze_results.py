"""
Training Log Analysis Script for SCRFD Experiments

This script analyzes training logs and generates visualizations for reports.

Usage:
    python tools/analyze_results.py \
        --exp-dir work_dirs/scrfd_500m_kd_10g \
        --output plots/

    python tools/analyze_results.py \
        --compare \
        --exp-dirs work_dirs/kd_10g work_dirs/kd_25g work_dirs/kd_retina \
        --labels "SCRFD-10G→500M" "SCRFD-2.5G→500M" "RetinaFace→500M" \
        --output comparison/
"""

import argparse
import os
import json
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def parse_log_file(log_file):
    """Parse mmdet log file and extract loss values."""
    losses = {
        'iter': [], 'epoch': [],
        'loss_cls': [], 'loss_bbox': [], 'loss_kps': [],
        'loss_cls_distill': [], 'loss_bbox_distill': [], 'loss_distill': []
    }

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse iteration
            if 'iter:' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('iter:'):
                        try:
                            losses['iter'].append(int(parts[i+1]))
                        except (IndexError, ValueError):
                            pass

                    # Parse losses
                    for key in ['loss_cls', 'loss_bbox', 'loss_kps',
                               'loss_cls_distill', 'loss_bbox_distill', 'loss_distill']:
                        if f'{key}:' in part:
                            try:
                                val_str = parts[i+1]
                                losses[key].append(float(val_str))
                            except (IndexError, ValueError):
                                pass

    return losses


def parse_csv_log(csv_file):
    """Parse the CSV log emitted by KDTensorboardLoggerHook."""
    losses = {
        'iter': [], 'epoch': [],
        'loss_cls': [], 'loss_bbox': [], 'loss_kps': [],
        'loss_cls_distill': [], 'loss_bbox_distill': [], 'loss_distill': []
    }

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in losses.keys():
                value = row.get(key, '')
                if value in ('', None):
                    continue
                if key in ('iter', 'epoch'):
                    losses[key].append(int(float(value)))
                else:
                    losses[key].append(float(value))

    return losses


def plot_loss_curves(losses, output_path, title="Training Loss Curves"):
    """Generate loss curve plots."""
    if not losses['iter']:
        print(f"No data found in {output_path}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Task losses
    ax1 = axes[0, 0]
    if losses['loss_cls']:
        ax1.plot(losses['iter'], losses['loss_cls'], label='cls', alpha=0.8)
    if losses['loss_bbox']:
        ax1.plot(losses['iter'], losses['loss_bbox'], label='bbox', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Task Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distillation losses
    ax2 = axes[0, 1]
    if losses.get('loss_cls_distill') and any(losses['loss_cls_distill']):
        ax2.plot(losses['iter'], losses['loss_cls_distill'], label='cls_distill', alpha=0.8)
    if losses.get('loss_bbox_distill') and any(losses['loss_bbox_distill']):
        ax2.plot(losses['iter'], losses['loss_bbox_distill'], label='bbox_distill', alpha=0.8)
    if losses.get('loss_distill') and any(losses['loss_distill']):
        ax2.plot(losses['iter'], losses['loss_distill'], label='distill_total', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Distillation Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Loss ratios
    ax3 = axes[1, 0]
    if losses.get('loss_cls') and losses.get('loss_cls_distill'):
        cls_total = [c + d for c, d in zip(losses['loss_cls'], losses['loss_cls_distill']) if d > 0]
        if cls_total:
            ratios = [d/c if c > 0 else 0 for c, d in zip(losses['loss_cls'], losses['loss_cls_distill']) if d > 0]
            iters = [i for i, d in enumerate(losses['loss_cls_distill']) if d > 0]
            if ratios:
                ax3.plot(iters, ratios, label='cls_distill_ratio', alpha=0.8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Ratio')
    ax3.set_title('Distillation/Task Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Total loss
    ax4 = axes[1, 1]
    if losses.get('loss_cls') and losses.get('loss_bbox'):
        total = [c + b for c, b in zip(losses['loss_cls'], losses['loss_bbox'])]
        ax4.plot(losses['iter'][:len(total)], total, label='total_task', alpha=0.8)
    if losses.get('loss_distill') and any(losses['loss_distill']):
        ax4.plot(losses['iter'], losses['loss_distill'], label='total_distill', alpha=0.8)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss')
    ax4.set_title('Total Loss Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {output_path}")


def compare_experiments(exp_dirs, labels, output_dir):
    """Compare multiple experiments."""
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Compare Easy AP
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['easy_AP', 'medium_AP', 'hard_AP']
    metric_names = ['Easy AP', 'Medium AP', 'Hard AP']

    for ax, metric, name in zip(axes, metrics, metric_names):
        values = []
        for exp_dir in exp_dirs:
            result_file = os.path.join(exp_dir.replace('work_dirs', 'results'),
                                      'results_summary.json')
            if os.path.exists(result_file):
                with open(result_file) as f:
                    data = json.load(f)
                    values.append(data.get(metric, 0))
            else:
                values.append(0)

        bars = ax.bar(labels, values, color=colors[:len(labels)])
        ax.set_ylabel(name)
        ax.set_title(f'{name} Comparison')
        ax.set_ylim(0, 1.0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ap_comparison.png'), dpi=150)
    plt.close()
    print(f"AP comparison saved to {output_dir}/ap_comparison.png")


def generate_summary_table(exp_dirs, output_file):
    """Generate LaTeX table for report."""
    rows = []

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)

        # Try to load results
        result_file = os.path.join(exp_dir.replace('work_dirs', 'results'),
                                  'results_summary.json')

        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            easy = data.get('easy_AP', 0)
            medium = data.get('medium_AP', 0)
            hard = data.get('hard_AP', 0)
            mAP = data.get('mAP', 0)
        else:
            easy = medium = hard = mAP = '-'

        # Try to get config name
        config_file = os.path.join(exp_dir, '*.py')
        config_files = glob.glob(config_file)
        config_name = os.path.basename(config_files[0]) if config_files else exp_name

        rows.append((config_name, f'{easy:.4f}' if isinstance(easy, float) else easy,
                    f'{medium:.4f}' if isinstance(medium, float) else medium,
                    f'{hard:.4f}' if isinstance(hard, float) else hard,
                    f'{mAP:.4f}' if isinstance(mAP, float) else mAP))

    # Generate LaTeX table
    latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Experimental Results on WIDERFace Dataset}\n"
        "\\begin{tabular}{|l|c|c|c|c|}\n"
        "\\hline\n"
        "\\textbf{Model} & \\textbf{Easy} & \\textbf{Medium} & "
        "\\textbf{Hard} & \\textbf{mAP} \\\\ \\hline\n"
    )
    for row in rows:
        latex += (
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} "
            "\\\\ \\hline\n"
        )

    latex += "\\end{tabular}\n\\end{table}\n"

    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to {output_file}")
    return latex


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SCRFD experiment results')
    parser.add_argument('--exp-dir', type=str, help='Experiment directory (work_dirs/...)')
    parser.add_argument('--exp-dirs', nargs='+', help='Multiple experiment directories')
    parser.add_argument('--labels', nargs='+', help='Labels for comparison')
    parser.add_argument('--output', type=str, default='analysis', help='Output directory')
    parser.add_argument('--compare', action='store_true', help='Compare multiple experiments')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.compare and args.exp_dirs:
        # Compare mode
        compare_experiments(args.exp_dirs, args.labels or [''] * len(args.exp_dirs), args.output)
        generate_summary_table(args.exp_dirs, os.path.join(args.output, 'results_table.tex'))
    elif args.exp_dir:
        # Single experiment mode
        csv_log = os.path.join(args.exp_dir, 'loss_log.csv')
        if os.path.exists(csv_log):
            print(f"Analyzing {csv_log}")
            losses = parse_csv_log(csv_log)
            plot_loss_curves(
                losses,
                os.path.join(args.output, 'loss_curves.png'),
                title=f"Training Curves - {os.path.basename(args.exp_dir)}")
        else:
            log_files = glob.glob(os.path.join(args.exp_dir, '*.log'))
            if log_files:
                log_file = log_files[0]
                print(f"Analyzing {log_file}")
                losses = parse_log_file(log_file)
                plot_loss_curves(
                    losses,
                    os.path.join(args.output, 'loss_curves.png'),
                    title=f"Training Curves - {os.path.basename(args.exp_dir)}")
            else:
                print(f"No log files found in {args.exp_dir}")
    else:
        print("Please specify --exp-dir or use --compare with --exp-dirs")


if __name__ == '__main__':
    main()
