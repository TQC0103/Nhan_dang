import argparse
import csv
import json
import os
import os.path as osp

import numpy as np

from mmdet.core.evaluation.widerface import get_preds, get_widerface_gts


DEFAULT_SIZE_BINS = [0.0, 8.0, 16.0, 32.0, 64.0, 128.0, 1e8]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze why hard subset performance differs between two WIDERFace models.')
    parser.add_argument('--baseline', required=True,
                        help='Baseline result dir or prediction dir. If a result dir is given, predictions/ is used.')
    parser.add_argument('--improved', required=True,
                        help='Improved result dir or prediction dir. If a result dir is given, predictions/ is used.')
    parser.add_argument('--gt-dir', required=True, help='WIDERFace gt directory containing *.mat files.')
    parser.add_argument('--out-dir', required=True, help='Output directory.')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU threshold for matching.')
    parser.add_argument('--size-bins', type=float, nargs='+', default=DEFAULT_SIZE_BINS,
                        help='Hard-face sqrt-area size bins in pixels.')
    parser.add_argument('--topk-images', type=int, default=30,
                        help='Number of best/worst per-image cases to export.')
    return parser.parse_args()


def resolve_prediction_dir(path):
    predictions_dir = osp.join(path, 'predictions')
    if osp.isdir(predictions_dir):
        return predictions_dir
    return path


def resolve_summary(path):
    summary_path = osp.join(path, 'results_summary.json')
    if osp.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as infile:
            return json.load(infile)
    aps_path = osp.join(path, 'aps')
    if osp.isfile(aps_path):
        with open(aps_path, 'r', encoding='utf-8') as infile:
            raw = infile.readline().strip()
        values = [float(item) for item in raw.split(',') if item.strip()]
        if len(values) == 3:
            return {
                'easy_AP': values[0],
                'medium_AP': values[1],
                'hard_AP': values[2],
                'mAP': sum(values) / 3.0,
            }
    return None


def bbox_iou_matrix_xywh(pred_xywh, gt_xyxy):
    if pred_xywh.shape[0] == 0 or gt_xyxy.shape[0] == 0:
        return np.zeros((pred_xywh.shape[0], gt_xyxy.shape[0]), dtype=np.float32)
    pred_xyxy = pred_xywh[:, :4].copy()
    pred_xyxy[:, 2] = pred_xyxy[:, 0] + pred_xyxy[:, 2]
    pred_xyxy[:, 3] = pred_xyxy[:, 1] + pred_xyxy[:, 3]
    ious = np.zeros((pred_xyxy.shape[0], gt_xyxy.shape[0]), dtype=np.float32)
    for gt_idx in range(gt_xyxy.shape[0]):
        gt = gt_xyxy[gt_idx]
        x1 = np.maximum(pred_xyxy[:, 0], gt[0])
        y1 = np.maximum(pred_xyxy[:, 1], gt[1])
        x2 = np.minimum(pred_xyxy[:, 2], gt[2])
        y2 = np.minimum(pred_xyxy[:, 3], gt[3])
        w = np.maximum(0.0, x2 - x1 + 1.0)
        h = np.maximum(0.0, y2 - y1 + 1.0)
        inter = w * h
        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0] + 1.0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1] + 1.0)
        gt_area = (gt[2] - gt[0] + 1.0) * (gt[3] - gt[1] + 1.0)
        union = pred_area + gt_area - inter
        valid = union > 0
        ious[valid, gt_idx] = inter[valid] / union[valid]
    return ious


def greedy_match(pred_xywh, gt_xyxy, iou_thr):
    num_pred = pred_xywh.shape[0]
    num_gt = gt_xyxy.shape[0]
    if num_pred == 0 or num_gt == 0:
        return {
            'matched_gt_mask': np.zeros((num_gt,), dtype=bool),
            'matched_pred_mask': np.zeros((num_pred,), dtype=bool),
            'matched_gt_indices': np.full((num_pred,), -1, dtype=np.int64),
            'ious': np.zeros((num_pred,), dtype=np.float32),
        }
    order = np.argsort(pred_xywh[:, 4])[::-1]
    iou_matrix = bbox_iou_matrix_xywh(pred_xywh, gt_xyxy)
    matched_gt_mask = np.zeros((num_gt,), dtype=bool)
    matched_pred_mask = np.zeros((num_pred,), dtype=bool)
    matched_gt_indices = np.full((num_pred,), -1, dtype=np.int64)
    matched_ious = np.zeros((num_pred,), dtype=np.float32)
    for pred_idx in order:
        gt_ious = iou_matrix[pred_idx]
        if gt_ious.shape[0] == 0:
            continue
        best_gt = int(np.argmax(gt_ious))
        best_iou = float(gt_ious[best_gt])
        if best_iou < iou_thr or matched_gt_mask[best_gt]:
            continue
        matched_gt_mask[best_gt] = True
        matched_pred_mask[pred_idx] = True
        matched_gt_indices[pred_idx] = best_gt
        matched_ious[pred_idx] = best_iou
    return {
        'matched_gt_mask': matched_gt_mask,
        'matched_pred_mask': matched_pred_mask,
        'matched_gt_indices': matched_gt_indices,
        'ious': matched_ious,
    }


def face_sizes_from_gt(gt_xyxy):
    if gt_xyxy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    widths = np.maximum(1e-6, gt_xyxy[:, 2] - gt_xyxy[:, 0])
    heights = np.maximum(1e-6, gt_xyxy[:, 3] - gt_xyxy[:, 1])
    return np.sqrt(widths * heights)


def assign_bin_indices(values, bin_edges):
    values = np.asarray(values, dtype=np.float32)
    indices = np.full((values.shape[0],), len(bin_edges) - 2, dtype=np.int64)
    for idx in range(len(bin_edges) - 1):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if right >= 1e8:
            mask = values >= left
        else:
            mask = (values >= left) & (values < right)
        indices[mask] = idx
    return indices


def init_bin_stats(bin_edges):
    stats = []
    for idx in range(len(bin_edges) - 1):
        stats.append({
            'bin_index': idx,
            'bin_left': float(bin_edges[idx]),
            'bin_right': float(bin_edges[idx + 1]),
            'gt_count': 0,
            'baseline_matched_gt': 0,
            'improved_matched_gt': 0,
            'baseline_matched_pred': 0,
            'improved_matched_pred': 0,
            'baseline_total_pred': 0,
            'improved_total_pred': 0,
        })
    return stats


def bin_label(left, right):
    if right >= 1e8:
        return '[{:.0f}, inf)'.format(left)
    return '[{:.0f}, {:.0f})'.format(left, right)


def update_bin_stats(bin_stats, gt_sizes, baseline_match, improved_match,
                     baseline_pred_count, improved_pred_count):
    if gt_sizes.shape[0] == 0:
        return
    gt_bins = assign_bin_indices(gt_sizes, [item['bin_left'] for item in bin_stats] + [bin_stats[-1]['bin_right']])
    for bin_idx in range(len(bin_stats)):
        gt_mask = gt_bins == bin_idx
        gt_count = int(gt_mask.sum())
        if gt_count == 0:
            continue
        bin_stats[bin_idx]['gt_count'] += gt_count
        bin_stats[bin_idx]['baseline_matched_gt'] += int(baseline_match['matched_gt_mask'][gt_mask].sum())
        bin_stats[bin_idx]['improved_matched_gt'] += int(improved_match['matched_gt_mask'][gt_mask].sum())
    baseline_matched_bins = gt_bins[baseline_match['matched_gt_indices'][baseline_match['matched_pred_mask']]]
    improved_matched_bins = gt_bins[improved_match['matched_gt_indices'][improved_match['matched_pred_mask']]]
    for bin_idx in range(len(bin_stats)):
        bin_stats[bin_idx]['baseline_matched_pred'] += int((baseline_matched_bins == bin_idx).sum())
        bin_stats[bin_idx]['improved_matched_pred'] += int((improved_matched_bins == bin_idx).sum())
        bin_stats[bin_idx]['baseline_total_pred'] += baseline_pred_count
        bin_stats[bin_idx]['improved_total_pred'] += improved_pred_count


def to_serializable_image_row(row):
    return {
        'image_id': row['image_id'],
        'hard_gt_count': row['hard_gt_count'],
        'baseline_recall': round(row['baseline_recall'], 6),
        'improved_recall': round(row['improved_recall'], 6),
        'recall_delta': round(row['recall_delta'], 6),
        'baseline_precision_proxy': round(row['baseline_precision_proxy'], 6),
        'improved_precision_proxy': round(row['improved_precision_proxy'], 6),
        'precision_proxy_delta': round(row['precision_proxy_delta'], 6),
        'baseline_pred_count': row['baseline_pred_count'],
        'improved_pred_count': row['improved_pred_count'],
        'baseline_tp': row['baseline_tp'],
        'improved_tp': row['improved_tp'],
        'baseline_fp': row['baseline_fp'],
        'improved_fp': row['improved_fp'],
        'mean_hard_face_size': round(row['mean_hard_face_size'], 4),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baseline_dir = resolve_prediction_dir(args.baseline)
    improved_dir = resolve_prediction_dir(args.improved)
    if not osp.isdir(baseline_dir):
        raise FileNotFoundError('Baseline prediction directory not found: {}'.format(baseline_dir))
    if not osp.isdir(improved_dir):
        raise FileNotFoundError('Improved prediction directory not found: {}'.format(improved_dir))

    baseline_preds = get_preds(baseline_dir)
    improved_preds = get_preds(improved_dir)
    _, _, hard_gts = get_widerface_gts(args.gt_dir)

    bin_stats = init_bin_stats(args.size_bins)
    image_rows = []

    total_hard_gt = 0
    baseline_total_tp = 0
    improved_total_tp = 0
    baseline_total_pred = 0
    improved_total_pred = 0

    for event_name in sorted(hard_gts.keys()):
        event_gts = hard_gts[event_name]
        baseline_event = baseline_preds.get(event_name, {})
        improved_event = improved_preds.get(event_name, {})
        for image_name in sorted(event_gts.keys()):
            gt_boxes = event_gts[image_name].astype(np.float32)
            baseline_boxes = baseline_event.get(image_name, np.empty((0, 5), dtype=np.float32))
            improved_boxes = improved_event.get(image_name, np.empty((0, 5), dtype=np.float32))
            if baseline_boxes.ndim == 1:
                baseline_boxes = baseline_boxes.reshape(1, -1)
            if improved_boxes.ndim == 1:
                improved_boxes = improved_boxes.reshape(1, -1)

            baseline_match = greedy_match(baseline_boxes, gt_boxes, args.iou_thr)
            improved_match = greedy_match(improved_boxes, gt_boxes, args.iou_thr)
            gt_sizes = face_sizes_from_gt(gt_boxes)
            update_bin_stats(
                bin_stats,
                gt_sizes,
                baseline_match,
                improved_match,
                baseline_boxes.shape[0],
                improved_boxes.shape[0],
            )

            hard_gt_count = int(gt_boxes.shape[0])
            baseline_tp = int(baseline_match['matched_gt_mask'].sum())
            improved_tp = int(improved_match['matched_gt_mask'].sum())
            baseline_pred_count = int(baseline_boxes.shape[0])
            improved_pred_count = int(improved_boxes.shape[0])
            baseline_fp = max(baseline_pred_count - baseline_tp, 0)
            improved_fp = max(improved_pred_count - improved_tp, 0)

            total_hard_gt += hard_gt_count
            baseline_total_tp += baseline_tp
            improved_total_tp += improved_tp
            baseline_total_pred += baseline_pred_count
            improved_total_pred += improved_pred_count

            baseline_recall = baseline_tp / hard_gt_count if hard_gt_count > 0 else 0.0
            improved_recall = improved_tp / hard_gt_count if hard_gt_count > 0 else 0.0
            baseline_precision_proxy = baseline_tp / baseline_pred_count if baseline_pred_count > 0 else 0.0
            improved_precision_proxy = improved_tp / improved_pred_count if improved_pred_count > 0 else 0.0

            image_rows.append({
                'image_id': '{}/{}'.format(event_name, image_name),
                'hard_gt_count': hard_gt_count,
                'baseline_recall': baseline_recall,
                'improved_recall': improved_recall,
                'recall_delta': improved_recall - baseline_recall,
                'baseline_precision_proxy': baseline_precision_proxy,
                'improved_precision_proxy': improved_precision_proxy,
                'precision_proxy_delta': improved_precision_proxy - baseline_precision_proxy,
                'baseline_pred_count': baseline_pred_count,
                'improved_pred_count': improved_pred_count,
                'baseline_tp': baseline_tp,
                'improved_tp': improved_tp,
                'baseline_fp': baseline_fp,
                'improved_fp': improved_fp,
                'mean_hard_face_size': float(gt_sizes.mean()) if gt_sizes.shape[0] > 0 else 0.0,
            })

    baseline_summary = resolve_summary(args.baseline)
    improved_summary = resolve_summary(args.improved)

    aggregate = {
        'hard_gt_count': total_hard_gt,
        'baseline_hard_recall_proxy': baseline_total_tp / max(total_hard_gt, 1),
        'improved_hard_recall_proxy': improved_total_tp / max(total_hard_gt, 1),
        'hard_recall_proxy_delta': improved_total_tp / max(total_hard_gt, 1) - baseline_total_tp / max(total_hard_gt, 1),
        'baseline_precision_proxy': baseline_total_tp / max(baseline_total_pred, 1),
        'improved_precision_proxy': improved_total_tp / max(improved_total_pred, 1),
        'precision_proxy_delta': improved_total_tp / max(improved_total_pred, 1) - baseline_total_tp / max(baseline_total_pred, 1),
        'baseline_pred_count': baseline_total_pred,
        'improved_pred_count': improved_total_pred,
        'baseline_tp': baseline_total_tp,
        'improved_tp': improved_total_tp,
        'baseline_fp': max(baseline_total_pred - baseline_total_tp, 0),
        'improved_fp': max(improved_total_pred - improved_total_tp, 0),
    }

    size_bin_rows = []
    for row in bin_stats:
        gt_count = max(row['gt_count'], 1)
        baseline_recall = row['baseline_matched_gt'] / gt_count
        improved_recall = row['improved_matched_gt'] / gt_count
        size_bin_rows.append({
            'size_bin': bin_label(row['bin_left'], row['bin_right']),
            'gt_count': row['gt_count'],
            'baseline_recall_proxy': baseline_recall,
            'improved_recall_proxy': improved_recall,
            'recall_proxy_delta': improved_recall - baseline_recall,
            'baseline_matched_gt': row['baseline_matched_gt'],
            'improved_matched_gt': row['improved_matched_gt'],
            'baseline_matched_pred': row['baseline_matched_pred'],
            'improved_matched_pred': row['improved_matched_pred'],
        })

    image_rows_sorted = sorted(image_rows, key=lambda item: (item['recall_delta'], item['hard_gt_count']), reverse=True)
    top_improved = [to_serializable_image_row(item) for item in image_rows_sorted[:args.topk_images]]
    top_regressed = [to_serializable_image_row(item) for item in sorted(
        image_rows, key=lambda item: (item['recall_delta'], -item['hard_gt_count']))[:args.topk_images]]

    output = {
        'baseline_summary': baseline_summary,
        'improved_summary': improved_summary,
        'aggregate': aggregate,
        'size_bins': size_bin_rows,
        'top_improved_images': top_improved,
        'top_regressed_images': top_regressed,
    }

    json_path = osp.join(args.out_dir, 'hard_subset_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as outfile:
        json.dump(output, outfile, indent=2, sort_keys=True)

    csv_path = osp.join(args.out_dir, 'hard_size_bins.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(size_bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(size_bin_rows)

    image_csv_path = osp.join(args.out_dir, 'top_improved_images.csv')
    with open(image_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(top_improved[0].keys()) if top_improved else ['image_id'])
        writer.writeheader()
        if top_improved:
            writer.writerows(top_improved)

    regressed_csv_path = osp.join(args.out_dir, 'top_regressed_images.csv')
    with open(regressed_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(top_regressed[0].keys()) if top_regressed else ['image_id'])
        writer.writeheader()
        if top_regressed:
            writer.writerows(top_regressed)

    md_lines = [
        '# Hard Subset Analysis',
        '',
        '## Aggregate',
        '',
        '| Metric | Baseline | Improved | Delta |',
        '| --- | ---: | ---: | ---: |',
    ]
    if baseline_summary and improved_summary:
        md_lines.append(
            '| hard_AP | {:.4f} | {:.4f} | {:+.4f} |'.format(
                float(baseline_summary.get('hard_AP', 0.0)),
                float(improved_summary.get('hard_AP', 0.0)),
                float(improved_summary.get('hard_AP', 0.0)) - float(baseline_summary.get('hard_AP', 0.0)),
            ))
        md_lines.append(
            '| mAP | {:.4f} | {:.4f} | {:+.4f} |'.format(
                float(baseline_summary.get('mAP', 0.0)),
                float(improved_summary.get('mAP', 0.0)),
                float(improved_summary.get('mAP', 0.0)) - float(baseline_summary.get('mAP', 0.0)),
            ))
    md_lines.append(
        '| hard_recall_proxy | {:.4f} | {:.4f} | {:+.4f} |'.format(
            aggregate['baseline_hard_recall_proxy'],
            aggregate['improved_hard_recall_proxy'],
            aggregate['hard_recall_proxy_delta'],
        ))
    md_lines.append(
        '| precision_proxy | {:.4f} | {:.4f} | {:+.4f} |'.format(
            aggregate['baseline_precision_proxy'],
            aggregate['improved_precision_proxy'],
            aggregate['precision_proxy_delta'],
        ))
    md_lines.append(
        '| prediction_count | {} | {} | {:+d} |'.format(
            aggregate['baseline_pred_count'],
            aggregate['improved_pred_count'],
            aggregate['improved_pred_count'] - aggregate['baseline_pred_count'],
        ))
    md_lines.extend([
        '',
        '## Hard Recall By Size Bin',
        '',
        '| Size Bin | GT | Baseline Recall Proxy | Improved Recall Proxy | Delta |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for row in size_bin_rows:
        md_lines.append(
            '| {} | {} | {:.4f} | {:.4f} | {:+.4f} |'.format(
                row['size_bin'],
                row['gt_count'],
                row['baseline_recall_proxy'],
                row['improved_recall_proxy'],
                row['recall_proxy_delta'],
            ))
    md_path = osp.join(args.out_dir, 'hard_subset_analysis.md')
    with open(md_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(md_lines) + '\n')

    print('Wrote hard subset analysis to {}'.format(args.out_dir))
    print('Prediction dirs:')
    print('  baseline: {}'.format(baseline_dir))
    print('  improved: {}'.format(improved_dir))


if __name__ == '__main__':
    main()
