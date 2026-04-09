import argparse
import csv
import io
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SCRFD search/NAS results')
    parser.add_argument(
        '--group',
        required=True,
        help='config group name or relative path, e.g. scrfdgen2.5g or '
             'configs/scrfdgen500m_kernel')
    parser.add_argument(
        '--result-dir',
        default='wouts',
        help='directory containing WIDERFace evaluation outputs')
    parser.add_argument(
        '--prefix',
        default=None,
        help='config/result prefix. Defaults to basename(group)')
    parser.add_argument(
        '--idx-from',
        type=int,
        default=0,
        help='first config index to inspect')
    parser.add_argument(
        '--idx-to',
        type=int,
        default=320,
        help='exclusive upper bound of config indices')
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=3,
        default=(3, 480, 640),
        metavar=('C', 'H', 'W'),
        help='input shape used for FLOPs computation')
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='number of best models to highlight')
    parser.add_argument(
        '--score-key',
        choices=['easy', 'medium', 'hard', 'mean'],
        default='hard',
        help='metric used to rank models')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='where to save plots/stat files. Defaults to '
             '<result-dir>/<prefix>_viz')
    return parser.parse_args()


def resolve_group(group):
    normalized = group.replace('\\', '/').rstrip('/')
    if normalized.startswith('configs/'):
        config_dir = normalized
        group_name = osp.basename(normalized)
    else:
        config_dir = osp.join('configs', normalized)
        group_name = osp.basename(normalized)
    return config_dir, group_name


def get_flops(cfg, input_shape):
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is not supported for {}.'.format(
                model.__class__.__name__))

    buf = io.StringIO()
    all_flops, params = get_model_complexity_info(
        model,
        tuple(input_shape),
        print_per_layer_stat=True,
        as_strings=False,
        ost=buf)
    lines = buf.getvalue().split('\n')
    names = ['(stem)', '(layer1)', '(layer2)', '(layer3)', '(layer4)',
             '(neck)', '(bbox_head)']
    name_ptr = 0
    line_num = 0
    parsed_flops = []
    while name_ptr < len(names) and line_num + 1 < len(lines):
        line = lines[line_num].strip()
        if line.startswith(names[name_ptr]):
            parsed_flops.append(
                float(lines[line_num + 1].split(',')[2].strip().split(' ')[0]))
            name_ptr += 1
        line_num += 1

    if len(parsed_flops) != len(names):
        raise RuntimeError('Could not parse per-stage FLOPs for config.')

    backbone_flops = np.array(parsed_flops[:-2], dtype=np.float32)
    neck_flops = float(parsed_flops[-2])
    head_flops = float(parsed_flops[-1])
    return all_flops / 1e9, backbone_flops, neck_flops, head_flops


def load_aps(aps_file):
    if not osp.exists(aps_file):
        return None
    with open(aps_file, 'r') as file_obj:
        values = file_obj.readline().strip().split(',')
    if len(values) < 3:
        return None
    return [float(values[0]), float(values[1]), float(values[2])]


def extract_backbone_metadata(cfg):
    backbone = cfg.model.backbone
    block_cfg = backbone.get('block_cfg', {})
    metadata = {
        'backbone_type': backbone.get('type', ''),
        'stage_blocks': list(block_cfg.get('stage_blocks', [])),
        'stage_planes': list(block_cfg.get('stage_planes', [])),
        'stage_kernel_sizes': list(block_cfg.get('stage_kernel_sizes', [])),
        'stem_kernel_size': block_cfg.get('stem_kernel_size'),
        'stem_dw_kernel_size': block_cfg.get('stem_dw_kernel_size'),
    }
    return metadata


def score_value(record, score_key):
    if score_key == 'easy':
        return record['aps'][0]
    if score_key == 'medium':
        return record['aps'][1]
    if score_key == 'hard':
        return record['aps'][2]
    return float(np.mean(record['aps']))


def collect_records(config_dir, group_name, prefix, result_dir, idx_from, idx_to,
                    input_shape):
    records = []
    for idx in range(idx_from, idx_to):
        config_file = osp.join(config_dir, '{}_{}.py'.format(prefix, idx))
        if not osp.exists(config_file):
            continue

        aps_file = osp.join(result_dir, group_name, '{}_{}'.format(prefix, idx), 'aps')
        aps = load_aps(aps_file)
        if aps is None:
            continue

        cfg = Config.fromfile(config_file)
        all_flops, backbone_flops, neck_flops, head_flops = get_flops(
            cfg, input_shape)
        metadata = extract_backbone_metadata(cfg)
        records.append({
            'idx': idx,
            'name': '{}_{}'.format(prefix, idx),
            'config_file': config_file,
            'aps': aps,
            'all_flops': float(all_flops),
            'backbone_flops': [float(x) for x in backbone_flops],
            'neck_flops': float(neck_flops),
            'head_flops': float(head_flops),
            'metadata': metadata,
        })

    return records


def write_stats(records, output_dir):
    jsonl_file = osp.join(output_dir, 'search_stats.jsonl')
    csv_file = osp.join(output_dir, 'search_stats.csv')

    with open(jsonl_file, 'w', encoding='utf-8') as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + '\n')

    headers = [
        'idx', 'name', 'easy_ap', 'medium_ap', 'hard_ap', 'mean_ap',
        'all_flops', 'neck_flops', 'head_flops',
        'stem_kernel_size', 'stem_dw_kernel_size',
        'stage_blocks', 'stage_planes', 'stage_kernel_sizes',
    ]
    with open(csv_file, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(headers)
        for record in records:
            writer.writerow([
                record['idx'],
                record['name'],
                record['aps'][0],
                record['aps'][1],
                record['aps'][2],
                float(np.mean(record['aps'])),
                record['all_flops'],
                record['neck_flops'],
                record['head_flops'],
                record['metadata']['stem_kernel_size'],
                record['metadata']['stem_dw_kernel_size'],
                '-'.join(map(str, record['metadata']['stage_blocks'])),
                '-'.join(map(str, record['metadata']['stage_planes'])),
                '-'.join(map(str, record['metadata']['stage_kernel_sizes'])),
            ])

    return jsonl_file, csv_file


def plot_ap_curves(records, output_dir):
    sorted_records = sorted(records, key=lambda item: item['idx'])
    indices = [record['idx'] for record in sorted_records]
    easy = [record['aps'][0] for record in sorted_records]
    medium = [record['aps'][1] for record in sorted_records]
    hard = [record['aps'][2] for record in sorted_records]

    plt.figure(figsize=(12, 6))
    plt.plot(indices, easy, label='Easy AP', linewidth=2)
    plt.plot(indices, medium, label='Medium AP', linewidth=2)
    plt.plot(indices, hard, label='Hard AP', linewidth=2)
    plt.xlabel('Candidate Index')
    plt.ylabel('AP')
    plt.title('SCRFD Search Accuracy by Candidate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, 'ap_vs_candidate.png'), dpi=200)
    plt.close()


def plot_flops_vs_ap(records, output_dir, score_key):
    plt.figure(figsize=(10, 7))
    hard_scores = [record['aps'][2] for record in records]
    sizes = [120 + 90 * record['neck_flops'] / max(record['all_flops'], 1e-6)
             for record in records]
    scatter = plt.scatter(
        [record['all_flops'] for record in records],
        [score_value(record, score_key) for record in records],
        c=hard_scores,
        s=sizes,
        cmap='viridis',
        alpha=0.85,
        edgecolors='black',
        linewidths=0.3)
    plt.xlabel('GFLOPs')
    plt.ylabel('{} AP'.format(score_key.capitalize()))
    plt.title('SCRFD Search: Accuracy vs FLOPs')
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Hard AP')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, 'ap_vs_flops.png'), dpi=200)
    plt.close()


def plot_compute_distribution(top_records, output_dir):
    labels = [record['name'] for record in top_records]
    stem = [record['backbone_flops'][0] for record in top_records]
    stage1 = [record['backbone_flops'][1] for record in top_records]
    stage2 = [record['backbone_flops'][2] for record in top_records]
    stage3 = [record['backbone_flops'][3] for record in top_records]
    stage4 = [record['backbone_flops'][4] for record in top_records]
    neck = [record['neck_flops'] for record in top_records]
    head = [record['head_flops'] for record in top_records]

    x = np.arange(len(labels))
    plt.figure(figsize=(14, 7))
    bottom = np.zeros(len(labels), dtype=np.float32)
    for values, title in [
            (stem, 'Stem'),
            (stage1, 'Layer1'),
            (stage2, 'Layer2'),
            (stage3, 'Layer3'),
            (stage4, 'Layer4'),
            (neck, 'Neck'),
            (head, 'Head')]:
        plt.bar(x, values, bottom=bottom, label=title)
        bottom += np.array(values, dtype=np.float32)

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('GFLOPs')
    plt.title('Top Candidates: Compute Distribution')
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, 'topk_compute_distribution.png'), dpi=200)
    plt.close()


def plot_kernel_distribution(top_records, output_dir):
    kernel_labels = ['stem', 'stem_dw', 'stage1', 'stage2', 'stage3', 'stage4']
    matrix = []
    row_labels = []
    for record in top_records:
        metadata = record['metadata']
        stage_kernel_sizes = list(metadata['stage_kernel_sizes'])
        while len(stage_kernel_sizes) < 4:
            stage_kernel_sizes.append(0)
        row_labels.append(record['name'])
        matrix.append([
            metadata['stem_kernel_size'] or 0,
            metadata['stem_dw_kernel_size'] or 0,
            *stage_kernel_sizes[:4]
        ])

    if not matrix:
        return

    matrix = np.array(matrix, dtype=np.float32)
    plt.figure(figsize=(10, max(5, len(row_labels) * 0.45)))
    image = plt.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=3, vmax=7)
    plt.colorbar(image, label='Kernel Size')
    plt.xticks(np.arange(len(kernel_labels)), kernel_labels)
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.title('Top Candidates: Kernel Choices')
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col, row, int(matrix[row, col]),
                ha='center', va='center', color='black', fontsize=9)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, 'topk_kernel_heatmap.png'), dpi=200)
    plt.close()


def write_topk_summary(top_records, output_dir, score_key):
    summary_file = osp.join(output_dir, 'topk_summary.md')
    with open(summary_file, 'w', encoding='utf-8') as file_obj:
        file_obj.write('# Top Search Candidates\n\n')
        file_obj.write(
            '| Rank | Name | Easy | Medium | Hard | Mean | GFLOPs | '
            'Stem | Stem-DW | Stage Kernels | Stage Blocks |\n')
        file_obj.write(
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n')
        for rank, record in enumerate(top_records, start=1):
            file_obj.write(
                '| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {} | {} |\n'.format(
                    rank,
                    record['name'],
                    record['aps'][0],
                    record['aps'][1],
                    record['aps'][2],
                    np.mean(record['aps']),
                    record['all_flops'],
                    record['metadata']['stem_kernel_size'],
                    record['metadata']['stem_dw_kernel_size'],
                    '-'.join(map(str, record['metadata']['stage_kernel_sizes'])),
                    '-'.join(map(str, record['metadata']['stage_blocks']))))
        file_obj.write('\n')
        file_obj.write(
            'Ranking metric: `{}`.\n'.format(score_key))
    return summary_file


def main():
    args = parse_args()
    config_dir, group_name = resolve_group(args.group)
    prefix = args.prefix or group_name
    output_dir = args.output_dir or osp.join(
        args.result_dir, '{}_viz'.format(prefix))
    os.makedirs(output_dir, exist_ok=True)

    records = collect_records(
        config_dir=config_dir,
        group_name=group_name,
        prefix=prefix,
        result_dir=args.result_dir,
        idx_from=args.idx_from,
        idx_to=args.idx_to,
        input_shape=args.input_shape)

    if not records:
        raise RuntimeError(
            'No valid search results were found. Expected aps files under '
            '{}/{}/{}_*/aps'.format(args.result_dir, group_name, prefix))

    records = sorted(
        records,
        key=lambda item: score_value(item, args.score_key),
        reverse=True)
    top_records = records[:min(args.topk, len(records))]

    jsonl_file, csv_file = write_stats(records, output_dir)
    plot_ap_curves(records, output_dir)
    plot_flops_vs_ap(records, output_dir, args.score_key)
    plot_compute_distribution(top_records, output_dir)
    plot_kernel_distribution(top_records, output_dir)
    summary_file = write_topk_summary(top_records, output_dir, args.score_key)

    print('Saved search stats to:', jsonl_file)
    print('Saved search stats CSV to:', csv_file)
    print('Saved plots to:', output_dir)
    print('Saved top-k summary to:', summary_file)
    print('Best candidate:', top_records[0]['name'],
          'score={:.4f}'.format(score_value(top_records[0], args.score_key)))


if __name__ == '__main__':
    main()
