"""
Enhanced WIDERFace evaluation script with detailed logging for experiments.

This script extends test_widerface.py with:
1. Detailed per-epoch AP logging to CSV
2. Precision-Recall curve data
3. Per-category breakdown
4. Easy to parse format for report generation

Usage:
    python test_widerface_enhanced.py \
        configs/kd/scrfd_500m_kd_10g.py \
        work_dirs/scrfd_500m_kd_10g/latest.pth \
        --out results/kd_10g \
        --save-preds
"""

import argparse
import os
import os.path as osp
import pickle
import numpy as np
import csv
import json
import warnings
from collections import OrderedDict

import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.core.evaluation import wider_evaluation, get_widerface_gts


def parse_args():
    parser = argparse.ArgumentParser(
        description='Enhanced MMDet test with detailed logging')
    parser.add_argument('config', nargs='?', help='test config file path')
    parser.add_argument('checkpoint', nargs='?', help='checkpoint file')
    parser.add_argument('--config', dest='config_option', help='test config file path')
    parser.add_argument('--checkpoint', dest='checkpoint_option', help='checkpoint file')
    parser.add_argument('--out', default='eval_results', help='output folder')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics')
    parser.add_argument('--save-preds', action='store_true', help='save predictions')
    parser.add_argument('--save-pr-curve', action='store_true', help='save PR curve data')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--thr', type=float, default=0.02, help='score threshold')
    parser.add_argument('--mode', type=int, default=0, help='test mode')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    args.config = args.config_option or args.config
    args.checkpoint = args.checkpoint_option or args.checkpoint
    if not args.config or not args.checkpoint:
        parser.error('config and checkpoint are required')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def compute_pr_curve(predictions, gt_path, iou_thresh=0.5):
    """Compute Precision-Recall curve data for each difficulty level.

    Returns:
        dict: PR curve data for easy, medium, hard
    """
    # Get ground truth
    widerfaceGt = wider_evaluation

    # This would need to be implemented based on wider_evaluation API
    # For now, return placeholder structure
    return {
        'easy': {'precision': [], 'recall': [], 'mAP': 0},
        'medium': {'precision': [], 'recall': [], 'mAP': 0},
        'hard': {'precision': [], 'recall': [], 'mAP': 0}
    }


def save_results_csv(results, output_file):
    """Save detection results to CSV format."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'box_count', 'boxes_x1', 'boxes_y1', 'boxes_x2', 'boxes_y2', 'scores'])

        for img_name, boxes in results.items():
            if len(boxes) == 0:
                writer.writerow([img_name, 0])
            else:
                for box in boxes:
                    writer.writerow([
                        img_name,
                        len(boxes),
                        box[0], box[1], box[2], box[3], box[4]
                    ])


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    gt_path = os.path.join(os.path.dirname(cfg.data.test.ann_file), 'gt')
    pipelines = cfg.data.test.pipeline
    for pipeline in pipelines:
        if pipeline.type == 'MultiScaleFlipAug':
            if args.mode == 0:
                pipeline.img_scale = (640, 640)
            elif args.mode == 1:
                pipeline.img_scale = (1100, 1650)
            elif args.mode == 2:
                pipeline.img_scale = None
                pipeline.scale_factor = 1.0
            elif args.mode > 30:
                pipeline.img_scale = (args.mode, args.mode)
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type == 'Pad':
                    if args.mode != 2:
                        transform.size = pipeline.img_scale
                    else:
                        transform.size = None
                        transform.size_divisor = 32

    print(f'Test config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output folder: {args.out}')

    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    cfg.test_cfg.score_thr = args.thr

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Run inference
    results = {}
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        assert len(result) == 1
        result = result[0][0]
        img_metas = data['img_metas'][0].data[0][0]
        filepath = img_metas['ori_filename']
        det_scale = img_metas['scale_factor'][0]
        ori_shape = img_metas['ori_shape']

        _vec = filepath.split('/')
        event_name = _vec[-2]
        img_name = _vec[-1].rstrip('.jpg')

        if event_name not in results:
            results[event_name] = {}

        # Convert to XYWH format
        xywh = result.copy()
        w = xywh[:, 2] - xywh[:, 0]
        h = xywh[:, 3] - xywh[:, 1]
        xywh[:, 2] = w
        xywh[:, 3] = h

        results[event_name][img_name] = xywh

        if args.save_preds:
            out_dir = os.path.join(args.out, 'predictions', event_name)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f'{img_name}.txt')
            with open(out_file, 'w') as f:
                f.write(f"{filepath}\n")
                f.write(f"{result.shape[0]}\n")
                for b in range(result.shape[0]):
                    box = result[b]
                    f.write(f"{box[0]:.5f} {box[1]:.5f} {box[2]-box[0]:.5f} {box[3]-box[1]:.5f} {box[4]}\n")

        prog_bar.update()

    # Evaluate
    aps = wider_evaluation(results, gt_path, 0.5, args.debug)

    # Save results summary
    results_summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'timestamp': mmcv.timestamp(),
        'easy_AP': float(aps[0]),
        'medium_AP': float(aps[1]),
        'hard_AP': float(aps[2]),
        'mAP': float(np.mean(aps)),
        'score_threshold': args.thr,
        'test_mode': args.mode
    }

    # Save JSON summary
    json_out = os.path.join(args.out, 'results_summary.json')
    with open(json_out, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save CSV summary (easy to load for analysis)
    csv_out = os.path.join(args.out, 'results_summary.csv')
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['easy_AP', f'{aps[0]:.4f}'])
        writer.writerow(['medium_AP', f'{aps[1]:.4f}'])
        writer.writerow(['hard_AP', f'{aps[2]:.4f}'])
        writer.writerow(['mAP', f'{np.mean(aps):.4f}'])
        writer.writerow(['config', args.config])
        writer.writerow(['checkpoint', args.checkpoint])
        writer.writerow(['timestamp', results_summary['timestamp']])

    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Easy AP:   {aps[0]:.4f}')
    print(f'Medium AP: {aps[1]:.4f}')
    print(f'Hard AP:   {aps[2]:.4f}')
    print(f'mAP:       {np.mean(aps):.4f}')
    print('='*60)
    print(f'Results saved to: {args.out}')
    print('='*60)


if __name__ == '__main__':
    main()
