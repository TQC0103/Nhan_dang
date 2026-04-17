import argparse
import csv
import json
import os
import os.path as osp
import statistics
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile end-to-end detector runtime, model complexity, and memory usage.')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--out', required=True, help='Output JSON file path')
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='PyTorch device, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--shape',
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('HEIGHT', 'WIDTH'),
        help='Input image shape')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for latency profiling')
    parser.add_argument('--warmup', type=int, default=30, help='Warmup iterations')
    parser.add_argument('--repeat', type=int, default=200, help='Measured iterations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for dummy input')
    parser.add_argument(
        '--skip-complexity',
        action='store_true',
        help='Skip params/FLOPs computation')
    args = parser.parse_args()
    return args


def import_custom_modules(cfg):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])


def strip_pretrained(cfg):
    cfg.model.pretrained = None
    if not cfg.model.get('neck'):
        return
    if isinstance(cfg.model.neck, list):
        neck_cfgs = cfg.model.neck
    else:
        neck_cfgs = [cfg.model.neck]
    for neck_cfg in neck_cfgs:
        if neck_cfg.get('rfp_backbone') and neck_cfg.rfp_backbone.get('pretrained'):
            neck_cfg.rfp_backbone.pretrained = None


def build_model(cfg, checkpoint_path, device):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    model = model.to(device)
    model.eval()
    return model


def make_dummy_batch(batch_size, height, width, device):
    img = torch.randn(batch_size, 3, height, width, device=device)
    img_metas = []
    for _ in range(batch_size):
        img_metas.append(
            dict(
                img_shape=(height, width, 3),
                ori_shape=(height, width, 3),
                pad_shape=(height, width, 3),
                scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                flip=False,
                flip_direction=None,
                border=(0, 0, 0, 0),
            ))
    return dict(img=[img], img_metas=[img_metas], return_loss=False, rescale=True)


def synchronize_if_needed(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def compute_latency_stats(latencies_ms, batch_size):
    if not latencies_ms:
        return {}
    mean_ms = statistics.mean(latencies_ms)
    return {
        'mean_ms': mean_ms,
        'ms_per_image': mean_ms / float(batch_size),
        'std_ms': statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        'min_ms': min(latencies_ms),
        'max_ms': max(latencies_ms),
        'p50_ms': float(np.percentile(latencies_ms, 50)),
        'p90_ms': float(np.percentile(latencies_ms, 90)),
        'p95_ms': float(np.percentile(latencies_ms, 95)),
        'fps': (1000.0 * float(batch_size) / mean_ms) if mean_ms > 0 else 0.0,
    }


def maybe_compute_complexity(model, shape):
    if get_model_complexity_info is None:
        return {}
    if not hasattr(model, 'forward_dummy'):
        return {}

    original_forward = model.forward
    model.forward = model.forward_dummy
    try:
        flops, params = get_model_complexity_info(
            model,
            (3, shape[0], shape[1]),
            print_per_layer_stat=False,
            as_strings=False)
        if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'extra_flops'):
            flops += model.bbox_head.extra_flops
            flops *= 0.75
    finally:
        model.forward = original_forward

    return {
        'flops_g': float(flops / 1e9),
        'params_m': float(params / 1e6),
    }


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    import_custom_modules(cfg)
    strip_pretrained(cfg)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but torch.cuda.is_available() is False')

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(device)

    model = build_model(cfg, args.checkpoint, device)
    dummy_batch = make_dummy_batch(args.batch_size, args.shape[0], args.shape[1], device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(**dummy_batch)
        synchronize_if_needed(device)

        latencies_ms = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            _ = model(**dummy_batch)
            synchronize_if_needed(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latency_summary = {
        'config': osp.abspath(args.config),
        'checkpoint': osp.abspath(args.checkpoint),
        'device': str(device),
        'shape': [int(args.shape[0]), int(args.shape[1])],
        'batch_size': int(args.batch_size),
        'warmup': int(args.warmup),
        'repeat': int(args.repeat),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'checkpoint_size_mb': (
            float(os.path.getsize(args.checkpoint) / (1024.0 * 1024.0))
            if osp.exists(args.checkpoint) else None
        ),
    }
    latency_summary.update(compute_latency_stats(latencies_ms, args.batch_size))

    if device.type == 'cuda':
        latency_summary['peak_memory_mb'] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    if not args.skip_complexity:
        latency_summary.update(maybe_compute_complexity(model, args.shape))

    out_path = osp.abspath(args.out)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as outfile:
        json.dump(latency_summary, outfile, indent=2, sort_keys=True)

    csv_path = osp.splitext(out_path)[0] + '.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['metric', 'value'])
        for key, value in sorted(latency_summary.items()):
            writer.writerow([key, value])

    print('Wrote runtime profile to {}'.format(out_path))


if __name__ == '__main__':
    main()
