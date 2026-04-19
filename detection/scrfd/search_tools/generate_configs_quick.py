#!/usr/bin/env python

import argparse
import glob
import hashlib
import os
import os.path as osp
import random

import numpy as np
from mmcv import Config

try:
    from search_tools import search_space as at
except ImportError:
    import search_space as at


@at.obj(
    block=at.Choice('BasicBlock', 'Bottleneck'),
    base_channels=at.Int(8, 64),
    stage_blocks=at.List(
        at.Int(1, 10),
        at.Int(1, 10),
        at.Int(1, 10),
        at.Int(1, 10),
    ),
    stage_planes_ratio=at.List(
        at.Real(1.0, 4.0),
        at.Real(1.0, 4.0),
        at.Real(1.0, 4.0),
    ),
)
class GenConfigBackbone:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.m = 1.0

    def merge_cfg(self, det_cfg):
        base_channels = max(8, int(self.base_channels * self.m) // 8 * 8)
        stage_planes = [base_channels]
        for ratio in self.stage_planes_ratio:
            planes = int(stage_planes[-1] * ratio) // 8 * 8
            stage_planes.append(planes)
        stage_blocks = [max(1, int(x * self.m)) for x in self.stage_blocks]
        block_cfg = dict(
            block=self.block,
            stage_blocks=tuple(stage_blocks),
            stage_planes=stage_planes,
        )
        det_cfg['model']['backbone']['block_cfg'] = block_cfg
        det_cfg['model']['backbone']['base_channels'] = base_channels
        neck_in_planes = (
            stage_planes
            if self.block == 'BasicBlock'
            else [4 * x for x in stage_planes]
        )
        det_cfg['model']['neck']['in_channels'] = neck_in_planes
        return det_cfg


@at.obj(
    stage_blocks_ratio=at.Real(0.5, 3.0),
    base_channels_ratio=at.Real(0.5, 3.0),
    fpn_channel=at.Int(8, 128),
    head_channel=at.Int(8, 256),
    head_stack=at.Int(1, 4),
)
class GenConfigAll:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def merge_cfg(self, det_cfg):
        block_cfg = dict(det_cfg['model']['backbone']['block_cfg'])
        stage_blocks = tuple(
            int(np.round(x * self.stage_blocks_ratio))
            for x in block_cfg['stage_blocks']
        )
        block_cfg['stage_blocks'] = stage_blocks
        stage_planes = [
            int(np.round(x * self.base_channels_ratio)) // 8 * 8
            for x in block_cfg['stage_planes']
        ]
        block_cfg['stage_planes'] = stage_planes
        det_cfg['model']['backbone']['block_cfg'] = block_cfg
        det_cfg['model']['backbone']['base_channels'] = stage_planes[0]
        neck_in_planes = (
            stage_planes
            if block_cfg['block'] == 'BasicBlock'
            else [4 * x for x in stage_planes]
        )
        det_cfg['model']['neck']['in_channels'] = neck_in_planes

        fpn_channel = self.fpn_channel // 8 * 8
        head_channel = self.head_channel // 8 * 8
        det_cfg['model']['neck']['out_channels'] = fpn_channel
        det_cfg['model']['bbox_head']['in_channels'] = fpn_channel
        det_cfg['model']['bbox_head']['feat_channels'] = head_channel
        det_cfg['model']['bbox_head']['stacked_convs'] = self.head_stack

        gn_num_groups = 8
        for candidate in [8, 16, 32, 64]:
            if head_channel % candidate != 0:
                break
            gn_num_groups = candidate
        det_cfg['model']['bbox_head']['norm_cfg']['num_groups'] = gn_num_groups
        return det_cfg


def get_args():
    parser = argparse.ArgumentParser(
        description='Quick SCRFD config generator without FLOPs filtering')
    parser.add_argument(
        '--group',
        type=str,
        required=True,
        help='output config directory, e.g. configs/scrfdgen2.5g_quick',
    )
    parser.add_argument(
        '--template-config',
        type=str,
        required=True,
        help='template config path to mutate',
    )
    parser.add_argument(
        '--mode',
        type=int,
        default=1,
        choices=[1, 2],
        help='1 for backbone-style sampling, 2 for full detector sampling',
    )
    parser.add_argument(
        '--num-configs',
        type=int,
        default=4,
        help='number of configs to generate',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='random seed',
    )
    parser.add_argument(
        '--allow-duplicates',
        action='store_true',
        default=False,
        help='allow duplicate configs instead of skipping repeated samples',
    )
    return parser.parse_args()


def _resolve_group(group):
    output_group = osp.normpath(group)
    os.makedirs(output_group, exist_ok=True)
    group_name = osp.basename(output_group.rstrip('/\\'))
    if not group_name:
        raise ValueError('Could not infer group name from --group')
    return output_group, group_name


def _next_index(output_group, group_name):
    pattern = osp.join(output_group, f'{group_name}_*.py')
    indices = []
    for path in glob.glob(pattern):
        stem = osp.splitext(osp.basename(path))[0]
        suffix = stem.rsplit('_', 1)[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices) + 1 if indices else 0


def _fingerprint_cfg(cfg):
    return hashlib.sha1(cfg.pretty_text.encode('utf-8')).hexdigest()


def _collect_existing_fingerprints(output_group, group_name):
    fingerprints = set()
    pattern = osp.join(output_group, f'{group_name}_*.py')
    for path in glob.glob(pattern):
        cfg = Config.fromfile(path)
        fingerprints.add(_fingerprint_cfg(cfg))
    return fingerprints


def _summary_line(cfg, mode):
    backbone_cfg = dict(cfg['model']['backbone']['block_cfg'])
    if mode == 1:
        return (
            f"block={backbone_cfg.get('block')} "
            f"stage_blocks={tuple(backbone_cfg.get('stage_blocks', []))} "
            f"stage_planes={list(backbone_cfg.get('stage_planes', []))}"
        )
    return (
        f"stage_blocks={tuple(backbone_cfg.get('stage_blocks', []))} "
        f"stage_planes={list(backbone_cfg.get('stage_planes', []))} "
        f"fpn={cfg['model']['neck'].get('out_channels')} "
        f"head={cfg['model']['bbox_head'].get('feat_channels')} "
        f"stack={cfg['model']['bbox_head'].get('stacked_convs')}"
    )


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_group, group_name = _resolve_group(args.group)
    template_config = osp.normpath(args.template_config)
    if not osp.exists(template_config):
        raise FileNotFoundError(f'Template config not found: {template_config}')

    generator = GenConfigBackbone() if args.mode == 1 else GenConfigAll()
    seen_fingerprints = set()
    if not args.allow_duplicates:
        seen_fingerprints = _collect_existing_fingerprints(output_group, group_name)

    write_index = _next_index(output_group, group_name)
    written = 0
    attempts = 0
    max_attempts = max(100, args.num_configs * 50)

    while written < args.num_configs and attempts < max_attempts:
        attempts += 1
        det_cfg = Config.fromfile(template_config)
        det_cfg = generator.rand.merge_cfg(det_cfg)
        fingerprint = _fingerprint_cfg(det_cfg)
        if not args.allow_duplicates and fingerprint in seen_fingerprints:
            continue

        output_cfg_file = osp.join(output_group, f'{group_name}_{write_index}.py')
        det_cfg.dump(output_cfg_file)
        seen_fingerprints.add(fingerprint)
        print(f'Wrote {output_cfg_file}')
        print(f'  {_summary_line(det_cfg, args.mode)}')
        write_index += 1
        written += 1

    if written < args.num_configs:
        raise SystemExit(
            f'Only generated {written}/{args.num_configs} unique configs '
            f'after {attempts} attempts.')

    print(
        f'Generated {written} config(s) in {output_group} '
        f'from template {template_config}')


if __name__ == '__main__':
    main()
