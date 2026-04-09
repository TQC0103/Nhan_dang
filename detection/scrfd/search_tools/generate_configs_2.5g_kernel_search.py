"""Architecture search for SCRFD with MobileNet 500M kernel search support."""

import argparse
import datetime
import importlib.util
import io
import os
import os.path as osp

import numpy as np
import torch
from mmcv import Config
from mmdet.models import build_detector

try:
    import autotorch as at
except ImportError:
    try:
        from search_tools import search_space as at
    except ImportError:
        import search_space as at

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def _round_channels(value):
    return max(8, int(np.round(value)) // 8 * 8)


def _is_mobilenet_backbone(backbone_cfg):
    backbone_type = backbone_cfg.get('type', '')
    return 'MobileNetV1' in backbone_type


def _get_mobilenet_neck_in_channels(stage_planes):
    return stage_planes[2:]


@at.obj(
    stem_channels=at.Int(8, 64),
    plane_ratios=at.List(
        at.Real(1.0, 4.0),
        at.Real(1.0, 4.0),
        at.Real(1.0, 4.0),
        at.Real(1.0, 4.0),
        at.Real(1.0, 4.0),
    ),
    stage_blocks=at.List(
        at.Int(1, 10),
        at.Int(1, 10),
        at.Int(1, 10),
        at.Int(1, 10),
    ),
    stem_kernel_size=at.Choice(3, 5, 7),
    stem_dw_kernel_size=at.Choice(3, 5, 7),
    stage_kernel_sizes=at.List(
        at.Choice(3, 5, 7),
        at.Choice(3, 5, 7),
        at.Choice(3, 5, 7),
        at.Choice(3, 5, 7),
    ),
)
class GenConfigMobileNetBackboneKernelSearch:
    """Search MobileNet widths/depths together with stem and stage kernels."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.m = 1.0

    def stage_blocks_multi(self, mult):
        self.m = mult

    def merge_cfg(self, det_cfg):
        stage_planes = [_round_channels(self.stem_channels * self.m)]
        for ratio in self.plane_ratios:
            stage_planes.append(_round_channels(stage_planes[-1] * ratio))
        stage_blocks = tuple(max(1, int(block * self.m)) for block in self.stage_blocks)

        block_cfg = dict(
            stage_blocks=stage_blocks,
            stage_planes=stage_planes,
            stem_kernel_size=int(self.stem_kernel_size),
            stem_dw_kernel_size=int(self.stem_dw_kernel_size),
            stage_kernel_sizes=list(self.stage_kernel_sizes),
        )

        det_cfg['model']['backbone']['type'] = 'MobileNetV1KS'
        det_cfg['model']['backbone']['block_cfg'] = block_cfg
        det_cfg['model']['backbone'].pop('base_channels', None)
        det_cfg['model']['neck']['in_channels'] = _get_mobilenet_neck_in_channels(
            stage_planes)
        return det_cfg


@at.obj(
    stage_blocks_ratio=at.Real(0.5, 3.0),
    base_channels_ratio=at.Real(0.5, 3.0),
    fpn_channel=at.Int(8, 128),
    head_channel=at.Int(8, 256),
    head_stack=at.Int(1, 4),
    stem_kernel_size=at.Choice(3, 5, 7),
    stem_dw_kernel_size=at.Choice(3, 5, 7),
    stage_kernel_sizes=at.List(
        at.Choice(3, 5, 7),
        at.Choice(3, 5, 7),
        at.Choice(3, 5, 7),
        at.Choice(3, 5, 7),
    ),
)
class GenConfigMobileNetAllKernelSearch:
    """Search the full SCRFD 500M design space including the first kernel."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def merge_cfg(self, det_cfg):
        block_cfg = dict(det_cfg['model']['backbone']['block_cfg'])
        stage_blocks = tuple(
            max(1, int(np.round(block * self.stage_blocks_ratio)))
            for block in block_cfg['stage_blocks'])
        stage_planes = [
            _round_channels(channel * self.base_channels_ratio)
            for channel in block_cfg['stage_planes']
        ]

        block_cfg.update(
            stage_blocks=stage_blocks,
            stage_planes=stage_planes,
            stem_kernel_size=int(self.stem_kernel_size),
            stem_dw_kernel_size=int(self.stem_dw_kernel_size),
            stage_kernel_sizes=list(self.stage_kernel_sizes),
        )

        det_cfg['model']['backbone']['type'] = 'MobileNetV1KS'
        det_cfg['model']['backbone']['block_cfg'] = block_cfg
        det_cfg['model']['backbone'].pop('base_channels', None)
        det_cfg['model']['neck']['in_channels'] = _get_mobilenet_neck_in_channels(
            stage_planes)

        fpn_channel = _round_channels(self.fpn_channel)
        head_channel = _round_channels(self.head_channel)
        det_cfg['model']['neck']['out_channels'] = fpn_channel
        det_cfg['model']['bbox_head']['in_channels'] = fpn_channel
        det_cfg['model']['bbox_head']['feat_channels'] = head_channel
        det_cfg['model']['bbox_head']['stacked_convs'] = int(self.head_stack)

        gn_num_groups = 8
        for candidate in [8, 16, 32, 64]:
            if head_channel % candidate != 0:
                break
            gn_num_groups = candidate
        det_cfg['model']['bbox_head']['norm_cfg']['num_groups'] = gn_num_groups
        return det_cfg


def get_args():
    parser = argparse.ArgumentParser(description='Auto-SCRFD kernel search')
    parser.add_argument(
        '--group',
        type=str,
        default='configs/scrfdgen2.5g',
        help='output config directory')
    parser.add_argument(
        '--template',
        type=int,
        default=0,
        help='template index inside --group when --template-config is omitted')
    parser.add_argument(
        '--template-config',
        type=str,
        default=None,
        help='explicit template config path')
    parser.add_argument(
        '--gflops',
        type=float,
        default=None,
        help='target flops in GFLOPs; inferred from template when omitted')
    parser.add_argument(
        '--mode',
        type=int,
        default=1,
        help='1: search backbone, 2: search full detector')
    parser.add_argument(
        '--kernel-search',
        action='store_true',
        default=False,
        help='enable MobileNet kernel search')
    parser.add_argument(
        '--eps',
        type=float,
        default=2e-2,
        help='relative tolerance for target flops')
    parser.add_argument(
        '--num-configs',
        type=int,
        default=64,
        help='number of configs to generate')
    return parser.parse_args()


def get_flops(cfg, input_shape):
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is not supported for '
            f'{model.__class__.__name__}')

    buf = io.StringIO()
    all_flops, params = get_model_complexity_info(
        model,
        input_shape,
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
        raise RuntimeError(
            'Could not parse per-stage FLOPs. Please verify the model summary '
            'format produced by mmcv.')

    backbone_flops = np.array(parsed_flops[:-2], dtype=np.float32)
    neck_flops = parsed_flops[-2]
    head_flops = parsed_flops[-1]
    return all_flops / 1e9, backbone_flops, neck_flops, head_flops


def is_flops_valid(flops, target_flops, eps):
    return (1.0 - eps) * target_flops <= flops <= (1.0 + eps) * target_flops


def _resolve_template(args):
    output_group = osp.normpath(args.group)
    os.makedirs(output_group, exist_ok=True)
    group_name = osp.basename(output_group)
    if not group_name:
        raise ValueError('Could not infer output group name from --group')

    if args.template_config:
        template_config = osp.normpath(args.template_config)
    else:
        template_config = osp.join(output_group, f'{group_name}_{args.template}.py')

    if not osp.exists(template_config):
        raise FileNotFoundError(f'Template config not found: {template_config}')

    return output_group, group_name, template_config


def _infer_target_gflops(template_config, group_name, override):
    if override is not None:
        return override
    marker = f'{template_config} {group_name}'.lower()
    if '500m' in marker:
        return 0.5
    return 2.5


def _load_original_search_generators():
    module_path = osp.join(osp.dirname(__file__), 'generate_configs_2.5g.py')
    spec = importlib.util.spec_from_file_location('generate_configs_2p5g', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GenConfigBackbone, module.GenConfigAll


def main():
    args = get_args()
    print(datetime.datetime.now())
    print(f'Kernel search enabled: {args.kernel_search}')

    output_group, group_name, template_config = _resolve_template(args)
    target_gflops = _infer_target_gflops(
        template_config, group_name, args.gflops)
    print(f'Template: {template_config}')
    print(f'Target GFLOPs: {target_gflops}')

    input_shape = (3, 480, 640)
    det_cfg = Config.fromfile(template_config)
    is_mobilenet = _is_mobilenet_backbone(det_cfg['model']['backbone'])

    if args.kernel_search:
        if not is_mobilenet:
            raise NotImplementedError(
                'Kernel search in this script is implemented for MobileNetV1 '
                'templates only. Use configs/scrfdgen500m/scrfdgen500m_0.py '
                'or another MobileNet template.')
        if args.mode == 1:
            generator = GenConfigMobileNetBackboneKernelSearch()
        elif args.mode == 2:
            generator = GenConfigMobileNetAllKernelSearch()
        else:
            raise ValueError(f'Unsupported mode: {args.mode}')
    else:
        original_backbone_generator, original_all_generator = (
            _load_original_search_generators())
        if args.mode == 1:
            generator = original_backbone_generator()
        elif args.mode == 2:
            generator = original_all_generator()
        else:
            raise ValueError(f'Unsupported mode: {args.mode}')

    write_index = 0
    while osp.exists(osp.join(output_group, f'{group_name}_{write_index}.py')):
        write_index += 1
    print('write-index from:', write_index)

    template_backbone_ratios = None
    if args.mode == 2:
        _, template_backbone_flops, _, _ = get_flops(det_cfg, input_shape)
        template_backbone_ratios = list(
            map(lambda value: value / template_backbone_flops[0],
                template_backbone_flops))
        print('template_backbone_ratios:', template_backbone_ratios)

    attempts = 0
    write_count = 0
    while write_count < args.num_configs:
        attempts += 1
        det_cfg = Config.fromfile(template_config)
        sampled_config = generator.rand
        det_cfg = sampled_config.merge_cfg(det_cfg)

        try:
            all_flops, backbone_flops, neck_flops, head_flops = get_flops(
                det_cfg, input_shape)
        except Exception as exc:
            print(f'Error computing FLOPs for candidate {attempts}: {exc}')
            continue

        if attempts % 10 == 0:
            print(
                attempts,
                all_flops,
                backbone_flops,
                neck_flops,
                head_flops,
                datetime.datetime.now())

        if args.mode == 2 and template_backbone_ratios is not None:
            backbone_ratios = list(
                map(lambda value: value / backbone_flops[0], backbone_flops))
            if any(
                    not is_flops_valid(
                        template_backbone_ratios[i],
                        backbone_ratios[i],
                        args.eps * 5)
                    for i in range(1, 5)):
                continue

        if not is_flops_valid(all_flops, target_gflops, args.eps):
            continue

        output_cfg_file = osp.join(output_group, f'{group_name}_{write_index}.py')
        det_cfg.dump(output_cfg_file)
        print(
            'SUCC',
            write_index,
            all_flops,
            backbone_flops,
            neck_flops,
            head_flops,
            datetime.datetime.now())
        write_index += 1
        write_count += 1


if __name__ == '__main__':
    main()
