from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from mmdet.utils import get_root_logger
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES


@BACKBONES.register_module()
class MobileNetV1KS(nn.Module):
    """MobileNetV1 with configurable kernel sizes for architecture search.

    This variant extends MobileNetV1 to support different kernel sizes per stage,
    and in the stem, enabling first-kernel search as well.

    Args:
        in_channels (int): Number of input image channels. Default: 3
        block_cfg (dict): Block configuration containing:
            - stage_blocks (list[int]): Number of blocks per stage
            - stage_planes (list[int]): Channel dimensions per stage
            - stem_kernel_size (int, optional): Kernel size of the first conv
            - stem_dw_kernel_size (int, optional): Kernel size of the first
              depthwise conv in the stem
            - stage_kernel_sizes (list[int], optional): Kernel size for each stage.
              If not provided, defaults to 3 for all stages.
        num_stages (int): Number of stages. Default: 4
        out_indices (tuple): Which indices to return. Default: (0, 1, 2, 3)
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=None,
                 block_cfg=None,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3)):
        super(MobileNetV1KS, self).__init__()
        self.out_indices = out_indices

        def conv_bn(inp, oup, stride, kernel_size=3):
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride, kernel_size=3):
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        if block_cfg is None:
            stage_planes = [8, 16, 32, 64, 128, 256]
            stage_blocks = [2, 4, 4, 2]
            stem_kernel_size = 3
            stem_dw_kernel_size = 3
            stage_kernel_sizes = [3, 3, 3, 3]
        else:
            stage_planes = block_cfg['stage_planes']
            stage_blocks = block_cfg['stage_blocks']
            stem_kernel_size = block_cfg.get(
                'stem_kernel_size',
                block_cfg.get('first_kernel_size', 3))
            stem_dw_kernel_size = block_cfg.get('stem_dw_kernel_size', 3)
            # Get kernel sizes, default to 3 if not specified
            stage_kernel_sizes = block_cfg.get('stage_kernel_sizes', [3] * len(stage_blocks))

        assert len(stage_planes) == 6
        assert len(stage_blocks) == 4
        assert len(stage_kernel_sizes) == 4 or len(stage_kernel_sizes) == 1

        # If only one kernel size provided, use for all stages
        if len(stage_kernel_sizes) == 1:
            stage_kernel_sizes = stage_kernel_sizes * 4

        self.stem_kernel_size = stem_kernel_size
        self.stem_dw_kernel_size = stem_dw_kernel_size
        self.stage_kernel_sizes = stage_kernel_sizes

        self.stem = nn.Sequential(
            conv_bn(in_channels, stage_planes[0], 2, kernel_size=stem_kernel_size),
            conv_dw(
                stage_planes[0],
                stage_planes[1],
                1,
                kernel_size=stem_dw_kernel_size),
        )

        self.stage_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            _layers = []
            kernel_size = stage_kernel_sizes[i]
            for n in range(num_blocks):
                if n == 0:
                    _layer = conv_dw(stage_planes[i + 1], stage_planes[i + 2], 2,
                                     kernel_size=kernel_size)
                else:
                    _layer = conv_dw(stage_planes[i + 2], stage_planes[i + 2], 1,
                                     kernel_size=kernel_size)
                _layers.append(_layer)

            _block = nn.Sequential(*_layers)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, _block)
            self.stage_layers.append(layer_name)

    def forward(self, x):
        output = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.stage_layers):
            stage_layer = getattr(self, layer_name)
            x = stage_layer(x)
            if i in self.out_indices:
                output.append(x)
        return tuple(output)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


# Alias for convenience
MobileNetV1_KernelSearch = MobileNetV1KS
