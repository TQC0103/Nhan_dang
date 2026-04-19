import os
import numpy as np
import mmcv

from .version import __version__, short_version


# Compatibility aliases for legacy NumPy usage in the upstream SCRFD/MMDet code.
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object


def _patch_parallel_stream_compat():
    torch_get_stream = None
    try:
        import torch
        from torch.nn.parallel import _functions as torch_parallel_functions

        original_torch_get_stream = torch_parallel_functions._get_stream

        def _stream_device_candidates(device):
            candidates = [device]
            if isinstance(device, int):
                candidates.extend([
                    torch.device('cuda', device),
                    f'cuda:{device}',
                ])
            elif isinstance(device, str):
                try:
                    parsed = torch.device(device)
                except Exception:
                    parsed = None
                if parsed is not None:
                    candidates.append(parsed)
                    if parsed.type == 'cuda':
                        if parsed.index is not None:
                            candidates.append(parsed.index)
                        elif torch.cuda.is_available():
                            candidates.append(torch.cuda.current_device())
            elif isinstance(device, torch.device) and device.type == 'cuda':
                if device.index is not None:
                    candidates.extend([
                        device.index,
                        f'cuda:{device.index}',
                    ])
                elif torch.cuda.is_available():
                    candidates.append(torch.cuda.current_device())

            unique_candidates = []
            for candidate in candidates:
                if candidate not in unique_candidates:
                    unique_candidates.append(candidate)
            return unique_candidates

        def torch_compat_get_stream(device):
            last_error = None
            for candidate in _stream_device_candidates(device):
                try:
                    return original_torch_get_stream(candidate)
                except Exception as error:
                    last_error = error
            if last_error is not None:
                raise last_error
            return original_torch_get_stream(device)

        torch_parallel_functions._get_stream = torch_compat_get_stream
        torch_get_stream = torch_compat_get_stream
    except Exception:
        pass

    try:
        import torch
        from mmcv.parallel import _functions as mmcv_parallel_functions

        def mmcv_compat_get_stream(device):
            if torch_get_stream is not None:
                return torch_get_stream(device)
            return None

        mmcv_parallel_functions._get_stream = mmcv_compat_get_stream
    except Exception:
        pass


_patch_parallel_stream_compat()


def _patch_torch_sharing_strategy():
    strategy = os.environ.get('SCRFD_TORCH_SHARING_STRATEGY')
    if not strategy:
        return
    try:
        import torch
        torch.multiprocessing.set_sharing_strategy(strategy)
    except Exception:
        pass


_patch_torch_sharing_strategy()


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.1.5'
mmcv_maximum_version = os.environ.get('SCRFD_MMCV_MAX_VERSION', '1.4.0')
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}. ' \
    f'For the experimental RTX 50 setup, set SCRFD_MMCV_MAX_VERSION=1.7.2.'

__all__ = ['__version__', 'short_version']

