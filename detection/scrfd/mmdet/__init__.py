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

