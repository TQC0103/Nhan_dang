import os
import sys


def train(group, prefix, idx, gpuid, use_dist=True, port_base=29100):
    assert idx>=0
    config_path = "./configs/%s/%s_%d.py" % (group, prefix, idx)
    extra_args = os.environ.get('SCRFD_TRAIN_EXTRA_ARGS', '').strip()
    extra_args = (' ' + extra_args) if extra_args else ''
    if use_dist:
        cmd = (
            "CUDA_VISIBLE_DEVICES='%d' PORT=%d "
            "bash ./tools/dist_train.sh %s 1 --no-validate%s"
            % (gpuid, port_base + idx, config_path, extra_args)
        )
    else:
        cmd = (
            "CUDA_VISIBLE_DEVICES='%d' PYTHONPATH=\"$(pwd)\":$PYTHONPATH "
            "python -u ./tools/train.py %s --no-validate%s"
            % (gpuid, config_path, extra_args)
        )
    print(cmd)
    os.system(cmd)


gpuid = int(sys.argv[1])
idx_from = int(sys.argv[2])
idx_to = int(sys.argv[3])
group = 'scrfdgen'
if len(sys.argv)>4:
    group = sys.argv[4]
use_dist = True
if len(sys.argv)>5:
    use_dist = bool(int(sys.argv[5]))
port_base = 29100
if len(sys.argv)>6:
    port_base = int(sys.argv[6])
idx_step = 1
if len(sys.argv)>7:
    idx_step = int(sys.argv[7])

for idx in range(idx_from, idx_to, idx_step):
    train(group, group, idx, gpuid, use_dist=use_dist, port_base=port_base)

