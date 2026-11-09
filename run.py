"""Portable server entry point. Check is read-only; training is explicit."""
from __future__ import annotations
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--action', choices=('check', 'preflight', 'train', 'all', 'calibrate', 'eval'), default='check')
    p.add_argument('--dataset-root', required=True, help='Unmodified pl_3d_h1_h4_npz_v1 directory on this server')
    p.add_argument('--gpu', type=int, default=0, help='Physical NVIDIA GPU index')
    p.add_argument('--resume', action='store_true', help='Only this package exact-protocol checkpoints')
    p.add_argument('--log-every', type=int, default=20)
    args = p.parse_args()
    if sys.version_info < (3, 8):
        p.error('Python >= 3.8 is required; older versions are not silently supported')
    if args.gpu < 0 or args.log_every < 1:
        p.error('GPU index must be nonnegative and log-every positive')
    if args.resume and args.action in ('check', 'preflight'):
        p.error('Do not combine --resume with check/preflight')
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.action == 'check' else str(args.gpu)
    from server_launch import launch
    launch(args)


if __name__ == '__main__':
    main()
