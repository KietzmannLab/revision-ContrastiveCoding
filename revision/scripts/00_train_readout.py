"""Train a linear ImageNet readout on top of a (frozen) DNN layer.

Converted from PROJECT_DNFFA/NOTEBOOKS/0-Train-Readout.ipynb.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from jsputils import classes  # noqa: E402

from dnffa import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--readout-from', default='relu7')
    parser.add_argument('--sparse-pos', action='store_true',
                        help='constrain readout weights to be sparse and positive')
    args = parser.parse_args()

    DNN = classes.DNNModel(args.model)
    print(DNN.model_name)
    DNN.train_linear_probe(readout_from=args.readout_from, sparse_pos=args.sparse_pos)


if __name__ == '__main__':
    main()
