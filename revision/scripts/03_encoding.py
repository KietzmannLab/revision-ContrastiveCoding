"""Fit sparse positive encoding models from selective DNN units to NSD ROIs.

Converted from PROJECT_DNFFA/NOTEBOOKS/3-Encoding.ipynb.

Writes one CSV per (subject, ROI, layer, domain) to ``analysis_outputs/3-Encoding``;
existing CSVs are skipped unless --overwrite is given.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from jsputils import classes  # noqa: E402

from dnffa import config, nsd  # noqa: E402

DOMAINS = ['faces', 'scenes', 'bodies', 'characters', 'objects']


def domains_for(model_name, roi):
    """vggface: whole layers only. Untrained model: only the ROI's preferred domain."""
    if 'vggface' in model_name:
        return ['layer']
    if 'random' in model_name:
        return [config.ROI_DOMAIN[roi]]
    return DOMAINS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--method', default='lasso', choices=['lasso', 'ols'])
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='lasso penalty (paper: 0.1 for trained, 0.001 for untrained models)')
    parser.add_argument('--rois', nargs='+', default=config.ROI_LIST)
    parser.add_argument('--subjects', nargs='+', default=config.SUBJECTS)
    parser.add_argument('--savedir', type=Path, default=None,
                        help='defaults to analysis_outputs/3-Encoding')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    savedir = args.savedir or config.analysis_dir(config.ENCODING_SUBDIR)
    savedir.mkdir(parents=True, exist_ok=True)
    layers = config.encoding_layers(args.model)
    positive = args.method == 'lasso'
    alphas = [args.alpha] if args.method == 'lasso' else [None]

    DNN = classes.DNNModel(args.model)
    if 'vggface' not in args.model:
        DNN.find_selective_units(config.FLOC_IMAGESET, overwrite=False, verbose=False, FDR_p=config.FDR_P)

    for roi in args.rois:
        for subj in progress_bar(args.subjects):
            ROI = nsd.load_roi(nsd.load_subject(subj), roi)
            encoder = classes.EncodingProcedure(ROI, DNN, method=args.method, positive=positive, alphas=alphas)
            encoder.encode_layers(str(savedir),
                                  layers=np.flip(layers),
                                  domains=domains_for(args.model, roi),
                                  overwrite=args.overwrite)


if __name__ == '__main__':
    main()
