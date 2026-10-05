"""Encoding baselines from low-level image features (GIST PCs and Gabor energy).

Converted from PROJECT_DNFFA/NOTEBOOKS/7-LowLevelModels.ipynb.

Two stages:
  export  - save each subject's train images and the shared test images as .mat files
  encode  - fit OLS encoding models from the GIST/Gabor features to each ROI

Between the two, run PROJECT_DNFFA/HELPERS/DNFFA_extract_gist.m in MATLAB on the exported
images; it writes ``{subj}_{GistPC,Gabor}.mat`` / ``special515_{GistPC,Gabor}.mat`` into
``analysis_outputs/3d-LowLevel``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scipy.io as sio  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from jsputils import classes  # noqa: E402

from dnffa import config, nsd  # noqa: E402

FEATURE_SPACES = ['GistPC', 'Gabor']


def export_images(feature_dir, subjs):
    for subj in progress_bar(subjs):
        test_fn = feature_dir / f'{config.TEST_IMAGESET}_images.mat'
        train_fn = feature_dir / f'{subj}_train-{config.TRAIN_IMAGESET}_images.mat'

        partitions = {}
        if not train_fn.exists():
            partitions['train'] = config.TRAIN_IMAGESET
        if not test_fn.exists():
            partitions['test'] = config.TEST_IMAGESET  # shared across subjects
        if not partitions:
            print(subj, 'skipping, already exists')
            continue

        images = nsd.load_nsd_images(subj, partitions)
        for partition, fn in [('train', train_fn), ('test', test_fn)]:
            if partition in images:
                sio.savemat(fn, {'images': images[partition]})
                print(subj, partition, images[partition].shape)


def load_gist_gabor_features(feature_dir, name, feature_space):
    """``name`` is a subject id or the test image set name."""
    struct = sio.loadmat(feature_dir / f'{name}_{feature_space}.mat')[feature_space]
    if not isinstance(struct, dict):
        struct = {n: struct[n][0, 0] for n in struct.dtype.names}
    return struct['featureMatrix']


def encode(feature_dir, encoding_dir, subjs, rois, overwrite):
    for feature_space in FEATURE_SPACES:
        test_features = load_gist_gabor_features(feature_dir, config.TEST_IMAGESET, feature_space)
        print(feature_space, 'test features:', test_features.shape)

        for roi in rois:
            for subj in progress_bar(subjs):
                ROI = nsd.load_roi(nsd.load_subject(subj), roi)
                train_features = load_gist_gabor_features(feature_dir, subj, feature_space)
                encoder = classes.EncodingProcedureGistGabor(ROI, feature_space, train_features, test_features,
                                                             method='ols', positive=False, alphas=[None])
                encoder.encode_features(str(encoding_dir), overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['export', 'encode'])
    parser.add_argument('--rois', nargs='+', default=config.ROI_LIST)
    parser.add_argument('--subjects', nargs='+', default=config.SUBJECTS)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    feature_dir = config.analysis_dir(config.LOW_LEVEL_SUBDIR)

    if args.stage == 'export':
        export_images(feature_dir, args.subjects)
    else:
        encode(feature_dir, config.analysis_dir(config.ENCODING_SUBDIR), args.subjects, args.rois, args.overwrite)


if __name__ == '__main__':
    main()
