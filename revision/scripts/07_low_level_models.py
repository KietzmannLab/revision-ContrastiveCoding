"""Encoding baselines from low-level image features (GIST PCs and Gabor energy).

Converted from PROJECT_DNFFA/NOTEBOOKS/7-LowLevelModels.ipynb.

Three stages, all reading/writing ``analysis_outputs/3d-LowLevel`` (``3d-LowLevel-laion``
with --dataset laion; encoding results go to ``3-Encoding`` / ``3-Encoding-laion``):
  export    - save each subject's train images and the shared test images as .mat files
  features  - compute Gabor and GIST-PC features from the exported images; writes
              ``{subj}_{GistPC,Gabor}.mat`` / ``{test set}_{GistPC,Gabor}.mat``
              (test set: special515 for NSD, shared-12rep for LAION-fMRI)
  encode    - fit OLS encoding models from the GIST/Gabor features to each ROI

``features --backend`` picks the implementation:
  python  - Python port in dnffa/gist.py (default)
  matlab  - the original PROJECT_DNFFA/HELPERS/Code-GistModel/computeGaborAndGistFeatures.m,
            called with the same parameters as DNFFA_extract_gist.m. Set the executable with
            --matlab-cmd (or DNFFA_MATLAB); octave-cli works too. The MATLAB code expects
            425x425 images (the NSD size; LAION-fMRI images are exported at that size).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scipy.io as sio  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from jsputils import classes  # noqa: E402

from dnffa import config, datasets, gist  # noqa: E402

FEATURE_SPACES = ['GistPC', 'Gabor']
GIST_CODE_DIR = config.REPO_DIR / 'PROJECT_DNFFA' / 'HELPERS' / 'Code-GistModel'


def export_images(ds, feature_dir, subjs):
    for subj in progress_bar(subjs):
        test_fn = feature_dir / f'{ds.test_imageset}_images.mat'
        train_fn = feature_dir / f'{subj}_train-{ds.train_imageset}_images.mat'

        partitions = {}
        if not train_fn.exists():
            partitions['train'] = ds.train_imageset
        if not test_fn.exists():
            partitions['test'] = ds.test_imageset  # shared across subjects
        if not partitions:
            print(subj, 'skipping, already exists')
            continue

        images = ds.load_images(subj, partitions)
        for partition, fn in [('train', train_fn), ('test', test_fn)]:
            if partition in images:
                sio.savemat(fn, {'images': images[partition]})
                print(subj, partition, images[partition].shape)


def image_sets(ds, subjs):
    """(name, exported image file) for the test set and each subject's train set."""
    sets = [(ds.test_imageset, f'{ds.test_imageset}_images.mat')]
    sets += [(subj, f'{subj}_train-{ds.train_imageset}_images.mat') for subj in subjs]
    return sets


def python_features(image_path, out_fns):
    images = sio.loadmat(image_path)['images']
    gabor, gist_pc = gist.compute_gabor_and_gist_features(images)
    sio.savemat(out_fns['Gabor'], {'Gabor': gabor})
    sio.savemat(out_fns['GistPC'], {'GistPC': gist_pc})


def matlab_features(image_path, out_fns, summary_fn, matlab_cmd):
    """Run the original computeGaborAndGistFeatures.m, as DNFFA_extract_gist.m does."""
    def q(path):
        return "'" + str(path).replace("'", "''") + "'"

    orientations = ' '.join(map(str, gist.ORIENTATIONS_PER_SCALE))
    code = (f"addpath({q(GIST_CODE_DIR)}); "
            f"[Gabor, GistPC, fh] = computeGaborAndGistFeatures({q(image_path)}, {gist.N_BLOCKS}, "
            f"[{orientations}], {gist.N_PCS}); "
            f"save({q(out_fns['GistPC'])}, 'GistPC', '-v7'); "
            f"save({q(out_fns['Gabor'])}, 'Gabor', '-v7'); "
            # the summary figure is not used downstream; don't fail without a display
            f"try, print(fh, '-dpng', {q(summary_fn)}); catch err, disp(err.message); end")
    if 'octave' in Path(matlab_cmd).name:
        cmd = [matlab_cmd, '--no-gui', '--quiet', '--eval', code]
    else:
        cmd = [matlab_cmd, '-batch', code]
    subprocess.run(cmd, check=True)


def compute_features(ds, feature_dir, subjs, overwrite, backend='python', matlab_cmd='matlab'):
    if backend == 'matlab' and shutil.which(matlab_cmd) is None:
        raise FileNotFoundError(f'MATLAB executable not found: {matlab_cmd} (set --matlab-cmd or DNFFA_MATLAB)')

    for name, image_fn in image_sets(ds, subjs):
        out_fns = {fs: feature_dir / f'{name}_{fs}.mat' for fs in FEATURE_SPACES}
        if all(fn.exists() for fn in out_fns.values()) and not overwrite:
            print(name, 'skipping, already exists')
            continue

        if backend == 'python':
            python_features(feature_dir / image_fn, out_fns)
        else:
            matlab_features(feature_dir / image_fn, out_fns, feature_dir / f'{name}_summary.png', matlab_cmd)
        print(name, backend, {fs: load_gist_gabor_features(feature_dir, name, fs).shape for fs in FEATURE_SPACES})


def load_gist_gabor_features(feature_dir, name, feature_space):
    """``name`` is a subject id or the test image set name."""
    struct = sio.loadmat(feature_dir / f'{name}_{feature_space}.mat')[feature_space]
    if not isinstance(struct, dict):
        struct = {n: struct[n][0, 0] for n in struct.dtype.names}
    return struct['featureMatrix']


def encode(ds, feature_dir, encoding_dir, subjs, rois, overwrite):
    for feature_space in FEATURE_SPACES:
        test_features = load_gist_gabor_features(feature_dir, ds.test_imageset, feature_space)
        print(feature_space, 'test features:', test_features.shape)

        for roi in rois:
            for subj in progress_bar(subjs):
                ROI = ds.load_roi(ds.load_subject(subj), roi)
                train_features = load_gist_gabor_features(feature_dir, subj, feature_space)
                encoder = classes.EncodingProcedureGistGabor(ROI, feature_space, train_features, test_features,
                                                             method='ols', positive=False, alphas=[None])
                encoder.encode_features(str(encoding_dir), overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['export', 'features', 'encode'])
    datasets.add_arguments(parser)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--backend', choices=['python', 'matlab'], default='python',
                        help='features stage: Python port or the original MATLAB code')
    parser.add_argument('--matlab-cmd', default=os.environ.get('DNFFA_MATLAB', 'matlab'),
                        help='MATLAB (or octave-cli) executable for --backend matlab')
    args = parser.parse_args()
    ds = datasets.from_args(parser, args)

    feature_dir = ds.analysis_dir(config.LOW_LEVEL_SUBDIR)

    if args.stage == 'export':
        export_images(ds, feature_dir, args.subjects)
    elif args.stage == 'features':
        compute_features(ds, feature_dir, args.subjects, args.overwrite, args.backend, args.matlab_cmd)
    else:
        encode(ds, feature_dir, ds.analysis_dir(config.ENCODING_SUBDIR), args.subjects, args.rois, args.overwrite)


if __name__ == '__main__':
    main()
