"""Estimate univariate and RSA noise ceilings of each NSD (or LAION-fMRI) ROI with GSN.

Converted from PROJECT_DNFFA/NOTEBOOKS/6-NoiseCeilings.ipynb.

Results go to ``analysis_outputs/3c-NoiseCeilings`` (``3c-NoiseCeilings-laion`` with --dataset laion)
and are required by 04_encoding_plots.py.
"""

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

from dnffa import config, datasets, nsd  # noqa: E402
from dnffa.plotting import plt, save  # noqa: E402

# Large ROIs are subsampled to keep GSN tractable
MAX_VOXELS = 1000
N_SIMS = 3
NVOX_PER_SAMPLE = 1000


def run_rsa_noise_ceiling(rsa_noise_ceiling, data, rdmfuns):
    """One noise ceiling per RDM function, as (ncs, ncdists, results).

    The notebook used a pre-release GSN whose ``rsa_noise_ceiling`` took a list of
    ``rdmfuns``. Public GSN (github.com/cvnlab/GSN) takes a single ``opt['rdmfun']``,
    so it is called once per function, with its default figure turned off.
    """
    if 'rdmfuns' in inspect.signature(rsa_noise_ceiling).parameters:
        return rsa_noise_ceiling(data, rdmfuns=rdmfuns, wantverbose=False)
    runs = [rsa_noise_ceiling(data, {'rdmfun': rdmfun, 'wantverbose': 0, 'wantfig': 0}) for rdmfun in rdmfuns]
    ncs, ncdists, results = zip(*runs)
    return np.array(ncs), list(ncdists), list(results)


def compute_gsn_noise_ceiling(data, rsa_noise_ceiling):
    """``data`` is (images x reps x voxels)."""
    data_gsn = np.transpose(data, (2, 0, 1))
    rdmfuns = [lambda x: np.mean(x.T, axis=1),
               lambda x: pdist(x.T, 'correlation')]

    out = {'ncs': [], 'ncdists': [], 'results': []}
    n_sims = N_SIMS if data_gsn.shape[0] > MAX_VOXELS else 1
    for _ in range(n_sims):
        if data_gsn.shape[0] > MAX_VOXELS:
            # random voxel subset (drawn right before each GSN call, as in the notebook)
            sample = data_gsn[np.random.choice(np.arange(data_gsn.shape[0]), NVOX_PER_SAMPLE)]
        else:
            sample = data_gsn
        ncs, ncdists, results = run_rsa_noise_ceiling(rsa_noise_ceiling, sample, rdmfuns)
        out['ncs'].append(ncs)
        out['ncdists'].append(ncdists)
        out['results'].append(results)
        print(ncs)
    return out


def compute_all(ds, subjs, rois, overwrite):
    rsa_noise_ceiling = nsd.import_rsa_noise_ceiling()
    for subj in progress_bar(subjs):
        subject = ds.load_subject(subj)
        for roi in progress_bar(rois):
            print(roi, subj)
            savefn = ds.noise_ceiling_path(roi, subj)
            if savefn.exists() and not overwrite:
                print('skipping')
                continue

            ROI = ds.load_roi(subject, roi)
            y = nsd.roi_voxels(ROI, mean=False)
            print(y['test'].shape)
            if y['test'].shape[2] > 2:
                np.save(savefn, compute_gsn_noise_ceiling(y['test'], rsa_noise_ceiling), allow_pickle=True)
            else:
                print('skipping, insufficient voxels')


def plot_noise_ceilings(df, roi_domain, savedir, space=0.3):
    unique_rois = df['ROI'].unique()
    unique_subjs = df['Subject'].unique()
    n_subj = len(unique_subjs)

    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
    for i, metric in enumerate(['Univariate', 'RSA']):
        ax2 = ax[i].twiny()
        for j, roi in enumerate(unique_rois):
            roi_data = df[df['ROI'] == roi]
            color = config.DOMAIN_COLORS[roi_domain[roi]]
            for k, subj in enumerate(unique_subjs):
                subj_data = roi_data[roi_data['Subject'] == subj]
                if subj_data.empty:
                    continue
                x_pos = j * (1 + space) + k * 0.1
                val = subj_data[metric].tolist()[0]
                ax[i].scatter(x_pos, val, color=color, s=30)
                ax[i].vlines(x_pos, 0, val, color=color, linestyle='dashed')

        ax[i].set_xticks([j * (1 + space) + k * 0.1 for j in range(len(unique_rois)) for k in range(n_subj)])
        ax[i].set_xticklabels([subj for _ in range(len(unique_rois)) for subj in unique_subjs],
                              rotation='vertical', fontsize='small')
        ax2.set_xticks([j * (1 + space) + (n_subj - 1) * 0.05 for j in range(len(unique_rois))])
        ax2.set_xticklabels(unique_rois, fontsize=14)

        ax[i].set_title(f'{metric} Noise Ceiling', fontsize=20)
        ax[i].set_ylabel('Noise Ceiling (r)', fontsize=14)
        ax[i].set_ylim([0, 1])
        ax[i].set_xlim([-0.5, len(unique_rois) * (1 + space)])
        ax2.set_xlim(ax[i].get_xlim())

    plt.tight_layout()
    save(savedir / 'noise_ceilings.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    datasets.add_arguments(parser)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    ds = datasets.from_args(parser, args)

    if not args.plot_only:
        compute_all(ds, args.subjects, args.rois, args.overwrite)

    df = ds.load_noise_ceilings(args.rois, args.subjects)
    plot_noise_ceilings(df, ds.roi_domain, ds.figure_dir('NoiseCeilings'))


if __name__ == '__main__':
    main()
