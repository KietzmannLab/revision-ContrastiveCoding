"""Estimate univariate and RSA noise ceilings of each NSD ROI with GSN.

Converted from PROJECT_DNFFA/NOTEBOOKS/6-NoiseCeilings.ipynb.

Results go to ``analysis_outputs/3c-NoiseCeilings`` and are required by 04_encoding_plots.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

from dnffa import config, nsd  # noqa: E402
from dnffa.plotting import plt, save  # noqa: E402

# Large ROIs are subsampled to keep GSN tractable
MAX_VOXELS = 1000
N_SIMS = 3
NVOX_PER_SAMPLE = 1000

ROI_PALETTE = ['tomato'] * 3 + ['limegreen'] * 2 + ['dodgerblue'] * 3 + ['purple'] * 3


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
        ncs, ncdists, results = rsa_noise_ceiling(sample, rdmfuns=rdmfuns, wantverbose=False)
        out['ncs'].append(ncs)
        out['ncdists'].append(ncdists)
        out['results'].append(results)
        print(ncs)
    return out


def compute_all(subjs, rois, overwrite):
    rsa_noise_ceiling = nsd.import_rsa_noise_ceiling()
    for subj in progress_bar(subjs):
        nsd_subj = nsd.load_subject(subj)
        for roi in progress_bar(rois):
            print(roi, subj)
            savefn = nsd.noise_ceiling_path(roi, subj)
            if savefn.exists() and not overwrite:
                print('skipping')
                continue

            ROI = nsd.load_roi(nsd_subj, roi)
            y = nsd.roi_voxels(ROI, mean=False)
            print(y['test'].shape)
            if y['test'].shape[2] > 2:
                np.save(savefn, compute_gsn_noise_ceiling(y['test'], rsa_noise_ceiling), allow_pickle=True)
            else:
                print('skipping, insufficient voxels')


def plot_noise_ceilings(df, savedir, space=0.3):
    unique_rois = df['ROI'].unique()
    unique_subjs = df['Subject'].unique()
    n_subj = len(unique_subjs)

    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
    for i, metric in enumerate(['Univariate', 'RSA']):
        ax2 = ax[i].twiny()
        for j, roi in enumerate(unique_rois):
            roi_data = df[df['ROI'] == roi]
            for k, subj in enumerate(unique_subjs):
                subj_data = roi_data[roi_data['Subject'] == subj]
                if subj_data.empty:
                    continue
                x_pos = j * (1 + space) + k * 0.1
                val = subj_data[metric].tolist()[0]
                ax[i].scatter(x_pos, val, color=ROI_PALETTE[j], s=30)
                ax[i].vlines(x_pos, 0, val, color=ROI_PALETTE[j], linestyle='dashed')

        ax[i].set_xticks([j * (1 + space) + k * 0.1 for j in range(len(unique_rois)) for k in range(n_subj)])
        ax[i].set_xticklabels([f'subj{str(k + 1).zfill(2)}' for _ in range(len(unique_rois)) for k in range(n_subj)],
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
    parser.add_argument('--rois', nargs='+', default=config.ROI_LIST)
    parser.add_argument('--subjects', nargs='+', default=config.SUBJECTS)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if not args.plot_only:
        compute_all(args.subjects, args.rois, args.overwrite)

    df = nsd.load_noise_ceilings(args.rois, args.subjects)
    plot_noise_ceilings(df, config.figure_dir('NoiseCeilings'))


if __name__ == '__main__':
    main()
