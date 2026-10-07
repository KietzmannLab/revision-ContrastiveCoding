"""Helpers for loading NSD brain/image data and noise ceilings."""

import sys

import h5py
import numpy as np
import pandas as pd
from jsputils import classes, nsdorg, paths

from . import config


def partition_dict(train=config.TRAIN_IMAGESET, val=config.VAL_IMAGESET, test=config.TEST_IMAGESET):
    # nsdorg mutates the dict it receives, so always hand out a fresh one
    return {'train': train, 'val': val, 'test': test}


def load_subject(subj):
    return classes.fMRISubject(subj, config.SPACE, config.BETA_VERSION)


def load_roi(nsd_subj, roi, ncsnr_threshold=config.NCSNR_THRESHOLD, partitions=None):
    """Load an ROI's betas, apply the ncsnr mask and attach train/val/test encoding data."""
    ROI = classes.BrainRegion(nsd_subj, roi)
    ROI.load_betas()
    ROI.get_ncsnr_mask(threshold=ncsnr_threshold)
    ROI.load_encoding_data(partitions or partition_dict())
    return ROI


def roi_voxels(ROI, partitions=('train', 'val', 'test'), mean=True):
    """lh+rh responses per partition, ncsnr-masked (same as EncodingProcedure.get_encoding_voxels)."""
    y = dict()
    for partition in partitions:
        y[partition] = np.concatenate((ROI.brain_data[partition]['lh'],
                                       ROI.brain_data[partition]['rh']), axis=2)
        if mean:
            y[partition] = np.mean(y[partition], axis=1)
        if ROI.ncsnr_mask is not None:
            y[partition] = y[partition][..., ROI.ncsnr_mask]
    return y


def load_nsd_images(subj, partitions):
    """Stimulus images for a subject, keyed by partition (no brain data loaded)."""
    stim_info_df = pd.read_csv(f'{paths.nsd()}/nsddata/experiments/nsd/nsd_stim_info_merged.csv')
    annotations = nsdorg.load_NSD_coco_annotations(config.SUBJECTS, savedir=paths.nsd_coco_annots())
    coco_dict = nsdorg.get_coco_dict(config.SUBJECTS, annotations)

    images = dict()
    with h5py.File(f'{paths.nsd_stimuli()}/nsd_stimuli.hdf5', 'r') as stim_f:
        for partition, imageset in partitions.items():
            cocos = coco_dict[subj][imageset] if imageset in coco_dict[subj] else coco_dict[imageset]
            nsd_ids = [stim_info_df.loc[stim_info_df['cocoId'] == coco, 'nsdId'].values[0] for coco in cocos]
            images[partition] = np.stack([stim_f['imgBrick'][i] for i in nsd_ids])
    return images


# ---------------------------------------------------------------------------
# Noise ceilings
# ---------------------------------------------------------------------------

def import_rsa_noise_ceiling():
    if config.GSN_DIR not in sys.path:
        sys.path.append(config.GSN_DIR)
    from gsn.rsa_noise_ceiling import rsa_noise_ceiling
    return rsa_noise_ceiling


def noise_ceiling_path(roi, subj, test_imageset=config.TEST_IMAGESET, ncsnr_threshold=config.NCSNR_THRESHOLD,
                       subdir=config.NOISE_CEILING_SUBDIR):
    nc_dir = config.analysis_dir(subdir)
    return nc_dir / f'GSN-NC_{roi}_{subj}_{test_imageset}_nc-{ncsnr_threshold}.npy'


def load_noise_ceilings(roi_list=config.ROI_LIST, subjs=config.SUBJECTS, **path_kwargs):
    """DataFrame with columns ROI, Subject, Univariate, RSA (averaged over GSN resamples).

    ``path_kwargs`` go to ``noise_ceiling_path`` (another dataset's test set / output folder).
    """
    rows = []
    for roi in roi_list:
        for subj in subjs:
            fn = noise_ceiling_path(roi, subj, **path_kwargs)
            if not fn.exists():
                print(fn, 'does not exist')
                continue
            nc = np.load(fn, allow_pickle=True).item()
            ncs = np.mean(np.vstack(nc['ncs']), axis=0) if len(nc['ncs']) > 1 else np.array(nc['ncs'][0])
            rows.append([roi, subj, ncs[0], ncs[1]])
    return pd.DataFrame(rows, columns=['ROI', 'Subject', 'Univariate', 'RSA'])


def noise_ceiling_range(noise_ceilings, roi, metric):
    """[min, max] across subjects of an ROI's noise ceiling for 'veUnivar' or 'veRSA'."""
    col = 'Univariate' if 'Univar' in metric else 'RSA'
    vals = noise_ceilings.loc[noise_ceilings['ROI'] == roi, col].values
    return [np.nanmin(vals), np.nanmax(vals)]
