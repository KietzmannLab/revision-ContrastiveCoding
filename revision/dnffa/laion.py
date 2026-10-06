"""LAION-fMRI loaders that mimic the NSD objects used by jsputils.

``jsputils.classes.EncodingProcedure`` (and the noise-ceiling code) only read a
few attributes from the ROI object they are given. ``LaionROI`` provides them:

    brain_data[partition]['lh' | 'rh']   images x repetitions x voxels (float32)
    image_data[partition]                images x H x W x 3 (uint8)
    ncsnr_mask                           bool over the ROI's voxels (lh then rh)
    subj, roi, ncsnr_threshold_          used in output filenames

Usage mirrors ``dnffa.nsd``::

    from dnffa import laion
    ROI = laion.load_roi(laion.load_subject('sub-01'), 'FFA-1')
    encoder = classes.EncodingProcedure(ROI, DNN, method='lasso', positive=True, alphas=[0.1])

Preprocessing follows ``jsputils.nsdorg.load_betas``: single-trial betas are
z-scored per voxel within each session (over all of that session's trials),
then grouped as images x repetitions in chronological order.

Differences from NSD that cannot be avoided:

* The data are volumetric (T1w, 1.8 mm). Voxels are split into 'lh'/'rh' by the
  sign of their world x coordinate (RAS). EncodingProcedure concatenates the two,
  so the split is bookkeeping only and does not affect results.
* LAION-fMRI ships GLMsingle noise ceilings (percent variance explained for an
  n-trial average) instead of ncsnr. Both use the NSD estimator (Allen et al.
  2022), NC = 100 * ncsnr^2 / (ncsnr^2 + 1/n), so ncsnr is recovered exactly by
  inverting that formula and the NSD threshold (ncsnr > 0.3) is applied as is.
* Voxels whose GLMsingle fit failed (NaN betas in any trial) are dropped from the
  ROI; scikit-learn cannot fit NaNs.
"""

import re
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats

if __package__:
    from . import config
else:  # run as a file (python dnffa/laion.py or the IDE's run button)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dnffa import config

DATA_DIR = '/share/klab/datasets/optimized_datasets/laion_fmri_data'

SUBJECTS = ['sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07']

# NSD-style ROI name -> LAION-fMRI label. LAION has a single FBA and no OWFA.
ROI_LABELS = {'FFA-1': 'FFA1', 'FFA-2': 'FFA2', 'OFA': 'OFA',
              'PPA': 'PPA', 'OPA': 'OPA',
              'EBA': 'EBA', 'FBA': 'FBA',
              'VWFA-1': 'VWFA1', 'VWFA-2': 'VWFA2'}
ROI_LIST = list(ROI_LABELS)

ROI_DOMAIN = {'OFA': 'faces', 'FFA-1': 'faces', 'FFA-2': 'faces',
              'OPA': 'scenes', 'PPA': 'scenes',
              'EBA': 'bodies', 'FBA': 'bodies',
              'VWFA-1': 'characters', 'VWFA-2': 'characters'}

# Default partitions, analogous to NSD's nonshared1000-3rep-batch0/1 and special515
TRAIN_IMAGESET = 'unique1000-4rep-batch0'
VAL_IMAGESET = 'unique1000-4rep-batch1'
TEST_IMAGESET = 'shared-12rep'

# Noise ceiling map used to derive ncsnr, and the number of trials it was computed for
NC_DESC = 'Noiseceiling4rep'
NC_NREPS = 4
NCSNR_THRESHOLD = config.NCSNR_THRESHOLD

CACHE_SUBDIR = 'laion-fmri'

_metadata_cache = {}


def initialize(data_dir=DATA_DIR):
    """Point laion_fmri at the local data copy (only writes its config if needed)."""
    from laion_fmri.config import dataset_initialize, get_data_dir
    try:
        if str(get_data_dir()) == str(data_dir):
            return
    except Exception:
        pass
    dataset_initialize(data_dir)


def partition_dict(train=TRAIN_IMAGESET, val=VAL_IMAGESET, test=TEST_IMAGESET):
    return {'train': train, 'val': val, 'test': test}


def ncsnr_from_noise_ceiling(nc_percent, n_reps=NC_NREPS):
    """Invert NC = 100 * ncsnr^2 / (ncsnr^2 + 1/n)."""
    f = np.clip(np.asarray(nc_percent, dtype=float) / 100, 0, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(f / (n_reps * (1 - f)))


# ---------------------------------------------------------------------------
# Image sets
# ---------------------------------------------------------------------------

def trial_table(subj):
    """laion_fmri trial table (one row per single-trial beta), cached per subject."""
    if subj not in _metadata_cache:
        from laion_fmri.subject import load_subject as _load
        initialize()
        _metadata_cache[subj] = _load(subj).metadata
    return _metadata_cache[subj]


def _images_with_reps(metadata, n_reps, pattern=None, shared=None):
    reps = metadata.groupby('image_name').size()
    names = reps.index[reps.values == n_reps]
    if shared is not None:
        status = metadata.groupby('image_name')['unique_or_shared'].first()
        names = names[(status.loc[names] == ('shared' if shared else 'unique')).values]
    if pattern is not None:
        names = names[[re.search(pattern, n) is not None for n in names]]
    return np.sort(np.asarray(names))


def _shared_across_subjects(n_reps, pattern):
    """Images with exactly ``n_reps`` repetitions in every subject (cf. NSD special515)."""
    common = None
    for subj in SUBJECTS:
        names = _images_with_reps(trial_table(subj), n_reps, pattern=pattern, shared=True)
        common = names if common is None else np.intersect1d(common, names)
    return np.sort(common)


def image_set(subj, name):
    """Sorted image names of an image set, and the number of repetitions per image.

    unique-4rep                 all of the subject's unique images shown 4x
    unique1000-4rep-batch{k}    k = 0..3, 1000 of them (same rule as NSD's nonshared1000-3rep-batch{k})
    shared-12rep                shared LAION images shown 12x to every subject
    shared-4rep                 shared non-OOD images shown 4x to every subject
    shared-4rep-ood             shared OOD images shown 4x to every subject
    """
    if name == 'unique-4rep':
        return _images_with_reps(trial_table(subj), 4, shared=False), 4
    m = re.fullmatch(r'unique1000-4rep-batch(\d)', name)
    if m:
        names = _images_with_reps(trial_table(subj), 4, shared=False)
        assert len(names) >= 4000, f'{subj} has only {len(names)} unique 4-rep images'
        return names[:4000][int(m.group(1))::4], 4
    if name == 'shared-12rep':
        return _shared_across_subjects(12, r'^shared_12rep_'), 12
    if name == 'shared-4rep':
        return _shared_across_subjects(4, r'^shared_4rep_(?!OOD_)'), 4
    if name == 'shared-4rep-ood':
        return _shared_across_subjects(4, r'^shared_4rep_OOD_'), 4
    raise ValueError(f'unknown LAION-fMRI image set: {name}')


# ---------------------------------------------------------------------------
# Subject: z-scored betas for the union of ROIs, cached to disk
# ---------------------------------------------------------------------------

class LaionSubject:
    """Counterpart of ``jsputils.classes.fMRISubject``."""

    def __init__(self, subj, rois=None):
        from laion_fmri.subject import load_subject as _load
        initialize()
        self.subj = subj
        self.space = 'T1w-1pt8'
        self.sub = _load(subj)
        self.metadata = trial_table(subj)
        self.rois = list(rois or ROI_LIST)
        self._betas = None

    def _cache_paths(self):
        cache_dir = config.analysis_dir(CACHE_SUBDIR)
        stem = f'{self.subj}_zscored-betas'
        return cache_dir / f'{stem}.npy', cache_dir / f'{stem}_voxels.npz'

    def roi_masks(self):
        """ROI -> bool mask over the brain-mask voxels."""
        return {roi: self.sub.get_roi_mask(ROI_LABELS[roi]) for roi in self.rois}

    def load_betas(self, streaming=True):
        """(n_trials x n_union_voxels) betas, z-scored within session, plus voxel info.

        Reads each session file once; the result is cached under
        ``analysis_outputs/laion-fmri`` and reused as long as it covers ``self.rois``.
        """
        if self._betas is not None:
            return self._betas

        betas_fn, voxels_fn = self._cache_paths()
        masks = self.roi_masks()
        union = np.any(np.vstack(list(masks.values())), axis=0)

        if betas_fn.exists() and voxels_fn.exists():
            cached_union = np.load(voxels_fn)['union']
            if np.all(cached_union[union]):
                print(f'{self.subj}: loading cached betas')
                self._betas = self._betas_dict(np.load(betas_fn, mmap_mode='r'), cached_union, masks)
                return self._betas
            union |= cached_union  # extend the cache instead of shrinking it

        sessions = list(dict.fromkeys(self.metadata['session']))
        n_trials = self.metadata['session'].value_counts().loc[sessions].to_numpy()
        betas = np.empty((len(self.metadata), int(union.sum())), dtype=np.float32)
        start = 0
        for ses, n in zip(sessions, n_trials):
            print(f'{self.subj} {ses}: loading betas for {int(union.sum())} voxels')
            ses_betas = self.sub.get_betas(session=ses, mask=union, streaming=streaming)
            assert ses_betas.shape[0] == n, (ses, ses_betas.shape, n)
            # rows are trials in session order; z-score each voxel within the session
            betas[start:start + n] = stats.zscore(ses_betas.astype(float), axis=0)
            start += n

        np.save(betas_fn, betas)
        np.savez(voxels_fn, union=union)
        self._betas = self._betas_dict(betas, union, masks)
        return self._betas

    def _betas_dict(self, betas, union, masks):
        x = self.sub.get_voxel_coordinates(mask=union)[:, 0]
        nc = self.sub.get_noise_ceiling(desc=NC_DESC)[union]
        return {'betas': betas,
                'union': union,
                'roi_in_union': {roi: m[union] for roi, m in masks.items()},
                'is_left': x < 0,
                'ncsnr': ncsnr_from_noise_ceiling(nc)}


def load_subject(subj, rois=None):
    return LaionSubject(subj, rois)


# ---------------------------------------------------------------------------
# ROI: the object handed to EncodingProcedure
# ---------------------------------------------------------------------------

class LaionROI:
    """Counterpart of ``jsputils.classes.BrainRegion`` for one LAION-fMRI ROI."""

    def __init__(self, subject, roi, ncsnr_threshold=0):
        self.subject = subject
        self.subj = subject.subj
        self.space = subject.space
        self.roi = roi
        self.ncsnr_threshold = ncsnr_threshold
        self.ncsnr_mask = None

    def load_betas(self):
        """Select this ROI's voxels: label & ncsnr > ncsnr_threshold & finite betas, lh then rh."""
        data = self.subject.load_betas()
        betas = data['betas']
        candidates = data['roi_in_union'][self.roi] & (data['ncsnr'] > self.ncsnr_threshold)

        cols = np.flatnonzero(candidates)
        finite = np.isfinite(betas[:, cols]).all(axis=0)
        if not finite.all():
            print(f'{self.subj} {self.roi}: dropping {np.sum(~finite)} voxels with failed GLMsingle fits')
        cols = cols[finite]

        is_left = data['is_left'][cols]
        self.voxel_idx = {'lh': cols[is_left], 'rh': cols[~is_left]}
        self.voxel_idx['full'] = np.concatenate((self.voxel_idx['lh'], self.voxel_idx['rh']))
        self.ncsnr = data['ncsnr'][self.voxel_idx['full']]
        print(f'{self.subj} {self.roi}: {len(self.voxel_idx["lh"])} lh + {len(self.voxel_idx["rh"])} rh voxels')

    def get_ncsnr_mask(self, threshold):
        self.ncsnr_threshold_ = threshold
        self.ncsnr_mask = self.ncsnr > threshold

    def load_encoding_data(self, partition_dict, image_size=None):
        """Fill brain_data, image_data and image_metadata for each partition.

        ``image_size`` optionally resizes the (1000 x 1000) stimuli, e.g. 425 to match NSD.
        """
        print('loading train, val, and test data')
        metadata = self.subject.metadata
        betas = self.subject.load_betas()['betas']

        self.brain_data, self.image_data, self.image_metadata = {}, {}, {}
        for partition, imageset in partition_dict.items():
            names, n_reps = image_set(self.subj, imageset)
            trials = metadata[metadata['image_name'].isin(names)]
            # images x repetitions trial indices, repetitions in chronological order
            grouped = trials.groupby('image_name').apply(lambda t: t.index.to_numpy())
            trial_idx = np.stack(grouped.loc[names].values)
            assert trial_idx.shape == (len(names), n_reps), (imageset, trial_idx.shape)

            self.brain_data[partition] = {
                hemi: np.asarray(betas[trial_idx.ravel()][:, self.voxel_idx[hemi]], dtype=np.float32)
                .reshape(len(names), n_reps, -1)
                for hemi in ['lh', 'rh']}
            self.image_data[partition] = self._load_images(trial_idx[:, 0], image_size)
            self.image_metadata[partition] = metadata.loc[trial_idx[:, 0],
                                                          ['image_name', 'stim_idx', 'dataset']].reset_index()

    def _load_images(self, trial_indices, image_size):
        images = None
        for i, t in enumerate(trial_indices):
            img = self.subject.sub.images.get(int(t), as_displayed=True)
            if image_size is not None:
                img = img.resize((image_size, image_size))
            img = np.asarray(img, dtype=np.uint8)
            if images is None:  # preallocate once (stimuli are all the same size)
                images = np.empty((len(trial_indices),) + img.shape, dtype=np.uint8)
            images[i] = img
        return images


def load_roi(subject, roi, ncsnr_threshold=NCSNR_THRESHOLD, partitions=None, image_size=None):
    """Same steps as ``dnffa.nsd.load_roi``: select voxels, ncsnr mask, attach encoding data."""
    ROI = LaionROI(subject, roi)
    ROI.load_betas()
    ROI.get_ncsnr_mask(threshold=ncsnr_threshold)
    ROI.load_encoding_data(partitions or partition_dict(), image_size=image_size)
    return ROI


if __name__ == '__main__':
    # quick look at the dataset (no betas are loaded)
    initialize()
    for subj in SUBJECTS:
        sub = load_subject(subj)
        print(subj, len(sub.sub.get_sessions()), 'sessions,', len(sub.metadata), 'trials')
    for name in ['unique1000-4rep-batch0', 'unique1000-4rep-batch1', 'shared-12rep', 'shared-4rep', 'shared-4rep-ood']:
        names, n_reps = image_set(SUBJECTS[0], name)
        print(f'{name}: {len(names)} images x {n_reps} reps')
    print('ROIs:', {roi: label for roi, label in ROI_LABELS.items()})
