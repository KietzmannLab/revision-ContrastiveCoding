"""Generalization of the Fig. 4 encoding results to the re:vision LAION-fMRI splits.

Refits the Fig. 4 models (contrastive selective units, untrained selective units, face-trained
AlexNet, GistPC, Gabor) on the predefined splits of ``laion_fmri.splits`` and tests whether the
contrastive model still predicts each ROI's univariate profile and RDM best
(https://re-vision-initiative.org/generalization):

  Method 1  tau             80/20 split with test images far from their nearest training images
  Method 2  clusters        cluster_k5_0 .. cluster_k5_4: train on 4 CLIP-feature clusters, test on
                            the held-out one; scores are averaged over the 5 folds
  Method 3  ood-<type>      train on the regular images, test on OOD images of one type
                            (default: the 64 ``unusual`` images; --ood-types)
  Baseline  random          random_0 .. random_4: one shuffled 5-fold CV partition, averaged

Each subject's own pool is used (1,121 shared + 4,712 unique images, plus the 371 OOD images).
The fits follow 03 / 07, with three changes the splits require:

* Layer selection: the replication picks each model's layer on a separate 1,000-image validation
  set. Here a seeded VAL_FRACTION of every split's training images (~930 images) is held out for
  that; models are fit on the rest and the selected fit is scored on the test set, so test images
  never influence model selection. The validation images are the same for all models and ROIs.
* Repetitions: every image is represented by its first 4 trials (laion.SPLIT_NREPS), also the
  shared images shown 12x, so train and test images have the same noise level.
* GistPC: the PCA is fit on the training images and applied to the test images. The original
  code fits it separately on each image set (see dnffa/gist.py), which is undefined for the
  64-image OOD test set and leaves train and test PCs unaligned.

The Barlow Twins models use the ROI's preferred-domain units (the only ones Fig. 4 shows), the
face-trained model whole layers. Mappers and penalties are those of 03 / 07.

Stages, all reading/writing ``analysis_outputs/3f-Generalization-laion``:
  features        DNN features of all images in each subject's pool (GPU). Writes one .npy per
                  subject, model, layer and unit set to ``features/``; the stimuli are cached in
                  ``analysis_outputs/laion-fmri``.
  lowlevel        Gabor features (dnffa.gist) of the same images (CPU, --n-jobs processes).
  fit             encoding models for every split, layer and ROI (CPU). Writes one CSV per subject,
                  ROI, split and model to ``results/`` (val and test scores of every layer) and the
                  test-set predictions of the layers selected on the validation images to
                  ``predictions/``.
  noise-ceilings  GSN noise ceilings of every test set, as 06 (CPU).
  test            per-subject permutation tests (as 08, fold-averaged for clusters / random),
                  paired t-tests across subjects (as 04), summary tables and figures in
                  ``figure_outputs/Figure4-Generalization-laion``.

--subjects / --rois / --models / --splits restrict the work, e.g. to split a stage into
parallel jobs. Requires 01's selective-unit cache; ``test`` also reads 03 / 06 / 07's LAION
outputs to show the replication next to the generalization tests, if they exist.
"""

import argparse
import importlib
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from jsputils import classes  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from threadpoolctl import ThreadpoolController, threadpool_limits  # noqa: E402

from dnffa import config, datasets, gist, laion, nsd  # noqa: E402
from dnffa.laion import _save_atomic  # noqa: E402
from dnffa.plotting import despine, plt, save  # noqa: E402
from dnffa.stats import pairwise_ttest_with_first, permutation_test_rsa, permutation_test_univar  # noqa: E402

plots = importlib.import_module('04_encoding_plots')
noise = importlib.import_module('06_noise_ceilings')
perm = importlib.import_module('08_encoding_permutation')

GENERALIZATION_SUBDIR = '3f-Generalization'
METRICS = perm.METRICS
REFERENCE_MODEL = perm.REFERENCE_MODEL
MODEL_NAMES = plots.MODEL_NAMES
DNN_MODELS = plots.DNN_MODELS
NC = config.NCSNR_THRESHOLD

# Mapper per model, as in the 03 / 07 runs that Fig. 4 uses
MAPPERS = {'alexnet-barlow-twins': ('lasso', True, 0.1),
           'alexnet-barlow-twins-random': ('lasso', True, 0.001),
           'alexnet-vggface': ('lasso', True, 0.1),
           'GistPC': ('ols', False, None),
           'Gabor': ('ols', False, None)}

VAL_FRACTION = 0.2
CLUSTER_SPLITS = [f'cluster_k5_{k}' for k in range(5)]
RANDOM_SPLITS = [f'random_{k}' for k in range(5)]
SPLITS = ['tau'] + CLUSTER_SPLITS + ['ood'] + RANDOM_SPLITS
OOD_TYPES = ['unusual']
# Order of the models within a condition in the summary figure (Fig. 4's order)
PLOT_ORDER = plots.PAPER_MODEL_ORDER
FEATURE_BATCH_SIZE = 250

METHOD_TITLES = {'tau': 'tau split', 'clusters': 'held-out clusters (mean of 5 folds)',
                 'random': 'random 5-fold baseline (mean of 5 folds)', 'replication': 'replication'}


def method_title(method):
    if method.startswith('ood-'):
        return f'OOD images: {method[4:]}'
    return METHOD_TITLES[method]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_set_names(split, ood_types):
    """Test sets of a split: the split itself, or one per OOD type for the ``ood`` split."""
    return [f'ood-{t}' for t in ood_types] if split == 'ood' else [split]


def split_of(test_set):
    return 'ood' if test_set.startswith('ood-') else test_set


def methods(splits, ood_types):
    """Reported method -> its test sets; a method's scores are averaged over its test sets."""
    out = {}
    if 'tau' in splits:
        out['tau'] = ['tau']
    if set(CLUSTER_SPLITS) <= set(splits):
        out['clusters'] = CLUSTER_SPLITS
    if 'ood' in splits:
        out.update({f'ood-{t}': [f'ood-{t}'] for t in ood_types})
    if set(RANDOM_SPLITS) <= set(splits):
        out['random'] = RANDOM_SPLITS
    return out


class SplitPool:
    """A subject's split pool (laion.split_pool) and index sets of its splits into it."""

    def __init__(self, subj):
        self.subj = subj
        self.names, self.trial_idx = laion.split_pool(subj)
        self._pos = {name: i for i, name in enumerate(self.names)}

    def indices(self, names):
        idx = np.array([self._pos[n] for n in names if n in self._pos], dtype=int)
        if len(idx) < len(names):
            print(f'\t{self.subj}: {len(names) - len(idx)} of {len(names)} images have < {laion.SPLIT_NREPS} trials')
        return idx

    def partitions(self, split, ood_types):
        """{'train', 'val', <test sets>} -> indices; 'val' is a seeded VAL_FRACTION of the split's train images."""
        if split == 'ood':
            train, _ = laion.split_ids(self.subj, 'ood')
            tests = {f'ood-{t}': laion.split_ids(self.subj, 'ood', None if t == 'all' else [t])[1] for t in ood_types}
        else:
            train, test = laion.split_ids(self.subj, split)
            tests = {split: test}
        train = self.indices(train)
        rng = np.random.default_rng(zlib.crc32(f'{self.subj}|{split}'.encode()))
        val = np.sort(rng.choice(train, int(round(VAL_FRACTION * len(train))), replace=False))
        parts = {'train': np.setdiff1d(train, val), 'val': val}
        parts.update({name: self.indices(names) for name, names in tests.items()})
        return parts


def roi_responses(subject, roi, trial_idx):
    """images x reps x voxels responses of an ROI (voxels as in 03: ncsnr > 0.3)."""
    ROI = laion.LaionROI(subject, roi)
    ROI.load_betas()
    ROI.get_ncsnr_mask(threshold=NC)
    return ROI.responses(trial_idx)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def unit_set(model, roi, roi_domain):
    """Units a model is fit with in an ROI: whole layers for the face-trained model, else the ROI's domain."""
    if model in DNN_MODELS:
        return 'layer' if 'vggface' in model else roi_domain[roi]
    return 'N/A'


def feature_path(feature_dir, subj, model, layer, domain):
    return feature_dir / f'{subj}_{model}_{layer}_{domain}.npy'


def names_path(feature_dir, subj):
    return feature_dir / f'{subj}_split-pool_names.npy'


def results_path(results_dir, subj, roi, split, model):
    return results_dir / f'{subj}_{roi}_nc-{NC}_{split}_{model}.csv'


def prediction_path(pred_dir, subj, roi, test_set, model, layer, domain):
    if model in DNN_MODELS:
        return pred_dir / f'{subj}_{roi}_nc-{NC}_{test_set}_{model}_{layer}_{domain}_test-pred.npy'
    return pred_dir / f'{subj}_{roi}_nc-{NC}_{test_set}_{model}_test-pred.npy'


def voxels_path(pred_dir, subj, roi, test_set):
    return pred_dir / f'{subj}_{roi}_nc-{NC}_{test_set}_test-voxels.npy'


def noise_ceiling_subdir(ds):
    return f'{ds.subdir(GENERALIZATION_SUBDIR)}/noise-ceilings'


def check_pool_names(feature_dir, pool):
    fn = names_path(feature_dir, pool.subj)
    if not fn.exists() or not np.array_equal(np.load(fn), pool.names):
        raise FileNotFoundError(f'{fn} is missing or out of date; run the features stage for {pool.subj}')


# ---------------------------------------------------------------------------
# features / lowlevel
# ---------------------------------------------------------------------------

def load_dnn(model):
    import torch
    if 'random' in model:
        torch.manual_seed(config.RANDOM_MODEL_SEED)  # same untrained weights as 03 / 08
    DNN = classes.DNNModel(model)
    if 'vggface' not in model:
        DNN.find_selective_units(config.FLOC_IMAGESET, overwrite=False, verbose=False, FDR_p=config.FDR_P)
    return DNN


def extract_features(ds, feature_dir, subjects, models, rois, overwrite, batch_size=FEATURE_BATCH_SIZE):
    import torch
    for model in [m for m in models if m in DNN_MODELS]:
        DNN = load_dnn(model)
        units = sorted({unit_set(model, roi, ds.roi_domain) for roi in rois})
        for subj in subjects:
            pool = SplitPool(subj)
            out_fns = {(layer, domain): feature_path(feature_dir, subj, model, layer, domain)
                       for layer in config.encoding_layers(model) for domain in units}
            names_fn = names_path(feature_dir, subj)
            up_to_date = names_fn.exists() and np.array_equal(np.load(names_fn), pool.names)
            if up_to_date and all(fn.exists() for fn in out_fns.values()) and not overwrite:
                print(model, subj, 'skipping, features exist')
                continue

            images = laion.split_pool_images(subj, pool.names)
            print(model, subj, f'extracting features of {len(images)} images', flush=True)
            feats = {key: [] for key in out_fns}
            for start in progress_bar(range(0, len(images), batch_size)):
                DNN.get_nsd_features(torch.from_numpy(images[start:start + batch_size]), field='split_features')
                for layer, domain in out_fns:
                    X = DNN.split_features[layer]
                    X = np.reshape(X, (X.shape[0], -1))
                    if domain != 'layer':
                        X = X[:, DNN.selective_units[domain][layer]['mask']]
                    feats[(layer, domain)].append(np.asarray(X, dtype=np.float32))
            for key, fn in out_fns.items():
                _save_atomic(fn, np.save, np.concatenate(feats[key]))
            _save_atomic(names_fn, np.save, pool.names)


def _gabor_block(images):
    G = gist.gabor_transfer_functions(gist.ORIENTATIONS_PER_SCALE, images.shape[1])
    buf = np.empty(G.shape)
    return np.stack([gist.local_features(np.asarray(im, dtype=np.float64).mean(axis=2), G, gist.N_BLOCKS, buf)
                     for im in images])


def gabor_path(feature_dir, owner):
    return feature_dir / f'{owner}_split-pool_Gabor.npy'


def extract_gabor(feature_dir, subjects, n_jobs, overwrite):
    """Gabor features (Gabor.featureMatrix of computeGaborAndGistFeatures) of every pool image.

    Shared images are computed once for all subjects, unique images per subject.
    """
    for subj in subjects:
        for owner, block in laion.split_pool_blocks(subj).items():
            fn = gabor_path(feature_dir, owner)
            names_fn = fn.with_name(f'{fn.stem}_names.npy')
            if fn.exists() and names_fn.exists() and np.array_equal(np.load(names_fn), block) and not overwrite:
                print(owner, 'skipping, Gabor features exist')
                continue
            images = laion.split_pool_images(subj, block)
            print(owner, f'computing Gabor features of {len(images)} images with {n_jobs} processes', flush=True)
            bounds = np.linspace(0, len(images), min(len(images), 8 * n_jobs) + 1).astype(int)
            parts = Parallel(n_jobs=n_jobs)(delayed(_gabor_block)(images[a:b]) for a, b in zip(bounds, bounds[1:]))
            _save_atomic(fn, np.save, np.concatenate(parts))
            _save_atomic(names_fn, np.save, block)


def load_gabor(feature_dir, pool):
    """(pool images x 7680) Gabor features in pool order."""
    rows = {}
    blocks = {}
    for owner, block in laion.split_pool_blocks(pool.subj).items():
        fn = gabor_path(feature_dir, owner)
        names_fn = fn.with_name(f'{fn.stem}_names.npy')
        if not (fn.exists() and names_fn.exists() and np.array_equal(np.load(names_fn), block)):
            raise FileNotFoundError(f'{fn} is missing or out of date; run the lowlevel stage for {pool.subj}')
        blocks[owner] = np.load(fn, mmap_mode='r')
        rows.update({name: (owner, i) for i, name in enumerate(block)})
    return np.stack([blocks[rows[n][0]][rows[n][1]] for n in pool.names])


def gist_pcs(gabor, train_idx, n_pcs=gist.N_PCS):
    """GistPC features of all pool images, with the (uncentered, as pca.m) PCA fit on the training images."""
    U, _, _ = np.linalg.svd(gabor[train_idx].T, full_matrices=False)
    return gabor @ U[:, :n_pcs]


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def ve_scores(pred, true):
    """veUnivar and veRSA as computed by jsputils' EncodingProcedure."""
    scores = {'veUnivar': stats.pearsonr(pred.mean(axis=1), true.mean(axis=1))[0]}
    pred_rdm = pdist(pred, 'correlation')
    prop_bad, valid = classes.check_nan_or_inf(pred_rdm)
    scores['veRSA'] = (np.nan if prop_bad > 0.1 else
                       stats.pearsonr(pred_rdm[valid], pdist(true, 'correlation')[valid])[0])
    return scores


def fit_and_score(X, y, parts, test_sets, mapper):
    """Fit on parts['train'] and score on 'val' and every test set; None if there are too few features / voxels."""
    if X.shape[1] <= 1 or y.shape[1] <= 1:  # as EncodingProcedure
        return None
    mdl = classes.get_mapper(*mapper)
    mdl.fit(X[parts['train']], y[parts['train']])
    if hasattr(mdl, 'coef_') and np.all(mdl.coef_ == 0):
        print('\twarning: all model weights are 0')
    scores, preds = {}, {}
    for pt in ['val'] + test_sets:
        pred = mdl.predict(X[parts[pt]])
        scores[pt] = ve_scores(pred, y[parts[pt]])
        if pt != 'val':
            preds[pt] = pred.astype(np.float32)
    return scores, preds


def _single_threaded_openmp(controllers):
    for controller in controllers:
        controller.set_num_threads(1)


def result_rows(subj, roi, split, model, layer, domain, mapper, parts, scores):
    method, positive, alpha = mapper
    base = {'subj': subj, 'ROI': roi, 'ncsnr_threshold': NC, 'model_name': model, 'layer': layer, 'domain': domain,
            'method': method, 'positive': positive, 'alpha': alpha, 'split': split, 'n_train': len(parts['train'])}
    return [{**base, 'partition': 'val' if pt == 'val' else 'test', 'test_set': None if pt == 'val' else pt,
             'n_images': len(parts[pt]), **s} for pt, s in scores.items()]


def fit_subject(ds, out_dir, subj, rois, splits, models, ood_types, n_jobs, overwrite):
    feature_dir, results_dir, pred_dir = (out_dir / d for d in ('features', 'results', 'predictions'))
    pool = SplitPool(subj)
    parts = {split: pool.partitions(split, ood_types) for split in splits}
    test_sets = {split: test_set_names(split, ood_types) for split in splits}
    subject = ds.load_subject(subj)

    y = {}
    for roi in rois:
        y[roi] = roi_responses(subject, roi, pool.trial_idx).mean(axis=1)
        for split in splits:
            for ts in test_sets[split]:
                fn = voxels_path(pred_dir, subj, roi, ts)
                if not fn.exists() or overwrite:
                    _save_atomic(fn, np.save, y[roi][parts[split][ts]])

    for model in models:
        todo = [(roi, split) for roi in rois for split in splits
                if overwrite or not results_path(results_dir, subj, roi, split, model).exists()]
        print(f'{subj} {model}: {len(todo)} of {len(rois) * len(splits)} ROI x split fits to run', flush=True)
        if not todo:
            continue
        mapper = MAPPERS[model]

        if model in DNN_MODELS:
            check_pool_names(feature_dir, pool)
            layers = list(np.flip(config.encoding_layers(model)))  # 03's order
        else:
            layers = [model]
            gabor = load_gabor(feature_dir, pool)

        if model in DNN_MODELS:
            units = {unit_set(model, roi, ds.roi_domain) for roi, _ in todo}
            X = {(layer, d): np.load(feature_path(feature_dir, subj, model, layer, d)) for layer in layers for d in units}

            def features(layer, roi, split):
                return X[(layer, unit_set(model, roi, ds.roi_domain))]
        elif model == 'GistPC':
            pcs = {split: gist_pcs(gabor, parts[split]['train']) for split in {s for _, s in todo}}

            def features(layer, roi, split):
                return pcs[split]
        else:
            def features(layer, roi, split):
                return gabor

        # All layers' fits share one pool of --n-jobs threads, largest first (fit time ~ training images x
        # features x voxels). Fitting layer by layer would leave most threads idle while each layer waits
        # for its largest ROI.
        tasks = [(layer, roi, split) for layer in layers for roi, split in todo]
        tasks.sort(key=lambda t: -len(parts[t[2]]['train']) * features(*t).shape[1] * y[t[1]].shape[1])
        print(f'{subj} {model}: {len(tasks)} fits ({len(layers)} layers x {len(todo)} ROI x split)', flush=True)
        start = time.time()
        # the fits run in parallel threads, so keep BLAS (OpenBLAS: one pool per process) to one thread
        # OpenMP's thread count is a per-thread setting: each worker thread sets its own to 1 when it starts,
        # or every worker would start a team of one thread per core (thousands of spinning threads).
        # The controllers are looked up here: looking them up in the workers (threadpool_limits) deadlocks
        # with numpy, as threadpoolctl holds the dynamic loader's lock while waiting for the GIL.
        openmp = ThreadpoolController().select(user_api='openmp').lib_controllers
        with threadpool_limits(limits=1, user_api='blas'), \
                ThreadPoolExecutor(max_workers=n_jobs, initializer=_single_threaded_openmp,
                                   initargs=(openmp,)) as executor:
            futures = {t: executor.submit(fit_and_score, features(*t), y[t[1]], parts[t[2]], test_sets[t[2]], mapper)
                       for t in tasks}
            for i, _ in enumerate(as_completed(futures.values()), start=1):
                if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                    print(f'\t{i} / {len(tasks)} fits done after {(time.time() - start) / 60:.1f} min', flush=True)
            out = {t: f.result() for t, f in futures.items()}

        # collect in 03's layer order, as the layer-by-layer loop did
        rows = {key: [] for key in todo}
        best = {}  # (roi, split, metric) -> (val score, layer, domain, test predictions); first max as 04's idxmax
        for layer in layers:
            for roi, split in todo:
                res = out.pop((layer, roi, split))
                if res is None:
                    continue
                scores, preds = res
                domain = unit_set(model, roi, ds.roi_domain)
                rows[(roi, split)].extend(result_rows(subj, roi, split, model, layer, domain, mapper,
                                                      parts[split], scores))
                for metric in METRICS:
                    val = scores['val'][metric]
                    val = -999 if np.isnan(val) else val
                    key = (roi, split, metric)
                    if key not in best or val > best[key][0]:
                        best[key] = (val, layer, domain, preds)

        for roi, split in todo:
            for metric in METRICS:
                if (roi, split, metric) not in best:
                    continue
                _, layer, domain, preds = best[(roi, split, metric)]
                for ts, pred in preds.items():
                    _save_atomic(prediction_path(pred_dir, subj, roi, ts, model, layer, domain), np.save, pred)
            # the CSV marks the fit as done, so it is written last
            pd.DataFrame(rows[(roi, split)]).to_csv(results_path(results_dir, subj, roi, split, model), index=False)


# ---------------------------------------------------------------------------
# noise-ceilings
# ---------------------------------------------------------------------------

def compute_noise_ceilings(ds, subjects, rois, splits, ood_types, overwrite):
    rsa_noise_ceiling = nsd.import_rsa_noise_ceiling()
    subdir = noise_ceiling_subdir(ds)
    for subj in subjects:
        pool = SplitPool(subj)
        parts = {split: pool.partitions(split, ood_types) for split in splits}
        subject = ds.load_subject(subj)
        for roi in rois:
            y = None
            for split in splits:
                for ts in test_set_names(split, ood_types):
                    fn = nsd.noise_ceiling_path(roi, subj, test_imageset=ts, subdir=subdir)
                    if fn.exists() and not overwrite:
                        continue
                    if y is None:
                        y = roi_responses(subject, roi, pool.trial_idx)
                    data = y[parts[split][ts]]
                    print(subj, roi, ts, data.shape, flush=True)
                    if data.shape[2] <= 2:
                        print('skipping, insufficient voxels')
                        continue
                    np.random.seed(zlib.crc32(f'{subj}|{roi}|{ts}'.encode()))  # GSN's voxel subsampling
                    _save_atomic(fn, np.save, noise.compute_gsn_noise_ceiling(data, rsa_noise_ceiling),
                                 allow_pickle=True)


def load_noise_ceilings(ds, test_sets, rois, subjects):
    """ROI, Subject, Univariate, RSA averaged over ``test_sets``; NaN where none exists."""
    subdir = noise_ceiling_subdir(ds)
    dfs = [nsd.load_noise_ceilings(rois, subjects, test_imageset=ts, subdir=subdir) for ts in test_sets]
    df = pd.concat(dfs) if dfs else pd.DataFrame(columns=['ROI', 'Subject', 'Univariate', 'RSA'])
    df = df.groupby(['ROI', 'Subject'])[['Univariate', 'RSA']].mean()
    index = pd.MultiIndex.from_product([rois, subjects], names=['ROI', 'Subject'])
    return df.reindex(index).reset_index()


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def load_results(results_dir, subjects, rois, splits, models):
    dfs, missing = [], []
    for subj in subjects:
        for roi in rois:
            for split in splits:
                for model in models:
                    fn = results_path(results_dir, subj, roi, split, model)
                    if fn.exists():
                        dfs.append(pd.read_csv(fn))
                    else:
                        missing.append(fn.name)
    if missing:
        raise FileNotFoundError(f'{len(missing)} fit results missing (run the fit stage), e.g. {missing[:3]}')
    return pd.concat(dfs, ignore_index=True)


def select_fits(results, metric):
    """Test rows of the layer with the best validation ``metric`` per subject, ROI, split and model."""
    keys = ['subj', 'ROI', 'split', 'model_name', 'domain']
    val = results[results['partition'] == 'val'].copy()
    val['score'] = val[metric].fillna(-999)
    # dropna=False: GistPC / Gabor have domain 'N/A', which read_csv turns into NaN (merge matches NaN keys)
    best = val.loc[val.groupby(keys, sort=False, dropna=False)['score'].idxmax(), keys + ['layer']]
    return results[results['partition'] == 'test'].merge(best, on=keys + ['layer'])


def method_scores(selected, method_test_sets):
    """Per method, subject, ROI and model: the score averaged over the method's test sets."""
    rows = []
    for method, test_sets in method_test_sets.items():
        df = selected[selected['test_set'].isin(test_sets)]
        n = df.groupby(['subj', 'ROI', 'model_name'])['test_set'].nunique()
        if (n != len(test_sets)).any():
            raise ValueError(f'{method}: not every fit has all of {test_sets}')
        mean = df.groupby(['subj', 'ROI', 'model_name'], sort=False)[METRICS].mean().reset_index()
        rows.append(mean.assign(method=method))
    return pd.concat(rows, ignore_index=True)


def run_permutation_tests(selected, pred_dir, method, test_sets, subjects, rois, n_perm, seed, alpha):
    """08's tests on the selected fits. For several test sets (folds), the statistic is the fold-averaged
    difference r(contrastive) - r(alternative), and its null the average of independent per-fold nulls."""
    rows, nulls = [], {}
    for metric in METRICS:
        sel = selected[metric].set_index(['subj', 'ROI', 'test_set', 'model_name'])
        for subj in subjects:
            for roi in rois:
                for model in perm.models_for(roi)[1:]:
                    observed, null, r_a, r_b, layers_a, layers_b = [], [], [], [], [], []
                    for k, ts in enumerate(test_sets):
                        true = np.load(voxels_path(pred_dir, subj, roi, ts))
                        fit_a, fit_b = sel.loc[(subj, roi, ts, REFERENCE_MODEL)], sel.loc[(subj, roi, ts, model)]
                        pred_a, pred_b = (np.load(prediction_path(pred_dir, subj, roi, ts, m, f['layer'], f['domain']))
                                          for m, f in [(REFERENCE_MODEL, fit_a), (model, fit_b)])
                        if metric == 'veUnivar':
                            obs, nul, _ = permutation_test_univar(pred_a.mean(axis=1), pred_b.mean(axis=1),
                                                                  true.mean(axis=1), n_perm=n_perm, seed=seed + k)
                        else:
                            obs, nul, _ = permutation_test_rsa(pred_a, pred_b, true, n_perm=n_perm, seed=seed + k)
                        observed.append(obs)
                        null.append(nul)
                        r_a.append(fit_a[metric])
                        r_b.append(fit_b[metric])
                        layers_a.append(str(fit_a['layer']))
                        layers_b.append(str(fit_b['layer']))
                    diff, null = np.mean(observed), np.mean(null, axis=0)
                    p = (1 + np.sum(null >= diff)) / (1 + n_perm) if np.isfinite(diff) else np.nan
                    print(f'{method} {metric} {subj} {roi} vs {model}: diff {diff:.3f}, p {p:.4g}', flush=True)
                    rows.append({'method': method, 'metric': metric, 'subj': subj, 'ROI': roi,
                                 'model_a': REFERENCE_MODEL, 'model_b': model,
                                 'layer_a': '|'.join(layers_a), 'domain_a': fit_a['domain'],
                                 'layer_b': '|'.join(layers_b), 'domain_b': fit_b['domain'],
                                 'n_test_sets': len(test_sets), 'r_a': np.mean(r_a), 'r_b': np.mean(r_b),
                                 'diff': diff, 'null_95': np.percentile(null, 95), 'p': p, 'n_perm': n_perm,
                                 'alpha_bonferroni': alpha, 'significant': bool(p < alpha)})
                    nulls[f'{metric}|{subj}|{roi}|{model}'] = null.astype(np.float32)
    return pd.DataFrame(rows), nulls


def group_ttests(scores, subjects, rois, alpha):
    """04's paired t-tests across subjects (contrastive vs each alternative), per method, metric and ROI."""
    rows = []
    for (method, roi), df in scores.groupby(['method', 'ROI'], sort=False):
        df = df.set_index(['model_name', 'subj'])
        models = perm.models_for(roi)
        for metric in METRICS:
            data = [df.loc[m].reindex(subjects)[metric].to_numpy() for m in models]
            _, p_values = pairwise_ttest_with_first(data, alpha=alpha)
            for i, model in enumerate(models[1:], start=1):
                rows.append({'method': method, 'metric': metric, 'ROI': roi, 'model_a': REFERENCE_MODEL,
                             'model_b': model, 'mean_a': np.nanmean(data[0]), 'mean_b': np.nanmean(data[i]),
                             'p': p_values[i], 'alpha_bonferroni': alpha, 'significant': bool(p_values[i] < alpha)})
    return pd.DataFrame(rows)


def replication_scores(ds, subjects, rois):
    """Fig. 4's best-layer test scores from 03 / 07 (as 08 selects them), or None if they are incomplete."""
    try:
        fits = perm.selected_fits(ds, ds.analysis_dir(config.ENCODING_SUBDIR))
    except (ValueError, FileNotFoundError) as err:
        print('replication results not available:', err)
        return None
    rows = []
    for metric in METRICS:
        df = fits[(fits['metric'] == metric) & fits['subj'].isin(subjects) & fits['ROI'].isin(rois)]
        rows.append(df[['subj', 'ROI', 'model_name']].assign(**{metric: df[metric].to_numpy()}))
    return rows[0].merge(rows[1], on=['subj', 'ROI', 'model_name']).assign(method='replication')


def plot_summary(scores, noise_ceilings, savedir, rois):
    """Per ROI: each model's mean +/- SEM test score (across subjects) for the replication and every
    generalization test; gray bars span the subjects' noise ceilings."""
    conditions = list(dict.fromkeys(scores['method']))
    offsets = np.linspace(-0.3, 0.3, len(PLOT_ORDER))
    for metric in METRICS:
        nc_col = 'Univariate' if 'Univar' in metric else 'RSA'
        fig, axs = plt.subplots(1, len(rois), figsize=(1.1 * len(conditions) * len(rois) + 1.5, 5.5), sharey=True)
        for ax, roi in zip(np.atleast_1d(axs), rois):
            models = [m for m in PLOT_ORDER if m in perm.models_for(roi)]
            for c, cond in enumerate(conditions):
                nc = noise_ceilings.get(cond)
                if nc is not None:
                    vals = nc.loc[nc['ROI'] == roi, nc_col].to_numpy(float)
                    if np.isfinite(vals).any():
                        lo, hi = np.nanmin(vals), np.nanmax(vals)
                        if hi - lo < 0.005:  # a single subject: draw its ceiling as a line
                            ax.hlines(hi, c - 0.42, c + 0.42, color='gray', linewidth=2, alpha=0.5, zorder=0)
                        else:
                            ax.fill_between([c - 0.42, c + 0.42], lo, hi, color='gray', alpha=0.2, linewidth=0,
                                            zorder=0)
                df = scores[(scores['method'] == cond) & (scores['ROI'] == roi)]
                for model in models:
                    vals = df.loc[df['model_name'] == model, metric].to_numpy(float)
                    if not np.isfinite(vals).any():
                        continue
                    x = c + offsets[PLOT_ORDER.index(model)]
                    ax.errorbar(x, np.nanmean(vals), yerr=stats.sem(vals, nan_policy='omit'), fmt='o', ms=6,
                                color=plots.MODEL_COLORS['face'][model], mec='k', mew=0.5, elinewidth=1.5,
                                capsize=0, zorder=3, label=plots.PAPER_MODEL_LABELS[model].replace('\n', ' '))
            ax.axhline(0, color='k', linewidth=1, zorder=1)
            ax.grid(axis='y', linewidth=0.25)
            ax.set_xlim(-0.6, len(conditions) - 0.4)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels([method_title(c) for c in conditions], rotation=60, ha='right', fontsize=9)
            ax.set_title(roi, fontsize=13)
            despine(ax)
        first = np.atleast_1d(axs)[0]
        first.set_ylabel('Model-brain correlation (r)', fontsize=12)
        handles, labels = [], []
        for ax in np.atleast_1d(axs):  # face ROIs have the most models
            h, lab = ax.get_legend_handles_labels()
            if len(set(lab)) > len(set(labels)):
                handles, labels = h, lab
        unique = dict(zip(labels, handles))
        fig.suptitle(f'{plots.PAPER_METRIC_TITLES[metric]}: generalization (mean +/- SEM across subjects; '
                     'gray: noise ceiling range)', fontsize=13, x=0.01, y=0.99, ha='left')
        fig.legend(unique.values(), unique.keys(), loc='upper left', bbox_to_anchor=(0.01, 0.93),
                   ncol=len(unique), frameon=False, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.86])
        save(savedir / f'generalization-summary-{metric}.tiff')



def print_summary(perm_results, ttests, n_subj):
    for method, df in perm_results.groupby('method', sort=False):
        summary = (df.groupby(['metric', 'ROI', 'model_b'], sort=False)['significant'].sum()
                   .astype(int).astype(str) + f'/{n_subj}')
        print(f'\n{method}: subjects with contrastive > alternative (permutation tests, Bonferroni):')
        print(summary.unstack('model_b').to_string())
    if ttests is not None and len(ttests):
        print('\ngroup t-tests significant (contrastive vs alternative, Bonferroni):')
        print(ttests.pivot_table(index=['method', 'metric', 'ROI'], columns='model_b', values='significant',
                                 aggfunc='first', sort=False).to_string())


def test(ds, out_dir, subjects, rois, splits, ood_types, test_methods, n_perm, seed, overwrite):
    pred_dir = out_dir / 'predictions'
    method_test_sets = methods(splits, ood_types)
    unknown = [m for m in test_methods or [] if m not in method_test_sets]
    if unknown:
        raise ValueError(f'unknown / incomplete methods {unknown}; available: {list(method_test_sets)}')
    test_methods = test_methods or list(method_test_sets)

    results = load_results(out_dir / 'results', subjects, rois, splits, MODEL_NAMES)
    selected = {metric: select_fits(results, metric) for metric in METRICS}
    per_test_set = pd.concat([df.assign(selected_by=m) for m, df in selected.items()], ignore_index=True)
    per_test_set.to_csv(out_dir / 'generalization-scores-per-test-set.csv', index=False)

    scores = pd.concat([method_scores(selected[metric], method_test_sets)[['method', 'subj', 'ROI', 'model_name',
                                                                           metric]]
                        .set_index(['method', 'subj', 'ROI', 'model_name']) for metric in METRICS],
                       axis=1).reset_index()
    alpha = plots.bonferroni_alpha(ds.rois)  # all of the dataset's ROIs, as 08
    ttests = group_ttests(scores, subjects, rois, alpha)
    replication = replication_scores(ds, subjects, rois)
    all_scores = scores if replication is None else pd.concat([replication, scores], ignore_index=True)
    all_scores.to_csv(out_dir / 'generalization-scores.csv', index=False)
    ttests.to_csv(out_dir / 'group-ttests.csv', index=False)

    perm_results = []
    for method in test_methods:
        results_fn, nulls_fn = out_dir / f'permutation-tests-{method}.csv', out_dir / f'permutation-nulls-{method}.npz'
        res = None
        if results_fn.exists() and nulls_fn.exists() and not overwrite:
            res = pd.read_csv(results_fn)
            if (set(res['subj']) == set(subjects) and set(res['ROI']) == set(rois) and
                    set(res['n_perm']) == {n_perm}):
                print('loading', results_fn)
            else:
                print(results_fn, 'is from a run with other subjects / ROIs / --n-perm; recomputing')
                res = None
        if res is None:
            res, nulls = run_permutation_tests(selected, pred_dir, method, method_test_sets[method], subjects, rois,
                                               n_perm, seed, alpha)
            res.to_csv(results_fn, index=False)
            np.savez_compressed(nulls_fn, **nulls)
        perm_results.append(res)

        # 08's figures for this method; noise ceilings averaged over the method's test sets
        savedir = ds.figure_dir('Figure4-Generalization') / method
        savedir.mkdir(exist_ok=True)
        nc = load_noise_ceilings(ds, method_test_sets[method], rois, subjects)
        suffix = f' ({method_title(method)})'
        perm.plot_differences(res, savedir, subjects, rois, title_suffix=suffix)
        perm.plot_paper_style(res, nc, savedir, subjects, rois, ds.roi_domain, title_suffix=suffix)
        perm.plot_paper_style_per_subject(res, nc, savedir, subjects, rois, ds.roi_domain, title_suffix=suffix)
    perm_results = pd.concat(perm_results, ignore_index=True)

    noise_ceilings = {m: load_noise_ceilings(ds, ts, rois, subjects) for m, ts in method_test_sets.items()}
    if replication is not None:
        noise_ceilings['replication'] = ds.load_noise_ceilings(rois, subjects)
    plot_summary(all_scores, noise_ceilings, ds.figure_dir('Figure4-Generalization'), rois)
    print_summary(perm_results, ttests, len(subjects))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['features', 'lowlevel', 'fit', 'noise-ceilings', 'test'])
    parser.add_argument('--subjects', nargs='+', choices=laion.SUBJECTS, default=laion.SUBJECTS)
    parser.add_argument('--rois', nargs='+', choices=laion.ROI_LIST, default=laion.ROI_LIST)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES,
                        help='features / fit: only these models')
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=SPLITS)
    parser.add_argument('--ood-types', nargs='+', default=OOD_TYPES,
                        help="OOD test sets of the 'ood' split, one per type ('all': every OOD image); "
                             'default: unusual')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='test: run the permutation tests only for these methods '
                             '(tau, clusters, ood-<type>, random; default: all)')
    parser.add_argument('--n-jobs', type=int, default=len(os.sched_getaffinity(0)),
                        help='lowlevel: processes; fit: threads (default: CPUs of the job)')
    parser.add_argument('--n-perm', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0, help='test: swap masks of fold k use seed + k')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    from laion_fmri.splits import list_ood_types
    bad = [t for t in args.ood_types if t != 'all' and t not in list_ood_types()]
    if bad:
        parser.error(f'unknown OOD types {bad}; choose from {list_ood_types() + ["all"]}')

    ds = datasets.get('laion')
    out_dir = ds.analysis_dir(GENERALIZATION_SUBDIR)
    for sub in ['features', 'results', 'predictions']:
        (out_dir / sub).mkdir(exist_ok=True)

    if args.stage == 'features':
        extract_features(ds, out_dir / 'features', args.subjects, args.models, args.rois, args.overwrite)
    elif args.stage == 'lowlevel':
        extract_gabor(out_dir / 'features', args.subjects, args.n_jobs, args.overwrite)
    elif args.stage == 'fit':
        for subj in args.subjects:
            fit_subject(ds, out_dir, subj, args.rois, args.splits, args.models, args.ood_types, args.n_jobs,
                        args.overwrite)
    elif args.stage == 'noise-ceilings':
        compute_noise_ceilings(ds, args.subjects, args.rois, args.splits, args.ood_types, args.overwrite)
    else:
        test(ds, out_dir, args.subjects, args.rois, args.splits, args.ood_types, args.methods, args.n_perm,
             args.seed, args.overwrite)


if __name__ == '__main__':
    main()
