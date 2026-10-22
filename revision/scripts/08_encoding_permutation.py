"""Single-subject permutation tests of the Fig. 4 model comparisons (finding 3).

The paper compares the contrastive model with each alternative model (face-trained,
untrained, GistPC, Gabor) by paired t-tests across NSD's 8 subjects. LAION-fMRI has only
5 subjects, so this script tests each subject separately. For one subject, ROI and pair of
models, the null distribution comes from swapping the two models' test-set predictions
independently per image (probability 0.5). Each swap recomputes both models' univariate
profile / RDM correlation with the brain data. The p-value is the share of permuted
correlation differences (contrastive - alternative) at least as large as the observed one:
(1 + #{null >= observed}) / (1 + n_perm).

The models and layers are the ones 04 plots: each model's best layer selected on the
validation set (``find_max_score_layers``), with the ROI's preferred-domain units for the
Barlow Twins models. Bonferroni correction as in 04 (one test per ROI and alternative
model), applied within each subject and metric.

Two stages, reading/writing ``analysis_outputs/3e-Permutation`` (``3e-Permutation-laion``
with --dataset laion):
  predict - refit the selected encoding models (same fits as 03 / 07) and save their test-set
            predictions and the ROI's test responses. DNN features are extracted on cuda:0, so
            run this on a GPU node. Each refit is checked against the score in the 03/07 CSV.
  test    - run the permutation tests, write ``permutation-tests.csv`` and the null
            distributions, and draw the figures into ``figure_outputs/Figure4-Permutation``
            (incl. 04's paper-style Fig. 4 with the permutation tests, for the group and per subject)
            (``-laion``). Results are cached; pass --overwrite to rerun the permutations.

--subjects / --rois / --models restrict the work (e.g. to split predict into parallel jobs);
the Bonferroni count always uses all of the dataset's ROIs.

Requires the outputs of 03 and 07 (``encode``), run with the same --dataset.
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from jsputils import classes  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

from dnffa import config, datasets, nsd  # noqa: E402
from dnffa.laion import _save_atomic  # noqa: E402
from dnffa.plotting import despine, noise_ceiling_band, plt, save  # noqa: E402
from dnffa.stats import permutation_test_rsa, permutation_test_univar  # noqa: E402

plots = importlib.import_module('04_encoding_plots')
low_level = importlib.import_module('07_low_level_models')

PERMUTATION_SUBDIR = '3e-Permutation'
METRICS = ['veUnivar', 'veRSA']
REFERENCE_MODEL = 'alexnet-barlow-twins'
RESULTS_FN = 'permutation-tests.csv'
NULLS_FN = 'permutation-nulls.npz'


def models_for(roi):
    """Contrastive model first, then the alternatives in the published Fig. 4 order."""
    colors = plots.MODEL_COLORS['face' if plots.is_face_roi(roi) else 'non-face']
    return [m for m in plots.PAPER_MODEL_ORDER if m in colors]


# ---------------------------------------------------------------------------
# Which fits to compare
# ---------------------------------------------------------------------------

def selected_fits(ds, encoding_dir):
    """One row per (metric, subject, ROI, model): the fit 04 plots for that metric.

    Columns: metric, subj, ROI, model_name, layer, domain, method, positive, alpha and the
    test-set veUnivar / veRSA from the 03/07 CSV. layer and domain are None for GistPC / Gabor.
    """
    combined_df = plots.load_encoding_results([encoding_dir], plots.MODEL_NAMES, ds.rois, ds.subjects)
    rows = []
    for metric in METRICS:
        df_max = plots.find_max_score_layers(combined_df, metric)
        for subj in ds.subjects:
            for roi in ds.rois:
                for model in models_for(roi):
                    sel = (df_max['subj'] == subj) & (df_max['ROI'] == roi) & (df_max['model_name'] == model)
                    if 'barlow-twins' in model:
                        sel &= df_max['domain'] == ds.roi_domain[roi]
                    if sel.sum() != 1:
                        raise ValueError(f'{sel.sum()} best-layer rows for {subj} {roi} {model} {metric}; '
                                         'are 03 and 07 complete?')
                    row = df_max[sel].iloc[0]
                    dnn = model in plots.DNN_MODELS
                    rows.append({'metric': metric, 'subj': subj, 'ROI': roi, 'model_name': model,
                                 'layer': row['layer'] if dnn else None,
                                 'domain': row['domain'] if dnn else None,
                                 'method': row['method'], 'positive': bool(row['positive']),
                                 'alpha': float(row['alpha']) if row['method'] == 'lasso' else None,
                                 'veUnivar': row['veUnivar'], 'veRSA': row['veRSA']})
    return pd.DataFrame(rows)


def fit_key(fit):
    nc = config.NCSNR_THRESHOLD
    if pd.isna(fit['layer']):
        return f"{fit['subj']}_{fit['ROI']}_nc-{nc}_{fit['model_name']}"
    return f"{fit['subj']}_{fit['ROI']}_nc-{nc}_{fit['model_name']}_{fit['layer']}_{fit['domain']}"


def prediction_path(pred_dir, fit):
    return pred_dir / f'{fit_key(fit)}_test-pred.npy'


def voxels_path(pred_dir, subj, roi):
    return pred_dir / f'{subj}_{roi}_nc-{config.NCSNR_THRESHOLD}_test-voxels.npy'


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def score(pred, y_test, metric):
    """veUnivar / veRSA of test-set predictions, as computed by 03 / 07."""
    if metric == 'veUnivar':
        return stats.pearsonr(pred.mean(axis=1), y_test.mean(axis=1))[0]
    return stats.pearsonr(pdist(pred, 'correlation'), pdist(y_test, 'correlation'))[0]


def check_refit(fit, pred, y_test):
    """Warn if the refit's test scores differ from the 03/07 CSV (they should match)."""
    for name in METRICS:
        refit, saved = score(pred, y_test, name), fit[name]
        if not np.isclose(refit, saved, atol=1e-3):
            print(f'\twarning: {fit_key(fit)} {name} refit {refit:.4f} != saved {saved:.4f}')


def fit_and_save(fit, X_train, X_test, y, pred_dir):
    alpha = None if fit['method'] == 'ols' else fit['alpha']
    mdl = classes.get_mapper(fit['method'], fit['positive'], alpha)
    mdl.fit(X_train, y['train'])
    pred = mdl.predict(X_test)
    check_refit(fit, pred, y['test'])
    _save_atomic(prediction_path(pred_dir, fit), np.save, pred.astype(np.float32))


def dnn_features(DNN, layer, domain, partition):
    """Same as EncodingProcedure.get_encoding_features for one partition."""
    X = getattr(DNN, f'nsd_{partition}_features')[layer]
    X = np.reshape(X, (X.shape[0], -1))
    if domain != 'layer':
        X = X[:, DNN.selective_units[domain][layer]['mask']]
    return X


def predict(ds, fits, pred_dir, overwrite):
    fits = fits.drop(columns='metric').drop_duplicates(['subj', 'ROI', 'model_name', 'layer', 'domain'])
    if not overwrite:
        fits = fits[[not prediction_path(pred_dir, fit).exists() for _, fit in fits.iterrows()]]
    print(len(fits), 'fits to run')

    low_level_dir = ds.analysis_dir(config.LOW_LEVEL_SUBDIR)
    for model, model_fits in fits.groupby('model_name', sort=False):
        if model in plots.DNN_MODELS:
            if 'random' in model:
                torch.manual_seed(config.RANDOM_MODEL_SEED)  # same weights as 03
            DNN = classes.DNNModel(model)
            if 'vggface' not in model:
                DNN.find_selective_units(config.FLOC_IMAGESET, overwrite=False, verbose=False, FDR_p=config.FDR_P)
        else:
            test_features = low_level.load_gist_gabor_features(low_level_dir, ds.test_imageset, model)

        for subj, subj_fits in model_fits.groupby('subj', sort=False):
            subject = ds.load_subject(subj)
            features_loaded = False
            for roi, roi_fits in subj_fits.groupby('ROI', sort=False):
                print(model, subj, roi, flush=True)
                ROI = ds.load_roi(subject, roi)
                y = nsd.roi_voxels(ROI, partitions=('train', 'test'))
                # parallel jobs (e.g. one per model) write the same file, so never leave it half-written
                _save_atomic(voxels_path(pred_dir, subj, roi), np.save, y['test'].astype(np.float32))

                if model not in plots.DNN_MODELS:
                    X_train = low_level.load_gist_gabor_features(low_level_dir, subj, model)
                    fit_and_save(roi_fits.iloc[0], X_train, test_features, y, pred_dir)
                    continue

                # the stimuli depend only on the subject, so features are extracted once per subject
                if not features_loaded:
                    for partition in ['train', 'test']:
                        DNN.get_nsd_features(torch.from_numpy(ROI.image_data[partition]),
                                             field=f'nsd_{partition}_features')
                    features_loaded = True
                for _, fit in roi_fits.iterrows():
                    fit_and_save(fit, dnn_features(DNN, fit['layer'], fit['domain'], 'train'),
                                 dnn_features(DNN, fit['layer'], fit['domain'], 'test'), y, pred_dir)


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def run_tests(ds, fits, pred_dir, subjects, rois, n_perm, seed):
    alpha = plots.bonferroni_alpha(ds.rois)  # all of the dataset's ROIs, also when testing a subset
    if 1 / (n_perm + 1) >= alpha:
        print(f'warning: with {n_perm} permutations the smallest p-value is above the corrected alpha {alpha:.2g}')

    fits = fits.set_index(['metric', 'subj', 'ROI', 'model_name'])
    rows, nulls = [], {}
    for metric in METRICS:
        for subj in subjects:
            for roi in rois:
                true = np.load(voxels_path(pred_dir, subj, roi))
                ref = fits.loc[(metric, subj, roi, REFERENCE_MODEL)]
                pred_ref = np.load(prediction_path(pred_dir, {**ref, 'subj': subj, 'ROI': roi,
                                                              'model_name': REFERENCE_MODEL}))
                for model in models_for(roi)[1:]:
                    alt = fits.loc[(metric, subj, roi, model)]
                    pred_alt = np.load(prediction_path(pred_dir, {**alt, 'subj': subj, 'ROI': roi,
                                                                  'model_name': model}))
                    if metric == 'veUnivar':
                        observed, null, p = permutation_test_univar(pred_ref.mean(axis=1), pred_alt.mean(axis=1),
                                                                    true.mean(axis=1), n_perm=n_perm, seed=seed)
                    else:
                        observed, null, p = permutation_test_rsa(pred_ref, pred_alt, true, n_perm=n_perm, seed=seed)
                    print(f'{metric} {subj} {roi} vs {model}: diff {observed:.3f}, p {p:.4g}', flush=True)
                    rows.append({'metric': metric, 'subj': subj, 'ROI': roi,
                                 'model_a': REFERENCE_MODEL, 'model_b': model,
                                 'layer_a': ref['layer'], 'domain_a': ref['domain'],
                                 'layer_b': alt['layer'], 'domain_b': alt['domain'],
                                 'r_a': score(pred_ref, true, metric), 'r_b': score(pred_alt, true, metric),
                                 'diff': observed,
                                 'null_95': np.percentile(null, 95), 'p': p, 'n_perm': n_perm,
                                 'alpha_bonferroni': alpha, 'significant': p < alpha})
                    nulls[f'{metric}|{subj}|{roi}|{model}'] = null.astype(np.float32)
    return pd.DataFrame(rows), nulls


def plot_differences(results, savedir, subjects, rois):
    """Per ROI: each subject's observed difference (contrastive - alternative) and its null's 95th
    percentile; filled dots are significant after Bonferroni correction."""
    n_subj = len(subjects)
    offsets = np.linspace(-0.25, 0.25, n_subj) if n_subj > 1 else [0]
    for metric in METRICS:
        df = results[results['metric'] == metric]
        width_ratios = [len(models_for(roi)) - 1 for roi in rois]
        fig, axs = plt.subplots(1, len(rois), figsize=(2.2 * sum(width_ratios) + 1, 5), sharey=True,
                                gridspec_kw={'width_ratios': width_ratios})
        for ax, roi in zip(np.atleast_1d(axs), rois):
            alts = models_for(roi)[1:]
            for i, model in enumerate(alts):
                for offset, subj in zip(offsets, subjects):
                    row = df[(df['ROI'] == roi) & (df['model_b'] == model) & (df['subj'] == subj)].iloc[0]
                    ax.plot(i + offset, row['null_95'], marker='_', color='gray', markersize=10, mew=1.5)
                    color = plots.MODEL_COLORS['face'][model]
                    ax.scatter(i + offset, row['diff'], s=40, zorder=3, edgecolor='k', linewidth=0.75,
                               color=color if row['significant'] else 'white')
                n_sig = df[(df['ROI'] == roi) & (df['model_b'] == model)]['significant'].sum()
                ax.text(i, 1.02, f'{n_sig}/{n_subj}', transform=ax.get_xaxis_transform(), ha='center', fontsize=11)
            ax.axhline(0, color='k', linewidth=1, zorder=1)
            ax.set_xlim(-0.6, len(alts) - 0.4)
            ax.set_xticks(range(len(alts)))
            ax.set_xticklabels([plots.PAPER_MODEL_LABELS[m] for m in alts], rotation=45, ha='right', fontsize=10)
            ax.set_title(roi, fontsize=14, pad=18)
            ax.grid(axis='y', linewidth=0.25)
            despine(ax)
        first = np.atleast_1d(axs)[0]
        first.set_ylabel('r(contrastive) - r(alternative)', fontsize=12)
        fig.suptitle(f'{plots.PAPER_METRIC_TITLES[metric]}: single-subject permutation tests '
                     f'(filled: p < {df["alpha_bonferroni"].iloc[0]:.2g}; gray tick: null 95th percentile)',
                     fontsize=13, x=0.01, ha='left')
        plt.tight_layout()
        save(savedir / f'permutation-differences-{metric}.tiff')


def plot_nulls(results, nulls, savedir, subjects, rois):
    """Null distributions with the observed difference, subjects x alternative models, example ROIs."""
    for metric in METRICS:
        for roi in [r for r in plots.EXAMPLE_ROIS if r in rois]:
            alts = models_for(roi)[1:]
            fig, axs = plt.subplots(len(subjects), len(alts), figsize=(3 * len(alts), 2 * len(subjects)),
                                    squeeze=False)
            for r, subj in enumerate(subjects):
                for c, model in enumerate(alts):
                    ax = axs[r, c]
                    row = results[(results['metric'] == metric) & (results['subj'] == subj) &
                                  (results['ROI'] == roi) & (results['model_b'] == model)].iloc[0]
                    ax.hist(nulls[f'{metric}|{subj}|{roi}|{model}'], bins=50, color='lightgray')
                    ax.axvline(row['diff'], color='red' if row['significant'] else 'k', linewidth=1.5)
                    ax.set_title(f"{subj} vs {plots.PAPER_MODEL_LABELS[model].replace(chr(10), ' ')}\n"
                                 f"diff {row['diff']:.3f}, p {row['p']:.2g}", fontsize=9)
                    ax.set_yticks([])
                    ax.tick_params(labelsize=8)
                    despine(ax)
            fig.suptitle(f'{roi} {metric}: null distributions (red: significant)', fontsize=12)
            plt.tight_layout()
            save(savedir / f'permutation-nulls-{metric}-{roi}.tiff')


def roi_scores(results, metric, roi, subjects):
    """Models of an ROI (contrastive first), each model's per-subject r, and per-subject significance
    of contrastive > model for the alternatives; arrays follow ``subjects``."""
    df = results[(results['metric'] == metric) & (results['ROI'] == roi)].set_index(['model_b', 'subj'])
    models = models_for(roi)
    scores = {models[0]: df.loc[models[1]].loc[subjects, 'r_a'].to_numpy()}
    significant = {}
    for model in models[1:]:
        scores[model] = df.loc[model].loc[subjects, 'r_b'].to_numpy()
        significant[model] = df.loc[model].loc[subjects, 'significant'].to_numpy(bool)
    return models, scores, significant


def bar_style(model, roi, roi_domain):
    """04's paper style: contrastive bar in the ROI's domain colour, face-trained model dashed."""
    color = config.DOMAIN_COLORS[roi_domain[roi]] if model == REFERENCE_MODEL else plots.MODEL_COLORS['face'][model]
    face_trained = model == 'alexnet-vggface'
    return {'color': color, 'alpha': 0.6 if face_trained else 1, 'edgecolor': 'k' if face_trained else 'none',
            'linestyle': '--', 'linewidth': 1}


def bracket_layout(values, nc_tops, max_brackets, step):
    """y of the top bracket and the y limits, as in 04: brackets above the data and noise ceilings."""
    data_top = max(np.nanmax(values), max(nc_tops))
    bars_bottom = data_top + 0.05
    if bars_bottom + step * (max_brackets - 1) > 0.98:  # would collide with the 1.0 gridline
        bars_bottom = max(bars_bottom, 1.04)
    bars_top = bars_bottom + step * (max_brackets - 1)
    return bars_top, [min(-0.05, np.nanmin(values) - 0.05), bars_top + 0.05]


def draw_bar_panel(ax, roi, models, heights, nc_range, brackets, bars_top, step, ylim, roi_domain, dots=None):
    """One ROI panel in 04's paper style.

    ``brackets``: (model index, label) pairs, drawn from the contrastive bar downwards from ``bars_top``.
    ``dots``: {model index: (per-subject values, filled mask)}; open dots are not significant.
    ``nc_range``: [min, max] noise ceiling; a single subject's ceiling (min == max) is drawn as a line.
    """
    n_models = len(models)
    for i, (model, height) in enumerate(zip(models, heights)):
        ax.bar(i, height, width=0.7, zorder=2, **bar_style(model, roi, roi_domain))
    rng = np.random.default_rng(0)  # dot jitter
    for i, (values, filled) in (dots or {}).items():
        x = i + rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(x[filled], values[filled], s=12, color='dimgray', zorder=3)
        ax.scatter(x[~filled], values[~filled], s=12, facecolor='white', edgecolor='dimgray', linewidth=0.8, zorder=3)

    if nc_range[0] == nc_range[1]:
        ax.hlines(nc_range[0], -0.5, n_models - 0.5, color='gray', linewidth=2, alpha=0.5, zorder=0)
    else:
        noise_ceiling_band(ax, nc_range, n_models, zorder=0)
    ax.hlines(np.arange(0.2, 1.01, 0.2), -0.5, n_models - 0.5, 'k', linewidth=0.5, zorder=1)
    ax.hlines(0, -0.5, n_models - 0.5, 'k', linewidth=1, zorder=1)

    for k, (i, label) in enumerate(brackets):
        y = bars_top - k * step
        ax.plot([0, i], [y, y], color='k', lw=0.75)
        if label:
            ax.text(i + 0.1, y, label, fontsize=8, va='center')

    ax.set_xlim(-0.6, n_models - 0.4)
    ax.set_ylim(ylim)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([plots.PAPER_MODEL_LABELS[m] for m in models], rotation=45, ha='right', fontsize=12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(length=0, labelsize=12)
    despine(ax)


def plot_paper_style(results, noise_ceilings, savedir, subjects, rois, roi_domain):
    """04's paper-style Fig. 4 with the permutation tests: bars are subject means, each bracket is
    labelled with the number of subjects where contrastive > that model, filled dots are those subjects."""
    rois = [roi for roi in plots.EXAMPLE_ROIS if roi in rois]
    step = 0.03
    for metric in METRICS:
        alpha = results.loc[results['metric'] == metric, 'alpha_bonferroni'].iloc[0]
        panels = [(roi, *roi_scores(results, metric, roi, subjects)) for roi in rois]
        nc_ranges = [nsd.noise_ceiling_range(noise_ceilings, roi, metric) for roi in rois]
        bars_top, ylim = bracket_layout(np.concatenate([np.concatenate(list(sc.values())) for _, _, sc, _ in panels]),
                                        [nc[1] for nc in nc_ranges], max(len(m) - 1 for _, m, _, _ in panels), step)

        fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 6), sharey=True,
                                gridspec_kw={'width_ratios': [len(m) for _, m, _, _ in panels]})
        for ax, (roi, models, scores, significant), nc_range in zip(np.atleast_1d(axs), panels, nc_ranges):
            dots = {0: (scores[models[0]], np.ones(len(subjects), dtype=bool))}
            dots.update({i: (scores[m], significant[m]) for i, m in enumerate(models) if i > 0})
            brackets = [(i, f'{significant[m].sum()}/{len(subjects)}') for i, m in enumerate(models) if i > 0]
            draw_bar_panel(ax, roi, models, [np.nanmean(scores[m]) for m in models], nc_range, brackets,
                           bars_top, step, ylim, roi_domain, dots=dots)
            ax.set_title(roi, fontsize=16)

        first = np.atleast_1d(axs)[0]
        first.set_ylabel('Model-brain correlation (r)', fontsize=14)
        first.text(-0.4, nc_ranges[0][1] - 0.01, 'Noise ceiling range', fontsize=8, color='dimgray', va='top')
        fig.suptitle(plots.PAPER_METRIC_TITLES[metric], fontsize=18, x=0.02, ha='left')
        fig.text(0.02, 0.01, f'Brackets: subjects with contrastive > model (permutation test, p < {alpha:.2g}, '
                 'Bonferroni). Filled dots: significant subjects; open dots: not significant.',
                 fontsize=10, color='dimgray')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        save(savedir / f'permutation-best-layers-{metric}-paper-style.tiff')


def plot_paper_style_per_subject(results, noise_ceilings, savedir, subjects, rois, roi_domain):
    """04's paper-style Fig. 4 for each subject (rows): that subject's scores and noise ceiling, with a
    bracket for each significant permutation test (as 04 draws the significant t-tests)."""
    rois = [roi for roi in plots.EXAMPLE_ROIS if roi in rois]
    step = 0.03
    for metric in METRICS:
        alpha = results.loc[results['metric'] == metric, 'alpha_bonferroni'].iloc[0]
        panels = [(roi, *roi_scores(results, metric, roi, subjects)) for roi in rois]
        nc_col = 'Univariate' if 'Univar' in metric else 'RSA'
        nc = noise_ceilings.set_index(['ROI', 'Subject'])[nc_col]
        bars_top, ylim = bracket_layout(np.concatenate([np.concatenate(list(sc.values())) for _, _, sc, _ in panels]),
                                        [nc.loc[roi].max() for roi in rois], max(len(m) - 1 for _, m, _, _ in panels),
                                        step)

        fig, axs = plt.subplots(len(subjects), len(panels), figsize=(4 * len(panels), 3.2 * len(subjects) + 1.5),
                                sharey=True, squeeze=False,
                                gridspec_kw={'width_ratios': [len(m) for _, m, _, _ in panels]})
        for s, subj in enumerate(subjects):
            for ax, (roi, models, scores, significant) in zip(axs[s], panels):
                brackets = [(i, None) for i, m in enumerate(models) if i > 0 and significant[m][s]]
                subj_nc = nc.get((roi, subj), np.nan)
                draw_bar_panel(ax, roi, models, [scores[m][s] for m in models], [subj_nc, subj_nc], brackets,
                               bars_top, step, ylim, roi_domain)
                if s == 0:
                    ax.set_title(roi, fontsize=16)
                if s < len(subjects) - 1:
                    ax.set_xticklabels([])
            axs[s, 0].set_ylabel(f'{subj}\nr', fontsize=14)

        fig.suptitle(f'{plots.PAPER_METRIC_TITLES[metric]}: single subjects', fontsize=18, x=0.02, ha='left')
        fig.text(0.02, 0.005, f'Brackets: contrastive > model (permutation test, p < {alpha:.2g}, Bonferroni). '
                 'Gray line: the subject\'s noise ceiling.', fontsize=10, color='dimgray')
        plt.tight_layout(rect=[0, 0.015, 1, 0.975])
        save(savedir / f'permutation-best-layers-{metric}-per-subject.tiff')


def print_summary(results, n_subj):
    summary = (results.groupby(['metric', 'ROI', 'model_b'], sort=False)['significant'].sum()
               .astype(int).astype(str) + f'/{n_subj}')
    print('\nsubjects with contrastive > alternative (Bonferroni-corrected):')
    print(summary.unstack('model_b').to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['predict', 'test'])
    datasets.add_arguments(parser)
    parser.add_argument('--models', nargs='+', choices=plots.MODEL_NAMES, default=plots.MODEL_NAMES,
                        help='predict: only refit these models (e.g. to split GPU jobs)')
    parser.add_argument('--n-perm', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0, help='same swap masks for every test')
    parser.add_argument('--overwrite', action='store_true',
                        help='predict: refit cached predictions; test: rerun the permutations')
    args = parser.parse_args()
    ds = datasets.from_args(parser, args)

    out_dir = ds.analysis_dir(PERMUTATION_SUBDIR)
    pred_dir = out_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    fits = selected_fits(ds, ds.analysis_dir(config.ENCODING_SUBDIR))

    if args.stage == 'predict':
        todo = fits['subj'].isin(args.subjects) & fits['ROI'].isin(args.rois) & fits['model_name'].isin(args.models)
        predict(ds, fits[todo], pred_dir, args.overwrite)
        return

    results_fn, nulls_fn = out_dir / RESULTS_FN, out_dir / NULLS_FN
    results = None
    if results_fn.exists() and nulls_fn.exists() and not args.overwrite:
        results = pd.read_csv(results_fn)
        # only reuse results of a run with the same subjects, ROIs and number of permutations
        if (set(results['subj']) == set(args.subjects) and set(results['ROI']) == set(args.rois) and
                set(results['n_perm']) == {args.n_perm}):
            print('loading', results_fn)
            nulls = dict(np.load(nulls_fn))
        else:
            print(results_fn, 'is from a run with other subjects / ROIs / --n-perm; recomputing')
            results = None
    if results is None:
        results, nulls = run_tests(ds, fits, pred_dir, args.subjects, args.rois, args.n_perm, args.seed)
        results.to_csv(results_fn, index=False)
        np.savez_compressed(nulls_fn, **nulls)

    subjects = [s for s in ds.subjects if s in set(results['subj'])]
    rois = [r for r in ds.rois if r in set(results['ROI'])]
    savedir = ds.figure_dir('Figure4-Permutation')
    plot_differences(results, savedir, subjects, rois)
    plot_nulls(results, nulls, savedir, subjects, rois)
    noise_ceilings = ds.load_noise_ceilings(rois, subjects)
    plot_paper_style(results, noise_ceilings, savedir, subjects, rois, ds.roi_domain)
    plot_paper_style_per_subject(results, noise_ceilings, savedir, subjects, rois, ds.roi_domain)
    print_summary(results, len(subjects))


if __name__ == '__main__':
    main()
