"""Summarize encoding-model results against noise ceilings (Fig. 4 and related supplements).

Converted from PROJECT_DNFFA/NOTEBOOKS/4-Encoding-Plots.ipynb.

Requires the outputs of 03_encoding.py (lasso; and --method ols into
``3-Encoding/alexnet-barlow-twins-ols`` for the OLS comparison), 06_noise_ceilings.py
and 07_low_level_models.py, all run with the same --dataset. With --dataset laion, inputs
and figures use the ``-laion`` folders (e.g. ``3-Encoding-laion``, ``Figure4-Encoding-laion``).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from dnffa import config, datasets, nsd  # noqa: E402
from dnffa.plotting import despine, noise_ceiling_band, plt, save  # noqa: E402
from dnffa.stats import pairwise_ttest_with_first  # noqa: E402

MODEL_NAMES = ['alexnet-barlow-twins', 'alexnet-vggface', 'alexnet-barlow-twins-random', 'GistPC', 'Gabor']
DNN_MODELS = ['alexnet-barlow-twins', 'alexnet-barlow-twins-random', 'alexnet-vggface']
DOMAIN_LIST = ['faces', 'bodies', 'objects', 'scenes', 'characters', 'layer']
PLOT_DOMAINS = ['faces', 'bodies', 'scenes', 'characters', 'objects']
EXAMPLE_ROIS = ['FFA-1', 'PPA', 'EBA', 'VWFA-1']
LAYERS = config.encoding_layers('alexnet-barlow-twins')
FT = 24

MODEL_COLORS = {
    'face': {'alexnet-barlow-twins': 'dodgerblue',
             'alexnet-vggface': 'red',
             'alexnet-barlow-twins-random': 'lightgray',
             'GistPC': 'violet',
             'Gabor': 'tan'},
    'non-face': {'alexnet-barlow-twins': 'dodgerblue',
                 'alexnet-barlow-twins-random': 'lightgray',
                 'GistPC': 'violet',
                 'Gabor': 'tan'},
}


def is_face_roi(roi):
    return 'FFA' in roi or 'OFA' in roi


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_encoding_results(loaddirs, model_names, rois, subjs):
    nc = config.NCSNR_THRESHOLD
    df_list = []
    for loaddir in loaddirs:
        for roi in rois:
            for subj in subjs:
                for model_name in model_names:
                    if 'alexnet' in model_name:
                        fns = [loaddir / f'{subj}_{roi}_nc-{nc}_{model_name}_{layer}_{domain}.csv'
                               for layer in np.flip(LAYERS) for domain in DOMAIN_LIST]
                    else:
                        fns = [loaddir / f'{subj}_{roi}_nc-{nc}_{model_name}.csv']
                    df_list.extend(pd.read_csv(fn, index_col=None, header=0) for fn in fns if fn.exists())
    return pd.concat(df_list, axis=0, ignore_index=True)


def find_max_score_layers(results_df, metric):
    """Test-set scores at the layer that maximizes ``metric`` on the validation set."""
    results_df = results_df.copy()
    results_df['row_order'] = np.arange(len(results_df))
    results_df = results_df.fillna(-999)

    parts = []
    for model_name in MODEL_NAMES:
        model_df = results_df[results_df['model_name'] == model_name]
        test_df = model_df[model_df['partition'] == 'test']
        if model_name not in DNN_MODELS:
            parts.append(test_df)
            continue

        val_df = model_df[model_df['partition'] == 'val']
        max_layers = (val_df.groupby(['subj', 'ROI', 'model_name', 'domain'])[metric]
                      .idxmax().apply(lambda x: val_df.loc[x, 'layer']))
        for (subj, roi, _, domain), layer in max_layers.items():
            parts.append(test_df[(test_df['subj'] == subj) & (test_df['ROI'] == roi) &
                                 (test_df['layer'] == layer) & (test_df['domain'] == domain)])

    return pd.concat(parts, axis=0).sort_values('row_order').drop(columns=['row_order'])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_domain_layer_curves(ax, roi_df, metric, domains, highlight=None):
    """Mean +/- SEM of ``metric`` per layer, one line per domain.

    With ``highlight=None`` lines use matplotlib's default style (full layer summaries);
    otherwise the highlighted domain is drawn thick and on top (OLS vs. lasso figure).
    """
    numeric_cols = roi_df.select_dtypes(include=np.number).columns.tolist()
    grouped = roi_df.groupby(['domain', 'layer'], sort=False)[numeric_cols].agg(['mean', 'sem'])
    for domain in domains:
        if domain not in grouped.index.get_level_values(0):
            print('no data for', domain)
            continue
        data = grouped.loc[domain]
        x = np.flip(data.index.to_numpy())
        y = np.flip(data[(metric, 'mean')].to_numpy())
        yerr = np.flip(data[(metric, 'sem')].to_numpy())
        color = config.DOMAIN_COLORS[domain]
        if highlight is None:
            ax.plot(x, y, color=color, label=domain, marker='.')
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
        else:
            lw, zo = (4, 10) if domain == highlight else (1.5, 0)
            ax.plot(x, y, color=color, label=domain, marker='.', linewidth=lw, zorder=zo)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2, linewidth=0, zorder=zo)


def plot_full_layer_summaries(combined_df, noise_ceilings, savedir):
    for model_name in ['alexnet-barlow-twins', 'alexnet-vggface']:
        if 'barlow' in model_name:
            domains, rois = PLOT_DOMAINS, EXAMPLE_ROIS
        else:
            domains, rois = ['layer'], ['FFA-1', 'FFA-2', 'OFA', 'PPA', 'VWFA-1']

        df = combined_df[(combined_df['model_name'] == model_name) & (combined_df['partition'] == 'test')]
        for metric in ['veUnivar', 'veRSA']:
            fig, axs = plt.subplots(1, len(rois), figsize=(30, 6))
            for ax, roi in zip(axs, rois):
                plot_domain_layer_curves(ax, df[df['ROI'] == roi], metric, domains)
                noise_ceiling_band(ax, nsd.noise_ceiling_range(noise_ceilings, roi, metric), len(LAYERS))
                ax.set_title(f'{roi} {metric}')
                ax.tick_params(axis='x', rotation=90)
                ax.set_ylabel('Prediction Levels')
                ax.grid(True)
                ax.set_ylim([-0.4, 1])
                ax.plot(np.arange(len(LAYERS)), np.zeros((len(LAYERS),)), 'k', linewidth=2)
            axs[-1].set_xlabel('Model Layers')
            fig.suptitle(f'{model_name} - {metric}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save(savedir / f'full-layer-summary-{model_name}-{metric}.png')


def plot_best_layer_violins(df_max, noise_ceilings, savedir, ds, metrics=('veUnivar', 'veRSA')):
    """Fig. 4: test-set prediction per ROI for each model at its best (val-selected) layer."""
    # one test per non-first model and ROI; NSD: (3 * 4) + (8 * 3) = 36
    n_face = sum(is_face_roi(roi) for roi in ds.rois)
    n_tests = (n_face * (len(MODEL_COLORS['face']) - 1) +
               (len(ds.rois) - n_face) * (len(MODEL_COLORS['non-face']) - 1))
    ttest_alpha = 0.05 / n_tests  # bonferroni

    for metric in metrics:
        df = df_max[metric][['subj', 'ROI', 'model_name', 'domain', 'veUnivar', 'veRSA']]
        fig, axs = plt.subplots(1, len(ds.rois), figsize=(20, 8), sharey=True)

        for roi, ax in zip(df['ROI'].unique(), axs.flatten()):
            df_roi = df[df['ROI'] == roi]
            these_colors = MODEL_COLORS['face' if is_face_roi(roi) else 'non-face']

            model_data = []
            for model in these_colors:
                if 'barlow-twins' in model:
                    sel = (df_roi['model_name'] == model) & (df_roi['domain'] == ds.roi_domain[roi])
                else:
                    sel = df_roi['model_name'] == model
                model_data.append(df_roi[sel][metric].values)

                report = ('vggface' in model or
                          (model == 'alexnet-barlow-twins' and
                           (is_face_roi(roi) or roi in ['PPA', 'EBA', 'VWFA-1'])))
                if report:
                    print(roi, model, metric, np.nanmean(model_data[-1]), np.nanstd(model_data[-1]))

            sns.violinplot(data=model_data, ax=ax, palette=list(these_colors.values()),
                           inner='point', linewidth=1, scale='width', width=0.6, cut=1.5)

            n_models = len(model_data)
            noise_ceiling_band(ax, nsd.noise_ceiling_range(noise_ceilings, roi, metric), n_models, zorder=0)
            ax.set_title(roi)
            ax.set_ylim([-0.4, 1.2])
            ax.hlines(np.arange(-0.2, 1.2, 0.2), -0.5, n_models - 0.5, 'k', linewidth=0.25)
            ax.hlines(0, -0.5, n_models - 0.5, 'k', linewidth=1, zorder=0)

            significant_pairs, _ = pairwise_ttest_with_first(model_data, alpha=ttest_alpha)
            line_y = 0.97
            for pair in significant_pairs:
                ax.plot(pair, [line_y, line_y], color='black', lw=0.75)
                line_y -= 0.015

            ax.set_xlabel('')
            ax.axis('off')

        plt.tight_layout()
        save(savedir / f'encoding-best-layers-{metric}.tiff')


def plot_ols_vs_lasso(combined_df_ols, noise_ceilings, savedir, roi_domain, model_name='alexnet-barlow-twins'):
    df = combined_df_ols[(combined_df_ols['model_name'] == model_name) & (combined_df_ols['partition'] == 'val')]
    fg = 0
    for analysis in ['Univar', 'RSA']:
        metric = f've{analysis}'
        for roi in EXAMPLE_ROIS:
            for method, label in [('ols', f'voxel-encoding {analysis} (OLS)'),
                                  ('lasso', f'voxel-encoding {analysis} (sparse positive)')]:
                plt.figure(figsize=(16, 7))
                ax = plt.gca()
                roi_df = df[(df['ROI'] == roi) & (df['method'] == method)]
                plot_domain_layer_curves(ax, roi_df, metric, PLOT_DOMAINS, highlight=roi_domain[roi])
                noise_ceiling_band(ax, nsd.noise_ceiling_range(noise_ceilings, roi, metric), len(LAYERS))

                plt.xticks(rotation=90, fontsize=FT)
                plt.grid('on')
                despine()
                plt.yticks(fontsize=FT)
                plt.legend(fontsize=FT - 3)
                plt.ylim([-0.25, 0.85])
                plt.plot(np.arange(len(LAYERS)), np.zeros((len(LAYERS),)), 'k', linewidth=2)
                plt.tight_layout()
                save(savedir / f'encoding_summary-{fg}-{roi}-{label}.tiff')
                fg += 1


def plot_top_layer_indices(df_max, savedir, roi_groups, metrics=('veUnivar', 'veRSA')):
    """Which layer was selected (on val) as most predictive, per subject and ROI."""
    for model_name in DNN_MODELS:
        layer_list = config.encoding_layers(model_name)
        layer_mapping = {name: i for i, name in enumerate(layer_list)}

        for metric in metrics:
            this_df = df_max[metric]
            this_df = this_df[this_df['model_name'] == model_name].copy()
            this_df['layer_num'] = this_df['layer'].map(layer_mapping)

            fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
            for ax, roi_group in zip(axs, roi_groups.values()):
                rois = roi_group['ROIs']
                domain = 'layer' if 'vggface' in model_name else roi_group['domain']

                df_roi = this_df[this_df['ROI'].isin(rois) & (this_df['domain'] == domain)]
                for subj, subj_df in df_roi.groupby('subj'):
                    x, y = [], []
                    for r, roi in enumerate(rois):
                        if roi in subj_df['ROI'].values:
                            x.append(r)
                            y.append(subj_df[subj_df['ROI'] == roi]['layer_num'].values[0] + np.random.normal(0, 0.1))
                        else:
                            x.append(np.nan)
                            y.append(np.nan)
                    ax.plot(np.array(x) + np.random.normal(0, 0.01, size=len(x)), y, marker='o', linestyle='-',
                            linewidth=4, markersize=5, label=subj, alpha=0.8)

                ax.set_xticks(range(len(rois)))
                ax.set_xticklabels(rois, fontsize=12)
                ax.set_yticks(range(len(layer_list)))
                ax.set_yticklabels(layer_list, fontsize=12)
                ax.grid(True)

            plt.tight_layout()
            save(savedir / f'top-layer-idx-{model_name}-{metric}.tiff')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    datasets.add_arguments(parser, rois_and_subjects=False)
    parser.add_argument('--skip-ols', action='store_true', help='skip the OLS vs. sparse-positive comparison')
    args = parser.parse_args()
    ds = datasets.from_args(parser, args)

    loaddir = ds.analysis_dir(config.ENCODING_SUBDIR)
    savedir = ds.figure_dir('Figure4-Encoding')

    combined_df = load_encoding_results([loaddir], MODEL_NAMES, ds.rois, ds.subjects)
    noise_ceilings = ds.load_noise_ceilings()

    plot_full_layer_summaries(combined_df, noise_ceilings, savedir)

    metrics = ['veUnivar', 'veRSA', 'cUnivar', 'cRSA']
    df_max = {metric: find_max_score_layers(combined_df, metric) for metric in metrics}

    plot_best_layer_violins(df_max, noise_ceilings, savedir, ds)

    if not args.skip_ols:
        loaddir_ols = loaddir / 'alexnet-barlow-twins-ols'
        combined_df_ols = load_encoding_results([loaddir, loaddir_ols], ['alexnet-barlow-twins'], EXAMPLE_ROIS,
                                                ds.subjects)
        plot_ols_vs_lasso(combined_df_ols, noise_ceilings, savedir, ds.roi_domain)

    plot_top_layer_indices(df_max, savedir, ds.roi_groups)


if __name__ == '__main__':
    main()
