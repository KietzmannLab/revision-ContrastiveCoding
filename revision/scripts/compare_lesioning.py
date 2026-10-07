"""Compare the replicated randomized-lesion control with the original paper's.

Reads two ``randomized_lesion_results.npy`` files written by ``02_lesioning.py`` (the
original one is on the paper's Dataverse, file 10261720). Each holds the unlesioned
per-category top-5 accuracies (``acc``) and, per domain, 10 iterations of lesioning a
random unit set matched in size to the selective set: per-category accuracies after the
lesion (``lsn_acc``) and, per layer, the correlation between the lesioned units'
activation and the accuracy cost (``rs``).

The random unit sets differ between runs, so iteration i of one file is not comparable
to iteration i of the other. Lesion effects are compared as distributions over
iterations (mean, spread, Welch t-test); only ``acc`` is compared category by category.

Writes two CSVs (accuracy cost per domain, r per domain x layer) and a figure with the
randomized readout effect of both runs, as in readout_effect_summary_randomized.
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from dnffa import config  # noqa: E402
from dnffa.plotting import format_layer_axis, plt, save  # noqa: E402
from dnffa.stats import finite_mask  # noqa: E402

DOMAINS = ['faces', 'scenes', 'bodies', 'characters', 'objects']

# Layer order of ``rs`` in 02_lesioning.py: the readout model's layers up to relu7, without flatten.
LAYERS = ['conv1', 'groupnorm1', 'relu1', 'maxpool1', 'conv2', 'groupnorm2', 'relu2', 'maxpool2',
          'conv3', 'groupnorm3', 'relu3', 'conv4', 'groupnorm4', 'relu4', 'conv5', 'groupnorm5',
          'relu5', 'maxpool5', 'avgpool5', 'fc6', 'batchnorm6', 'relu6', 'fc7', 'batchnorm7', 'relu7']

ALPHA = 0.05


def load_results(path):
    results = np.load(path, allow_pickle=True).item()
    for domain in DOMAINS:
        results[domain] = {k: np.asarray(v, dtype=np.float64) for k, v in results[domain].items()}
    return results


def welch(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def compare_unlesioned(orig, mine):
    a, b = orig['acc'], mine['acc']
    diff = np.abs(a - b)
    return {
        'mean_acc_original': np.mean(a),
        'mean_acc_replication': np.mean(b),
        'r': np.corrcoef(a, b)[0, 1],
        'max_abs_diff': np.max(diff),
        'n_categories_differ': int(np.sum(diff > 1e-9)),
        'n_categories': len(a),
    }


def compare_costs(orig, mine):
    """Accuracy cost (unlesioned - lesioned top-5, mean over categories), per domain."""
    rows = []
    for domain in DOMAINS:
        # iterations x categories
        cost_orig = orig['acc'][None] - orig[domain]['lsn_acc']
        cost_mine = mine['acc'][None] - mine[domain]['lsn_acc']
        per_iter_orig, per_iter_mine = cost_orig.mean(axis=1), cost_mine.mean(axis=1)
        rows.append({
            'domain': domain,
            'n_iters_original': len(per_iter_orig),
            'n_iters_replication': len(per_iter_mine),
            'cost_mean_original': per_iter_orig.mean(),
            'cost_sd_original': per_iter_orig.std(),
            'cost_mean_replication': per_iter_mine.mean(),
            'cost_sd_replication': per_iter_mine.std(),
            'cost_welch_p': welch(per_iter_orig, per_iter_mine),
            # agreement of the per-category cost profile, averaged over iterations
            'cost_profile_r': np.corrcoef(cost_orig.mean(axis=0), cost_mine.mean(axis=0))[0, 1],
        })
    return pd.DataFrame(rows)


def compare_rs(orig, mine):
    """Activation/cost correlation per domain x layer, as distributions over iterations."""
    rows = []
    for domain in DOMAINS:
        rs_orig, rs_mine = orig[domain]['rs'], mine[domain]['rs']  # iterations x layers
        for i, layer in enumerate(LAYERS):
            a, b = rs_orig[:, i], rs_mine[:, i]
            fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
            rows.append({
                'domain': domain,
                'layer': layer,
                'r_mean_original': np.mean(fa) if len(fa) else np.nan,
                'r_sem_original': np.std(fa) / np.sqrt(len(fa)) if len(fa) else np.nan,
                'r_mean_replication': np.mean(fb) if len(fb) else np.nan,
                'r_sem_replication': np.std(fb) / np.sqrt(len(fb)) if len(fb) else np.nan,
                'n_nan_original': int(np.sum(~np.isfinite(a))),
                'n_nan_replication': int(np.sum(~np.isfinite(b))),
                'welch_p': welch(a, b),
            })
    df = pd.DataFrame(rows)
    df['r_mean_diff'] = df['r_mean_replication'] - df['r_mean_original']
    return df


def print_summary(unlesioned, costs, rs):
    print('unlesioned top-5 accuracy per category:')
    print(f'  mean {unlesioned["mean_acc_original"]:.5f} (original) vs. {unlesioned["mean_acc_replication"]:.5f} '
          f'(replication); r = {unlesioned["r"]:.6f}; {unlesioned["n_categories_differ"]} of '
          f'{unlesioned["n_categories"]} categories differ, by at most {unlesioned["max_abs_diff"]:.4f}\n')

    print('accuracy cost of the randomized lesions (mean over categories; mean +- sd over iterations):')
    with pd.option_context('display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(costs.drop(columns=['n_iters_original', 'n_iters_replication']).to_string(index=False))

    tested = rs['welch_p'].notna()
    n_sig = int((rs.loc[tested, 'welch_p'] < ALPHA).sum())
    print(f'\nactivation/cost r: {n_sig} of {tested.sum()} domain x layer combinations differ at p < {ALPHA} '
          f'(Welch, uncorrected; ~{ALPHA * tested.sum():.0f} expected by chance)')
    print(f'largest |mean r difference|: {rs["r_mean_diff"].abs().max():.4f}\n')

    by_domain = rs.groupby('domain', sort=False).agg(
        mean_abs_r_diff=('r_mean_diff', lambda d: d.abs().mean()),
        max_abs_r_diff=('r_mean_diff', lambda d: d.abs().max()),
        n_p_below_alpha=('welch_p', lambda p: int((p < ALPHA).sum())),
    )
    profile_r = {d: np.corrcoef(*[g[c].values[finite_mask(g['r_mean_original'].values, g['r_mean_replication'].values)]
                                  for c in ['r_mean_original', 'r_mean_replication']])[0, 1]
                 for d, g in rs.groupby('domain', sort=False)}
    by_domain['layer_profile_r'] = pd.Series(profile_r)
    with pd.option_context('display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(by_domain.to_string())


def plot_comparison(orig, mine, path):
    """readout_effect_summary_randomized for both runs: -r per layer, mean +- sem over iterations."""
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(9 * len(DOMAINS), 9), sharey=True)
    x = np.arange(len(LAYERS))
    for ax, domain in zip(axes, DOMAINS):
        plt.sca(ax)
        color = config.DOMAIN_COLORS[domain]
        for results, label, style in [(orig, 'original', '-'), (mine, 'replication', '--')]:
            rs = -1 * results[domain]['rs']
            n = np.sum(np.isfinite(rs), axis=0)
            with warnings.catch_warnings():  # layers without selective units are NaN in every iteration
                warnings.simplefilter('ignore', RuntimeWarning)
                mean, sem = np.nanmean(rs, axis=0), np.nanstd(rs, axis=0) / np.sqrt(n)
            plt.plot(mean, color=color, linewidth=3, linestyle=style, label=label)
            plt.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2 if style == '-' else 0.1)
        plt.title(config.domain_label(domain), fontsize=24)
        format_layer_axis(LAYERS, ylim=[-1, 0.2], fontsize=14, zero_line_width=2, legend_fontsize=14)
    save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--original', type=Path,
                        default=config.ANALYSIS_DIR / 'original_lesioning' / 'randomized_lesion_results.npy')
    parser.add_argument('--replication', type=Path,
                        default=config.ANALYSIS_DIR / config.LESIONING_SUBDIR / 'randomized_lesion_results.npy')
    args = parser.parse_args()

    orig, mine = load_results(args.original), load_results(args.replication)
    for results, path in [(orig, args.original), (mine, args.replication)]:
        n_layers = results[DOMAINS[0]]['rs'].shape[1]
        if n_layers != len(LAYERS):
            raise SystemExit(f'{path}: {n_layers} layers in rs, expected {len(LAYERS)}')

    unlesioned = compare_unlesioned(orig, mine)
    costs = compare_costs(orig, mine)
    rs = compare_rs(orig, mine)
    print_summary(unlesioned, costs, rs)

    outdir = config.analysis_dir('lesioning_comparison')
    costs.to_csv(outdir / 'randomized_costs.csv', index=False)
    rs.to_csv(outdir / 'randomized_rs.csv', index=False)
    fig_path = config.figure_dir('Lesioning-Comparison') / 'readout_effect_randomized.png'
    plot_comparison(orig, mine, fig_path)
    print(f'\nwrote {outdir / "randomized_costs.csv"}\nwrote {outdir / "randomized_rs.csv"}\nwrote {fig_path}')


if __name__ == '__main__':
    main()
