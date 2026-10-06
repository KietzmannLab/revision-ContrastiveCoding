"""Compare replicated DNN selective units with the original paper's.

Reads the per-domain ``.npy`` files written by ``01_dnn_localizer.py``
(``<model>_<imageset>-<domain>_FDR-<fdr>.npy``) from two directories and, for every
domain and layer, reports how many units each run calls selective, how far the two
masks overlap, and how closely the t-values agree. Writes a CSV with one row per
domain x layer and a figure with proportion selective and mask overlap per layer.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dnffa import config  # noqa: E402
from dnffa.plotting import format_layer_axis, plt, save  # noqa: E402
from dnffa.stats import finite_mask  # noqa: E402

# Layers shown in the Fig. 1 pies; the printed summary is restricted to these.
SUMMARY_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']


def find_domains(directory, model, imageset, fdr):
    pattern = re.compile(rf'^{re.escape(model)}_{re.escape(imageset)}-(.+)_FDR-{re.escape(fdr)}\.npy$')
    return {m.group(1) for f in directory.glob('*.npy') if (m := pattern.match(f.name))}


def load_units(directory, model, imageset, domain, fdr):
    return np.load(directory / f'{model}_{imageset}-{domain}_FDR-{fdr}.npy', allow_pickle=True).item()


def compare_layer(orig, mine):
    """Agreement statistics between two ``{'mask', 'tval', ...}`` dicts for one layer."""
    m_orig, m_mine = orig['mask'].astype(bool), mine['mask'].astype(bool)
    n_both = int(np.sum(m_orig & m_mine))
    n_either = int(np.sum(m_orig | m_mine))

    t_orig, t_mine = orig['tval'].astype(np.float64), mine['tval'].astype(np.float64)
    valid = finite_mask(t_orig, t_mine)
    t_diff = np.abs(t_orig[valid] - t_mine[valid])
    if np.sum(valid) > 1 and np.std(t_orig[valid]) > 0 and np.std(t_mine[valid]) > 0:
        t_r = np.corrcoef(t_orig[valid], t_mine[valid])[0, 1]
    else:
        t_r = np.nan

    row = {
        'n_units': len(m_orig),
        'n_sel_original': int(np.sum(m_orig)),
        'n_sel_replication': int(np.sum(m_mine)),
        'prop_sel_original': np.mean(m_orig),
        'prop_sel_replication': np.mean(m_mine),
        'n_sel_both': n_both,
        'n_only_original': int(np.sum(m_orig & ~m_mine)),
        'n_only_replication': int(np.sum(~m_orig & m_mine)),
        # 1 if both masks are empty: nothing selective in either run counts as full agreement
        'jaccard': n_both / n_either if n_either else 1.0,
        'mask_identical': bool(np.array_equal(m_orig, m_mine)),
        'tval_r': t_r,
        'tval_max_abs_diff': np.max(t_diff) if len(t_diff) else np.nan,
        'tval_mean_abs_diff': np.mean(t_diff) if len(t_diff) else np.nan,
        'n_nan_tval_original': int(np.sum(~np.isfinite(t_orig))),
        'n_nan_tval_replication': int(np.sum(~np.isfinite(t_mine))),
    }
    if 'mask-2x' in orig and 'mask-2x' in mine:
        row['n_mask2x_differ'] = int(np.sum(orig['mask-2x'].astype(bool) != mine['mask-2x'].astype(bool)))
    return row


def compare(original_dir, replication_dir, model, imageset, fdr):
    domains = find_domains(original_dir, model, imageset, fdr)
    missing = domains - find_domains(replication_dir, model, imageset, fdr)
    if missing:
        print(f'no replicated units for {sorted(missing)}; skipping those domains')
    domains = sorted(domains - missing)
    if not domains:
        raise SystemExit(f'no {model}_{imageset}-*_FDR-{fdr}.npy files found in both directories')

    rows = []
    for domain in domains:
        orig = load_units(original_dir, model, imageset, domain, fdr)
        mine = load_units(replication_dir, model, imageset, domain, fdr)
        if list(orig) != list(mine):
            print(f'{domain}: layer lists differ; comparing the {len(set(orig) & set(mine))} shared layers')
        for layer in [l for l in orig if l in mine]:
            if orig[layer]['mask'].shape != mine[layer]['mask'].shape:
                print(f'{domain} {layer}: shapes differ ({orig[layer]["mask"].shape} vs. '
                      f'{mine[layer]["mask"].shape}); skipped')
                continue
            rows.append({'domain': domain, 'layer': layer, **compare_layer(orig[layer], mine[layer])})
    return pd.DataFrame(rows)


def print_summary(df):
    n_identical = df['mask_identical'].sum()
    print(f'\nmasks identical in {n_identical} of {len(df)} domain x layer combinations')
    print(f'lowest t-value correlation: {df["tval_r"].min():.6f}; '
          f'largest |t difference|: {df["tval_max_abs_diff"].max():.4g}\n')

    cols = ['domain', 'layer', 'n_sel_original', 'n_sel_replication', 'n_only_original',
            'n_only_replication', 'jaccard', 'tval_r', 'tval_max_abs_diff']
    with pd.option_context('display.width', 200, 'display.max_rows', None, 'display.float_format', '{:.4f}'.format):
        print(df.loc[df['layer'].isin(SUMMARY_LAYERS), cols].to_string(index=False))


def plot_comparison(df, model, path):
    layers = [l for l in df['layer'].unique() if l != 'flatten']
    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    for domain, d in df.groupby('domain'):
        d = d.set_index('layer').loc[layers]
        color = config.DOMAIN_COLORS.get(domain, 'k')
        label = config.domain_label(domain)
        axes[0].plot(d['prop_sel_original'].values, color=color, linewidth=4, label=f'{label} (original)')
        axes[0].plot(d['prop_sel_replication'].values, color=color, linewidth=2, linestyle='--',
                     label=f'{label} (replication)')
        axes[1].plot(d['jaccard'].values, color=color, linewidth=4, label=label)

    plt.sca(axes[0])
    plt.title(f'{model}: proportion selective', fontsize=20)
    format_layer_axis(layers, ylim=[0, max(0.05, df['prop_sel_original'].max() * 1.1)], fontsize=14,
                      legend_fontsize=11)
    plt.sca(axes[1])
    plt.title('mask overlap (Jaccard)', fontsize=20)
    format_layer_axis(layers, ylim=[0, 1.05], fontsize=14, legend_fontsize=11)
    save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dir', type=Path, default=config.ANALYSIS_DIR / 'original_selective_units')
    parser.add_argument('--replication-dir', type=Path, default=config.ANALYSIS_DIR / 'selective_units')
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--imageset', default=config.FLOC_IMAGESET)
    parser.add_argument('--fdr', default=str(config.FDR_P)[2:],
                        help="FDR tag in the filenames, as written by jsputils (str(FDR_p)[2:], e.g. '05')")
    args = parser.parse_args()

    df = compare(args.original_dir, args.replication_dir, args.model, args.imageset, args.fdr)
    print_summary(df)

    stem = f'{args.model}_{args.imageset}_FDR-{args.fdr}'
    csv_path = config.analysis_dir('selective_units_comparison') / f'{stem}.csv'
    df.to_csv(csv_path, index=False)
    fig_path = config.figure_dir('Selective-Units-Comparison') / f'{stem}.png'
    plot_comparison(df, args.model, fig_path)
    print(f'\nwrote {csv_path}\nwrote {fig_path}')


if __name__ == '__main__':
    main()
