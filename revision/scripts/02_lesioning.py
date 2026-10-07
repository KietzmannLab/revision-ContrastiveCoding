"""Lesion category-selective units and measure ImageNet recognition deficits (Fig. 2).

Converted from PROJECT_DNFFA/NOTEBOOKS/2-Lesioning.ipynb.

As in the notebook, only the randomized-lesion control is cached (in
``analysis_outputs/2-Lesioning``); pass --overwrite to recompute it.

Note: the randomized control draws from numpy's global RNG, whose state is set by the
``np.random.seed(0)`` inside ``scatter_corr``. The order of steps in ``main`` therefore
mirrors the notebook's cell order and should not be rearranged.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from jsputils import classes, feature_extractor  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from dnffa import config  # noqa: E402
from dnffa.plotting import despine, format_layer_axis, plt, save, scatter_corr  # noqa: E402
from dnffa.stats import finite_mask, valid_pearsonr  # noqa: E402

READOUT_DESCRIPTION = ('mdl-alexnet-barlow-twins_from-relu7_mlr-0.05_ilr-0.001_eps-10'
                       '_sparse-pos-True_l1p-1e-05_l1n-1e-05')

LESION_DOMAINS = ['faces', 'scenes', 'bodies', 'characters', 'objects', 'scrambled']
CORE_DOMAINS = ['faces', 'bodies', 'scenes', 'characters']
CV_DOMAINS = ['faces', 'scenes', 'bodies', 'characters']  # order used for the cv sweep / example images
SCATTER_COLORS = ['red', 'dodgerblue', 'limegreen', 'purple']
BAR_COLORS = ['tomato', 'dodgerblue', 'limegreen', 'purple']
FT = 24
N_PROBE_PER_CLASS = 5  # images collected per class
N_EXAMPLES_PER_CATEG = 6  # images saved per category (as in the notebook; the 6th is the next class's first image)

# ImageNet normalization (to undo it when saving example images)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def free_gpu():
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Model setup and lesion sweeps
# ---------------------------------------------------------------------------

def build_lesion_model(model_name, device):
    DNN = classes.DNNModel(model_name)
    DNN.append_readout_layer(readout_from='relu7')
    DNN.load_readout_weights(description=READOUT_DESCRIPTION, device='cpu')

    weights = DNN.readout_model.readout.weight.detach().numpy()
    print('readout weights: prop. |w| < 0.001 =', np.mean(np.abs(weights) < 0.001),
          '| mean |w| =', np.mean(np.abs(weights)), '| std |w| =', np.std(np.abs(weights)))

    DNN.find_selective_units(config.FLOC_IMAGESET, overwrite=False, verbose=False, FDR_p=config.FDR_P)

    DNN.model = DNN.readout_model
    DNN.layer_names_fmt, _ = feature_extractor.get_pretty_layer_names(DNN.readout_model)

    LSN = classes.LesionModel(DNN, device)
    LSN.model.return_acts = False
    LSN.model.masks['apply'] = False
    return DNN, LSN, weights


def lesion_accs(LSN, domain, cv=False):
    LSN.model.return_acts = False
    LSN.apply_channelized_lesions(domain, method='relus')
    LSN.model.masks['apply'] = True
    LSN.get_imagenet_accs(topk=5, cv=cv)
    return LSN.imagenet_accs


def unlesioned_accs(LSN, cv=False):
    LSN.model.masks['apply'] = False
    LSN.model.return_acts = False
    LSN.get_imagenet_accs(topk=5, cv=cv)
    return LSN.imagenet_accs


def run_lesion_sweep(LSN, prelesion_accs):
    results = {'acc': prelesion_accs}
    for domain in progress_bar(LESION_DOMAINS):
        results[domain] = {'lsn_acc': lesion_accs(LSN, domain)}
    return results


def run_cv_lesion_sweep(LSN):
    """Per-category accuracies on two halves of the val set, with and without lesions."""
    accs = unlesioned_accs(LSN, cv=True)
    results_cv = {'acc_splitA': accs[:, 0], 'acc_splitB': accs[:, 1]}
    for domain in progress_bar(CV_DOMAINS):
        accs = lesion_accs(LSN, domain, cv=True)
        results_cv[domain] = {'lsn_acc_splitA': accs[:, 0], 'lsn_acc_splitB': accs[:, 1]}
    return results_cv


def run_randomized_sweep(LSN, prelesion_accs, layer_list, analysis_layers, n_iters):
    """Control: lesion random unit sets matched in size to the selective sets."""
    domains = LESION_DOMAINS[:-1]
    results_rand = {'acc': prelesion_accs}
    for domain in domains:
        results_rand[domain] = {'lsn_acc': [], 'rs': []}

    for _ in progress_bar(range(n_iters)):
        LSN.randomize_selective_unit_indices()
        LSN.get_selective_unit_acts(layers=analysis_layers)
        free_gpu()

        for domain in progress_bar(domains):
            postlesion_accs = lesion_accs(LSN, domain)
            results_rand[domain]['lsn_acc'].append(postlesion_accs)
            y = results_rand['acc'] - postlesion_accs
            rs = [valid_pearsonr(LSN.selective_unit_acts[domain][layer], y) for layer in layer_list]
            results_rand[domain]['rs'].append(rs)
    return results_rand


def cached(path, compute, overwrite):
    if path.exists() and not overwrite:
        print('loading cached', path)
        return np.load(path, allow_pickle=True).item()
    out = compute()
    np.save(path, out, allow_pickle=True)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_readout_weights(weights, savedir):
    plt.imshow(weights, aspect='auto', cmap='RdBu_r', clim=(-0.01, 0.01))
    plt.colorbar()
    save(savedir / 'readout_weights.png')


def plot_act_cost_scatters(LSN, results, savedir, layer='relu6'):
    """Category activation of selective units vs. accuracy drop after lesioning them."""
    rs = []
    for d, domain in enumerate(CORE_DOMAINS):
        x = LSN.selective_unit_acts[domain][layer].copy()
        y = -100 * (results['acc'] - results[domain]['lsn_acc'])
        valid = finite_mask(x, y)
        if np.sum(valid) <= 750:
            print('skipping', domain, layer, np.sum(valid))
            continue
        rs.append(valid_pearsonr(x, y))
        plt.figure(figsize=(24, 24))
        scatter_corr(x[valid], y[valid], linecols=[SCATTER_COLORS[d]], bigks=[10])
        plt.xticks(fontsize=75)
        plt.yticks(fontsize=75)
        save(savedir / f'scatter_corr_{domain}.tiff')
    print('act/cost correlation:', np.mean(rs), np.std(rs))


def plot_readout_effect_by_layer(LSN, results, layer_list, savedir):
    plt.figure(figsize=(16, 9))
    for domain in ['faces', 'scenes', 'bodies', 'characters', 'objects']:
        y = -100 * (results['acc'] - results[domain]['lsn_acc'])
        rs = [valid_pearsonr(LSN.selective_unit_acts[domain][layer], y) for layer in layer_list]
        plt.plot(rs, label=config.domain_label(domain), color=config.DOMAIN_COLORS[domain], linewidth=5)
    format_layer_axis(layer_list, ylim=[-1, 0.2], fontsize=FT, zero_line_width=3, legend_fontsize=FT - 3)
    save(savedir / 'readout_effect_summary.tiff')


def plot_cost_correlations(results, savedir):
    """Similarity of lesion-cost profiles across domains + two example dissociations."""
    domains = ['faces', 'bodies', 'scenes', 'characters', 'objects']
    costs = np.vstack([results['acc'] - results[d]['lsn_acc'] for d in domains])
    cost_corrs = pdist(costs, 'correlation')
    print('cost profile correlation:', np.mean(1 - cost_corrs), np.std(1 - cost_corrs))

    plt.figure(figsize=(20, 16))
    sns.heatmap(1 - squareform(cost_corrs), annot=True, cmap='RdBu_r', vmin=-1, vmax=1, annot_kws={'size': 40})
    plt.xticks([])
    plt.yticks([])
    save(savedir / 'cost_corr_domain_summary.png')

    for dA, dB in [(0, 2), (0, 3)]:
        plt.figure(figsize=(24, 24))
        scatter_corr(-100 * costs[dA], -100 * costs[dB], e1=0.5, e2=0.5, bigks=[10, 10],
                     linecols=[SCATTER_COLORS[dA], SCATTER_COLORS[dB]])
        plt.xticks(fontsize=75)
        plt.yticks(fontsize=75)
        save(savedir / f'dissociation_{domains[dA]}-{domains[dB]}.tiff')


def cv_domain_costs(results_cv, domain, split, prop=False):
    cost = results_cv[f'acc_split{split}'] - results_cv[domain][f'lsn_acc_split{split}']
    if prop:
        cost = cost / results_cv[f'acc_split{split}']
    cost[~np.isfinite(cost)] = 0
    return cost


def plot_dissociation_bars(results_cv, savedir, ks=(5, 10, 25, 50, 75, 100), prop=False):
    """Cross-validated: define top-k impaired categories on split A, measure drops on split B."""
    domains = CORE_DOMAINS
    plt.figure(figsize=(12, len(ks) * 5))

    for c, k in enumerate(ks, start=1):
        costs = {(d, sp): cv_domain_costs(results_cv, d, sp, prop) for d in domains for sp in 'AB'}
        top_k = {d: np.argsort(costs[(d, 'A')])[-k:] for d in domains}

        mean_drop = {l: {p: np.mean(costs[(l, 'B')][top_k[p]]) for p in domains} for l in domains}
        sem_drop = {l: {p: np.std(costs[(l, 'B')][top_k[p]]) / np.sqrt(k) for p in domains} for l in domains}

        if k == 10:
            for d in domains:
                vals = costs[(d, 'B')][top_k[d]]
                print('k=10', d, np.mean(vals), np.std(vals))

        x = np.arange(len(domains))
        bar_width = 0.18
        plt.subplot(len(ks), 1, c)
        for i, probe_domain in enumerate(domains):
            means = -100 * np.array([mean_drop[l][probe_domain] for l in domains])
            sems = 100 * np.array([sem_drop[l][probe_domain] for l in domains])
            plt.bar(x + i * bar_width, means, width=bar_width, color=BAR_COLORS[i], yerr=sems)

        plt.hlines(0, 0, 3.5, 'k', linewidth=0.5)
        plt.xticks([])
        plt.box('off')
        despine()
        plt.yticks(fontsize=25)
        plt.grid('on')
        plt.ylim([-110, 25] if prop else [-70, 15])

    plt.tight_layout()
    save(savedir / 'dissociation_bars_k_summary.tiff')


def load_imagenet_categories():
    with open(config.IMAGENET_CLASS_LABELS, 'r') as f:
        class_index = json.load(f)
    return np.array([class_index[idx] for idx in range(len(class_index))])


def print_most_impaired_categories(results_cv, categories, k=6):
    for domain in CORE_DOMAINS:
        costs = results_cv['acc_splitA'] - results_cv[domain]['lsn_acc_splitA']
        idx = np.flip(np.argsort(costs)[-k:])
        print(domain, categories[idx], '\naccuracy drops:', costs[idx], '\n')


def load_probe_images(n_per_class=N_PROBE_PER_CLASS):
    """First ``n_per_class`` ImageNet val images of every class, ordered by class.

    Same result as the notebook (which first loaded all 50k val images), but keeps only
    the selected images in memory. The loader is still consumed completely, as before.
    """
    loader = classes.DataLoaderFFCV('val')
    probe_images = None
    counts = np.zeros(1000, dtype=int)
    for images, targets, _, _ in progress_bar(loader.data_loader):
        images = images.cpu()
        if probe_images is None:
            probe_images = torch.empty(1000 * n_per_class, *images.shape[1:])
        for img, t in zip(images, targets.cpu().numpy().astype(int)):
            if counts[t] < n_per_class:
                probe_images[t * n_per_class + counts[t]] = img
                counts[t] += 1
    return probe_images


def save_impaired_category_examples(results_cv, categories, savedir, n_categs=8):
    savedir = savedir / 'impaired-examples'
    savedir.mkdir(exist_ok=True)
    probe_images = load_probe_images()
    for domain in CV_DOMAINS:
        costs = results_cv['acc_splitA'] - results_cv[domain]['lsn_acc_splitA']
        rankings = np.flip(np.argsort(costs))
        for i in range(n_categs):
            categ_idx = rankings[i]
            loss = costs[categ_idx]
            for j in range(N_EXAMPLES_PER_CATEG):
                img = probe_images[categ_idx * N_PROBE_PER_CLASS + j].numpy().transpose(1, 2, 0)
                img = np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
                plt.figure(figsize=(20, 20))
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                save(savedir / f'{domain}-impaired-{i}-{categories[categ_idx]}-{round(loss, 2)}-{j}.tiff')
    del probe_images
    free_gpu()


def plot_randomized_readout_effect(results_rand, layer_list, n_iters, savedir):
    plt.figure(figsize=(16, 9))
    for domain in ['faces', 'scenes', 'bodies', 'characters', 'objects']:
        rs = -1 * np.stack(results_rand[domain]['rs'], axis=1)
        rs_mean = np.mean(rs, axis=1)
        rs_sem = np.std(rs, axis=1) / np.sqrt(n_iters)
        color = config.DOMAIN_COLORS[domain]
        plt.plot(rs_mean, label=config.domain_label(domain), color=color, linewidth=3)
        plt.fill_between(np.arange(len(layer_list)), rs_mean - rs_sem, rs_mean + rs_sem, color=color, alpha=0.3)
    format_layer_axis(layer_list, ylim=[-1, 0.2], fontsize=FT, zero_line_width=3, legend_fontsize=FT - 3)
    save(savedir / 'readout_effect_summary_randomized.tiff')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-random-iters', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true', help='recompute the cached randomized-lesion control')
    parser.add_argument('--skip-examples', action='store_true',
                        help='do not export example images of the most impaired categories '
                             '(deviates from the notebook run; may affect the randomized control)')
    parser.add_argument('--skip-randomized', action='store_true')
    args = parser.parse_args()

    savedir = config.figure_dir('Figure2-Lesioning')
    cachedir = config.analysis_dir(config.LESIONING_SUBDIR)

    free_gpu()
    DNN, LSN, weights = build_lesion_model(args.model, args.device)
    plot_readout_weights(weights, savedir)

    analysis_layers = DNN.layer_names_fmt[:-1]
    layer_list = [l for l in analysis_layers if 'flatten' not in l]

    prelesion_accs = unlesioned_accs(LSN)
    print('pre-lesion top-5 acc:', np.mean(prelesion_accs), np.std(prelesion_accs))

    LSN.get_selective_unit_acts(layers=analysis_layers)
    free_gpu()

    results = run_lesion_sweep(LSN, prelesion_accs)
    free_gpu()

    plot_act_cost_scatters(LSN, results, savedir)
    plot_readout_effect_by_layer(LSN, results, layer_list, savedir)
    plot_cost_correlations(results, savedir)

    results_cv = run_cv_lesion_sweep(LSN)
    print('unlesioned acc (split B):', np.mean(results_cv['acc_splitB']))
    plot_dissociation_bars(results_cv, savedir)

    categories = load_imagenet_categories()
    print_most_impaired_categories(results_cv, categories)
    if not args.skip_examples:
        save_impaired_category_examples(results_cv, categories, savedir)

    # Must come last: randomizing overwrites LSN's selective-unit indices and activations.
    if not args.skip_randomized:
        results_rand = cached(cachedir / 'randomized_lesion_results.npy',
                              lambda: run_randomized_sweep(LSN, prelesion_accs, layer_list,
                                                           analysis_layers, args.n_random_iters),
                              args.overwrite)
        plot_randomized_readout_effect(results_rand, layer_list, args.n_random_iters, savedir)


if __name__ == '__main__':
    main()
