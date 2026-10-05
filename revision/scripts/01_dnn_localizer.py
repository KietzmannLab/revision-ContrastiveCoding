"""Identify category-selective DNN units and plot their distribution (Fig. 1, Supp. Fig. 1).

Converted from PROJECT_DNFFA/NOTEBOOKS/1-DNN-Localizer.ipynb.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import scipy.stats as stats  # noqa: E402
from jsputils import classes  # noqa: E402

from dnffa import config  # noqa: E402
from dnffa.plotting import format_layer_axis, plt, save  # noqa: E402

PIE_DOMAINS = ['faces', 'bodies', 'objects', 'scenes', 'characters']  # no scrambled
ACT_DOMAINS = ['faces', 'bodies', 'scenes', 'characters']
SUMMARY_DOMAINS = ['faces', 'bodies', 'scenes', 'characters', 'objects']
COMPARISON_MODELS = ['alexnet-barlow-twins', 'alexnet-ipcl', 'alexnet-supervised', 'alexnet-barlow-twins-random']
FT = 24


def localize(model_name, imageset=config.FLOC_IMAGESET):
    DNN = classes.DNNModel(model_name)
    DNN.find_selective_units(imageset, overwrite=False, verbose=False, FDR_p=config.FDR_P)
    return DNN


def print_tval_summary(DNN):
    """Mean/std t-value of selective units per domain, pooled over layers."""
    for domain in ACT_DOMAINS:
        floc_dict = DNN.selective_units[domain]
        all_tvals = [floc_dict[layer]['tval'][floc_dict[layer]['mask']] for layer in floc_dict]
        all_tvals = np.concatenate([t for t in all_tvals if len(t) > 0])
        print(domain, np.mean(all_tvals), np.std(all_tvals))


def plot_activation_heatmap(DNN, model_name, savedir, layer='fc6', n_images=400):
    """Fig. 1: z-scored probe-set responses of selective units, grouped by domain."""
    acts = DNN.probe_features[layer]
    acts = acts.reshape(acts.shape[0], -1)[:n_images]
    print(acts.shape)

    sorted_acts, nsel, sel_masks = [], [], []
    for domain in ACT_DOMAINS:
        mask = DNN.selective_units[domain][layer]['mask'].astype(bool)
        print(domain, np.sum(mask))
        sel_masks.append(mask)
        nsel.append(np.sum(mask))
        sorted_acts.append(acts[:, mask].T)

    non_sel = np.logical_not(np.any(np.vstack(sel_masks), axis=0))
    sorted_acts.append(acts[:, non_sel].T)
    sorted_acts = stats.zscore(np.vstack(sorted_acts), axis=1)

    plt.figure(figsize=(12, 14))
    plt.imshow(sorted_acts, aspect='auto', clim=(-1.5, 1.5), cmap='magma')
    plt.colorbar()
    x = 0
    for ns in nsel:
        plt.plot(np.arange(n_images), np.ones((n_images,)) * x + ns, 'cyan', linewidth=3)
        x += ns
    plt.xticks(np.arange(0, 480, 80))
    save(savedir / f'{model_name}_{layer}-heatmap.tiff')


def plot_selectivity_pies(DNN, model_name, savedir):
    """Fig. 1: proportion of units selective for each domain, per layer."""
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    colors = ['darkgray', 'purple', 'limegreen', 'orange', 'dodgerblue', 'tomato']
    explodes = np.flip([0.05, 0.05, 0.05, 0.05, 0.05, 0])

    for layer in layers:
        sel_props = [np.mean(DNN.selective_units[domain][layer]['mask']) for domain in PIE_DOMAINS]
        sel_props.append(1 - np.sum(sel_props))
        sel_props = np.flip(sel_props)
        assert np.isclose(np.sum(sel_props), 1)
        print(layer, dict(zip(['none'] + list(np.flip(PIE_DOMAINS)), sel_props)))

        plt.figure(figsize=(12, 12))
        plt.pie(sel_props, colors=colors, explode=explodes, startangle=70)
        save(savedir / f'{model_name}_{layer}-selectivity-pie.tiff')


def plot_selectivity_by_layer(savedir):
    """Supp. Fig. 1: proportion of selective units per layer for several models."""
    for model_name in COMPARISON_MODELS:
        DNN = localize(model_name)
        layers = [l for l in DNN.selective_units[SUMMARY_DOMAINS[0]]
                  if 'flatten' not in l and 'dropout' not in l]

        plt.figure(figsize=(16, 9))
        for domain in SUMMARY_DOMAINS:
            props = [np.mean(DNN.selective_units[domain][layer]['mask']) for layer in layers]
            plt.plot(props, label=config.domain_label(domain), color=config.DOMAIN_COLORS[domain], linewidth=4)
        plt.legend(fontsize=FT, loc='upper left')
        format_layer_axis(layers, ylim=[0, 0.4], fontsize=FT)
        save(savedir / f'{model_name}_{config.FLOC_IMAGESET}_summary.tiff')


def plot_trained_vs_untrained_tvals(savedir):
    """Supp. Fig. 1: mean t-values of selective units, trained vs. untrained model."""
    model_names = ['alexnet-barlow-twins', 'alexnet-barlow-twins-random']
    image_sets = ['vpnl-floc', 'classic-categ']

    floc_info = dict()
    for model_name in model_names:
        floc_info[model_name] = dict()
        for image_set in image_sets:
            DNN = localize(model_name, image_set)
            floc_info[model_name][image_set] = DNN.selective_units

    layer = 'fc6'
    for model_name in model_names:
        units = floc_info[model_name]['vpnl-floc']
        props = [np.nanmean(units[d][layer]['mask']) for d in ACT_DOMAINS]
        tvals = [np.nanmean(units[d][layer]['tval'][units[d][layer]['mask']]) for d in ACT_DOMAINS]
        print(model_name, layer, 'prop. selective:', np.sum(props), 'mean t:', np.nanmean(tvals))

    layers = [l for l in DNN.layer_names_fmt if l != 'flatten']
    for model_name in model_names:
        for image_set in image_sets:
            plt.figure(figsize=(16, 9))
            for domain in SUMMARY_DOMAINS:
                mean_tvals = []
                for layer in layers:
                    # units are always defined on the vpnl-floc localizer
                    mask = floc_info[model_name]['vpnl-floc'][domain][layer]['mask']
                    mean_tvals.append(np.nanmean(floc_info[model_name][image_set][domain][layer]['tval'][mask]))
                plt.plot(mean_tvals, label=domain, color=config.DOMAIN_COLORS[domain], linewidth=4)
            format_layer_axis(layers, ylim=[-8, 17], fontsize=FT, zero_line_width=6)
            save(savedir / f'{model_name}_{image_set}_tvalues.tiff')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--probe-imageset', default='classic-categ')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip-supplementary', action='store_true')
    args = parser.parse_args()

    savedir = config.figure_dir('Figure1-Categ-Selective-Units')

    DNN = localize(args.model)
    print_tval_summary(DNN)

    probe = classes.ImageSet(args.probe_imageset, transforms=DNN.transforms)
    DNN.get_floc_features(probe, field='probe_features', device=args.device, invert=False)

    plot_activation_heatmap(DNN, args.model, savedir)
    plot_selectivity_pies(DNN, args.model, savedir)

    if not args.skip_supplementary:
        plot_selectivity_by_layer(savedir)
        plot_trained_vs_untrained_tvals(savedir)


if __name__ == '__main__':
    main()
