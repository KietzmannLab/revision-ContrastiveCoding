"""Visualize how category content is channelled through the network hierarchy (Fig. 5).

Converted from PROJECT_DNFFA/NOTEBOOKS/5-Content-Channeling.ipynb.

Fig. 5A: PCA per layer -> correlation RDM over all (image, layer) PC vectors -> 2D MDS, then
each image's trajectory across layers is drawn in 2D and as a rotating 3D plot.

Fig. 5B (not in the notebook; rebuilt from the paper's Materials and Methods): 2D PCA of relu6
responses to 2 ImageNet val images per class, the 240 probe images projected into that space,
and the tuning vectors of the 25 most face-, scene- and object-selective relu6 units drawn as
arrows from the origin.
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastprogress import progress_bar  # noqa: E402
from jsputils import classes, feature_extractor  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402
from mpl_toolkits import mplot3d  # noqa: E402,F401  (registers the 3d projection)
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import MDS  # noqa: E402

from dnffa import config  # noqa: E402
from dnffa.plotting import despine, plt, save  # noqa: E402

LAYERS = ['conv1', 'conv3', 'conv5', 'fc7']
IMG_DOMAINS = [0, 3, 2, 6]  # indices into the probe set's domains
N_PER_DOMAIN = 30
N_PCS = 10
SEED = 0

# Fig. 5B
PC_LAYER = 'relu6'
N_IMAGENET_PER_CLASS = 2
UNIT_DOMAINS = ['faces', 'scenes', 'objects']  # localizer (vpnl-floc) domains drawn as arrows
UNIT_COLORS = {'faces': 'red', 'scenes': 'darkblue', 'objects': 'lightslategray'}
N_TOP_UNITS = 25
# The paper scales the one-hot unit vectors by a constant 2.5e4 "so the tuning vectors lie meaningfully
# onto the image data points". With our relu6 activations that makes the arrows ~300x longer than the
# spread of the probe images, so by default the scale is chosen so the longest arrow reaches the
# farthest probe image instead. Either way all arrows get the same factor, so angles and relative
# lengths are unchanged.
PAPER_ARROW_SCALE = 2.5e4


def pca_mds_trajectories(probe_features, layers, img_domains, n_pcs):
    """Returns MDS coordinates stacked as (layer0 images, layer1 images, ...)."""
    incl_idx = np.concatenate([range(N_PER_DOMAIN * d, N_PER_DOMAIN * (d + 1)) for d in img_domains])

    layer_pcs = []
    for layer in layers:
        X = probe_features[layer]
        X = X.reshape((X.shape[0], -1))[incl_idx]
        layer_pcs.append(PCA(n_components=n_pcs, random_state=SEED).fit_transform(X))
    Y = np.vstack(layer_pcs)

    uber_rdm = squareform(pdist(Y, 'correlation'))
    mds_coords = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED).fit_transform(uber_rdm)
    return Y, uber_rdm, mds_coords


def plot_pcs_and_rdm(Y, uber_rdm, savedir):
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(Y, aspect='auto')
    plt.clim([-100, 100])
    plt.colorbar()
    plt.title('PCs from all imgs/layers')
    plt.subplot(122)
    plt.imshow(uber_rdm)
    plt.colorbar()
    plt.title('uberRDM over all img/layer PCs')
    save(savedir / 'pcs_and_uberRDM.png')


def plot_trajectories_2d(mds_coords, categ_colors, n_img, savedir, linewidths=(1, 2, 5, 8)):
    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(n_img):
        subset = mds_coords[np.arange(i, mds_coords.shape[0], n_img)]
        color = categ_colors[i // N_PER_DOMAIN]
        ax.scatter(subset[0, 0], subset[0, 1], 60, c=color.reshape(1, 3))
        ax.scatter(subset[-1, 0], subset[-1, 1], 240, c=color.reshape(1, 3))
        points = subset.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        ax.add_collection(LineCollection(segments, linewidths=linewidths, color=color, alpha=0.3))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    save(savedir / 'channels_2d.png')


def plot_trajectories_3d(mds_coords, categ_colors, n_img, savedir, linewidths=(1, 2, 3, 4)):
    """Each layer gets its own x position; saves one frame per 10 degrees of rotation."""
    plt.subplots(figsize=(20, 20))
    plt.axis('off')
    plt.box('off')
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2.25, 1, 1))
    xpos = np.arange(len(LAYERS))

    for i in range(n_img):
        subset = mds_coords[np.arange(i, mds_coords.shape[0], n_img)]
        color = categ_colors[i // N_PER_DOMAIN]
        for j in range(subset.shape[0]):
            ax.scatter3D(xpos[j], subset[j, 0], subset[j, 1], c=color.reshape(1, 3), s=40 * j + 10)
            if j < subset.shape[0] - 1:
                ax.plot3D([xpos[j], xpos[j + 1]], [subset[j, 0], subset[j + 1, 0]], [subset[j, 1], subset[j + 1, 1]],
                          color=color, alpha=0.4, linewidth=linewidths[j])

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.set_axis_off()

    for azim in range(0, 360, 10):
        ax.view_init(elev=0, azim=azim)
        plt.savefig(savedir / f'channels_{azim}_deg.png')
    plt.close('all')


# ---------------------------------------------------------------------------
# Fig. 5B: category-selective tuning vectors in the relu6 PC space
# ---------------------------------------------------------------------------

class ImageList(torch.utils.data.Dataset):
    def __init__(self, files, transforms):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.files[idx]).convert('RGB'))


def imagenet_val_subset(n_per_class):
    """First ``n_per_class`` images (sorted by file name) of each of the 1000 val classes."""
    class_dirs = sorted(d for d in config.IMAGENET_VAL_DIR.iterdir() if d.is_dir())
    assert len(class_dirs) == 1000, f'expected 1000 class folders in {config.IMAGENET_VAL_DIR}'
    return [f for d in class_dirs for f in sorted(d.iterdir())[:n_per_class]]


def layer_features(DNN, files, layer, device, batch_size=250):
    """(n_images, n_units) activations of one layer, computed in batches."""
    torch_layer = DNN.layer_names_torch[DNN.layer_names_fmt.index(layer)]
    model = copy.deepcopy(DNN.model)
    loader = torch.utils.data.DataLoader(ImageList(files, DNN.transforms), batch_size=batch_size,
                                         num_workers=4, shuffle=False)
    feats = [feature_extractor.get_features(model, images, [torch_layer], [layer], device=device)[layer]
             for images in progress_bar(loader)]
    return np.concatenate(feats).reshape(len(files), -1)


def top_selective_units(selective_units, domain, layer, k):
    """Indices of the k selective units with the highest localizer t-values, most selective first."""
    tvals = selective_units[domain][layer]['tval'].astype(float)
    tvals[np.isnan(tvals) | ~selective_units[domain][layer]['mask']] = -np.inf
    idx = np.argsort(tvals)[::-1][:k]
    return idx[np.isfinite(tvals[idx])]


def plot_feature_space(probe_pcs, probe, unit_vectors, pca, arrow_scale, savedir):
    fig, ax = plt.subplots(figsize=(12, 12))
    for domain, vectors in unit_vectors.items():
        for x, y in vectors:
            ax.annotate('', xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=UNIT_COLORS[domain], alpha=0.6, lw=1.5))
        ax.plot([], [], color=UNIT_COLORS[domain], lw=2, label=f'{config.domain_label(domain)} units')
    for d, domain in enumerate(probe.floc_domains):
        sel = probe.img_domain_indices == d
        ax.scatter(probe_pcs[sel, 0], probe_pcs[sel, 1], s=80, color=probe.domain_colors[d],
                   edgecolors='k', linewidths=0.5, label=domain, zorder=3)
    ax.axhline(0, color='k', lw=0.5, zorder=0)
    ax.axvline(0, color='k', lw=0.5, zorder=0)
    lim = 1.05 * np.abs(np.vstack([probe_pcs, *unit_vectors.values()])).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_title(f'top {N_TOP_UNITS} selective {PC_LAYER} units per domain, arrow scale {arrow_scale:.3g}', fontsize=14)
    evr = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f'{PC_LAYER} PC1 ({evr[0]:.1f}% of ImageNet variance)', fontsize=16)
    ax.set_ylabel(f'{PC_LAYER} PC2 ({evr[1]:.1f}%)', fontsize=16)
    ax.legend(fontsize=12, frameon=False, loc='best')
    despine(ax)
    plt.tight_layout()
    save(savedir / f'feature_space_{PC_LAYER}_PCs.png')


def contrastive_feature_space(DNN, probe, device, savedir, n_per_class=N_IMAGENET_PER_CLASS,
                              n_top=N_TOP_UNITS, arrow_scale=None):
    imagenet_feats = layer_features(DNN, imagenet_val_subset(n_per_class), PC_LAYER, device)
    pca = PCA(n_components=2, random_state=SEED).fit(imagenet_feats)
    probe_pcs = pca.transform(DNN.probe_features[PC_LAYER].reshape(len(probe.images), -1))

    DNN.find_selective_units(config.FLOC_IMAGESET, FDR_p=config.FDR_P)
    # one-hot unit vector . (n_units, 2) component matrix = rows of components_.T
    unit_vectors = {domain: pca.components_.T[top_selective_units(DNN.selective_units, domain, PC_LAYER, n_top)]
                    for domain in UNIT_DOMAINS}
    if arrow_scale is None:
        longest = max(np.linalg.norm(v, axis=1).max() for v in unit_vectors.values())
        arrow_scale = np.linalg.norm(probe_pcs, axis=1).max() / longest
    unit_vectors = {domain: arrow_scale * v for domain, v in unit_vectors.items()}

    np.savez(savedir / f'feature_space_{PC_LAYER}_PCs.npz', probe_pcs=probe_pcs,
             probe_domains=probe.floc_domains[probe.img_domain_indices],
             components=pca.components_, mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_, arrow_scale=arrow_scale,
             **{f'{domain}_unit_vectors': v for domain, v in unit_vectors.items()})
    plot_feature_space(probe_pcs, probe, unit_vectors, pca, arrow_scale, savedir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--probe-imageset', default='mc8-shined')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--panels', nargs='+', choices=['A', 'B'], default=['A', 'B'],
                        help='A: MDS trajectories across layers; B: relu6 PC space with selective-unit tuning vectors')
    parser.add_argument('--n-imagenet-per-class', type=int, default=N_IMAGENET_PER_CLASS,
                        help='ImageNet val images per class used to fit the relu6 PCA (5B)')
    parser.add_argument('--arrow-scale', type=float, default=None,
                        help='length factor for the selective-unit tuning vectors (5B); default: longest arrow '
                             f'reaches the farthest probe image. The paper used {PAPER_ARROW_SCALE:g}.')
    args = parser.parse_args()

    savedir = config.figure_dir('Figure5-Content-Channeling')

    DNN = classes.DNNModel(args.model)
    probe = classes.ImageSet(args.probe_imageset, transforms=DNN.transforms)
    DNN.get_floc_features(probe, field='probe_features', device=args.device, invert=False)

    if 'A' in args.panels:
        Y, uber_rdm, mds_coords = pca_mds_trajectories(DNN.probe_features, LAYERS, IMG_DOMAINS, N_PCS)
        plot_pcs_and_rdm(Y, uber_rdm, savedir)

        categ_colors = probe.domain_colors[np.array(IMG_DOMAINS)]
        n_img = N_PER_DOMAIN * len(IMG_DOMAINS)
        plot_trajectories_2d(mds_coords, categ_colors, n_img, savedir)
        plot_trajectories_3d(mds_coords, categ_colors, n_img, savedir)

    if 'B' in args.panels:
        contrastive_feature_space(DNN, probe, args.device, savedir, n_per_class=args.n_imagenet_per_class,
                                  arrow_scale=args.arrow_scale)


if __name__ == '__main__':
    main()
