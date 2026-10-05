"""Visualize how category content is channelled through the network hierarchy (Fig. 5).

Converted from PROJECT_DNFFA/NOTEBOOKS/5-Content-Channeling.ipynb.

PCA per layer -> correlation RDM over all (image, layer) PC vectors -> 2D MDS, then each
image's trajectory across layers is drawn in 2D and as a rotating 3D plot.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from jsputils import classes  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits import mplot3d  # noqa: E402,F401  (registers the 3d projection)
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import MDS  # noqa: E402

from dnffa import config  # noqa: E402
from dnffa.plotting import plt, save  # noqa: E402

LAYERS = ['conv1', 'conv3', 'conv5', 'fc7']
IMG_DOMAINS = [0, 3, 2, 6]  # indices into the probe set's domains
N_PER_DOMAIN = 30
N_PCS = 10
SEED = 0


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--probe-imageset', default='mc8-shined')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    savedir = config.figure_dir('Figure5-Content-Channeling')

    DNN = classes.DNNModel(args.model)
    probe = classes.ImageSet(args.probe_imageset, transforms=DNN.transforms)
    DNN.get_floc_features(probe, field='probe_features', device=args.device, invert=False)

    Y, uber_rdm, mds_coords = pca_mds_trajectories(DNN.probe_features, LAYERS, IMG_DOMAINS, N_PCS)
    plot_pcs_and_rdm(Y, uber_rdm, savedir)

    categ_colors = probe.domain_colors[np.array(IMG_DOMAINS)]
    n_img = N_PER_DOMAIN * len(IMG_DOMAINS)
    plot_trajectories_2d(mds_coords, categ_colors, n_img, savedir)
    plot_trajectories_3d(mds_coords, categ_colors, n_img, savedir)


if __name__ == '__main__':
    main()
