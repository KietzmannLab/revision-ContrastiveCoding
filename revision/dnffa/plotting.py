"""Plotting helpers shared by the figure scripts."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def despine(ax=None):
    ax = ax or plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_layer_axis(layer_labels, ylim, fontsize=24, zero_line_width=None, legend_fontsize=None):
    """Common styling of the 'value per layer' line plots used throughout the paper."""
    n = len(layer_labels)
    plt.xticks(np.arange(n), np.array(layer_labels), rotation=90, fontsize=fontsize)
    plt.grid('on')
    despine()
    plt.ylim(ylim)
    plt.yticks(fontsize=fontsize)
    if zero_line_width:
        plt.plot(np.arange(n), np.zeros((n,)), color='k', linewidth=zero_line_width)
    if legend_fontsize:
        plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()


def save(path, close=True):
    plt.savefig(path)
    if close:
        plt.close('all')


def scatter_corr(x, y, linecols=('r',), bigks=(None,), e1=None, e2=0.5):
    """Scatter with jitter and a linear fit; highlights the top-k points per axis."""
    np.random.seed(0)
    ep1 = np.zeros((len(x),)) if e1 is None else np.random.normal(0, e1, len(x))
    ep2 = np.random.normal(0, e2, len(y))
    sizes = np.array([150] * len(y))
    colors = ['darkgray'] * len(y)
    if bigks[0]:
        for bk, bigk in enumerate(bigks):
            if bk == 0 and len(bigks) == 1:
                idx = np.argsort(y)
            elif bk == 0 and len(bigks) == 2:
                idx = np.argsort(x)
            else:
                idx = np.argsort(y)

            sizes[idx[:bigk]] = 400
            for k in range(bigk):
                colors[idx[:bigk][k]] = linecols[bk]
    colors = np.array(colors)

    linecol = 'k' if len(linecols) == 2 else linecols[0]

    xj = x + ep1
    yj = y + ep2
    plt.scatter(xj, yj, sizes, c=colors)
    plt.plot(np.unique(xj), np.poly1d(np.polyfit(xj, yj, 1))(np.unique(xj)), color=linecol, linewidth=20)
    print(f'r = {round(np.corrcoef(x, y)[1, 0], 3)}')


def noise_ceiling_band(ax, nc_range, n, **kwargs):
    """Gray band spanning the across-subject noise-ceiling range."""
    ax.fill_between(np.arange(-0.5, n + 0.5), nc_range[0], nc_range[1],
                    color='gray', alpha=0.2, **kwargs)
