"""Gabor and GIST-PC image features: a Python port of PROJECT_DNFFA/HELPERS/Code-GistModel.

Reproduces computeGaborAndGistFeatures.m (and createRosasGabor.m, gabor_set.m,
localFeatures.m, downN.m, pca.m) as used by DNFFA_extract_gist.m, including its quirks:

- The GIST PCA is fit separately on every image set it is called on (each subject's train
  images, and the test images on their own), so train and test GistPC features live in
  different, unaligned PC bases. The PCA is also not mean-centered.
- Images must be square and are not resized; the filters are built for the image size
  (MATLAB hard-codes 425, the NSD size).

PC signs are arbitrary (as with MATLAB's eigs), so GistPC columns may differ in sign from
the MATLAB output. Since train and test PCs come from separate fits, this can change GistPC
test scores.
"""

import numpy as np

# Parameters from DNFFA_extract_gist.m / computeGaborAndGistFeatures.m
N_BLOCKS = 16
ORIENTATIONS_PER_SCALE = (12, 8, 6, 4)
N_PCS = 50


def gabor_transfer_functions(orientations_per_scale, image_size):
    """createRosasGabor.m + gabor_set.m: (n, n, n_filters) frequency-domain filters, DC at [0, 0]."""
    params = []  # [sigma_r, fr, sigma_theta, theta] per filter, high to low spatial frequency
    for scale, n_or in enumerate(orientations_per_scale):
        for j in range(n_or):
            params.append([0.35, 0.3 / (1.85 ** scale), 16 * n_or ** 2 / 32 ** 2, np.pi / n_or * j])

    n = image_size
    fx, fy = np.meshgrid(np.arange(n), np.arange(n))
    fx = fx - n / 2
    fy = fy - n / 2
    fr = np.sqrt(fx ** 2 + fy ** 2)
    t = np.angle(fx + 1j * fy)

    G = np.zeros((n, n, len(params)))
    for i, (sigma_r, f0, sigma_theta, theta) in enumerate(params):
        tr = t + theta
        tr = tr + 2 * np.pi * (tr < -np.pi) - 2 * np.pi * (tr > np.pi)
        if f0 > 0:
            H = np.exp(-10 * sigma_r * (fr / n / f0 - 1) ** 2 - 2 * sigma_theta * np.pi * tr ** 2)
        else:
            H = np.exp(-10 * sigma_r * fr ** 2 - 2 * sigma_theta * np.pi * tr ** 2)
        G[:, :, i] = np.fft.fftshift(H)
    return G


def block_average(x, n_blocks):
    """downN.m: mean over non-overlapping spatial blocks -> (n_blocks, n_blocks, n_filters)."""
    edges_r = np.fix(np.linspace(0, x.shape[0], n_blocks + 1)).astype(int)
    edges_c = np.fix(np.linspace(0, x.shape[1], n_blocks + 1)).astype(int)
    y = np.zeros((n_blocks, n_blocks, x.shape[2]))
    for r in range(n_blocks):
        for c in range(n_blocks):
            y[r, c] = x[edges_r[r]:edges_r[r + 1], edges_c[c]:edges_c[c + 1]].mean(axis=(0, 1))
    return y


def local_features(img, G, n_blocks, out=None):
    """localFeatures.m: block-averaged Gabor energy of one grayscale image, flattened as vC(:).

    Filters one Gabor at a time into ``out`` (a reusable G-shaped float buffer) rather than
    allocating (n, n, n_filters) complex temporaries per image, which is very slow on nodes
    under memory pressure.
    """
    if out is None:
        out = np.empty(G.shape)
    img = img - img.mean()
    F = np.fft.fft2(img)
    for i in range(G.shape[2]):
        out[:, :, i] = np.abs(np.fft.ifft2(F * G[:, :, i]))
    return block_average(out, n_blocks).flatten(order='F')


def compute_gabor_and_gist_features(images, n_blocks=N_BLOCKS,
                                    orientations_per_scale=ORIENTATIONS_PER_SCALE, n_pcs=N_PCS):
    """computeGaborAndGistFeatures.m for an (n_images, h, w, 3) image block.

    Returns (Gabor, GistPC) dicts with the fields of the MATLAB structs (except Gabor.G).
    """
    G = gabor_transfer_functions(orientations_per_scale, images.shape[1])
    if images.shape[1:3] != G.shape[:2]:
        raise ValueError(f'images must be square, got {images.shape[1:3]}')

    # (n_features, n_images), as in MATLAB
    buf = np.empty(G.shape)
    features = np.stack([local_features(np.asarray(im, dtype=np.float64).mean(axis=2), G, n_blocks, buf)
                         for im in images], axis=1)

    # pca.m: top eigenvectors of features @ features.T (no centering), via SVD
    U, s, _ = np.linalg.svd(features, full_matrices=False)
    pc = U[:, :n_pcs]
    latent = s[:n_pcs] ** 2

    gabor = {'featureMatrix': features.T,
             'paramString': f'[{"  ".join(map(str, orientations_per_scale))}] x {n_blocks}',
             'NBlocks': n_blocks,
             'NrOrientationsPerScale': np.array(orientations_per_scale)}
    gist_pc = {'featureMatrix': (pc.T @ features).T,
               'pcLocalWeights': pc,
               'NumPC': n_pcs,
               'latent': latent,
               'vaf': latent / latent.sum()}
    return gabor, gist_pc
