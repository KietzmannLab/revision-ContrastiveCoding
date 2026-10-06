# DNFFA analyses as Python scripts

Script versions of the notebooks in `PROJECT_DNFFA/NOTEBOOKS/`. The analysis logic is unchanged; the heavy lifting is still done by the `jsputils` submodule.

```
revision/
├── dnffa/                     # shared code
│   ├── config.py              # output paths, subjects, ROIs, domain colours, layer lists, NSD settings
│   ├── nsd.py                 # load ROIs/betas, NSD images, GSN noise ceilings
│   ├── laion.py               # LAION-fMRI loader with the same ROI interface as nsd.py
│   ├── gist.py                # Python port of the MATLAB Gabor / GIST-PC features (HELPERS/Code-GistModel)
│   ├── plotting.py            # despine/layer-axis styling, scatter_corr, noise-ceiling band
│   └── stats.py               # finite-masked Pearson r, paired t-tests vs. first model
└── scripts/
    ├── download_models.py        # pre-download public model weights (run once)
    ├── make_imagenet_ffcv.py     # write the ImageNet val set to FFCV for 02 (run once)
    ├── 00_train_readout.py       # 0-Train-Readout.ipynb
    ├── 01_dnn_localizer.py       # 1-DNN-Localizer.ipynb      -> Figure 1, Supp. Fig. 1
    ├── 02_lesioning.py           # 2-Lesioning.ipynb          -> Figure 2
    ├── 03_encoding.py            # 3-Encoding.ipynb
    ├── 04_encoding_plots.py      # 4-Encoding-Plots.ipynb     -> Figure 4
    ├── 05_content_channeling.py  # 5-Content-Channeling.ipynb -> Figure 5
    ├── 06_noise_ceilings.py      # 6-NoiseCeilings.ipynb
    └── 07_low_level_models.py    # 7-LowLevelModels.ipynb
```

## Setup

### Environment

`environment/environment.yml` pins the original environment: Python 3.9, PyTorch 1.13.1, CUDA 11.7. It can't be used directly with `conda env create`:

- The conda part has to be solved with `CONDA_OVERRIDE_CUDA=11.8`. The pinned `nccl` requires a CUDA version between 11.2 and 12, which fails on machines without a GPU and on GPU nodes whose driver reports CUDA 12.
- The pip list contains `python-graphviz`, the conda package name. On PyPI the package is called `graphviz`.
- `fastprogress` is missing, but jsputils imports it.
- `ffcv` compiles against the env's compilers and opencv, so the pip packages are installed with the env activated.
- `pycortex` 1.2.6 does not compile with Cython 3. Install with `--no-build-isolation` so pip uses the env's pinned Cython 0.29 and numpy instead of fetching the latest ones.

Older conda versions (e.g. 4.10) may run out of memory loading the conda-forge index. micromamba avoids this. Run the following from the repo root:

```bash
# micromamba (once)
(cd ~ && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba)
export MAMBA_ROOT_PREFIX=~/micromamba

# conda packages; the env goes into ~/.conda/envs so `conda activate revision` finds it
sed '/^  - pip:/,$d; /^prefix:/d' environment/environment.yml > /tmp/revision-conda.yml
CONDA_OVERRIDE_CUDA=11.8 ~/bin/micromamba create -y -p ~/.conda/envs/revision -f /tmp/revision-conda.yml

# pip packages
conda activate revision
sed -n '/^  - pip:/,$p' environment/environment.yml | grep '^      - ' \
    | sed 's/^      - //; s/^python-graphviz/graphviz/' > /tmp/revision-pip.txt
pip install --no-build-isolation -r /tmp/revision-pip.txt fastprogress==1.0.3

# jsputils
pip install -e jsputils

# LAION-fMRI loader (see below)
pip install --ignore-requires-python --no-deps \
    "laion-fmri @ git+https://github.com/ViCCo-Group/LAION-fMRI.git@main"
pip install awscli

# libGL: opencv (imported by ffcv) needs libGL.so.1, which the klab compute nodes don't have.
# The solver can't add it (every libglx build pins a newer xorg-libx11), so unpack the
# libraries into the env by hand; they work with the env's libX11 1.8.4.
(cd "$(mktemp -d)" && for p in libglvnd libglx libgl; do
    curl -fsSLO https://conda.anaconda.org/conda-forge/linux-64/$p-1.7.0-ha4b6fd6_0.conda
    unzip -oq $p-1.7.0-ha4b6fd6_0.conda && zstd -dc pkg-$p-*.tar.zst | tar -x
done && cp -a lib/libGL* "$CONDA_PREFIX/lib/")

# check (cuda is True on a GPU node)
python -c "from jsputils import classes; import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import laion_fmri, numpy; print(laion_fmri.__file__, numpy.__version__)"
```

#### LAION-fMRI

The [LAION-fMRI](https://github.com/ViCCo-Group/LAION-fMRI) package requires Python ≥ 3.10 and numpy ≥ 1.24. This env has Python 3.9 and numpy 1.23.5, so a plain `pip install` is refused, or it upgrades numpy and breaks torch 1.13. Neither floor is actually needed. The package's offline test suite (500 tests, `pytest -m "not network"`) passes on this env. That's why the install skips both checks:

- `--ignore-requires-python` skips the Python version check. Modules that use `X | None` hints also import `from __future__ import annotations`, so they run on 3.9.
- `--no-deps` keeps numpy at 1.23.5. The other dependencies (h5py, nibabel, pandas, Pillow) are already in the env at versions that meet the requirements. `awscli` is the only one missing, so it's installed separately. It only adds packages and changes none of the existing ones.

`pip check` will keep reporting `laion-fmri 0.1.0 has requirement numpy>=1.24`. That's expected.

Downloads run `python -m awscli s3 sync` (anonymous, us-west-2). With `http(s)_proxy` set to the university proxy (e.g. by `~/startup_conda.sh`), parallel S3 downloads can fail with `Failed to connect to proxy URL`. If the node has direct internet access, unset the proxy variables before downloading:

```bash
unset http_proxy https_proxy
laion-fmri config --help    # set the data directory
laion-fmri download --help
```

### Paths

Data locations (NSD, …) are set in `jsputils/jsputils/paths.py`. Six environment variables control the rest:

| variable | default | purpose |
|---|---|---|
| `DNFFA_OUTPUT_DIR` | `revision/outputs` | root for `analysis_outputs/` and `figure_outputs/`. Point it at `PROJECT_DNFFA/NOTEBOOKS` to reuse results the notebooks already cached. |
| `DNFFA_GSN_DIR` | `/home/jovyan/work/DropboxSandbox/GSN` | checkout of [GSN](https://github.com/cvnlab/GSN), needed for noise ceilings |
| `DNFFA_DATA_DIR` | `revision/data` | stimulus sets (`vpnl-floc`, `classic-categ`), one folder per set. `dnffa.config` points jsputils' `image_set_dir()` here; `selective_unit_dir()` goes to `analysis_outputs/selective_units`. |
| `DNFFA_IMAGENET_VAL_DIR` | `/share/klab/datasets/imagenet/val` | raw ImageNet val images (one folder per wnid), input to `make_imagenet_ffcv.py` |
| `DNFFA_IMAGENET_FFCV` | `revision/data/imagenet1k-ffcv/imagenet1k_val_..._includes_index.ffcv` | ImageNet val set in FFCV format, used by 02. `dnffa.config` points jsputils' `ffcv_imagenet1k_valset()` here. |
| `DNFFA_WEIGHTS_DIR` | `weights` (repo root) | VGGFace AlexNet and ImageNet readout checkpoints. `dnffa.config` points jsputils' `weight_savedir()` and `training_checkpoint_dir()` here. |

### Model weights

| model | source |
|---|---|
| `alexnet-barlow-twins`, `alexnet-supervised`, `alexnet-ipcl` | public; downloaded by `scripts/download_models.py` |
| `alexnet-barlow-twins-random` | random initialization, no file |
| `alexnet-vggface` | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5848EQ) → `weights/alexnet_faces_final.pth.tar` |
| readout used by 02 | Dataverse → `weights/mdl-alexnet-barlow-twins_from-relu7_..._sparse-pos-True_l1p-1e-05_l1n-1e-05/checkpoint.pth` |

`weights/` is git-ignored. Download the two Dataverse files from the repo root:

```bash
READOUT=mdl-alexnet-barlow-twins_from-relu7_mlr-0.05_ilr-0.001_eps-10_sparse-pos-True_l1p-1e-05_l1n-1e-05
curl -fL --create-dirs -o weights/alexnet_faces_final.pth.tar \
     https://dataverse.harvard.edu/api/access/datafile/10261439
# file 10261445 is Dataverse's checkpoint-1.pth; the checkpoint.pth next to it on Dataverse is empty
curl -fL --create-dirs -o "weights/$READOUT/checkpoint.pth" \
     https://dataverse.harvard.edu/api/access/datafile/10261445
md5sum weights/alexnet_faces_final.pth.tar "weights/$READOUT/checkpoint.pth"
# 6d4181cfff8300844a65c4daa7306359  weights/alexnet_faces_final.pth.tar
# 836b80b9c8b606a1c00bca08299e7764  weights/mdl-.../checkpoint.pth
```

For the three public models, run the download script once on a machine with internet access. Weights go to `$TORCH_HOME/hub` (default `~/.cache/torch/hub`):

```bash
python revision/scripts/download_models.py
```

## Running

Every script has `--help`. Run them from any directory. Steps that take a long time cache their results and skip work that is already done; pass `--overwrite` to recompute.

```bash
python revision/scripts/00_train_readout.py --sparse-pos     # readout used by 02
python revision/scripts/01_dnn_localizer.py
python revision/scripts/make_imagenet_ffcv.py                 # ImageNet val -> FFCV, needed by 02
python revision/scripts/02_lesioning.py

# encoding models (outputs feed 04)
python revision/scripts/03_encoding.py                                    # alexnet-barlow-twins, lasso
python revision/scripts/03_encoding.py --model alexnet-barlow-twins-random --alpha 0.001
python revision/scripts/03_encoding.py --model alexnet-vggface
python revision/scripts/03_encoding.py --method ols --rois FFA-1 PPA EBA VWFA-1 \
    --savedir "$DNFFA_OUTPUT_DIR/analysis_outputs/3-Encoding/alexnet-barlow-twins-ols"
python revision/scripts/07_low_level_models.py export
python revision/scripts/07_low_level_models.py features   # Python port, ~1.5 s/image
#   or the original MATLAB code (same output files):
#   python revision/scripts/07_low_level_models.py features --backend matlab --matlab-cmd /path/to/matlab
python revision/scripts/07_low_level_models.py encode
python revision/scripts/06_noise_ceilings.py

python revision/scripts/04_encoding_plots.py
python revision/scripts/05_content_channeling.py
```

## LAION-fMRI

`dnffa/laion.py` loads LAION-fMRI (via the `laion_fmri` package) into objects with the same attributes as NSD's `BrainRegion`, so `jsputils`' `EncodingProcedure` and the noise-ceiling code work on it unchanged:

```python
from dnffa import laion
ROI = laion.load_roi(laion.load_subject('sub-01'), 'FFA-1')   # cf. nsd.load_roi(nsd.load_subject('subj01'), 'FFA-1')
encoder = classes.EncodingProcedure(ROI, DNN, method='lasso', positive=True, alphas=[0.1])
```

`python -m dnffa.laion` (run from `revision/`) prints the sessions, image sets and ROIs without loading betas.

| | NSD (`dnffa.nsd`) | LAION-fMRI (`dnffa.laion`) |
|---|---|---|
| subjects | `subj01`–`subj08` | `sub-01`, `sub-03`, `sub-05`, `sub-06`, `sub-07` |
| ROIs | 11, incl. `FBA-1`, `FBA-2`, `OWFA` | 9: a single `FBA`, no `OWFA` |
| train / val | `nonshared1000-3rep-batch0/1` | `unique1000-4rep-batch0/1`: same selection rule, 1000 unique images × 4 reps each |
| test | `special515`: 515 images × 3 reps | `shared-12rep`: 866 images seen 12× by every subject |
| voxel filter | ncsnr > 0.3 | ncsnr > 0.3, with ncsnr recovered from the `Noiseceiling4rep` map |
| betas | z-scored per voxel within session | same |

Notes:
- **Speed:** the first `load_subject(...).load_betas()` reads all 30 sessions once (about 30 s each). It caches z-scored betas for the union of the 9 ROIs (~665 MB per subject) in `analysis_outputs/laion-fmri`; later calls load that cache.
- **Volumetric data:** voxels are assigned to `lh`/`rh` by the sign of their world x coordinate. `EncodingProcedure` concatenates the two, so this does not affect results.
- **Failed fits:** voxels with failed GLMsingle fits (NaN betas) are dropped from the ROI. The two sessions checked for sub-01 had none.

## Differences from the notebooks

The scripts are meant to reproduce the notebooks exactly: same parameters, same order of computations, same random-number usage, same output filenames. The only change that affects results is:

- 3: `roi_list` listed `'VWFA-1'` twice and left out `'VWFA-2'`. This is treated as a typo, and the script uses the same 11 ROIs as notebooks 4, 6 and 7.

Fixes so the code runs against the current `jsputils`, with no effect on results:

- `BrainRegion.load_encoding_data` now takes a `{'train', 'val', 'test'}` partition dict. The notebooks still passed three positional arguments.
- 3/7: the unused `gpu_encoding` import is removed because that module no longer exists.
- 6: the notebook used `DNN` without ever defining it, only to call `get_encoding_voxels`. `nsd.roi_voxels` does the same computation directly.
- 7: `nsdorg.get_NSD_train_test_images` no longer exists, and `scipy.io` was never imported. `nsd.load_nsd_images` exports the images in the same COCO order that `load_encoding_data` uses for the brain data. The old function is gone, so I couldn't check its image order.
- 7: the Gabor / GIST-PC features are computed by the `features` stage instead of running `DNFFA_extract_gist.m` by hand. By default it uses a Python port (`dnffa/gist.py`); `--backend matlab` calls the original `computeGaborAndGistFeatures.m` with the same parameters. On test images the port matches the original `.m` code run in Octave to ~1e-14, except that some GistPC columns have flipped signs (PC signs are arbitrary). The original quirks are kept on purpose: the GIST PCA is fit separately on each image set, so train and test PCs are not aligned, and it is not mean-centered. Because of that, GistPC test scores depend on which arbitrary sign each PC gets, so they can differ from the paper's numbers. Gabor features are unaffected.

Changes with no effect on results:

- 2: probe images are collected while the data streams in, instead of first loading all 50k validation images (about 30 GB). The selected images are identical. The loader is still read to the end, and 6 images per category are still saved: as in the notebook, the 6th is the first image of the next class.
- 1: `--skip-classic-categ` skips the parts that need the `classic-categ` images (the probe-set heatmap and the classic-categ panels of the trained vs. untrained t-value plot). Selective units are always defined on `vpnl-floc`, so the pies and per-layer summaries are unaffected.
- 2: the randomized control uses numpy's global random state, which comes from the `np.random.seed(0)` inside `scatter_corr`. `main()` therefore keeps the notebook's cell order. `--skip-examples` and `--skip-randomized` are optional shortcuts that leave the notebook path.
- Figures the notebooks only showed inline are now saved too: readout weights, full layer summaries, the 2D trajectory plot, and the noise-ceiling plot. Notebook 6 drew the noise-ceiling plot twice in slightly different styles; only the second version is kept.
- Figures are drawn with the non-interactive Agg backend. With matplotlib-inline ≥ 0.1.4 (the environment pins 0.1.6), the inline backend does not override rcParams either, so saved figures are the same.
