# DNFFA analyses as Python scripts

Script versions of the notebooks in `PROJECT_DNFFA/NOTEBOOKS/`. The analysis logic is unchanged; the heavy lifting is still done by the `jsputils` submodule.

```
revision/
├── dnffa/                     # shared code
│   ├── config.py              # output paths, subjects, ROIs, domain colours, layer lists, NSD settings
│   ├── nsd.py                 # load ROIs/betas, NSD images, GSN noise ceilings
│   ├── plotting.py            # despine/layer-axis styling, scatter_corr, noise-ceiling band
│   └── stats.py               # finite-masked Pearson r, paired t-tests vs. first model
└── scripts/
    ├── download_models.py        # pre-download public model weights (run once)
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

# check (cuda is True on a GPU node)
python -c "from jsputils import classes; import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Paths

Data locations (NSD, stimulus sets, …) are set in `jsputils/jsputils/paths.py`. Three environment variables control the rest:

| variable | default | purpose |
|---|---|---|
| `DNFFA_OUTPUT_DIR` | `revision/outputs` | root for `analysis_outputs/` and `figure_outputs/`. Point it at `PROJECT_DNFFA/NOTEBOOKS` to reuse results the notebooks already cached. |
| `DNFFA_GSN_DIR` | `/home/jovyan/work/DropboxSandbox/GSN` | checkout of [GSN](https://github.com/cvnlab/GSN), needed for noise ceilings |
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
python revision/scripts/02_lesioning.py

# encoding models (outputs feed 04)
python revision/scripts/03_encoding.py                                    # alexnet-barlow-twins, lasso
python revision/scripts/03_encoding.py --model alexnet-barlow-twins-random --alpha 0.001
python revision/scripts/03_encoding.py --model alexnet-vggface
python revision/scripts/03_encoding.py --method ols --rois FFA-1 PPA EBA VWFA-1 \
    --savedir "$DNFFA_OUTPUT_DIR/analysis_outputs/3-Encoding/alexnet-barlow-twins-ols"
python revision/scripts/07_low_level_models.py export
#   -> run PROJECT_DNFFA/HELPERS/DNFFA_extract_gist.m in MATLAB (first set `imdir` to analysis_outputs/3d-LowLevel/)
python revision/scripts/07_low_level_models.py encode
python revision/scripts/06_noise_ceilings.py

python revision/scripts/04_encoding_plots.py
python revision/scripts/05_content_channeling.py
```

## Differences from the notebooks

The scripts are meant to reproduce the notebooks exactly: same parameters, same order of computations, same random-number usage, same output filenames. The only change that affects results is:

- 3: `roi_list` listed `'VWFA-1'` twice and left out `'VWFA-2'`. This is treated as a typo, and the script uses the same 11 ROIs as notebooks 4, 6 and 7.

Fixes so the code runs against the current `jsputils`, with no effect on results:

- `BrainRegion.load_encoding_data` now takes a `{'train', 'val', 'test'}` partition dict. The notebooks still passed three positional arguments.
- 3/7: the unused `gpu_encoding` import is removed because that module no longer exists.
- 6: the notebook used `DNN` without ever defining it, only to call `get_encoding_voxels`. `nsd.roi_voxels` does the same computation directly.
- 7: `nsdorg.get_NSD_train_test_images` no longer exists, and `scipy.io` was never imported. `nsd.load_nsd_images` exports the images in the same COCO order that `load_encoding_data` uses for the brain data. The old function is gone, so I couldn't check its image order.

Changes with no effect on results:

- 2: probe images are collected while the data streams in, instead of first loading all 50k validation images (about 30 GB). The selected images are identical. The loader is still read to the end, and 6 images per category are still saved: as in the notebook, the 6th is the first image of the next class.
- 2: the randomized control uses numpy's global random state, which comes from the `np.random.seed(0)` inside `scatter_corr`. `main()` therefore keeps the notebook's cell order. `--skip-examples` and `--skip-randomized` are optional shortcuts that leave the notebook path.
- Figures the notebooks only showed inline are now saved too: readout weights, full layer summaries, the 2D trajectory plot, and the noise-ceiling plot. Notebook 6 drew the noise-ceiling plot twice in slightly different styles; only the second version is kept.
- Figures are drawn with the non-interactive Agg backend. With matplotlib-inline ≥ 0.1.4 (the environment pins 0.1.6), the inline backend does not override rcParams either, so saved figures are the same.
