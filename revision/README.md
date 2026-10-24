# DNFFA analyses as Python scripts

Script versions of the notebooks in `PROJECT_DNFFA/NOTEBOOKS/`. The analysis logic is unchanged; the heavy lifting is still done by the `jsputils` submodule.

```
revision/
├── dnffa/                     # shared code
│   ├── config.py              # output paths, subjects, ROIs, domain colours, layer lists, NSD settings
│   ├── nsd.py                 # load ROIs/betas, NSD images, GSN noise ceilings
│   ├── laion.py               # LAION-fMRI loader with the same ROI interface as nsd.py
│   ├── datasets.py            # --dataset {nsd,laion} for scripts 03, 04, 06, 07
│   ├── gist.py                # Python port of the MATLAB Gabor / GIST-PC features (HELPERS/Code-GistModel)
│   ├── plotting.py            # despine/layer-axis styling, scatter_corr, noise-ceiling band
│   └── stats.py               # finite-masked Pearson r, paired t-tests vs. first model, per-image swap permutation tests
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
    ├── 07_low_level_models.py    # 7-LowLevelModels.ipynb
    ├── 08_encoding_permutation.py  # single-subject permutation tests of the Fig. 4 comparisons (new)
    └── 09_generalization.py      # Fig. 4 on the re:vision generalization splits of LAION-fMRI (new)
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

#### GPU env (`revision-cu128`)

PyTorch 1.13 has no kernels for the klab GPUs (H100 sm_90, B200 sm_100, RTX PRO 6000 sm_120). It only runs on them through PTX JIT compilation, which takes long enough at startup that the lesioning script barely gets going. `environment/env_build_cu128.sh` builds a second env with the same analysis stack on PyTorch 2.7.1 / CUDA 12.8:

```bash
bash environment/env_build_cu128.sh    # -> ~/.conda/envs/revision-cu128
srun -p klab-gpu --gres=gpu:1 -c 16 --mem=50G ~/.conda/envs/revision-cu128/bin/python revision/scripts/02_lesioning.py
```

Differences from the pinned env: Python 3.10, numpy 1.26.4, opencv 4 (ffcv 1.0.2 doesn't find opencv 5), cupy 13 (cupy 14 needs numpy 2), latest pycortex. `dnffa.config` handles two behaviour changes so jsputils stays unmodified: it restores `torch.load`'s old `weights_only=False` default, and it caps the ImageNet loader's 64 threads at the CPUs the job has (newer numba refuses more). Tested with `02_lesioning.py`: one ImageNet val pass takes ~10 s on an RTX PRO 6000.

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
| `DNFFA_GSN_DIR` | `GSN` (repo root) | checkout of [GSN](https://github.com/cvnlab/GSN), needed for noise ceilings. GSN is imported from the checkout, no install needed; see [Noise ceilings (GSN)](#noise-ceilings-gsn) for setup. |
| `DNFFA_DATA_DIR` | `revision/data` | stimulus sets (`vpnl-floc`, `classic-categ`), one folder per set. `dnffa.config` points jsputils' `image_set_dir()` here; `selective_unit_dir()` goes to `analysis_outputs/selective_units`. |
| `DNFFA_IMAGENET_VAL_DIR` | `/share/klab/datasets/imagenet/val` | raw ImageNet val images (one folder per wnid), input to `make_imagenet_ffcv.py` |
| `DNFFA_IMAGENET_FFCV` | `revision/data/imagenet1k-ffcv/imagenet1k_val_..._includes_index.ffcv` | ImageNet val set in FFCV format, used by 02. `dnffa.config` points jsputils' `ffcv_imagenet1k_valset()` here. |
| `DNFFA_WEIGHTS_DIR` | `weights` (repo root) | VGGFace AlexNet and ImageNet readout checkpoints. `dnffa.config` points jsputils' `weight_savedir()` and `training_checkpoint_dir()` here. |

### Noise ceilings (GSN)

06 uses `rsa_noise_ceiling` from [GSN](https://github.com/cvnlab/GSN). The notebooks used an unreleased GSN version whose `rsa_noise_ceiling` took a list of `rdmfuns`; the public one takes a single `opt['rdmfun']`, which 06 handles by calling it once per function. The public Python port also has bugs in its split-half simulations (float trial counts used as slice indices, `np.iqr`, in-place `ndarray.resize`, 1-based loop counters, and a "random" permutation that returns the same split every time). `revision/patches/gsn-rsa-noise-ceiling-random-splits.patch` fixes them. Set it up once from the repo root:

```bash
git clone https://github.com/cvnlab/GSN GSN       # git-ignored; the default DNFFA_GSN_DIR
git -C GSN checkout f33935e                       # commit the patch was made against
git -C GSN apply ../revision/patches/gsn-rsa-noise-ceiling-random-splits.patch
```

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

### Running the encoding analyses (Fig. 4) on LAION-fMRI

Scripts 03, 04, 06, 07 and 08 take `--dataset {nsd,laion}` (default `nsd`, which is unchanged). With `--dataset laion` they use the subjects, ROIs and image sets in the table below, and read/write sibling folders with a `-laion` suffix (`analysis_outputs/3-Encoding-laion`, `3c-NoiseCeilings-laion`, `3d-LowLevel-laion`, `figure_outputs/Figure4-Encoding-laion`, `NoiseCeilings-laion`), so NSD and LAION results never mix. `--rois` / `--subjects` default to all of the dataset's ROIs / subjects; NSD-only names such as `FBA-1` are rejected.

03 extracts DNN features on `cuda:0`, so run it on a GPU node. The other steps run on CPU.

```bash
GPU="srun -p klab-gpu --gres=gpu:1 -c 16 --mem=64G ~/.conda/envs/revision-cu128/bin/python"

# 03: encoding models (the first run also builds the beta and image caches)
$GPU revision/scripts/03_encoding.py --dataset laion
$GPU revision/scripts/03_encoding.py --dataset laion --model alexnet-barlow-twins-random --alpha 0.001
$GPU revision/scripts/03_encoding.py --dataset laion --model alexnet-vggface
$GPU revision/scripts/03_encoding.py --dataset laion --method ols --rois FFA-1 PPA EBA VWFA-1 \
    --savedir "${DNFFA_OUTPUT_DIR:-revision/outputs}/analysis_outputs/3-Encoding-laion/alexnet-barlow-twins-ols"

# 07: Gabor / GIST-PC baselines
python revision/scripts/07_low_level_models.py export   --dataset laion
python revision/scripts/07_low_level_models.py features --dataset laion
python revision/scripts/07_low_level_models.py encode   --dataset laion

# 06: GSN noise ceilings on the shared-12rep test set
python revision/scripts/06_noise_ceilings.py --dataset laion

# 04: Figure 4 (add --skip-ols if the OLS run above was skipped)
python revision/scripts/04_encoding_plots.py --dataset laion

# 08: single-subject permutation tests of the Fig. 4 comparisons
$GPU revision/scripts/08_encoding_permutation.py predict --dataset laion
srun -p klab-cpu -c 16 --mem=32G env OPENBLAS_NUM_THREADS=16 \
    ~/.conda/envs/revision/bin/python revision/scripts/08_encoding_permutation.py test --dataset laion
```

#### Single-subject statistics (08)

Fig. 4 tests contrastive vs. each alternative model with paired t-tests across subjects. With LAION-fMRI's 5 subjects, 08 tests each subject on its own instead:

- **Models:** the fits 04 plots, i.e. each model's best layer on the validation set, with the ROI's preferred-domain units for the Barlow Twins models. `predict` refits them (same data, mapper and seed as 03 / 07; each refit is checked against the 03/07 CSV score). The untrained model only matches if 03 was run with the seeded init (see [Differences from the notebooks](#differences-from-the-notebooks)); results from an older 03 run give refit warnings for it, and 03 has to be rerun for that model and caches the test-set predictions in `analysis_outputs/3e-Permutation-laion/predictions`. Only `predict` needs the GPU.
- **Null:** for each subject, ROI, metric and alternative model, the two models' predictions are swapped independently per test image with probability 0.5, and both models' correlations with the brain data are recomputed (for veRSA: both predicted RDMs). 10,000 permutations by default (`--n-perm`).
- **p-value:** one-sided, the share of permuted differences r(contrastive) − r(alternative) at least as large as the observed one, (1 + #{null ≥ observed}) / (1 + n_perm).
- **Correction:** Bonferroni with 04's count (30 tests for LAION), within each subject and metric.
- **Outputs:** `permutation-tests.csv` (one row per test) and `permutation-nulls.npz` in the analysis folder. In `figure_outputs/Figure4-Permutation-laion`: `permutation-best-layers-{metric}-paper-style.tiff` (04's paper-style Fig. 4, with each bracket labelled with the number of subjects where contrastive > that model, and filled / open dots for significant / non-significant subjects), `permutation-best-layers-{metric}-per-subject.tiff` (the same layout with one row per subject, that subject's noise ceiling and a bracket for each significant test), `permutation-differences-{metric}.tiff` (every subject's observed difference per ROI and model, with the number of significant subjects) and `permutation-nulls-{metric}-{ROI}.tiff` (null histograms for the example ROIs).
- **Parallel runs:** the slow step is rerunning 03 for the untrained model (~3 h in one job). Thanks to the fixed seed, parallel jobs now fit the same network, so 03 can be split by subject or ROI. 03 extracts features per subject × ROI anyway, so splitting costs nothing extra. `08 predict` takes ~25 min for the DNNs and ~10 min for GistPC / Gabor (CPU only). It can be split with `--models` / `--subjects` / `--rois`; the jobs write separate files, and the shared test-response files are written atomically. Run `08 test` as one job once all predictions exist: it writes a single results file, and its matrix products already use all CPUs of the job (~10 min with 16).

  ```bash
  SRUN_GPU="srun -p klab-gpu --gres=gpu:1 -c 16 --mem=64G"
  PY_GPU=~/.conda/envs/revision-cu128/bin/python
  for subj in sub-01 sub-03 sub-05 sub-06 sub-07; do
      $SRUN_GPU $PY_GPU revision/scripts/03_encoding.py --dataset laion --subjects $subj \
          --model alexnet-barlow-twins-random --alpha 0.001 --overwrite &
  done; wait
  python revision/scripts/04_encoding_plots.py --dataset laion

  for model in alexnet-barlow-twins alexnet-barlow-twins-random alexnet-vggface; do
      $SRUN_GPU $PY_GPU revision/scripts/08_encoding_permutation.py predict --dataset laion --models $model &
  done
  srun -p klab-cpu -c 4 --mem=32G ~/.conda/envs/revision/bin/python \
      revision/scripts/08_encoding_permutation.py predict --dataset laion --models GistPC Gabor &
  wait
  srun -p klab-cpu -c 16 --mem=32G env OPENBLAS_NUM_THREADS=16 ~/.conda/envs/revision/bin/python \
      revision/scripts/08_encoding_permutation.py test --dataset laion
  ```
- **Speed:** a veRSA test (866 test images, 10k permutations) takes ~5 s on one core. Rebuilding each permuted RDM would take hours, so the sums in the Pearson r are written as quadratic forms in the swap mask instead. They give the same values as `pdist` on the swapped predictions to ~1e-14. The test stage is mostly matrix products: on the shared login node, OpenBLAS's default of one thread per core made them ~100× slower, so run it as a job with `OPENBLAS_NUM_THREADS` set to the job's CPU count.

For one subject × ROI, 03 (one model, all layers and domains) took 4 min on an H100/RTX node, including building that subject's image cache; the 45 combinations take up to ~3 h per model. The three model runs can go in parallel; split further with `--subjects` / `--rois`.

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
- **Images:** the stimuli are 1000 × 1000. The scripts resize them to 425 × 425 (`laion.ENCODING_IMAGE_SIZE`, the NSD size): the DNNs resize to 224 anyway, and the Gabor / GIST filter bank is built for the image size. Loading takes ~50 ms per image, so resized image sets are cached next to the betas (`{sub-XX|shared}_{image set}_images-425.npy`, ~540 MB per 1000 images, ~6 GB for all subjects). Cache files are written atomically, so parallel jobs can share them.
- **Fig. 4 statistics:** the Bonferroni correction counts one test per ROI and non-reference model: 36 for NSD as in the paper, 30 for LAION's 9 ROIs (3 face ROIs × 4 + 6 others × 3).
- **Volumetric data:** voxels are assigned to `lh`/`rh` by the sign of their world x coordinate. `EncodingProcedure` concatenates the two, so this does not affect results.
- **Failed fits:** voxels with failed GLMsingle fits (NaN betas) are dropped from the ROI. The two sessions checked for sub-01 had none.

### Generalization (09)

09 tests whether the Fig. 4 result holds beyond the replication's image sets. It uses the [re:vision generalization methods](https://re-vision-initiative.org/generalization) with the predefined splits of `laion_fmri.splits` ([docs](https://laion-fmri.hebartlab.com/train_test_splits.html)). The Fig. 4 models are refit on each split's training images and tested on its held-out images:

| method | splits | test set | reported score |
|---|---|---|---|
| 1: within-distribution | `tau` | ~1,170 images far from their nearest training images | test score |
| 2: OOD clusters | `cluster_k5_0` … `cluster_k5_4` | one held-out CLIP k-means cluster (615–1,506 images) | mean over the 5 folds |
| 3: OOD images | `ood` | the 64 `unusual` OOD images (`--ood-types`; `all` = all 371) | test score |
| baseline | `random_0` … `random_4` | a random fifth of the pool | mean over the 5 folds |

Each subject's own pool is used: its 1,121 shared + 4,712 unique regular images, plus the 371 shared OOD images. 09 only runs on LAION-fMRI. Changes from 03 / 07, all forced by the splits:

- **Layer selection:** in the replication, each model's layer is the best one on a separate 1,000-image validation set. The splits have no validation set, so 09 holds out a seeded 20% of each split's training images (~930 images). The models are fit on the other 80%, the layer is selected on the held-out 20%, and that fit is scored on the test set. Test images never influence which layer is chosen. The validation images depend only on subject and split, so all models and ROIs share them.
- **Repetitions:** the pools mix images shown 4× and 12× (plus a few with missing trials). Every image is represented by its first 4 trials (`laion.SPLIT_NREPS`), so training and test images have the same noise level. Images with fewer than 4 trials (≤ 5 per subject) are dropped. This also means the noise ceilings are for 4-trial averages, lower than the replication's 12-trial ones.
- **GistPC:** the PCA is fit on the training images and applied to the test images. The original code fits a separate PCA on every image set (see [Differences from the notebooks](#differences-from-the-notebooks)). That can't work for the 64-image OOD test set, and it leaves train and test PCs unaligned. GistPC is therefore a somewhat stronger baseline here than in the replication.
- **Gabor:** unchanged from 07: OLS on 7,680 features. The minimum-norm OLS solution predicts worst when the number of training images approaches the number of features. With ~3,700 training images instead of 1,000 it is much closer to that point, so Gabor scores drop (sub-01 FFA-1: r = 0.17 in the replication, 0.07 on `tau`). This makes Gabor a weaker baseline here than in the replication.
- **Units:** the Barlow Twins models are fit only on the ROI's preferred-domain units (the only fits Fig. 4 shows); the face-trained model uses whole layers. Mappers and penalties are those of the 03 / 07 runs above: lasso α = 0.1 for the trained models, 0.001 for the untrained one, OLS for GistPC / Gabor.

Statistics follow 04 and 08. The per-subject permutation tests swap the two models' predictions per test image. For the fold-based methods, the statistic is the fold-averaged difference r(contrastive) − r(alternative), and its null is the average of independent per-fold nulls (fold k uses seed + k). The Bonferroni count is the same as for Fig. 4 (30 tests). Paired t-tests across the 5 subjects are run as in 04.

`scripts/run_generalization.sh` submits the whole pipeline as `sbatch` jobs (logs in `logs/generalization/`). At most 3 CPU jobs and 1 GPU job run at a time. Each CPU job uses 64 cores (`CPUS`), and the jobs are chained into three lanes, so each lane runs one job at a time:

```
GPU:    features (~30 min) ··························· (the fits wait for it)
lane A: fit face-trained sub-01 -> sub-03 -> sub-06 ───────────┐
lane B: lowlevel -> fit other models -> fit face-trained sub-05 ┼─> test
lane C: noise ceilings -> fit face-trained sub-07 ──────────────┘
```

```bash
bash revision/scripts/run_generalization.sh
# settings via the environment, e.g.
CPUS=128 MEM=400G bash revision/scripts/run_generalization.sh
OOD_TYPES="unusual all" N_PERM=10000 bash revision/scripts/run_generalization.sh
```

Times measured with 64 cores per job:

| job | time |
|---|---|
| `features` (GPU) | 9 min for all subjects and models |
| `lowlevel` | 1.7 h for all subjects |
| `noise-ceilings` | 12.6 h for all subjects (one process per subject) |
| `fit`, other four models | ~40 min per subject (estimated from a 16-core run) |
| `fit`, face-trained | ~3 h per subject (estimated from a 16-core benchmark of its largest layer); its whole conv layers have up to 65k units |

A fresh run takes ~15 h because of the noise ceilings. Without them, lanes A and B take ~6 h. The `test` job adds ~1 h.

Jobs depend on each other with `afterany`, so a failed job does not block the rest. Every stage skips outputs that already exist, so after a failure, fix the cause and run the script again; the `test` job lists any results that are still missing. Stages can also be run by hand with `--subjects`, `--rois`, `--models` and `--splits`, e.g. `09_generalization.py fit --subjects sub-01 --models alexnet-vggface --splits tau ood`.

Outputs, in `analysis_outputs/3f-Generalization-laion`:

- `features/`: DNN features (one `.npy` per subject, model, layer and unit set; ~8 GB per subject for the face-trained model's whole layers) and the Gabor features.
- `results/`: one CSV per subject, ROI, split and model, with val and test scores for every layer.
- `predictions/`: test-set predictions of the layers selected on the validation images, plus the test responses.
- `noise-ceilings/`: one GSN noise ceiling per test set.
- `generalization-scores.csv`: per method, subject, ROI and model, including the replication's Fig. 4 scores from 03 / 07 if they exist. `generalization-scores-per-test-set.csv` has every fold.
- `group-ttests.csv`, and `permutation-tests-{method}.csv` with `permutation-nulls-{method}.npz`.

Figures go to `figure_outputs/Figure4-Generalization-laion`. `generalization-summary-{metric}.tiff` shows every model's mean ± SEM per ROI for the replication and each method, with the noise-ceiling range. Each method's subfolder has 08's figures for that method: paper style, per subject, and differences.

`fit` runs all fits of a model (layers × ROIs × splits) in one pool of `--n-jobs` threads (default: the job's CPUs), largest first. Lasso releases the GIL, so the threads run in parallel. Each fit is single-threaded: OpenBLAS is limited to one thread for the whole process, and OpenMP to one thread in each worker thread. Without these limits, every worker starts a team with one thread per core: a 64-core job then ran ~4,000 threads and was ~10–30× slower. Each concurrent face-trained conv-layer fit needs ~2 GB of memory.

## Differences from the notebooks

The scripts are meant to reproduce the notebooks exactly: same parameters, same order of computations, same random-number usage, same output filenames. The only change that affects results is:

- 3: `roi_list` listed `'VWFA-1'` twice and left out `'VWFA-2'`. This is treated as a typo, and the script uses the same 11 ROIs as notebooks 4, 6 and 7.
- 3: the untrained model (`alexnet-barlow-twins-random`) is initialized after `torch.manual_seed(config.RANDOM_MODEL_SEED)`. jsputils does not seed it, so in the notebook every run (and every parallel job) drew a different untrained network, and 08 could not refit the same one. The seeded weights are identical in the pinned env and `revision-cu128`. The untrained model's selective units are still loaded from the cache written by 01, which was computed with 01's own unseeded draw.

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
- 5: the notebook only has Fig. 5A (MDS trajectories). 05 also draws Fig. 5B (`feature_space_relu6_PCs.png`, `--panels B`), rebuilt from the paper's Materials and Methods: 2D PCA of relu6 responses to the first 2 ImageNet val images per class (sorted file names, from `DNFFA_IMAGENET_VAL_DIR`), the 240 `mc8-shined` images projected into it, and arrows for the 25 face-, scene- and object-selective relu6 units with the highest localizer t-values (from the `vpnl-floc` cache written by 01). The paper scaled the arrows by a constant 2.5e4; with our activations that makes them ~300× longer than the spread of the images, so by default the factor is chosen so the longest arrow reaches the farthest image (`--arrow-scale 2.5e4` restores the paper's). The schematic dashed decision boundaries and the face masking of the published panel were added by hand and are not drawn.
- 4: the notebook only draws the Fig. 4 violins (all ROIs, Barlow Twins in `dodgerblue`, VGGFace second). The published Fig. 4 was restyled outside the notebook, so 04 also saves `encoding-best-layers-{metric}-paper-style.tiff` in that layout: the 4 example ROIs, mean bars with subject dots, the contrastive bar in the ROI's domain colour, and the face-trained model last. It uses the same data and t-tests as the violins.
- Figures are drawn with the non-interactive Agg backend. With matplotlib-inline ≥ 0.1.4 (the environment pins 0.1.6), the inline backend does not override rcParams either, so saved figures are the same.
