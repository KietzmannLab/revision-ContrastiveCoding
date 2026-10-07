#!/bin/bash
# Build the `revision-cu128` env: same analysis stack as `revision`, but with PyTorch 2.7 /
# CUDA 12.8, which has native kernels for the klab GPUs (H100 sm_90, B200 sm_100,
# RTX PRO 6000 sm_120). The pinned env (PyTorch 1.13 / CUDA 11.x) only runs on them via
# slow PTX JIT compilation.
#
# Run from the repo root on a node with internet access:
#   bash environment/env_build_cu128.sh
set -euo pipefail

ENV=${1:-$HOME/.conda/envs/revision-cu128}
MAMBA=${MAMBA:-$HOME/bin/micromamba}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}

# conda-forge: python + compiled deps. ffcv builds against opencv and libjpeg-turbo via
# pkg-config (ffcv 1.0.2 only knows opencv 4); libgl is listed so opencv finds libGL.so.1 on the compute nodes.
"$MAMBA" create -y -p "$ENV" -c conda-forge \
    python=3.10 numpy=1.26 scipy pandas scikit-learn matplotlib seaborn h5py nibabel \
    ipython tqdm gitpython pycocotools xarray psutil numba \
    opencv=4 libgl pkg-config libjpeg-turbo compilers setuptools wheel pip

PIP="$ENV/bin/python -m pip"

# Keep numpy 1.x (the original env used 1.23.5; ffcv 1.0.2 and the analysis code predate
# numpy 2). cupy >= 14 needs numpy 2, so cap it too. Applies to every pip call below.
PIP_CONSTRAINT=$(mktemp)
printf 'numpy==1.26.4\ncupy-cuda12x<14\n' > "$PIP_CONSTRAINT"
export PIP_CONSTRAINT

# PyTorch with CUDA 12.8 (bundles its own CUDA runtime libraries)
$PIP install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# ffcv compiles against the env's opencv/libjpeg-turbo, so build it with the env's tools
# (activate the env so pkg-config and the compilers are found)
eval "$("$MAMBA" shell hook -s bash)"
micromamba activate "$ENV"
$PIP install pkgconfig
$PIP install --no-build-isolation ffcv==1.0.2

$PIP install fastargs==1.2.0 fastprogress==1.0.3 torchmetrics wandb graphviz \
    torchlens==0.1.0 cupy-cuda12x pycortex awscli
$PIP install "laion-fmri @ git+https://github.com/ViCCo-Group/LAION-fMRI.git@main"

# jsputils (setup.py imports GitPython)
$PIP install --no-build-isolation -e jsputils

# check (cuda is True on a GPU node)
"$ENV/bin/python" -c "from jsputils import classes; import torch, ffcv, laion_fmri; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
