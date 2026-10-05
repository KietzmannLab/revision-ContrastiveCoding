"""Paths and experiment constants shared across all analysis scripts."""

import os
from pathlib import Path

from jsputils import paths as jsputils_paths

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REVISION_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = REVISION_DIR.parent

# Root for everything the scripts write. The notebooks wrote to
# ``PROJECT_DNFFA/NOTEBOOKS/{analysis,figure}_outputs``; point
# DNFFA_OUTPUT_DIR there to reuse previously cached results.
OUTPUT_DIR = Path(os.environ.get('DNFFA_OUTPUT_DIR', REVISION_DIR / 'outputs'))
ANALYSIS_DIR = OUTPUT_DIR / 'analysis_outputs'
FIGURE_DIR = OUTPUT_DIR / 'figure_outputs'

# Location of the GSN package (https://github.com/cvnlab/GSN), used for noise ceilings.
GSN_DIR = os.environ.get('DNFFA_GSN_DIR', '/home/jovyan/work/DropboxSandbox/GSN')

IMAGENET_CLASS_LABELS = REPO_DIR / 'PROJECT_DNFFA' / 'NOTEBOOKS' / 'imagenet_class_labels.json'

# Weights that are not fetched automatically: the VGGFace AlexNet and the ImageNet
# readout checkpoints (``<readout description>/checkpoint.pth``). jsputils looks for
# them in ``paths.weight_savedir()`` / ``paths.training_checkpoint_dir()``; both are
# redirected here so the jsputils submodule can stay unmodified.
WEIGHTS_DIR = Path(os.environ.get('DNFFA_WEIGHTS_DIR', REPO_DIR / 'weights'))


def _weights_dir():
    return str(WEIGHTS_DIR)


jsputils_paths.weight_savedir = _weights_dir
jsputils_paths.training_checkpoint_dir = _weights_dir


def analysis_dir(name):
    path = ANALYSIS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(name):
    path = FIGURE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# Sub-directory names (kept identical to the original notebook outputs)
ENCODING_SUBDIR = '3-Encoding'
NOISE_CEILING_SUBDIR = '3c-NoiseCeilings'
LOW_LEVEL_SUBDIR = '3d-LowLevel'
LESIONING_SUBDIR = '2-Lesioning'

# ---------------------------------------------------------------------------
# Models and image sets
# ---------------------------------------------------------------------------

MODEL_NAME = 'alexnet-barlow-twins'
FLOC_IMAGESET = 'vpnl-floc'
FDR_P = 0.05

# ---------------------------------------------------------------------------
# Category domains
# ---------------------------------------------------------------------------

DOMAIN_COLORS = {
    'faces': 'tomato',
    'bodies': 'dodgerblue',
    'objects': 'orange',
    'scenes': 'limegreen',
    'characters': 'purple',
    'scrambled': 'navy',
    'layer': 'tomato',
}

# In figures the 'characters' domain is labelled 'words'
DOMAIN_LABELS = {'characters': 'words'}


def domain_label(domain):
    return DOMAIN_LABELS.get(domain, domain)


# ---------------------------------------------------------------------------
# NSD / fMRI
# ---------------------------------------------------------------------------

SUBJECTS = [f'subj0{s}' for s in range(1, 9)]

ROI_LIST = ['FFA-1', 'FFA-2', 'OFA',
            'PPA', 'OPA',
            'EBA', 'FBA-1', 'FBA-2',
            'VWFA-1', 'VWFA-2', 'OWFA']

ROI_DOMAIN = {'OFA': 'faces', 'FFA-1': 'faces', 'FFA-2': 'faces',
              'OPA': 'scenes', 'PPA': 'scenes',
              'EBA': 'bodies', 'FBA-1': 'bodies', 'FBA-2': 'bodies',
              'OWFA': 'characters', 'VWFA-1': 'characters', 'VWFA-2': 'characters'}

ROI_GROUPS = {'Face-selective ROIs': {'ROIs': ['OFA', 'FFA-1', 'FFA-2'], 'domain': 'faces'},
              'Body-selective ROIs': {'ROIs': ['FBA-1', 'FBA-2', 'EBA'], 'domain': 'bodies'},
              'Scene-selective ROIs': {'ROIs': ['OPA', 'PPA'], 'domain': 'scenes'},
              'Word-selective ROIs': {'ROIs': ['OWFA', 'VWFA-1', 'VWFA-2'], 'domain': 'characters'}}

TRAIN_IMAGESET = 'nonshared1000-3rep-batch0'
VAL_IMAGESET = 'nonshared1000-3rep-batch1'
TEST_IMAGESET = 'special515'

SPACE = 'nativesurface'
BETA_VERSION = 'betas_fithrf_GLMdenoise_RR'
NCSNR_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Layers used for encoding
# ---------------------------------------------------------------------------

ENCODING_LAYERS = {
    'alexnet-barlow-twins': ['conv3', 'groupnorm3', 'relu3',
                             'conv4', 'groupnorm4', 'relu4',
                             'conv5', 'groupnorm5', 'relu5', 'maxpool5',
                             'fc6', 'batchnorm6', 'relu6',
                             'fc7', 'batchnorm7', 'relu7',
                             'fc8', 'batchnorm8'],
    'alexnet-vggface': ['conv3', 'relu3',
                        'conv4', 'relu4',
                        'conv5', 'relu5', 'maxpool5',
                        'fc6', 'relu6',
                        'fc7', 'relu7',
                        'fc8'],
}


def encoding_layers(model_name):
    """Layer list for a model; variants (e.g. '-random') share their base model's layers."""
    for key, layers in ENCODING_LAYERS.items():
        if key in model_name:
            return list(layers)
    raise ValueError(f'no encoding layer list defined for {model_name}')
