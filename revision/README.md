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

Use the conda environment from `environment/environment.yml`, then install `jsputils`:

```bash
pip install -e jsputils
```

Data locations (NSD, stimulus sets, checkpoints, …) are set in `jsputils/jsputils/paths.py`. Two environment variables control the rest:

| variable | default | purpose |
|---|---|---|
| `DNFFA_OUTPUT_DIR` | `revision/outputs` | root for `analysis_outputs/` and `figure_outputs/`. Point it at `PROJECT_DNFFA/NOTEBOOKS` to reuse results the notebooks already cached. |
| `DNFFA_GSN_DIR` | `/home/jovyan/work/DropboxSandbox/GSN` | checkout of [GSN](https://github.com/cvnlab/GSN), needed for noise ceilings |

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
