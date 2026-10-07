"""fMRI datasets the encoding scripts (03, 04, 06, 07) can run on, selected with ``--dataset``.

``nsd`` (default) is the paper's setup. ``laion`` runs the same analyses on LAION-fMRI
through ``dnffa.laion``. LAION outputs go to sibling folders with a ``-laion`` suffix
(e.g. ``analysis_outputs/3-Encoding-laion``, ``figure_outputs/Figure4-Encoding-laion``),
so results from the two datasets never mix.
"""

from dataclasses import dataclass
from typing import Optional

from . import config, laion, nsd

NAMES = ['nsd', 'laion']


@dataclass(frozen=True)
class Dataset:
    name: str
    subjects: list
    rois: list
    roi_domain: dict
    roi_groups: dict
    train_imageset: str
    test_imageset: str
    image_size: Optional[int] = None  # resize stimuli to this size (None: keep as stored)

    def subdir(self, name):
        return name if self.name == 'nsd' else f'{name}-{self.name}'

    def analysis_dir(self, name):
        return config.analysis_dir(self.subdir(name))

    def figure_dir(self, name):
        return config.figure_dir(self.subdir(name))

    def load_subject(self, subj):
        return nsd.load_subject(subj) if self.name == 'nsd' else laion.load_subject(subj)

    def load_roi(self, subject, roi, partitions=None):
        """ROI with ncsnr mask and train/val/test encoding data (``nsd.load_roi`` / ``laion.load_roi``)."""
        if self.name == 'nsd':
            return nsd.load_roi(subject, roi, partitions=partitions)
        return laion.load_roi(subject, roi, partitions=partitions, image_size=self.image_size)

    def load_images(self, subj, partitions):
        """Stimuli keyed by partition, in the same order as the ROI's brain data."""
        if self.name == 'nsd':
            return nsd.load_nsd_images(subj, partitions)
        return laion.load_images(subj, partitions, image_size=self.image_size)

    def noise_ceiling_path(self, roi, subj):
        return nsd.noise_ceiling_path(roi, subj, test_imageset=self.test_imageset,
                                      subdir=self.subdir(config.NOISE_CEILING_SUBDIR))

    def load_noise_ceilings(self, rois=None, subjs=None):
        return nsd.load_noise_ceilings(rois or self.rois, subjs or self.subjects, test_imageset=self.test_imageset,
                                       subdir=self.subdir(config.NOISE_CEILING_SUBDIR))


def get(name):
    if name == 'nsd':
        return Dataset('nsd', config.SUBJECTS, config.ROI_LIST, config.ROI_DOMAIN, config.ROI_GROUPS,
                       config.TRAIN_IMAGESET, config.TEST_IMAGESET)
    if name == 'laion':
        return Dataset('laion', laion.SUBJECTS, laion.ROI_LIST, laion.ROI_DOMAIN, laion.ROI_GROUPS,
                       laion.TRAIN_IMAGESET, laion.TEST_IMAGESET, image_size=laion.ENCODING_IMAGE_SIZE)
    raise ValueError(f'unknown dataset: {name}')


def add_arguments(parser, rois_and_subjects=True):
    """--dataset, plus --rois / --subjects that default to all of the dataset's ROIs / subjects."""
    parser.add_argument('--dataset', choices=NAMES, default='nsd',
                        help='fMRI dataset (default: nsd); laion outputs go to folders with a -laion suffix')
    if rois_and_subjects:
        parser.add_argument('--rois', nargs='+', default=None, help="default: all of the dataset's ROIs")
        parser.add_argument('--subjects', nargs='+', default=None, help="default: all of the dataset's subjects")


def from_args(parser, args):
    """Resolve the dataset and fill in / check ``args.rois`` and ``args.subjects`` (if present)."""
    ds = get(args.dataset)
    if not hasattr(args, 'rois'):
        return ds
    args.rois = args.rois or list(ds.rois)
    args.subjects = args.subjects or list(ds.subjects)
    for kind, given, valid in [('ROI', args.rois, ds.rois), ('subject', args.subjects, ds.subjects)]:
        unknown = [x for x in given if x not in valid]
        if unknown:
            parser.error(f'unknown {ds.name} {kind}(s) {unknown}; choose from {list(valid)}')
    return ds
