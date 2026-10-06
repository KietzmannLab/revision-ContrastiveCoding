"""Write the ImageNet validation set to an FFCV file for the lesioning analysis (02).

Run once. jsputils reads ImageNet through ``DataLoaderFFCV('val')``, which expects the
file the original lab built (``imagenet1k_val_jpg_q100_s256_lmax512_crop_includes_index.ffcv``).
That file is not public, so this script rebuilds it from the raw val images.

Input:  ``config.IMAGENET_VAL_DIR`` (one folder per wnid, 50 images each)
Output: ``config.IMAGENET_FFCV_VALSET`` (set DNFFA_IMAGENET_FFCV to change it)

Fields, in order:
  image   RGBImageField, stored as JPEG (quality 100, longest side <= 512)
  label   IntField, class index from the sorted wnid folders (standard ImageNet order)
  index   IntField, position of the image in the file
  val_id  IntField, number in the original filename (ILSVRC2012_val_<val_id>.JPEG)

The loaders only decode ``image`` and ``label``, but both jsputils and 02_lesioning.py
unpack four values per batch, so the file needs the two extra fields.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffcv.fields import IntField, RGBImageField  # noqa: E402
from ffcv.writer import DatasetWriter  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from dnffa import config  # noqa: E402

N_CLASSES = 1000
N_VAL_IMAGES = 50000


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the sample index and the val image number."""

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        val_id = int(Path(self.samples[index][0]).stem.split('_')[-1])
        return image, label, index, val_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--val-dir', type=Path, default=config.IMAGENET_VAL_DIR)
    parser.add_argument('--out', type=Path, default=config.IMAGENET_FFCV_VALSET)
    parser.add_argument('--max-resolution', type=int, default=512)
    parser.add_argument('--jpeg-quality', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        sys.exit(f'{args.out} exists; pass --overwrite to rebuild it')

    dataset = IndexedImageFolder(str(args.val_dir))
    print(f'{len(dataset)} images, {len(dataset.classes)} classes in {args.val_dir}')
    assert len(dataset.classes) == N_CLASSES, 'expected one folder per ImageNet class'
    assert len(dataset) == N_VAL_IMAGES, 'expected the full 50k val set'

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(str(args.out), {
        'image': RGBImageField(write_mode='jpg', max_resolution=args.max_resolution,
                               jpeg_quality=args.jpeg_quality),
        'label': IntField(),
        'index': IntField(),
        'val_id': IntField(),
    }, num_workers=args.num_workers)
    writer.from_indexed_dataset(dataset)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
