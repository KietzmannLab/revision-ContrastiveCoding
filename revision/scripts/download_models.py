"""Pre-download the public model weights that jsputils fetches at load time.

Run once. Weights go to the torch hub cache 
(``$TORCH_HOME/hub``, default ``~/.cache/torch/hub``), where
``jsputils.nnutils.load_model`` finds them later. If you set TORCH_HOME here, set it
in your jobs as well.

  alexnet-barlow-twins  Konkle lab S3 checkpoint (same URL as jsputils/models.py)
  alexnet-supervised    torchvision ImageNet weights
  alexnet-ipcl          torch.hub harvard-visionlab/open_ipcl

For alexnet-ipcl the repo is also added to torch hub's trusted list. Since torch 2.x,
``torch.hub.load`` without ``trust_repo`` asks for confirmation the first time it sees a
repo, which fails in non-interactive jobs (EOFError).

The VGGFace AlexNet and the readout checkpoint are not public model-zoo files; they have
to be downloaded by hand from the paper's Dataverse (doi:10.7910/DVN/5848EQ).
"""

import argparse

import torch
import torchvision

BARLOW_TWINS_URL = ('https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/barlow_twins/'
                    'barlow_alexnet_gn_imagenet_final.pth.tar')


def download_barlow_twins():
    torch.hub.load_state_dict_from_url(BARLOW_TWINS_URL, map_location='cpu')


def download_supervised():
    torchvision.models.alexnet(weights='DEFAULT')


def download_ipcl():
    torch.hub.load('harvard-visionlab/open_ipcl', 'alexnetgn_ipcl_ref01', trust_repo=True)


DOWNLOADS = {
    'alexnet-barlow-twins': download_barlow_twins,
    'alexnet-supervised': download_supervised,
    'alexnet-ipcl': download_ipcl,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', choices=list(DOWNLOADS), default=list(DOWNLOADS))
    args = parser.parse_args()

    print('torch hub dir:', torch.hub.get_dir())
    for model_name in args.models:
        print(f'--- {model_name}')
        DOWNLOADS[model_name]()
    print('done')


if __name__ == '__main__':
    main()
