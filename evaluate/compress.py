"""
Calculate the compression
"""
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

from neuralcompress_v3.models import BiModel
from enuralcompress_v3.utils import load_tpc_wedges

# uncomment if multiple GPU is available
# and you want to use an specific one
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
torch.cuda.set_device(0)

ROOT = Path('/home/yhuang2/PROJs/sparse_poi/')


def load_model(model,
               checkpoint,
               prefix = 'model',
               epoch  = 'last',
               device = 'cuda'):

    ckpt_fname = Path(checkpoint)/f'{prefix}_{epoch}.pth'
    model.load_state_dict(torch.load(ckpt_fname))
    return model.to(device)



def to_numpy(data):
    """
    Input
    data: (possibly cuda) tensor of shape (wedge, channel(=1), layer, azimuth, beam)
    Output:
        numpy array of shape (wedge, azimuth, beam, layer)
    """
    return data.detach().cpu().numpy()


def main(split):

    model = BiModel()
    model_identifier = 'bi_lambda-30_lb-10'
    checkpoint = ROOT/f'train/training_results/{model_identifier}/checkpoints'
    encoder = load_model(model, checkpoint).encoder

    # load meta of the wedges
    df = pd.read_csv(ROOT/'evaluate/csvs/occupancy_by_wedge.csv')
    df = df[ df.split == split ].reset_index(drop=True)
    print(f'number of samples = {len(df)}')

    # set up the folder for compression and csv
    compression_folder = Path('./compression')/model_identifier
    compression_folder.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(df.iterrows(), total=len(df))

    for _, row in pbar:

        experiment = row['experiment']
        event      = row['event']
        side       = row['side']
        sector     = row['sector']

        data = load_tpc_wedges(experiment = experiment,
                               event      = event,
                               side       = side,
                               sector     = sector,
                               log        = True,
                               device     = 'cuda')
        prob, regr = encoder(data)

        result = {'data': to_numpy(data),
                  'prob': to_numpy(prob),
                  'regr': to_numpy(regr)}

        fname = f'wedge-exp_{experiment}-ent_{event}-side_{side:02d}-sector_{sector:02d}'
        np.savez_compressed(compression_folder/fname, **result)


if __name__ == '__main__':
    split = sys.argv[1]
    assert split in ('train', 'test'), \
        f'split can either be train or test but got {split}'

    main(split)
