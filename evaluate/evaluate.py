"""
Calculate the reconstruction
"""
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

from neuralcompress_v3.models import BiModel
from neuralcompress_v3.utils import (Checkpointer,
                                     load_tpc_wedges)


os.environ['OMP_NUM_THREADS'] = '24'
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"


def get_stats(data, reco_clf, reco_reg, gate, cutoff=None):
    """
    Evaluate reconstruction performance
    """

    # classification accuracy
    label_grt = (data > 0)
    label_prd = (reco_clf > .5)

    true_pos  = (label_grt * label_prd).sum()
    pos       = label_prd.sum()
    true      = label_grt.sum()

    precision = (true_pos / pos).item()
    recall    = (true_pos / true).item()

    # combine reg. and clf. to get reconstruction
    reco = reco_reg * (reco_clf > .5)

    dmax = data.max().item()

    # regression accuracy
    l1 = torch.abs(reco - data).mean().item()
    l2 = torch.pow(reco - data, 2).mean()
    psnr = 10. * torch.log10((dmax ** 2) * torch.rsqrt(l2)).item()

    result = {'precision' : [precision],
              'recall'    : [recall],
              'l1'        : [l1],
              'l2'        : [l2.item()],
              'psnr'      : [psnr],
              'cutoff'    : [np.nan]}

    if cutoff is None:
        cutoff = []

    for threshold in cutoff:
        reco[reco < threshold] = 0

        label_prd = (reco > 0)
        true_pos = (label_grt * label_prd).sum()
        pos = label_prd.sum()

        precision = (true_pos / pos).item()
        recall    = (true_pos / true).item()

        l1 = torch.abs(reco - data).mean().item()
        l2 = torch.pow(reco - data, 2).mean()
        psnr = 10. * torch.log10((dmax ** 2) * torch.rsqrt(l2)).item()

        result['precision'].append(precision)
        result['recall'].append(recall)
        result['l1'].append(l1)
        result['l2'].append(l2.item())
        result['psnr'].append(psnr)
        result['cutoff'].append(threshold)

    # compression efficiency
    keep_ratio = (gate.sum() / true).item()

    result['keep_ratio'] = [keep_ratio] * (len(cutoff) + 1)

    return result


def to_numpy(data):
    return data.detach().cpu().numpy()


def evaluate(checkpoint_path, split, half=True, save=False):

    model = BiModel()
    checkpointer = Checkpointer(model, checkpoint_path)
    checkpointer.load()

    model.eval()


    # load meta of the wedges
    df = pd.read_csv(ROOT/'evaluate/csvs/occupancy_by_wedge.csv')
    df = df[ df.split == split ].reset_index(drop=True)
    print(f'number of samples = {len(df)}')

    # set up the folder for reconstruction and csv
    reconstruction_folder = Path('./reconstruction')
    reconstruction_folder.mkdir(parents=True, exist_ok=True)
    csv_folder = Path('./results')
    csv_folder.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(df.iterrows(), total=len(df))

    dfs = []
    for row_idx, row in pbar:

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

        data_precision = 'half' if half else 'full'
        result = model.inference(data,
                                 threshold=None,
                                 precision=data_precision)

        # evaluation
        reco_clf = result['reco_clf'].squeeze(1)
        reco_reg = result['reco_reg'].squeeze(1)

        gate = result['gate'].squeeze(1)

        cutoff = list(np.linspace(3, 6, int((6 - 3)/.1) + 1, endpoint=True))
        stats = get_stats(data     = data,
                          reco_clf = reco_clf,
                          reco_reg = reco_reg,
                          gate     = gate,
                          cutoff   = cutoff)

        pbar.set_postfix({key: val[0] for key, val in stats.items()})

        df = pd.DataFrame(data=stats)
        row_repeated = pd.concat([row.to_frame().T] * len(df), ignore_index=True)
        dfs.append(df.join(row_repeated))

        if save:
            # save result
            for key, val in result.items():
                result[key] = to_numpy(val.squeeze())

            result['data'] = to_numpy(data.squeeze())

            fname = f'wedge-exp_{experiment}-ent_{event}-side_{side:02d}-sector_{sector:02d}'
            np.savez_compressed(reconstruction_folder/fname, **result)

    df_evaluation = pd.concat(dfs)
    print(df_evaluation)

    csv_fname = csv_folder/f'performance_{split}-me_{data_precision}.csv'
    df_evaluation.to_csv(csv_fname, float_format='%.6f', index=False)


def main():
    parser = argparse.ArgumentParser(
        description="evaluate a pretrained BCAE_VS model"
    )
    parser.add_argument('checkpoint_path',
                        type    = str,
                        help    = "The path to the model checkpoint.")
    parser.add_argument('--split',
                        type    = str,
                        default = 'test',
                        choices = ('test', 'train'),
                        help    = ('The split of the dataset to evaluate '
                                   'the model on. (default = test)'))
    parser.add_argument('--device',
                        type    = str,
                        default = 'cuda',
                        choices = ('cpu', 'cuda'),
                        help    = ('The device for running the evaluation. '
                                   '(default = cuda)'))
    parser.add_argument('--gpu-id',
                        type    = int,
                        default = 0,
                        help    = ('The id of the GPU for running the '
                                   'evaluation. Only active if device=cuda.'
                                   '(default = 0)'))
    parser.add_argument('--precision',
                        type    = str,
                        default = 'half',
                        choices = ('full', 'half'),
                        help    = ('The precision the encoded (compressed) '
                                   'data will be converted to and/or saved '
                                   'with. (default = half)'))
    parser.add_argument('--compressed-path',
                        type    = str,
                        default = None,
                        help    = ("If given, the compressed data (code) will "
                                   "saved to this path. "
                                   "(default = None, do not save)"))
    parser.add_argument('--result-csv-path',
                        type    = str,
                        default = './result.csv',
                        help    = ("The path of the CSV file the evaluation "
                                   "result will be saved to. "
                                   "(default = ./result.csv)"))
    args = parser.parse_args()

    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu_id)

    evalute(checkpoint_path  = args.checkpoint_path,
            split            = args.split,
            half             = (args.precision == 'half')
            compression_path = args.compression_path,
            result_csv_path  = args.result_csv_path,
            device           = args.device)


if __name__ == '__main__':
    main()
