"""
Load TPC wedges
"""
from itertools import product
from pathlib import Path
import numpy as np

import torch

from .utils import get_data_root


def __to_list(arg, default):

    if arg is None:
        return default

    if isinstance(arg, int):
        return [arg]
    elif isinstance(arg, [list, tuple]):
        return arg

    raise TypeError("argument must be an integer, a list, or a tuple")


def load_tpc_wedges(experiment,
                    event,
                    side   = None,
                    sector = None,
                    log    = True,
                    device = 'cuda'):
    """
    Load TPC wedges
    """

    data_root = Path(get_data_root(verbose=False))
    fname_template = f'12-2_{{sector}}-{{side}}/AuAu200_170kHz_10C_Iter2_{experiment}.xml_TPCMLDataInterface_{event}.npy'

    data = []
    sides = __to_list(side, [0, 1])
    sectors = __to_list(sector, list(range(12)))

    for _side, _sector in product(sides, sectors):
        fname = data_root/fname_template.format(sector=_sector, side=_side)
        data.append(np.load(fname).astype(np.float32))

    data = torch.tensor(np.array(data))
    if log:
        data = torch.log2(data + 1.)

    if device == 'cuda':
        return data.cuda()
    return data
