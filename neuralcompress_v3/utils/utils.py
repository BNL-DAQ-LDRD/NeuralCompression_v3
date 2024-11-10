"""
Utility functions for model training
"""

import os

def get_data_root():
    """
    Get root to data as an environment variable
    """
    # Get the value of DATAROOT
    data_root = os.getenv('DATAROOT')

    assert data_root is not None, \
        ('DATAROOT is not set!'
         'please run export DATAROOT=/path/to/your/data'
         'to set the environment variable.')

    print(f'\nDATAROOT is set to: {data_root}\n')

    return data_root


def flatten_dict(mydict):
    result = {}
    for key, val in mydict.items():
        if isinstance(val, dict):
            flattened = flatten_dict(val)
            for sub_key, sub_val in flattened.items():
                result[f'{key}-{sub_key}'] = sub_val
        else:
            result[key] = val
    return result


def get_lr(optim):
    """
    Get the current learning rate
    """
    for param_group in optim.param_groups:
        return param_group['lr']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
