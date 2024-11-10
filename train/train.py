"""
Train a BCAE_VS model on time projection chamber (TPC) data.
BCAE_VS stands for Bicephalous Convlutional Autoencoder with
Variable compression ratio for Sparse input.
"""

import os
import sys
from pathlib import Path
from pprint import pprint
import yaml
import pandas as pd

# == torch ==================================================
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

# == package defined ========================================
from neuralcompress_v3.datasets.dataset_tpc import DatasetTPC
from neuralcompress_v3.models import BiModel, BiLoss
from neuralcompress_v3.utils import (Cumulator,
                                     Checkpointer,
                                     get_lr,
                                     flatten_dict,
                                     get_data_root)

os.environ['OMP_NUM_THREADS'] = '24'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def run_epoch(model,
              loss_fn,
              dataloader, *,
              log,
              prob_lambda,
              prob_lower_bound,
              batches_per_epoch,
              device,
              optimizer=None,
              desc=None):
    """
    run one epoch on a data loader
    """
    if batches_per_epoch is None:
        batches_per_epoch = float('inf')

    total_loss = 0
    cumulator_reco_soft = Cumulator()
    cumulator_reco_hard = Cumulator()
    cumulator_comp = Cumulator()

    for idx, adc in enumerate(dataloader, 1):

        if idx >= batches_per_epoch:
            break

        # ground truth
        adc = adc.to(device)
        if log:
            adc = torch.log2(adc + 1)

        tag = (adc > 0)
        true = tag.sum()

        # running the model
        result = model(adc, return_hard=True)

        # compression loss and metrics
        prob_loss = result['prob'].sum() / true
        prob_coef = prob_lambda * max(0., prob_loss.item() - prob_lower_bound)

        # compression performance
        cumulator_comp.update({
            'keep_ratio_soft': (result['gate'].sum() / true).item(),
            'keep_ratio_hard': (result['gate_hard'].sum() / true).item(),
            'prob_loss': prob_loss.item(),
            'prob_coef': prob_coef,
        })

        # reconstruction loss with soft (differentiable) cutoff
        reco_clf_soft = result['reco_clf'].squeeze(1)
        reco_reg_soft = result['reco_reg'].squeeze(1)

        results_reco_soft = loss_fn.forward(reco_clf_soft,
                                            reco_reg_soft,
                                            tag, adc)
        reco_loss_soft = results_reco_soft['reco_loss']

        # Detach the reco loss
        # This is very important since if we don't detach the loss
        # the computation graph will be retained because of
        # the use of cumulator.
        results_reco_soft['reco_loss'] = reco_loss_soft.item()
        cumulator_reco_soft.update(results_reco_soft)

        # overall loss and back-propogation
        loss = reco_loss_soft + prob_coef * prob_loss
        total_loss += loss.item()

        if optimizer is not None:
            if not isinstance(optimizer, (tuple, list)):
                optimizer = [optimizer]

            for opt in optimizer:
                opt.zero_grad()

            loss.backward()

            for opt in optimizer:
                opt.step()

        # reconstruction error with hard cutoff
        reco_clf_hard = result['reco_clf_hard'].squeeze(1)
        reco_reg_hard = result['reco_reg_hard'].squeeze(1)
        results_reco_hard = loss_fn.evaluate(reco_clf_hard,
                                             reco_reg_hard,
                                             tag, adc)
        cumulator_reco_hard.update(results_reco_hard)

        # summary
        summary = {'loss': total_loss / idx,
                   'comp': cumulator_comp.get_average(),
                   'reco_soft': cumulator_reco_soft.get_average(),
                   'reco_hard': cumulator_reco_hard.get_average()}
        print(f'\n{desc} Step {idx}')
        pprint(summary, sort_dicts=False)

        torch.cuda.empty_cache()

    return summary


def train():
    """
    Train a model:
        - load configurations
        - set up model, resume if needed
        - train model and save checkpoint
    """

    config_fname = sys.argv[1]

    with open(config_fname, 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    # checkpointing parameters
    checkpoint_path = config['checkpointing']['checkpoint_path']
    save_frequency  = config['checkpointing']['save_frequency']
    resume          = config['checkpointing']['resume']

    # device parameters
    device = config['device']['device']
    gpu_id = config['device']['gpu_id']
    if device == 'cuda':
        torch.cuda.set_device(gpu_id)

    # model parameters
    log              = config['model']['log']
    prob_lambda      = config['model']['prob_lambda']
    prob_lower_bound = config['model']['prob_lower_bound']
    reco_loss_type   = config['model']['reco_loss_type']

    # training parameters
    num_epochs        = config['train']['num_epochs']
    num_warmup_epochs = config['train']['num_warmup_epochs']
    batch_size        = config['train']['batch_size']
    batches_per_epoch = config['train']['batches_per_epoch']
    learning_rate     = config['train']['learning_rate']
    weight_decay      = config['train']['weight_decay']
    sched_steps       = config['train']['sched_steps']
    sched_gamma       = config['train']['sched_gamma']

    # Create model and resume if needed
    model = BiModel()
    checkpointer = Checkpointer(model,
                                checkpoint_path = checkpoint_path,
                                save_frequency  = save_frequency)
    resume_epoch = 0
    if resume:
        resume_epoch = checkpointer.load()

    model = model.to(device)

    # loss function
    loss_fn = BiLoss(reg_loss=reco_loss_type)

    # optimizer
    optimizer_encoder = AdamW(model.encoder.parameters(),
                              lr = learning_rate)
    optimizer_decoder = AdamW(model.decoder.parameters(),
                              lr = learning_rate,
                              weight_decay = weight_decay)
    # schedular
    milestones = range(num_warmup_epochs, num_epochs, sched_steps)
    scheduler_encoder = MultiStepLR(optimizer_encoder,
                                    milestones = milestones,
                                    gamma      = sched_gamma)
    scheduler_decoder = MultiStepLR(optimizer_decoder,
                                    milestones = milestones,
                                    gamma      = sched_gamma)

    # data loader
    data_root = get_data_root()
    train_ds  = DatasetTPC(data_root, split='train', dimension=2)
    valid_ds  = DatasetTPC(data_root, split='test',  dimension=2)
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ldr = DataLoader(valid_ds, batch_size=1,          shuffle=True)

    # training
    train_log = Path(checkpoint_path)/'train_log.csv'
    valid_log = Path(checkpoint_path)/'valid_log.csv'
    for epoch in range(resume_epoch + 1, num_epochs + 1):

        current_lr = get_lr(optimizer_encoder)
        print(f'current learning rate = {current_lr:.10f}')

        # train
        desc = f'Train Epoch {epoch} / {num_epochs}'
        train_stat = run_epoch(model,
                               loss_fn,
                               train_ldr,
                               log               = log,
                               prob_lambda       = prob_lambda,
                               prob_lower_bound  = prob_lower_bound,
                               desc              = desc,
                               optimizer         = (optimizer_encoder,
                                                    optimizer_decoder),
                               batches_per_epoch = batches_per_epoch,
                               device            = device)
        train_stat = flatten_dict(train_stat)

        # validation
        with torch.no_grad():
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(model,
                                   loss_fn,
                                   valid_ldr,
                                   log               = log,
                                   prob_lambda       = prob_lambda,
                                   prob_lower_bound  = prob_lower_bound,
                                   desc              = desc,
                                   batches_per_epoch = batches_per_epoch,
                                   device            = device)
            valid_stat = flatten_dict(valid_stat)

        # update learning rate
        scheduler_encoder.step()
        scheduler_decoder.step()

        # save checkpoints
        checkpointer.save(epoch)

        # log the results
        for log, stat in zip([train_log, valid_log],
                             [train_stat, valid_stat]):

            stat.update({'lr': current_lr,
                         'epoch': epoch})
            dataframe = pd.DataFrame(data=stat, index=[1])
            dataframe.to_csv(log,
                             index        = False,
                             float_format = '%.6f',
                             mode         = 'a' if log.exists() else 'w',
                             header       = not log.exists())


if __name__ == '__main__':
    train()
