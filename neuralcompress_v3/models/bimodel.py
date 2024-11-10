"""
Bicephalous convolutional autoencoder with a
sparse encoder and a dense decoder.
"""
import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder


class BiModel(nn.Module):
    """
    Autoencoder
    """
    def __init__(self,
                 norm  = 'batch',
                 activ = None,
                 alpha = 4.,
                 eps   = 1e-6):

        super().__init__()

        if activ is None:
            activ = {'name': 'leakyrelu',
                     'negative_slope': .1}

        self.eps     = eps
        self.alpha   = alpha
        self.encoder = Encoder()
        self.decoder = nn.ModuleDict(
            {'reg': Decoder(norm, activ, output_activ=None),
             'clf': Decoder(norm, activ, output_activ='sigmoid')}
        )

    def forward(self, data, return_hard=False):
        """
        1. get the probability and regression values by encoding
        2. get the soft mask from from the probability
        3. reconstruct with the regression value and the (soft) gate
        4. if return_hard is set to true, calculate the hard gate
           and hard reconstruction from the hard gate.
        """

        # get the probability and regression values from the encoder
        prob, regr = self.encoder(data)

        # get the (soft) gate from the probability
        # for gradient descent steps
        logit = torch.logit(prob, eps=self.eps)

        logit_rnd = torch.logit(torch.rand_like(prob), eps=self.eps)
        logit_dff = logit - logit_rnd

        gate = torch.sigmoid(self.alpha * logit_dff)

        # reconstruct with the regression value and the (soft) gate
        reco_reg = self.decoder['reg'](regr * gate)
        reco_clf = self.decoder['clf'](regr * gate)

        result = {'prob': prob,
                  'gate': gate,
                  'reco_reg': reco_reg,
                  'reco_clf': reco_clf}

        if return_hard:
            # Computate the hard gate (step function) and
            # the reconstruction from the regression value
            # and the hard gate.
            gate_hard = logit_dff > 0

            with torch.no_grad():
                reco_reg_hard = self.decoder['reg'](regr * gate_hard)
                reco_clf_hard = self.decoder['clf'](regr * gate_hard)

            result['gate_hard'] = gate_hard
            result['reco_reg_hard'] = reco_reg_hard
            result['reco_clf_hard'] = reco_clf_hard

        return result

    def inference(self, data, threshold=None, precision='full'):
        """
        Inference encoding for production.
        NOTE: The gate and reco(nstruction) here correspond to
        to the hard gate and hard reconsturction in the
        forward function above.

        If threshold is None, use random thresholding
        of the probability value, other wise, use a
        fix threshold.
        """
        assert precision in ('full', 'half'),\
            (f'precision can only be either full '
             'or half, but got {precision}')

        with torch.no_grad():
            prob, regr = self.encoder(data)

            logit = torch.logit(prob, eps = self.eps)

            if threshold is None:
                logit_threshold = torch.logit(torch.rand_like(prob), eps=self.eps)
            else:
                logit_threshold = torch.logit(torch.tensor(threshold))

            gate = (logit - logit_threshold) > 0

            if precision == 'half':
                regr = regr.half().float()

            reco_reg = self.decoder['reg'](regr * gate)
            reco_clf = self.decoder['clf'](regr * gate)

        return {'prob': prob,
                'regr': regr,
                # hard gate and reconstruction for the hard gate
                'gate': gate,
                'reco_reg': reco_reg,
                'reco_clf': reco_clf}
