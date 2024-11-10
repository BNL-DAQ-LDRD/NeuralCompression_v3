"""
A dense decoder
"""
from functools import partial

from torch import nn

from .utils import (get_norm_layer3d,
                    get_activ_layer)


class Conv3dBlock(nn.Conv3d):
    """
    3d convolution block with optional normalization
    and activation for the dense decoder below.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding = 0,
                 norm    = None,
                 activ   = None,
                 **kwargs):

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         **kwargs)

        self.pad   = nn.ReflectionPad3d(padding)
        self.norm  = get_norm_layer3d(norm, out_channels)
        self.activ = get_activ_layer(activ)

    def forward(self, input):
        return self.activ(self.norm(super().forward(self.pad(input))))


class Decoder(nn.Module):
    """
    Dense convolution decorder
    """
    def __init__(self, norm, activ, output_activ=None):

        super().__init__()

        layer = partial(Conv3dBlock, norm=norm, activ=activ)

        self.model = nn.Sequential(
            layer( 1, 16, kernel_size = 3, padding = 1),
            layer(16, 16, kernel_size = 3, padding = 1),
            layer(16, 16, kernel_size = 3, padding = 1),
            layer(16,  8, kernel_size = 3, padding = 1),
            layer( 8,  8, kernel_size = 3, padding = 1),
            layer( 8,  8, kernel_size = 3, padding = 1),
            layer( 8,  4, kernel_size = 3, padding = 1),
            layer( 4,  2, kernel_size = 3, padding = 1),
            nn.Conv3d(2, 1, kernel_size = 1)
        )
        if output_activ is None:
            self.output_activ = nn.Identity()
        else:
            self.output_activ = get_activ_layer(output_activ)

    def forward(self, data):
        return self.output_activ(self.model(data))
