"""
A sparse encoder
"""
import torch
from torch import nn

import MinkowskiEngine as ME


class Encoder(nn.Module):
    """
    Encoder with sparse convolution
    """
    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(
            ME.MinkowskiConvolution(in_channels  = 1,
                                    out_channels = 2,
                                    kernel_size  = 3,
                                    dilation     = 1,
                                    dimension    = 3),
            # ME.MinkowskiBatchNorm(2),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels  = 2,
                                    out_channels = 2,
                                    kernel_size  = 3,
                                    dilation     = 2,
                                    dimension    = 3),
            # ME.MinkowskiBatchNorm(2),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels  = 2,
                                    out_channels = 2,
                                    kernel_size  = 3,
                                    dilation     = 4,
                                    dimension    = 3),
            # ME.MinkowskiBatchNorm(2),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels  = 2,
                                    out_channels = 2,
                                    kernel_size  = 3,
                                    dilation     = 2,
                                    dimension    = 3),
            # ME.MinkowskiBatchNorm(2),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels  = 2,
                                    out_channels = 2,
                                    kernel_size  = 1,
                                    dilation     = 1,
                                    dimension    = 3),
            ME.MinkowskiSigmoid())

        # self.device = device

    def forward(self, data):
        """
        1. Turn data into sparse encoding.
        2. Run the sparse encoder.
        3. Turn the output into a dense tensor.
        4. Get a probability and regression value for
           each non-zero value in the data.
        """

        # sparse encoding
        data_sparse = data.to_sparse_coo()

        coordinates = data_sparse.indices().T.contiguous().int()
        features = data_sparse.values().unsqueeze(-1)

        mink_batch = ME.SparseTensor(coordinates = coordinates,
                                     features    = features,)
                                     # device      = self.device)
        # run model
        output = self.model(mink_batch)

        # turn the output to dense tensor
        batch_size, spatial_shape = data.shape[0], data.shape[1:]
        output_shape = torch.Size((batch_size, 2) + spatial_shape)
        output = output.dense(output_shape)[0]

        # get the probability and regression value
        prob = output[:, :1, ...]
        regr = output[:, 1:, ...]

        return prob, regr
