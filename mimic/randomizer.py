import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torchvision
from robomimic.models.base_nets import Randomizer


class ImgColorJitterRandomizer(Randomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """

    def __init__(
        self,
        input_shape,
        num_rands=1,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super(ImgColorJitterRandomizer, self).__init__()
        assert len(input_shape) == 3  # (C, H, W)
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.num_rands = num_rands

    def output_shape_in(self, input_shape=None):
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        return list(input_shape)

    def forward_in(self, inputs):
        out = []
        for _ in range(self.num_rands):
            if self.training and np.random.rand() > self.epsilon:
                out.append(self.color_jitter(inputs))
            else:
                out.append(torch.clone(inputs))
        return torch.cat(out, dim=0)

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = inputs.shape[0] // self.num_rands
        out = TensorUtils.reshape_dimensions(
            inputs, begin_axis=0, end_axis=0, target_dims=(batch_size, self.num_rands)
        )
        return out.mean(dim=1)

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, num_rands={}, epsilon={})".format(
            self.input_shape, self.num_rands, self.epsilon
        )
        return msg


class ImgGaussianBlurRandomizer(Randomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """

    def __init__(
        self,
        input_shape,
        num_rands=1,
        kernel_size=(5, 5),
        sigma=(0.1, 2.0),
        epsilon=0.05,
    ):
        super(ImgGaussianBlurRandomizer, self).__init__()
        assert len(input_shape) == 3  # (C, H, W)
        self.blur = torchvision.transforms.GaussianBlur(
            kernel_size=kernel_size, sigma=sigma
        )
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.num_rands = num_rands

    def output_shape_in(self, input_shape=None):
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        return list(input_shape)

    def forward_in(self, inputs):
        out = []
        for _ in range(self.num_rands):
            if self.training and np.random.rand() > self.epsilon:
                out.append(self.blur(inputs))
            else:
                out.append(torch.clone(inputs))
        return torch.cat(out, dim=0)

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = inputs.shape[0] // self.num_rands
        out = TensorUtils.reshape_dimensions(
            inputs, begin_axis=0, end_axis=0, target_dims=(batch_size, self.num_rands)
        )
        return out.mean(dim=1)

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, num_rands={}, epsilon={})".format(
            self.input_shape, self.num_rands, self.epsilon
        )
        return msg
