import numpy as np
import torch
from torch import nn
from torch import distributions

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class StandardUniform(Distribution):
    def __init__(
        self,
        shape: list = [1],
    ):
        """Multidimensionqal uniform distribution defined on a box."""
        super().__init__()
        self._shape = torch.Size(shape)
        self.register_buffer(
            "_log_z", torch.tensor(0.0, dtype=torch.float64), persistent=False
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = torch.zeros(inputs.shape[0])
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.rand(num_samples, *self._shape, device=self._log_z.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.rand(
                context_size * num_samples, *self._shape, device=context.device
            )
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            # return self._log_z.new_zeros(self._shape)
            return self._log_z.new_full(self._shape, 0.5)
        else:
            # The value of the context is ignored, only its size is taken into account.
            # return context.new_zeros(context.shape[0], *self._shape)
            return context.new_full(context.shape[0], *self._shape, 0.5)


# Need to see if pyro can gang together distributions. I want to send such an object to the NF code.
class Uniform(Distribution):
    def __init__(
        self,
        # shape: list = [1],
        low: torch.Tensor = torch.Tensor([0]),
        high: torch.Tensor = torch.Tensor([1]),
    ):
        """Multidimensional uniform distribution defined on a box."""
        super().__init__()
        shape = low.shape
        self._low = low
        self._high = high
        self._neg_energy = (-torch.log(torch.dot((high - low), (high - low)))).item()
        self._shape = shape
        self.register_buffer(
            "_log_z", torch.tensor(0.0, dtype=torch.float64), persistent=False
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = torch.full(torch.Size([inputs.shape[0]]), self._neg_energy)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return (
                torch.rand(num_samples, *self._shape, device=self._log_z.device)
                * (self._high - self._low)
            ) + self._low
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.rand(
                context_size * num_samples, *self._shape, device=context.device
            )
            samples = (samples * (self._high - self._low)) + self._low
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            # return self._log_z.new_zeros(self._shape)
            # return self._log_z.new_full(self._shape,0.5)
            return self._log_z.new_tensor(0.5 * (self._high + self._low))
        else:
            # The value of the context is ignored, only its size is taken into account.
            # return context.new_zeros(context.shape[0], *self._shape)
            # return context.new_full(context.shape[0], *self._shape,0.5)
            return context.new_tensor(
                context.shape[0], self._high - self._low
            )  # will this even work?
