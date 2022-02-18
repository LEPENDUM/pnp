from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from torch import Tensor
from abc import ABC, abstractmethod

import warnings

import torch
import numpy as np


def _generate(*output_size, params, distrib_selector, device):
    # Makes sure to send a fully unpacked argument list of non iterable values for the output size argument:
    while hasattr(output_size[0], '__getitem__'):
        if len(output_size) > 1:
            raise TypeError("The other parameters than the output size must be given as keyword arguments.")
        output_size = output_size[0]

    device = torch.device(device) if device is not None else torch.device('cpu')
    if device.type == 'cuda':
        return distrib_selector.gen_cuda(*output_size, params=params, device=device)
    elif device.type == 'cpu':
        return distrib_selector.gen_cpu(*output_size, params=params, device=device)
    else:
        raise RuntimeError(
            f'Unsupported device type: \'{device.type}\'. Supported device types are \'cpu\' or \'cuda\'.')


class RandomGenerator(ABC):

    def __init__(self, seed: Optional[int] = None, use_global_generator: bool = False):
        # pre-instanciate all the distribution selectors
        self._uniform_selector = _UniformDistribSelector(self)
        self._normal_selector = _NormalDistribSelector(self)
        self._poisson_selector = _PoissonDistribSelector(self)

        self._use_global_generator = use_global_generator
        if use_global_generator:
            self._rng = self._get_global_generator()
        else:
            self._rng = self._create_generator()
        if seed is not None:
            self.seed(seed)
        # else:
        #     self.set_to_global_state()


    def uniform(
            self,
            *output_size,
            mini: float = 0,
            maxi: float = 1,
            device: Optional[torch.device] = None
    ) -> Tensor:
        """Generate random numbers with uniform distribution between the value 'mini' and the value 'maxi'."""
        params = {'mini': mini, 'maxi': maxi}
        return _generate(*output_size, params=params, distrib_selector=self._uniform_selector, device=device)

    def normal(
            self,
            *output_size,
            mean: float = 0,
            std: float = 1,
            device: Optional[torch.device] = None
    ) -> Tensor:
        """Generate random numbers with normal distribution with mean 'mean' and standard deviation 'std'."""
        params = {'mean': mean, 'std': std}
        return _generate(*output_size, params=params, distrib_selector=self._normal_selector, device=device)

    def poisson(
            self,
            rate: Tensor,
            device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Generate random numbers with poisson distribution with rate 'rate'. By defaut, the returned tensor is
        generated on the same device as the input rate tensor.
        """
        params = {'rate': rate}
        if device is None:
            device = rate.device
        return _generate(None, params=params, distrib_selector=self._poisson_selector, device=device)

    @abstractmethod
    def seed(self, seed):
        pass

    @abstractmethod
    def _get_global_generator(self):
        pass

    @abstractmethod
    def _create_generator(self):
        pass

    @abstractmethod
    def set_to_global_state(self):
        """
        Sets the state of the RandomGenerator to the same state as the global generator warped by the RandomGenerator
        type (this should have no effect if the RandomGenerator is already configured with use_global_generator=True).
        """
        pass

    @abstractmethod
    def _uniform_cpu(self, *output_size, mini: float, maxi: float, device: torch.device) -> Tensor:
        pass

    def _uniform_gpu(self, *output_size, mini: float, maxi: float, device: torch.device) -> Tensor:
        return self._uniform_cpu(*output_size, mini=mini, maxi=maxi, device=torch.device('cpu')).to(device)

    @abstractmethod
    def _normal_cpu(self, *output_size, mean: float, std: float, device: torch.device) -> Tensor:
        pass

    def _normal_gpu(self, *output_size, mean: float, std: float, device: torch.device) -> Tensor:
        return self._normal_cpu(*output_size, mean=mean, std=std, device=torch.device('cpu')).to(device)

    @abstractmethod
    def _poisson_cpu(self, rate: Tensor, device: torch.device) -> Tensor:
        pass

    def _poisson_gpu(self, rate: Tensor, device: torch.device) -> Tensor:
        return self._poisson_cpu(rate=rate, device=torch.device('cpu')).to(device)


########################################################################################################################
#           Classes for selecting the cpu/gpu function calls of each random distribution
########################################################################################################################

class _RandomDistribSelector(ABC):

    def __init__(self, rg: RandomGenerator):
        self.rg = rg

    @abstractmethod
    def gen_cpu(self, *output_size, params, device):
        pass

    @abstractmethod
    def gen_cuda(self, *output_size, params, device):
        pass


class _UniformDistribSelector(_RandomDistribSelector):

    def gen_cpu(self, *output_size, params, device):
        return self.rg._uniform_cpu(*output_size, **params, device=device)

    def gen_cuda(self, *output_size, params, device):
        return self.rg._uniform_gpu(*output_size, **params, device=device)


class _NormalDistribSelector(_RandomDistribSelector):

    def gen_cpu(self, *output_size, params, device):
        return self.rg._normal_cpu(*output_size, **params, device=device)

    def gen_cuda(self, *output_size, params, device):
        return self.rg._normal_gpu(*output_size, **params, device=device)


class _PoissonDistribSelector(_RandomDistribSelector):

    def gen_cpu(self, *output_size, params, device):
        return self.rg._poisson_cpu(**params, device=device)

    def gen_cuda(self, *output_size, params, device):
        return self.rg._poisson_gpu(device=device, **params)


########################################################################################################################
#                                    Concrete Random generator implementations
########################################################################################################################

class TorchRandomGenerator(RandomGenerator):

    def __init__(self, seed: Optional[int] = None, use_global_generator: bool = False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._num_cuda_devices = torch.cuda.device_count()
        self._rngs_cuda = None
        super(TorchRandomGenerator, self).__init__(seed, use_global_generator)

    def seed(self, seed):
        if self._use_global_generator:
            torch.manual_seed(seed)
        else:
            self._rng.manual_seed(seed)
            for i in range(self._num_cuda_devices):
                self._rngs_cuda[i].manual_seed(seed)

    def _get_global_generator(self):
        return None

    def _create_generator(self):
        rng = torch.Generator()
        if self._num_cuda_devices:
            self._rngs_cuda = [torch.Generator(torch.device(i)) for i in range(self._num_cuda_devices)]
        return rng

    def set_to_global_state(self):
        if not self._use_global_generator:
            self._rng.set_state(torch.get_rng_state())
            for i in range(self._num_cuda_devices):
                self._rngs_cuda[i].set_state(torch.cuda.get_rng_state(i))

    def _uniform_cpu(self, *output_size, mini, maxi, device):
        return torch.rand(*output_size, device=device, generator=self._rng) * (maxi - mini) + mini

    def _uniform_gpu(self, *output_size, mini, maxi, device):
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        return torch.cuda.FloatTensor(*output_size, device=device).uniform_(mini, maxi, generator=self._rngs_cuda[idx])

    def _normal_cpu(self, *output_size, mean, std, device):
        return torch.randn(*output_size, device=device, generator=self._rng) * std + mean

    def _normal_gpu(self, *output_size, mean, std, device):
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        return torch.cuda.FloatTensor(*output_size, device=device).normal_(mean, std, generator=self._rngs_cuda[idx])

    def _poisson_cpu(self, rate: Tensor, device: torch.device) -> Tensor:
        return torch.poisson(rate.to(device), generator=self._rng)

    def _poisson_gpu(self, rate: Tensor, device: torch.device) -> Tensor:
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        return torch.poisson(rate.to(device), generator=self._rngs_cuda[idx])


class NumpyRandomGenerator(RandomGenerator):

    def seed(self, seed):
        self._rng.seed(seed)

    def _get_global_generator(self):
        return np.random

    def _create_generator(self):
        return np.random.RandomState()

    def set_to_global_state(self):
        if not self._use_global_generator:
            self._rng.set_state(np.random.get_state())

    def _uniform_cpu(self, *output_size, mini, maxi, device):
        return torch.from_numpy(self._rng.uniform(mini, maxi, output_size).astype(np.float32))

    def _normal_cpu(self, *output_size, mean, std, device):
        return torch.from_numpy(self._rng.normal(mean, std, output_size).astype(np.float32))

    def _poisson_cpu(self, rate: Tensor, device: torch.device) -> Tensor:
        return torch.from_numpy(self._rng.poisson(rate.to(device).detach().numpy())).to(rate.dtype)
