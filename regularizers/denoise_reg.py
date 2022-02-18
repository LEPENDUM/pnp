from __future__ import annotations
from typing import TYPE_CHECKING, List, Type, Dict, Any
if TYPE_CHECKING:
    from torch import Tensor

from abc import ABC, abstractmethod
from pnpcore import Regularizer, Preconditioner, DiagonalPreconditioner, CodeOptimPrecoMsg
from utils.image_utils import convert_channels_number, view_as_tensor4
from models.image_module import ImageModule

import torch
import torch.fft
import torch.nn as nn


class DenoisingRegularizer(Regularizer):

    def __init__(self, denoiser: ImageModule, denoiser_caller: DenoiserCaller):
        super(DenoisingRegularizer, self).__init__()
        self._denoiser_caller = denoiser_caller
        self._denoiser = denoiser
        if not issubclass(denoiser.module_type(), self._denoiser_caller.expected_denoiser_type()):
            raise TypeError(f'The given module of type \'{denoiser.module_type().__name__}\' is not compatible with '
                            f'the caller that expects a type \'{denoiser_caller.expected_denoiser_type().__name__}\'.')

    def on_preco_change(self) -> None:
        self._denoiser_caller.on_preco_change(self.get_preconditioner(), self._denoiser)

    def denoise_operator(self, Pu: Tensor, sigma: float, u: Tensor = None, iter_num: int = 0) -> Tensor:
        return self._denoiser_caller.call_denoiser(self._denoiser, Pu, sigma, u, self.get_preconditioner(), iter_num)

    def missing_config_elements_msgs(self) -> List[str]:
        msgs = super(DenoisingRegularizer, self).missing_config_elements_msgs()
        if self._denoiser_caller is None:
            msgs.append('No denoiser adapter object specified for the DenoisingRegularizer.')
        if self._denoiser is None:
            expected_str = '.'
            if self._denoiser_caller is not None:
                expected_str = ' (expected a ' + self._denoiser_caller.expected_denoiser_type().__name__ + ' object).'
            msgs.append('No denoiser object specified for the DenoisingRegularizer' + expected_str)
        return msgs

    def compatible_preco_types(self) -> List[Type]:
        return self._denoiser_caller.compatible_preco_types()

    def name(self) -> str:
        return self._denoiser.name()

    def code_optim_preco_msg(self):
        return self._denoiser_caller.code_optim_preco_msg()


class DenoiserCaller(ABC):
    """
    base class to define how to call a given type of denoiser from the denoising regularizer.
    """
    def compatible_preco_types(self) -> List[Type]:
        return [Preconditioner]

    @abstractmethod
    def expected_denoiser_type(self) -> Type:
        pass

    @abstractmethod
    def on_preco_change(self, preconditioner: Preconditioner, denoiser: ImageModule) -> None:
        pass

    @abstractmethod
    def call_denoiser(
            self,
            denoiser: ImageModule,
            Pu: Tensor,
            sigma: float,
            u: Tensor,
            preconditioner: Preconditioner,
            iter_num: int = 0) -> Tensor:
        pass

    def code_optim_preco_msg(self):
        return []


########################################################################################################################
# Regularization based on denoiser with inputs:
#   1- Noisy image (with diagonal noise covariance matrix)
#   2- Standard deviation map input
########################################################################################################################

class DenoiserStdMapCaller(DenoiserCaller):

    def __init__(self):
        self._std_map = None

    def expected_denoiser_type(self) -> Type:
        return DenoiserStdMap

    def compatible_preco_types(self) -> List[Type]:
        return [DiagonalPreconditioner]

    def on_preco_change(self, preconditioner: Preconditioner, denoiser: ImageModule) -> None:
        std_map = view_as_tensor4(preconditioner.multiplier())
        self._std_map = convert_channels_number(std_map, denoiser.num_channels('std_map'))

    def call_denoiser(self, denoiser, Pu, sigma, u, preconditioner, iter_num=0) -> Tensor:
        n_channels = Pu.shape[1]
        Pu = convert_channels_number(Pu, denoiser.num_channels('image'))
        denoised = denoiser(args_dict={'iter_num': iter_num - 1}, image=Pu, std_map=self._std_map * sigma)
        return convert_channels_number(denoised, n_channels)


class DenoiserStdMap(ImageModule):
    """
    Base abstract class for adapting a denoiser module to inputs given as: the image to denoise and a pixel-wise map of
    noise standard deviations (both as tensors with 4 dimensions: batch, channels, vertical size, horizontal size).
    """
    def __init__(self,
                 denoise_module: nn.Module,
                 num_channels_image: int,
                 num_channels_std_map: int,
                 denoiser_name: str,
                 ):
        super(DenoiserStdMap, self).__init__()
        if num_channels_image != 1 and num_channels_image != 3:
            raise ValueError('The number of channels of the input image must be 1 or 3.')
        if num_channels_std_map != 1 and num_channels_std_map != 3:
            raise ValueError('The number of channels of the standard deviation map must be 1 or 3.')
        self._name = denoiser_name
        self.__dict_input_channels = {'image': num_channels_image, 'std_map': num_channels_std_map}
        self._denoise_module = denoise_module

    def dict_input_channels(self) -> Dict[str, Any]:
        return self.__dict_input_channels

    def name(self) -> str:
        return self._name

    def forward(self, args_dict: Dict[str, Any] = None, **image_inputs: Tensor) -> Tensor:
        return self.call_denoiser(image_inputs['image'], image_inputs['std_map'])

    @abstractmethod
    def call_denoiser(self, image: Tensor, std_map: Tensor) -> Tensor:
        pass


class DenoiserStdMapCat(DenoiserStdMap):
    """
    Concrete DenoiserStdMap adapter class for denoising modules that concatenate the image and standard deviation map
    inputs in the channel dimension.
    """
    def call_denoiser(self, image, std_map):
        std_shape_exp = [image.shape[0], -1, image.shape[2], image.shape[3]]
        x = torch.cat((image, std_map.expand(std_shape_exp)), dim=1)
        return self._denoise_module(x)


'''
class RegularizerStdMap(Regularizer):

    def __init__(self, denoiser: ImageModule):
        super(RegularizerStdMap, self).__init__()
        self._denoiser = denoiser
        self._std_map = None

    def on_preco_change(self) -> None:
        std_map = view_as_tensor4(self.get_preconditioner().multiplier())
        self._std_map = convert_channels_number(std_map, self._denoiser.num_channels('std_map'))

    def denoise_operator(self, Pu: Tensor, sigma: float, u: Tensor = None, iter_num: int = 0) -> Tensor:
        n_channels = Pu.shape[1]
        Pu = convert_channels_number(Pu, self._denoiser.num_channels('image'))
        denoised = self._denoiser(args_dict={'iter_num': iter_num-1}, image=Pu, std_map=self._std_map * sigma)
        return convert_channels_number(denoised, n_channels)

    def missing_config_elements_msgs(self) -> List[str]:
        msgs = super(RegularizerStdMap, self).missing_config_elements_msgs()
        if self._denoiser is None:
            msgs.append('no StdMapDenoiser object specified for the RegularizerStdMap.')
        return msgs

    def compatible_preco_types(self) -> List[Type]:
        return [DiagonalPreconditioner]

    def name(self) -> str:
        return self._denoiser.name()


class DenoiserStdMapCat(DenoiserStdMap):
    """
    DenoiserStdMap adapter class for denoising modules that concatenate the image and standard deviation map
    inputs in the channel dimension.
    """

    def __init__(self,
                 denoise_module: nn.Module,
                 num_channels_image: int,
                 num_channels_std_map: int,
                 denoiser_name: str,
                 num_channels_image_transformed: int = 0
                 ):
        super(DenoiserStdMapCat, self).__init__(denoise_module, num_channels_image, num_channels_std_map, denoiser_name)
        if num_channels_image_transformed not in [0, 1, 3]:
            raise ValueError(
                'The number of channels of the input transformed image '
                'must be either 0 (no input transformed image), 1 or 3.')
        self.num_channels_image_transformed = num_channels_image_transformed

    def call_denoiser(self, image, std_map):
        std_shape_exp = [image.shape[0], -1, image.shape[2], image.shape[3]]
        if self.num_channels_image_transformed:
            image = torch.cat((image, convert_channels_number(image, self.num_channels_image_transformed)), dim=1)
        x = torch.cat((image, std_map.expand(std_shape_exp)), dim=1)
        return self._denoise_module(x)
'''


########################################################################################################################
# Regularization based on denoiser with inputs:
#   1- Noisy image (with possibly non-diagonal noise covariance matrix)
#   2- Transformed version of the image (with constant noise level)
#   3- standard deviation map (or value) of the transformed image.
########################################################################################################################

class DenoiserTransCovCaller(DenoiserCaller):

    def __init__(self, inf_t1: float = float('inf'), inf_t2: float = float('inf'), inf_pow: float = 1.0):
        super(DenoiserTransCovCaller, self).__init__()
        if inf_t1 > inf_t2:
            raise ValueError("1st threshold should be lower or equal to the 2nd threshod.")
        self._inf_t1 = inf_t1
        self._inf_t2 = inf_t2
        self._inf_pow = inf_pow

    def use_inf_noise_scaling(self) -> bool:
        return self._inf_t2 < float('inf')

    def infinite_noise_scale(self, sigma: float, preconditioner: Preconditioner) -> Tensor:
        n = torch.abs(sigma * preconditioner.multiplier())  # noise s.t.d expected by the denoiser at each frequency
        return torch.clamp(1 - torch.clamp((n - self._inf_t1) / (self._inf_t2 - self._inf_t1), 0) ** self._inf_pow, 0)

    def preprocess_img(self, Pu: Tensor, sigma: float, preconditioner: Preconditioner) -> Tensor:
        is_image_fourier_Pu = CodeOptimPrecoMsg.IMAGE_FOURIER_DOMAIN_PX in preconditioner.code_optim_settings()
        if self.use_inf_noise_scaling():
            if not is_image_fourier_Pu:
                Pu = torch.fft.fftn(Pu, dim=(2, 3))
                is_image_fourier_Pu = True
            Pu *= self.infinite_noise_scale(sigma, preconditioner)
        if is_image_fourier_Pu:
            Pu = torch.fft.irfftn(Pu, Pu.shape[2:4], dim=(2, 3))
        return Pu

    def expected_denoiser_type(self) -> Type:
        return DenoiserTransCov

    def on_preco_change(self, preconditioner: Preconditioner, denoiser: ImageModule) -> None:
        pass

    def call_denoiser(self, denoiser, Pu, sigma, u, preconditioner, iter_num=0) -> Tensor:
        Pu = self.preprocess_img(Pu, sigma, preconditioner)
        n_channels = Pu.shape[1]
        Pu = convert_channels_number(Pu, denoiser.num_channels('image'))
        u = convert_channels_number(u, denoiser.num_channels('image_trans'))
        std = torch.tensor(sigma, device=Pu.device).reshape([1] * Pu.ndim)
        std = convert_channels_number(std, denoiser.num_channels('std_map'))
        denoised = denoiser(args_dict={'iter_num': iter_num - 1}, image=Pu, image_trans=u, std_map=std)
        return convert_channels_number(denoised, n_channels)

    def code_optim_preco_msg(self):
        # Asks the preconditioner to return the preconditioned variable in the Fourier domain directly if it saves
        # computations (skip unnecessary inverse and forward Fourier Transform in the case of a FourierPreconditioner).
        if self.use_inf_noise_scaling():
            return [CodeOptimPrecoMsg.IMAGE_FOURIER_DOMAIN_PX]
        return []


class DenoiserTransCov(ImageModule):
    """
    Base abstract class for adapting a denoiser module to inputs given as:
     - the image to denoise,
     - a transformed version of the image with constant noise standard deviation,
     - a standard deviation value (given as a tensor).
    The images are given as tensors with 4 dimensions: batch, channels, vertical size, horizontal size.
    """
    def __init__(self,
                 denoise_module: nn.Module,
                 num_channels_image: int,
                 num_channels_image_transformed: int,
                 num_channels_std_map: int,
                 denoiser_name: str,
                 ):
        super(DenoiserTransCov, self).__init__()
        if num_channels_image != 1 and num_channels_image != 3:
            raise ValueError('The number of channels of the input image must be 1 or 3.')
        if num_channels_image_transformed != 1 and num_channels_image_transformed != 3:
            raise ValueError('The number of channels of the transformed input image must be 1 or 3.')
        if num_channels_std_map != 1 and num_channels_std_map != 3:
            raise ValueError('The number of channels of the standard deviation map must be 1 or 3.')
        self._name = denoiser_name
        self._denoise_module = denoise_module
        self.__dict_input_channels = {
            'image': num_channels_image,
            'image_trans': num_channels_image_transformed,
            'std_map': num_channels_std_map}

    def dict_input_channels(self) -> Dict[str, Any]:
        return self.__dict_input_channels

    def name(self) -> str:
        return self._name

    def forward(self, args_dict: Dict[str, Any] = None, **image_inputs: Tensor) -> Tensor:
        return self.call_denoiser(image_inputs['image'], image_inputs['image_trans'], image_inputs['std_map'])

    @abstractmethod
    def call_denoiser(self, image: Tensor, image_trans: Tensor, std_map: Tensor) -> Tensor:
        pass


class DenoiserTransCovCat(DenoiserTransCov):
    """
    Concrete DenoiserTransCovCat adapter class for denoising modules that concatenate the images and standard deviation
    map inputs in the channel dimension.
    """
    def call_denoiser(self, image, image_trans, std_map):
        std_shape_exp = [image.shape[0], -1, image.shape[2], image.shape[3]]
        image = torch.cat((image, image_trans), dim=1)
        x = torch.cat((image, std_map.expand(std_shape_exp)), dim=1)
        return self._denoise_module(x)


'''
class RegularizerTransCov(Regularizer):

    def __init__(self, denoiser: ImageModule):
        super(RegularizerTransCov, self).__init__()
        self._denoiser = denoiser

    def denoise_operator(self, Pu: Tensor, sigma: float, u: Tensor = None, iter_num: int = 0) -> Tensor:
        n_channels = Pu.shape[1]
        Pu = convert_channels_number(Pu, self._denoiser.num_channels('image'))
        u = convert_channels_number(u, self._denoiser.num_channels('image_trans'))
        std = torch.tensor(sigma, device=Pu.device).reshape([1]*Pu.ndim)
        std = convert_channels_number(std, self._denoiser.num_channels('std_map'))

        denoised = self._denoiser(args_dict={'iter_num': iter_num-1}, image=Pu, image_trans=u, std_map=std)
        return convert_channels_number(denoised, n_channels)

    def missing_config_elements_msgs(self) -> List[str]:
        msgs = super(RegularizerTransCov, self).missing_config_elements_msgs()
        if self._denoiser is None:
            msgs.append('no DenoiserTransCov object specified for the RegularizerTransCov.')
        return msgs

    def name(self) -> str:
        return self._denoiser.name()
'''


########################################################################################################################
# Classes for calling denoiser modules to perform denoising of Indenpendent and Identically Distributied (IID) noise,
# i.e. using a single standard deviation parameter.
########################################################################################################################

class DenoiserIID(ImageModule):
    """
    Base abstract class for adapting a denoiser module to perform iid denoising with inputs given as:
     - the image to denoise,
     - the standard deviation (either fixed in the constructor or given in the input args_dict as the value 'sigma').
    The image is given as a tensor with 4 dimensions: batch, channels, vertical size, horizontal size.
    """
    def __init__(self,
                 denoise_module: nn.Module,
                 num_channels_image: int,
                 denoiser_name: str,
                 fixed_sigma: float = None
                 ):
        super(DenoiserIID, self).__init__()
        if num_channels_image != 1 and num_channels_image != 3:
            raise ValueError('The number of channels of the input image must be 1 or 3.')
        self._name = denoiser_name
        self.__dict_input_channels = {'image': num_channels_image}
        self._denoise_module = denoise_module
        self._fixed_sigma = fixed_sigma

    def dict_input_channels(self) -> Dict[str, Any]:
        return self.__dict_input_channels

    def name(self) -> str:
        return self._name

    def forward(self, args_dict: Dict[str, Any] = None, **image_inputs: Tensor) -> Tensor:
        sigma_in = args_dict.get('sigma')
        sigma = sigma_in if self._fixed_sigma is None else self._fixed_sigma
        if sigma is None:
            raise ValueError('No standard deviation parameter specified for the denoising.')
        elif self._fixed_sigma is not None and sigma_in is not None and sigma_in != sigma:
            raise ValueError(
                f'The denoiser is setup for a fixed standard deviation value: fixed_sigma={sigma}. '
                f'It can\'t be used for denoising with the standard deviation sigma={sigma_in}.')

        return self.call_denoiser(image_inputs['image'], sigma)

    @abstractmethod
    def call_denoiser(self, image: Tensor, sigma: float) -> Tensor:
        pass


class DenoiserIIDFromStdMapCat(DenoiserIID):
    """
    Adapt the iid denoising inputs to a denoiser module taking as input:
    the concatenation of the noisy input image and a standard deviation map.
    """
    def __init__(self,
                 denoise_module: nn.Module,
                 num_channels_image: int,
                 num_channels_std_map: int,
                 denoiser_name: str,
                 fixed_sigma: float = None
                 ):
        super(DenoiserIIDFromStdMapCat, self).__init__(denoise_module, num_channels_image, denoiser_name, fixed_sigma)
        if num_channels_std_map != 1 and num_channels_std_map != 3:
            raise ValueError('The number of channels of the standard deviation map must be 1 or 3.')
        self._num_channels_std_map = num_channels_std_map

    def call_denoiser(self, image: Tensor, sigma: float) -> Tensor:
        std_map = torch.tensor(sigma, device=image.device).reshape([1] * image.ndim)
        std_shape_exp = [image.shape[0], self._num_channels_std_map, image.shape[2], image.shape[3]]
        x = torch.cat((image, std_map.expand(std_shape_exp)), dim=1)
        return self._denoise_module(x)


class DenoiserIIDFromTransCovCat(DenoiserIID):
    """
    Adapt the iid denoising inputs to a denoiser module taking as input:
    the concatenation of the noisy input image, an inverse transformed input image, and a standard deviation map.
    """
    def __init__(self,
                 denoise_module: nn.Module,
                 num_channels_image: int,
                 num_channels_image_transformed: int,
                 num_channels_std_map: int,
                 denoiser_name: str,
                 fixed_sigma: float = None
                 ):
        super(DenoiserIIDFromTransCovCat, self).__init__(denoise_module, num_channels_image, denoiser_name, fixed_sigma)
        if num_channels_image_transformed != 1 and num_channels_image_transformed != 3:
            raise ValueError('The number of channels of the transformed input image must be 1 or 3.')
        if num_channels_std_map != 1 and num_channels_std_map != 3:
            raise ValueError('The number of channels of the standard deviation map must be 1 or 3.')
        self._num_channels_std_map = num_channels_std_map
        self._num_channels_image_transformed = num_channels_image_transformed

    def call_denoiser(self, image: Tensor, sigma: float) -> Tensor:
        std_map = torch.tensor(sigma, device=image.device).reshape([1] * image.ndim)
        std_shape_exp = [image.shape[0], self._num_channels_std_map, image.shape[2], image.shape[3]]
        image = torch.cat((image, convert_channels_number(image, self._num_channels_image_transformed)), dim=1)
        x = torch.cat((image, std_map.expand(std_shape_exp)), dim=1)
        return self._denoise_module(x)


########################################################################################################################
########################################################################################################################


class DenoiserFactory(ABC):
    """
    Base factory class for creating denoisers (either the raw module or the ImageModule for adapting to regularizer's
    inputs). The class also keeps the loaded model instance.
    """
    def __init__(self):
        self.__raw_module = None

    @abstractmethod
    def denoiser_iid(self) -> DenoiserIID:
        pass

    @abstractmethod
    def denoiser_non_iid(self) -> ImageModule:
        pass

    @abstractmethod
    def denoiser_caller(self) -> DenoiserCaller:
        pass

    @abstractmethod
    def _create_raw_module(self) -> nn.Module:
        pass
    def raw_module(self) -> nn.Module:
        if self.__raw_module is None:
            self.__raw_module = self._create_raw_module()
        return self.__raw_module


'''
class DenoiserStdMap(ImageModule):
    """
    Base abstract class for adapting a denoiser module to inputs given as: the image to denoise and a pixel-wise map of
    noise standard deviations (both as tensors with 4 dimensions: batch, channels, vertical size, horizontal size).
    """
    def __init__(self,
                 denoise_module: nn.Module,
                 num_channels_image: int,
                 num_channels_std_map: int,
                 denoiser_name: str,
                 ):
        super(DenoiserStdMap, self).__init__()
        if num_channels_image != 1 and num_channels_image != 3:
            raise ValueError('The number of channels of the input image must be 1 or 3.')
        if num_channels_std_map != 1 and num_channels_std_map != 3:
            raise ValueError('The number of channels of the standard deviation map must be 1 or 3.')
        self._name = denoiser_name
        self.__dict_input_channels = {'image': num_channels_image, 'std_map': num_channels_std_map}
        self._denoise_module = denoise_module

    def dict_input_channels(self) -> Dict[str, Any]:
        return self.__dict_input_channels

    def name(self) -> str:
        return self._name

    def forward(self, args_dict: Dict[str, Any] = None, **image_inputs: Tensor) -> Tensor:
        return self.call_denoiser(image_inputs['image'], image_inputs['std_map'])

    @abstractmethod
    def call_denoiser(self, image: Tensor, std_map: Tensor) -> Tensor:
        pass
'''
