from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Tuple, Optional, Type
if TYPE_CHECKING:
    from torch import Tensor

from abc import ABC, abstractmethod
from utils.image_utils import augment_img_tensor4
import torch.nn as nn
import torch
import numpy as np


class ImageModule(nn.Module, ABC):

    def module_type(self) -> Type:
        return type(self)  # Can be overriden by wrapper modules to return the type of the warped module instead.

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def dict_input_channels(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def forward(self, args_dict: Dict[str, Any] = None, **image_inputs: Tensor) -> Tensor:
        pass

    def num_channels(self, input_key: str) -> int:
        try:
            nc = self.dict_input_channels()[input_key]
        except KeyError:
            raise KeyError(f"The image module has no input key \'{input_key}\'. Accepted input keys are "
                           f"{list(self.dict_input_channels().keys())}.")
        return nc


'''
    def format_inputs_channels(self, inputs: Dict[str, Tensor]):
        for input_key, input_tensor in inputs.items():
            inputs[input_key] = convert_channels_number(input_tensor, self.num_channels(input_key))

    def required_inputs_keys(self) -> List[str]:
        return [*self._dict_input_channels()]

    def _check_required_inputs(self, inputs: Dict[str, Tensor]):
        """Check all required inputs were given."""
        missing_inputs_keys = [key for key in self.required_inputs_keys() if key not in inputs]
        if len(missing_inputs_keys) > 0:
            raise KeyError(
                f'The follwong required inputs are missing from the given inputs dictionary: {missing_inputs_keys}'
            )

    def forward(self, **inputs, check_inputs: bool=True) -> Tensor:
        if check_inputs:
            self._check_required_inputs(inputs)
        # Select only required inputs and force all input tensor to have 4 dimensions.
        # inputs = {key: view_as_tensor4(inputs[key]) for key in self.required_inputs_keys()}
        self.format_inputs_channels(inputs)
        return self._forward(**inputs)  # TODO?: format_outputs, e.g. channels conversions,...?
'''


class RunImageModuleStrategy(ABC):
    @abstractmethod
    def run_module(
            self,
            module: ImageModule,
            args_dict: Dict[str, Any],
            image_inputs: Dict[str, Tensor],
            scalar_inputs: Dict[str, Tensor]
    ) -> Tensor:
        pass


class DirectRun(RunImageModuleStrategy):
    """Direct run strategy: directly applies the model to the inputs."""
    def run_module(self, module, args_dict, image_inputs, scalar_inputs):
        return module(args_dict, **image_inputs, **scalar_inputs)


class SplitRun(RunImageModuleStrategy):
    """
    Split run strategy: splits the image inputs recursively to run the model on each portion and recompose the final
    image. Padding may be done at each portion to satisfy input size constraints of the model.
    """

    def __init__(self, refield: int = 32, min_size: int = 256, sf: int = 1, modulo: int = 16):
        self.refield = refield
        self.min_size = min_size
        self.sf = sf
        self.modulo = modulo

    def run_module(self, module, args_dict, image_inputs, scalar_inputs):
        return _test_split_fn(
            module, args_dict, image_inputs, scalar_inputs,
            self.refield, self.min_size, self.sf, self.modulo
        )


class PadRun(RunImageModuleStrategy):
    """
    Padding run strategy: pads the image before running the model to satisfy input size constraints of the model.
    """
    def __init__(self, modulo: int = 16):
        self.modulo = modulo

    def run_module(self, module, args_dict, image_inputs, scalar_inputs):
        h, w = next(iter(image_inputs.items()))[1].size()[-2:]
        return _test_pad(module, args_dict, image_inputs, scalar_inputs, h, w, self.modulo)[..., :h, :w]


def _test_pad(module, args_dict, image_inputs, scalar_inputs, h, w,  modulo):
    padBottom = int(np.ceil(h / modulo) * modulo - h)
    padRight = int(np.ceil(w / modulo) * modulo - w)
    L = {k: torch.nn.ReplicationPad2d((0, padRight, 0, padBottom))(image_inputs[k]) for k in image_inputs}
    E = module(args_dict, **L, **scalar_inputs)
    return E


def _test_split_fn(module, args_dict, image_inputs, scalar_inputs, refield, min_size, sf, modulo):
    """
    model: callable function/module to run.
    image_inputs: dictionary of input images (keys must match with keyword arguments in model).
    scalar_inputs: dictionary of input scalar data (keys must match with keyword arguments in model).
    refield: effective receptive filed of the network, 32 is enough.
    min_size: min_sizeXmin_size image, e.g., 256X256 image.
    sf: scale factor for super-resolution, otherwise 1.
    modulo: 1 if split
    """
    L = next(iter(image_inputs.items()))[1]
    h, w = L.size()[-2:]
    if h*w <= min_size**2:
        E = _test_pad(module, args_dict, image_inputs, scalar_inputs, h=h, w=w, modulo=modulo)[..., :h*sf, :w*sf]
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)

        Ls = [{k: image_inputs[k][..., y_slice, x_slice] for k in image_inputs}
              for (y_slice, x_slice) in ((top, left), (top, right), (bottom, left), (bottom, right))]

        if h * w <= 4*(min_size**2):
            Es = [module(args_dict, **Ls[i], **scalar_inputs) for i in range(4)]
        else:
            Es = [_test_split_fn(
                module, args_dict, Ls[i], scalar_inputs, refield=refield, min_size=min_size, sf=sf, modulo=modulo)
                for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



class ImageModuleWrapper(ImageModule):
    """
    Base Wrapper class for image processing modules suitable for modules that have images both at inputs and output.
    It augments a module with the following functionalities:
        - 8 times data augmentation may be applied before runnning the module and inverted at the output.
        One of 8 possibilities (flip, mirror, rotation, ...) is selected at each run.
        - Strategies to run models with constraints on the input images dimensions.
    """

    def __init__(self,
                 wrapped_module: ImageModule,
                 run_strategy: Optional[RunImageModuleStrategy] = None,
                 x8_augment: bool = True,
                 ):
        super(ImageModuleWrapper, self).__init__()
        self._module = wrapped_module
        self.x8_augment = x8_augment
        self.run_strategy = DirectRun() if run_strategy is None else run_strategy

    def forward(self, args_dict: Dict[str, Any] = None, **image_inputs: Tensor) -> Tensor:
        if args_dict is None:
            args_dict = {}
        iter_num = args_dict.get('iter_num', 0)
        image_inputs, scalar_inputs = self._separate_images_and_scalars(image_inputs)
        if self.x8_augment:
            x8_mode = iter_num % 8
            image_inputs = {k: augment_img_tensor4(image_inputs[k], mode=x8_mode) for k in image_inputs}
        output = self.run_strategy.run_module(self._module, args_dict, image_inputs, scalar_inputs)
        if self.x8_augment:
            x8_invert_mode = 8 - x8_mode if x8_mode == 3 or x8_mode == 5 else x8_mode
            output = augment_img_tensor4(output, mode=x8_invert_mode)
        return output

    @staticmethod
    def _separate_images_and_scalars(inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Separate image Tensors from scalar ones (i.e. singleton in spatial dimensions). The function assumes that the
        input tensors are already formatted as Tensors with 4 dimensions (batch,channels,y,x).
        An error is raised if several Tensors have different sizes in spatial dimensions (x,y).
        :param inputs:
        :return: Dictionaries of image Tensors and scalar Tensors (with same keys as the input dictionary).
        """
        spatial_dims = None
        is_scalar = {key: False for key in inputs}
        for key in inputs:
            t_spatial_dims = list(inputs[key].shape[-2:])
            if t_spatial_dims == [1, 1]:
                is_scalar[key] = True
            elif t_spatial_dims != spatial_dims:
                if spatial_dims is not None:
                    raise ValueError('The image module wrapper is not compatible for several input '
                                     'images with different and non-singleton spatial dimensions.')
                spatial_dims = t_spatial_dims
        image_inputs_dict = {key: inputs[key] for key in inputs if not is_scalar[key]}
        scalar_inputs_dict = {key: inputs[key] for key in inputs if is_scalar[key]}
        return image_inputs_dict, scalar_inputs_dict

    def dict_input_channels(self) -> Dict[str, int]:
        return self._module.dict_input_channels()

    def name(self):
        return self._module.name()

    def module_type(self) -> Type:
        return self._module.module_type()
