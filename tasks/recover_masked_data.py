from __future__ import annotations
from typing import TYPE_CHECKING, List, Type, Tuple
if TYPE_CHECKING:
    from torch import Tensor
    from utils.random import RandomGenerator

from abc import ABC, abstractmethod
from pnpcore import Task, PrecoCreatorForTask, DiagonalPreconditioner
from utils.random import TorchRandomGenerator
from utils.filenames import join_names

from utils.image_utils import imread, imsave
from utils.fourier_utils import gen_gaussian2d_fourier, convolve_old
import math
import torch
import torch.nn as nn
import numpy as np
import cv2
import os


class RecoverMaskedData(Task):

    def __init__(
            self,
            mask_generator: MaskGenerator,
            sigma_noise: float = None,
            random_generator: RandomGenerator = TorchRandomGenerator(),
            gen_reproducible_inputs: bool = True
    ):
        super(RecoverMaskedData, self).__init__()
        if sigma_noise is not None and sigma_noise < 0:
            raise ValueError(
                'The parameter sigma_noise (noise standard deviation of known samlpes) '
                'for the task RecoverMaskedData should be non-negative.')
        self.register_buffer('_samples', None, persistent=False)
        self.register_buffer('_mask', None, persistent=False)
        self._mask_generator = mask_generator
        self.register_optional_input_attr('_sigma_noise', 'sigma_noise', 0, sigma_noise)
        self.random_generator = random_generator
        self._mask_generator.set_random_generator(random_generator)
        self.gen_reproducible_inputs = gen_reproducible_inputs
        self.__PtAt = None
        self.__PtAtAP = None
        self.__pre_denoise = None

    def _no_denoise_known_px(self) -> bool:
        return self._sigma_noise == 0

    def name(self):
        return self._mask_generator.name()

    def params_strings(self):
        return self._mask_generator.params_strings() + [self.sigma_str()]

    def pre_post_denoise_strings(self):
        if self._no_denoise_known_px():
            return ['no-denoise-known-px']
        return []

    def missing_config_elements_msgs(self) -> List[str]:
        msgs = super(RecoverMaskedData, self).missing_config_elements_msgs()
        if self._mask_generator is None:
            msgs.append('no mask generator specified for the task RecoverMaskedData')
        return msgs

    @property
    def data_term_inverse_weight(self) -> float:
        return self._sigma_noise ** 2

    def preco_types_for_prox(self) -> List[Type]:
        return [DiagonalPreconditioner]

    def _proximal_operator(self, u: Tensor, weight: float) -> Tensor:
        return (self.__PtAt * self._samples + weight * u) / (self.__PtAtAP + weight)

    def update_prox_on_preco_change(self) -> None:
        self.__PtAt = self._mask * self.get_preconditioner().multiplier()
        self.__PtAtAP = self.__PtAt ** 2

    def pre_denoising(self, current_estimate: Tensor) -> None:
        if self._no_denoise_known_px():
            self.__pre_denoise = current_estimate * self._mask

    def post_denoising(self, current_estimate: Tensor) -> None:
        if self._no_denoise_known_px():
            current_estimate *= 1 - self._mask
            current_estimate += self.__pre_denoise

    def post_process(self, current_estimate: Tensor) -> None:
        if self._sigma_noise == 0:
            current_estimate *= 1 - self._mask
            current_estimate += self._samples

    def reformat_ground_truth(self, ground_truth: Tensor) -> Tensor:
        return ground_truth  # No reformating needed for this task.

    def required_inputs_keys(self) -> List:
        # Only inlcude the mask generator's inputs without adding the sigma_noise parameter which is optional.
        return self._mask_generator.required_inputs_keys()

    def generate_required_inputs(self, ground_truth: Tensor) -> dict:
        """Generate inputs from ground truth data for the task RecoverMaskedData.
        :param ground_truth: Ground truth data as a Tensor of floats (with batch dimension first).
        :return: Dictionary of inputs to be given to the solver for reconstruction simulations.
        The content of the inputs dictionary is controlled by the mask generator object given to the task.
        """
        if self.gen_reproducible_inputs:
            self.random_generator.seed(0)
        if self._sigma_noise > 0:
            ground_truth = ground_truth + self.random_generator.normal(
                ground_truth.shape, std=self._sigma_noise, device=ground_truth.device)

        mask = self._mask_generator.generate_mask(ground_truth)
        return self.samples_and_mask_to_inputs(ground_truth * mask, mask)

    def samples_and_mask_to_inputs(self, samples: Tensor, mask: Tensor) -> dict:
        inputs = self._mask_generator.samples_and_mask_to_inputs(samples, mask)
        return inputs

    def inputs_to_samples_and_mask(self, inputs: dict) -> Tuple[Tensor, Tensor]:
        return self._mask_generator.inputs_to_samples_and_mask(inputs)

    def set_required_inputs(self, inputs: dict) -> None:
        self._mask_generator.set_params_from_inputs(inputs)
        self._samples, self._mask = self._mask_generator.inputs_to_samples_and_mask(inputs)

    def save_inputs_at_batch_id(self, directory: str, name_prefix: str, extension: str, batch_id: int) -> None:
        inputs = self.samples_and_mask_to_inputs(self._samples[batch_id:batch_id + 1], self._mask[batch_id:batch_id + 1])
        self._mask_generator.save_inputs(directory, join_names(name_prefix, self.sigma_str()), extension, inputs)

    def load_required_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        inputs = self._mask_generator.load_inputs(directory, join_names(name_prefix, self.sigma_str()), extension)
        return inputs

    def sigma_str(self):
        if self._sigma_noise > 0:
            return f'sig255-noise={self._sigma_noise*255:g}'
        return ''


########################################################################################################################
#                               Preconditioner creator for masked data recovery
########################################################################################################################

class MaskPrecoCreator(PrecoCreatorForTask):

    def __init__(self, p_max, sigma_blur_last_iter):
        super(MaskPrecoCreator, self).__init__()
        if p_max <= 1:
            raise ValueError("In MaskPrecoCreator, the maximum preconditing "
                             "value \'p_max\' must be strictly higer than 1.")
        if sigma_blur_last_iter < 0:
            raise ValueError("In MaskPrecoCreator, the blurring parameter at the last "
                             "iteration \'sigma_blur_last_iter\' must be non-negative.")
        self._epsilon = 1 / (p_max-1)
        self._sigma_blur_last_iter = sigma_blur_last_iter
        self._mask_blur = None
        self._gauss_filter = None
        self._sig_gauss = 0

    def name(self) -> str:
        return 'mask-preco'

    def params_strings(self) -> List[str]:
        p_max = (1 + self._epsilon) / self._epsilon
        return [f'siglast={self._sigma_blur_last_iter:g}', f'pmax={p_max:g}']

    def _mask_to_preco(self, ref_val=1):
        return (ref_val + self._epsilon) / (self._mask_blur + self._epsilon)


    def create_preco_data(self, solver, preco) -> None:
        self._mask_blur = solver.task._mask
        preco._P = self._mask_to_preco()
        self._sig_gauss = self._sigma_blur_last_iter / math.sqrt(solver.max_iterations)
        if self._sig_gauss > 0:
            self._gauss_filter = gen_gaussian2d_fourier(
                self._mask_blur.shape[-1], self._mask_blur.shape[-2], self._sig_gauss,
                half_last_dim=False, device=self._mask_blur.device)

    def update_preco_data(self, solver, preco) -> None:
        if self._sig_gauss > 0:
            self._mask_blur = convolve_old(self._mask_blur, self._gauss_filter)
            max_val = self._mask_blur.amax(dim=tuple(range(1, self._mask_blur.ndim)), keepdim=True)
            preco._P = self._mask_to_preco(ref_val=max_val)
        else:
            preco.skip_update_callbacks()  # prevent unnecessary update computations in the rest of the solver

    @staticmethod
    def task_type():
        return RecoverMaskedData

    @staticmethod
    def preconditioner_type():
        return DiagonalPreconditioner


########################################################################################################################


class MaskGenerator(ABC):
    """
    Base Mask generator class for the RecoverMaskedData task.
    """
    def __init__(self):
        self._rng = None

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def params_strings(self) -> List[str]:
        pass

    def set_random_generator(self, rng: RandomGenerator) -> None:
        self._rng = rng

    @abstractmethod
    def generate_mask(self, ground_truth_tensor: Tensor) -> Tensor:
        pass

    @abstractmethod
    def required_inputs_keys(self) -> List:
        pass

    @abstractmethod
    def inputs_to_samples_and_mask(self, inputs: dict) -> Tuple[Tensor, Tensor]:
        """
        Retrieves the mask and samples from the inputs dictionary.
        :param inputs: dictionary of input data in the format accepted by the MaskGenerator instance.
        :return: Sequence of the samples Tensor and the mask Tensor (in that order).
        """
        pass

    @abstractmethod
    def samples_and_mask_to_inputs(self, samples: Tensor, mask: Tensor) -> dict:
        """
        Creates the inputs dictionary from the samples and mask Tensors.
        :return: dictionary of input data in the format accepted by the MaskGenerator instance.
        """
        pass

    @abstractmethod
    def set_params_from_inputs(self, inputs: dict) -> None:
        """
        Sets the mask generator parameters from the inputs (if needed).
        """
        pass

    @abstractmethod
    def save_inputs(self, directory: str, name_prefix: str, extension: str, inputs) -> None:
        pass

    @abstractmethod
    def load_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        pass


########################################################################################################################
#                                       Image completion with random mask
########################################################################################################################

class ImageCompletionMaskGenerator(MaskGenerator):
    """
    Mask generator for image completion.
    Generates masks of known pixels uniformly sampled at random with a rate 'rate_known' (between 0 and 1).
    """

    def __init__(self, rate_known: float, colored_mask: bool = False):
        super(ImageCompletionMaskGenerator, self).__init__()
        if rate_known < 0 or rate_known > 1:
            raise ValueError('rate_known (percentage of known pixels in the mak generation) should be between 0 and 1.')
        self.rate_known = rate_known
        self.colored_mask = colored_mask

    def name(self) -> str:
        return 'completion'

    def params_strings(self) -> List[str]:
        return [f'r={100 * self.rate_known:.3g}']

    def generate_mask(self, ground_truth_tensor: Tensor) -> Tensor:
        mask_shape = list(ground_truth_tensor.shape)
        if not self.colored_mask:
            mask_shape[1] = 1
        mask = self._rng.uniform(mask_shape, device=ground_truth_tensor.device)
        mask = (mask > (1 - self.rate_known)).type_as(ground_truth_tensor)
        return mask

    def required_inputs_keys(self) -> List:
        return ['samples', 'mask']

    def inputs_to_samples_and_mask(self, inputs: dict) -> Tuple[Tensor, Tensor]:
        return inputs['samples'], inputs['mask']

    def samples_and_mask_to_inputs(self, samples: Tensor, mask: Tensor) -> dict:
        return {'samples': samples, 'mask': mask}

    def set_params_from_inputs(self, inputs: dict) -> None:
        pass

    def file_names(self, directory: str, name_prefix: str) -> dict:
        file_prefix = os.path.join(directory, join_names(name_prefix, self.params_strings()))
        return {
            'mask': join_names(file_prefix, 'mask'),
            'samples': file_prefix
        }

    def save_inputs(self, directory: str, name_prefix: str, extension: str, inputs) -> None:
        files_dict = self.file_names(directory, name_prefix)
        for key in files_dict:
            imsave(inputs[key], files_dict[key] + extension)

    def load_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        files_dict = self.file_names(directory, name_prefix)
        inputs = {key: imread(files_dict[key] + extension) for key in files_dict}
        return inputs


########################################################################################################################
#                                               Demosaicing
########################################################################################################################

class BayerMaskGenerator(MaskGenerator):
    """
    Mask generator with Bayer pattern for demosaicing task.
    """

    def __init__(self, pattern='rggb'):
        super(BayerMaskGenerator, self).__init__()
        pattern = pattern.lower()
        self.check_pattern(pattern)
        self.pattern = pattern

    @staticmethod
    def check_pattern(pattern: str):
        patterns = ['rggb', 'bggr', 'grbg', 'gbrg']
        if pattern not in patterns:
            raise ValueError(f'Unknown Bayer pattern \'{pattern}\'. Possible patterns are {patterns}')

    @staticmethod
    def bayer_mask(img_shape, pattern: str, device=None) -> Tensor:
        if img_shape[1] != 3:
            raise ValueError('The image should have 3 color channels for generating the corresponding bayer mask.')
        channel_order = {'r': 0, 'g': 1, 'b': 2}
        mask = torch.zeros(img_shape, device=device)
        mask[:, channel_order[pattern[0]], 0::2, 0::2] = 1
        mask[:, channel_order[pattern[1]], 0::2, 1::2] = 1
        mask[:, channel_order[pattern[2]], 1::2, 0::2] = 1
        mask[:, channel_order[pattern[3]], 1::2, 1::2] = 1
        return mask

    def name(self) -> str:
        return 'demosaic-bayer'

    def params_strings(self) -> List[str]:
        return [self.pattern]

    def generate_mask(self, ground_truth_tensor: Tensor) -> Tensor:
        return self.bayer_mask(ground_truth_tensor.shape, self.pattern, ground_truth_tensor.device)

    def required_inputs_keys(self) -> List:
        return ['samples', 'pattern']

    def inputs_to_samples_and_mask(self, inputs: dict) -> Tuple[Tensor, Tensor]:
        self.check_pattern(inputs['pattern'])
        samples = inputs['samples']
        return samples, self.bayer_mask(samples.shape, inputs['pattern'], device=samples.device)

    def samples_and_mask_to_inputs(self, samples: Tensor, mask: Tensor) -> dict:
        return {'samples': samples, 'pattern': self.pattern}

    def set_params_from_inputs(self, inputs: dict) -> None:
        pattern = inputs['pattern']
        self.check_pattern(pattern)
        self.pattern = pattern

    def file_names(self, directory: str, name_prefix: str) -> dict:
        file_prefix = os.path.join(directory, join_names(name_prefix, self.params_strings()))
        return {'samples': file_prefix}

    def save_inputs(self, directory: str, name_prefix: str, extension: str, inputs) -> None:
        files_dict = self.file_names(directory, name_prefix)
        imsave(inputs['samples'], files_dict['samples'] + extension)

    def load_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        files_dict = self.file_names(directory, name_prefix)
        return {
            'samples': imread(files_dict['samples'] + extension),
            'pattern': self.pattern
        }


########################################################################################################################
#                                               2D Grid interpolation
########################################################################################################################

class GridMaskGenerator(MaskGenerator):
    """
    Mask generator with grid patterns for the image interpolation task.
    """

    def __init__(self, sampling_factor: int):
        super(GridMaskGenerator, self).__init__()
        self.check_factor(sampling_factor)
        self.sampling_factor = sampling_factor

    @staticmethod
    def check_factor(sampling_factor: int):
        if sampling_factor < 1:
            raise ValueError(f'The sampling factor must be a strictly positive integer.')

    @staticmethod
    def check_size_compatibility(lr_x: int, lr_y: int, hr_x: int, hr_y: int, sampling_factor: int):
        border_x = lr_x * sampling_factor - hr_x
        border_y = lr_y * sampling_factor - hr_y
        if border_x < 0 or border_x >= sampling_factor or border_y < 0 or border_y >= sampling_factor:
            raise ValueError(f'Incompatibility between input (low resolution) size, target (high resolution) size '
                             f'and sampling factor.')

    @staticmethod
    def grid_mask(fullsize_shape, sampling_factor: int, device=None) -> Tensor:
        mask = torch.zeros(fullsize_shape[0], 1, fullsize_shape[2], fullsize_shape[3], device=device)
        mask[:, :, 0::sampling_factor, 0::sampling_factor] = 1
        return mask

    def name(self) -> str:
        return 'grid-interpolation'

    def params_strings(self) -> List[str]:
        return [f'x{self.sampling_factor}']

    def generate_mask(self, ground_truth_tensor: Tensor) -> Tensor:
        return self.grid_mask(ground_truth_tensor.shape, self.sampling_factor, device=ground_truth_tensor.device)

    def required_inputs_keys(self) -> List:
        return ['samples', 'factor', 'fullsize_x', 'fullsize_y']

    def inputs_to_samples_and_mask(self, inputs: dict) -> Tuple[Tensor, Tensor]:
        lr, factor, fullsize_x, fullsize_y = (inputs[k] for k in ('samples', 'factor', 'fullsize_x', 'fullsize_y'))
        self.check_factor(factor)
        self.check_size_compatibility(lr.shape[3], lr.shape[2], fullsize_x, fullsize_y, factor)
        samples = torch.zeros(lr.shape[0], lr.shape[1], fullsize_y, fullsize_x)
        samples[:, :, 0::factor, 0::factor] = lr
        return samples, self.grid_mask(samples.shape, factor, device=samples.device)

    def samples_and_mask_to_inputs(self, samples: Tensor, mask: Tensor) -> dict:
        low_res = samples[:, :, 0::self.sampling_factor, 0::self.sampling_factor]
        return {'samples': low_res, 'factor': self.sampling_factor,
                'fullsize_x': samples.shape[3], 'fullsize_y': samples.shape[2]}

    def set_params_from_inputs(self, inputs: dict) -> None:
        factor = inputs['factor']
        self.check_factor(factor)
        self.sampling_factor = factor

    def file_names(self, directory: str, name_prefix: str) -> dict:
        file_prefix = os.path.join(directory, join_names(name_prefix, self.params_strings()))
        return {'samples': file_prefix}

    def save_inputs(self, directory: str, name_prefix: str, extension: str, inputs) -> None:
        files_dict = self.file_names(directory, name_prefix)
        for key in files_dict:
            imsave(inputs[key], files_dict[key] + extension)

    def load_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        files_dict = self.file_names(directory, name_prefix)
        samples = imread(files_dict['samples'] + extension)
        return {
            'samples': samples,
            'factor': self.sampling_factor,
            'fullsize_x': self.sampling_factor * samples.shape[3],
            'fullsize_y': self.sampling_factor * samples.shape[2]
        }

########################################################################################################################
#                                           Initialization functions
########################################################################################################################


class InitMaskFill:
    """
    Initialization callable class for filling masked data with a constant value.
    """
    def __init__(self, mask_generator: MaskGenerator, fill_value: float = 0):
        self.fill_value = fill_value
        self.mask_generator = mask_generator
    def __call__(self, inputs) -> Tensor:
        samples, mask = self.mask_generator.inputs_to_samples_and_mask(inputs)
        return samples * mask + (1 - mask) * self.fill_value


class InitDemosaic:
    """
    Initialization callable class for demosaicing problem with Bayer patterns.
    """
    def __call__(self, inputs) -> Tensor:
        pattern = inputs['pattern']
        samples = inputs['samples']
        BayerMaskGenerator.check_pattern(pattern)
        mask = BayerMaskGenerator.bayer_mask(samples.shape, pattern, device=samples.device)
        cfa = torch.sum(samples * mask, 1, keepdim=True)
        rgb = cfa.repeat(1, 3, 1, 1)
        cfa = nn.functional.pad(cfa, (2, 2, 2, 2), mode='reflect')

        kgrb = 1 / 8 * torch.FloatTensor(
            [[0,    0, -1,  0,  0],
             [0,    0,  2,  0,  0],
             [-1,   2,  4,  2, -1],
             [0,    0,  2,  0,  0],
             [0,    0, -1,  0,  0]]).type_as(rgb)
        krbg0 = 1 / 8 * torch.FloatTensor(
            [[0,    0, 1 / 2,   0,  0],
             [0,   -1,     0,  -1,  0],
             [-1,   4,     5,   4, -1],
             [0,   -1,     0,  -1,  0],
             [0,    0, 1 / 2,   0,  0]]).type_as(rgb)
        krbg1 = krbg0.t()
        krbbr = 1 / 8 * torch.FloatTensor(
            [[0,        0,  -3 / 2, 0,      0],
             [0,        2,       0, 2,      0],
             [-3 / 2,   0,       6, 0, -3 / 2],
             [0,        2,       0, 2,      0],
             [0,        0,  -3 / 2, 0,      0]]).type_as(rgb)

        k = torch.stack((kgrb, krbg0, krbg1, krbbr), 0).unsqueeze(1)

        conv_cfa = nn.functional.conv2d(cfa, k, padding=0, bias=None)

        yx_offset = {'rggb': (0, 0), 'bggr': (1, 1), 'grbg': (0, 1), 'gbrg': (1, 0)}
        y0 = yx_offset[pattern][0]
        y1 = (1 + yx_offset[pattern][0]) % 2
        x0 = yx_offset[pattern][1]
        x1 = (1 + yx_offset[pattern][1]) % 2

        # fill G
        rgb[:, 1, y0::2, x0::2] = conv_cfa[:, 0, y0::2, x0::2]
        rgb[:, 1, y1::2, x1::2] = conv_cfa[:, 0, y1::2, x1::2]

        # fill R
        rgb[:, 0, y0::2, x1::2] = conv_cfa[:, 1, y0::2, x1::2]
        rgb[:, 0, y1::2, x0::2] = conv_cfa[:, 2, y1::2, x0::2]
        rgb[:, 0, y1::2, x1::2] = conv_cfa[:, 3, y1::2, x1::2]

        # fill B
        rgb[:, 2, y0::2, x1::2] = conv_cfa[:, 2, y0::2, x1::2]
        rgb[:, 2, y1::2, x0::2] = conv_cfa[:, 1, y1::2, x0::2]
        rgb[:, 2, y0::2, x0::2] = conv_cfa[:, 3, y0::2, x0::2]

        return rgb


class InitImageInterp:
    """
    Initialization callable class for grid interpolation task.
    """
    def __init__(self, interp_mode: str = 'bicubic'):
        cv_flags = {'bilinear':     cv2.INTER_LINEAR,
                    'nearest':      cv2.INTER_NEAREST,
                    'bicubic':      cv2.INTER_CUBIC,
                    'lanczos4':     cv2.INTER_LANCZOS4,
                    'area':         cv2.INTER_AREA,
                    }
        if interp_mode not in cv_flags:
            raise KeyError(f'Unkwon interpolation mode \'{interp_mode}\'. Available modes are: {list(cv_flags.keys())}')
        self.interp_flag = cv_flags[interp_mode]

    def __call__(self, inputs) -> Tensor:
        lr, factor, fullsize_x, fullsize_y = (inputs[k] for k in ('samples', 'factor', 'fullsize_x', 'fullsize_y'))
        GridMaskGenerator.check_factor(factor)
        GridMaskGenerator.check_size_compatibility(lr.shape[3], lr.shape[2], fullsize_x, fullsize_y, factor)
        X, Y = np.meshgrid(np.linspace(0, (fullsize_x - 1) / factor, fullsize_x),
                           np.linspace(0, (fullsize_y - 1) / factor, fullsize_y))
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        lr = lr.permute(2, 3, 1, 0).cpu().numpy()
        interp = torch.empty(fullsize_y, fullsize_x, lr.shape[2], lr.shape[3])
        for batch in range(lr.shape[3]):
            interp[:, :, :, batch] = torch.from_numpy(cv2.remap(
                lr[:, :, :, batch], map1=X, map2=Y, interpolation=self.interp_flag, borderMode=cv2.BORDER_REPLICATE
            ))

        return interp.permute(3, 2, 0, 1)
