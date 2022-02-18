from __future__ import annotations
from typing import TYPE_CHECKING, List, Type
if TYPE_CHECKING:
    from torch import Tensor
    from utils.random import RandomGenerator
    from models.image_module import ImageModule

from pnpcore import Task, PrecoCreatorForTask, DiagonalPreconditioner
from utils.random import TorchRandomGenerator
from utils.filenames import join_names
from utils.image_utils import imread, imsave, convert_channels_number
from regularizers.denoise_reg import DenoiserIID

import math
import torch
import os


class PoissonDenoise(Task):

    def __init__(self,
                 peak: float,
                 random_generator: RandomGenerator = TorchRandomGenerator(),
                 gen_reproducible_inputs: bool = True):
        super(PoissonDenoise, self).__init__()
        self.check_peak(peak)
        self.register_buffer('_noisy_image', None, persistent=False)
        self.peak = peak
        self.random_generator = random_generator
        self.gen_reproducible_inputs = gen_reproducible_inputs

    @staticmethod
    def check_peak(peak: float):
        if peak <= 0:
            raise ValueError('The parameter peak for the task PoissonDenoise should be a strictly positive value.')

    def name(self):
        return 'poisson-denoise'

    def params_strings(self) -> List[str]:
        return [f'peak={self.peak:g}']


    def _proximal_operator(self, u: Tensor, weight: float) -> Tensor:
        P = self.get_preconditioner().multiplier()
        x_prox = - (P / self.peak - (weight * u)) * (1 / 2 / weight)
        x_prox += torch.sqrt(x_prox ** 2 + self._noisy_image * (1 / self.peak / weight))
        return x_prox

    @property
    def data_term_inverse_weight(self) -> float:
        return 1.0

########################################################################################################################
    # TODO: this is not really clean (maybe instead give a scale value in denoiser that scales both the image and
    #  the noise s.t.d. AND output data in range [0,255])
    #  ==> the task would need an access to the regularizer at some point??

    def pre_denoising(self, current_estimate: Tensor) -> None:
        self.get_preconditioner()._P /= self.peak

    def post_denoising(self, current_estimate: Tensor) -> None:
        self.get_preconditioner()._P *= self.peak
        # current_estimate.clamp_(0, 1)

########################################################################################################################

    def preco_types_for_prox(self) -> List[Type]:
        return [DiagonalPreconditioner]

    def set_required_inputs(self, inputs: dict) -> None:
        self._noisy_image = inputs['noisy_image']
        peak = inputs['peak']
        self.check_peak(peak)
        self.peak = peak

    def required_inputs_keys(self) -> List:
        return ['noisy_image', 'peak']

    def generate_required_inputs(self, ground_truth: Tensor) -> dict:
        if self.gen_reproducible_inputs:
            self.random_generator.seed(0)
        noisy_image = self.random_generator.poisson(ground_truth * self.peak) / self.peak
        return {'noisy_image': noisy_image, 'peak': self.peak}

    def filename(self, directory: str, name_prefix: str, extension: str):
        return os.path.join(directory, join_names(name_prefix, self.params_strings())) + extension

    def save_inputs_at_batch_id(self, directory: str, name_prefix: str, extension: str, batch_id: int) -> None:
        imsave(self._noisy_image[batch_id:batch_id+1], self.filename(directory, name_prefix, extension))

    def load_required_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        return {
            'noisy_image': imread(self.filename(directory, name_prefix, extension)),
            'peak': self.peak
        }

    def reformat_ground_truth(self, ground_truth: Tensor) -> Tensor:
        return ground_truth


########################################################################################################################
#                                           Preconditioner Creator
########################################################################################################################

class PoissonPrecoCreator(PrecoCreatorForTask):
    def __init__(self, do_update: bool = True, min_preco: float = 1e-3):
        super(PoissonPrecoCreator, self).__init__()
        self.do_update = do_update
        if min_preco <= 0 or min_preco > 1:
            raise ValueError(
                'The minimum preconditioning value \'min_preco\' should be strictly positive and lower than 1.')
        self.min_preco = min_preco

    def name(self) -> str:
        return 'poisson-preco'

    def params_strings(self) -> List[str]:
        return []

    def create_preco_data(self, solver, preco) -> None:
        preco._P = torch.sqrt(solver.current_estimate.clamp(self.min_preco, 1) * solver.task.peak)

    def update_preco_data(self, solver, preco) -> None:
        if self.do_update:
            self.create_preco_data(solver, preco)
        else:
            preco.skip_update_callbacks()

    @staticmethod
    def task_type():
        return PoissonDenoise

    @staticmethod
    def preconditioner_type():
        return DiagonalPreconditioner


########################################################################################################################
#                                           Initialization functions
########################################################################################################################

def init_noisy(inputs):
    return inputs['noisy_image']


class InitAnscombeDenoise:
    def __init__(self, denoiser: ImageModule):
        if not issubclass(denoiser.module_type(), DenoiserIID):
            raise TypeError(f'The initilization requires a denoiser of type \'DenoiserIID\'. '
                            f'Received a module of type \'{denoiser.module_type().__name__}\' instead.')
        self.denoiser = denoiser

    def get_device(self):
        try:
            p = next(self.denoiser.named_parameters())
            return p[1].device
        except StopIteration:
            return None

    def __call__(self, inputs) -> Tensor:
        device = self.get_device()
        img = inputs['noisy_image'].to(device)
        peak = inputs['peak']
        PoissonDenoise.check_peak(peak)
        max_val = 2 * math.sqrt(peak + 3 / 8)
        img = 2 * torch.sqrt(img * peak + 3 / 8) / max_val
        img = convert_channels_number(img, self.denoiser.num_channels('image'))
        img = self.denoiser(image=img, args_dict={'sigma': 1/max_val})
        img = ((img * max_val / 2) ** 2 - 3 / 8) / peak
        img = img.clamp(0, 1)
        return img
