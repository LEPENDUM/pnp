from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
if TYPE_CHECKING:
    from pnpcore import Task, Preconditioner, Regularizer
    from torch import Tensor

import torch
import math
from pnpcore import Solver, no_preco, AlgorithmStopped
import warnings


class ADMMSolver(Solver):
    """
    ADMM Solver implementation for inverse problems (includes preconditioning).
    The ADMM hyper-parameters are:
     * sigma_den_0: standard dev. parameter of the denoiser at the first iteration.
     * sigma_den_N: standard dev. parameter of the denoiser at the last iteration.
     * min_reg_weight_sqrt: minimum value for the square root of the regularizer weight in the minimization.
       This parameter is needed for the case where the degradation model (i.e. defined by the task) does not include the
       application of noise. In theory, the regularization weight should be zero in this case in the ADMM formulation,
       which is a problem in practice since regularization is still needed.
       A small but non-zero value of 'min_reg_weight_sqrt' makes it possible to always use regularization.
       By default, this parameter is set to either 1/255 or sigma_den_N if sigma_den_N<1/255.
    """

    _lagrange_mult: Optional[Tensor]
    _x_data_prox: Optional[Tensor]

    def __init__(
            self,
            sigma_den_0: float,
            sigma_den_N: float,
            min_reg_weight_sqrt: float = None,
            task: Task = None,
            regularizer: Regularizer = None,
            preconditioner: Preconditioner = no_preco
    ):
        super(ADMMSolver, self).__init__()
        if sigma_den_0 <= 0:
            raise ValueError('ADMM parameter sigma_den_0 should be strictly positive.')
        if sigma_den_N <= 0:
            raise ValueError('ADMM parameter sigma_den_N should be strictly positive.')
        if sigma_den_N > sigma_den_0:
            warnings.warn(
                f'The ADMM parameter sigma_den_N={sigma_den_N} (denoiser s.t.d. at last iteration) is set higher than '
                f'sigma_den_0={sigma_den_0}. This might be a wrong setting (will result internally in decreasing the '
                f'ADMM penalty parameter at each iteration).')
        if min_reg_weight_sqrt is None:
            min_reg_weight_sqrt = min(1 / 255, sigma_den_N)
        if min_reg_weight_sqrt < 0:
            raise ValueError('ADMM parameter min_reg_weight_sqrt should be non-negative.')


        self.task = task
        self.regularizer = regularizer
        self.preconditioner = preconditioner
        self.sigma_den_0 = sigma_den_0
        self.sigma_den_N = sigma_den_N
        self.min_reg_weight_sqrt = min_reg_weight_sqrt
        self._reg_weight_sqrt = None
        self._rho = None
        self._rho_0 = None
        self._alpha = None
        self._lagrange_mult = None
        self._x_data_prox = None

    def name(self):
        return 'admm'

    def use_task_gradients(self) -> bool:
        return False

    def use_task_proximal(self) -> bool:
        return True

    def params_strings(self) -> List[str]:
        sigs_str = f'sig255=[{self.sigma_den_0*255:g}-{self.sigma_den_N*255:g}]'
        return [sigs_str] + self.task.pre_post_denoise_strings()

    @property
    def rho_0(self):
        return self._rho_0

    @property
    def alpha(self):
        return self._alpha

    def sigma_denoise(self, iteration: int) -> float:
        return self._reg_weight_sqrt / math.sqrt(self._rho_0 * self._alpha ** (iteration - 1))

    def prepare_solver_data(self):
        self.compute_params()
        self._lagrange_mult = torch.zeros(self._Px.shape, device=self._Px.device)
        self._rho = self._rho_0


    def compute_params(self):
        if self.min_reg_weight_sqrt == 0 and self.task.data_term_inverse_weight == 0:
            raise ValueError('The ADMM requires a small but non-zero parameter value \'min_reg_weight_sqrt\' '
                             'to handle constrained tasks (i.e. with infinite data term weight).')
        self._reg_weight_sqrt = max(self.min_reg_weight_sqrt, math.sqrt(self.task.data_term_inverse_weight))
        if self._reg_weight_sqrt > self.sigma_den_N:
            warnings.warn(
                f'The ADMM parameter sigma_den_N={self.sigma_den_N} (denoiser s.t.d. at last iteration) is lower than '
                f'the square root of the regularization weight (={self._reg_weight_sqrt}). '
                f'This might be a wrong setting (the last iterations are unlikely to improve the solution).')
        self._rho_0 = (self._reg_weight_sqrt / self.sigma_den_0) ** 2
        if math.isinf(self.max_iterations) or self.max_iterations <= 1:
            self._alpha = 1
        else:
            self._alpha = (self.sigma_den_0 / self.sigma_den_N) ** (2 / (self.max_iterations-1))



    def iterate(self):
        # Step 1: Apply Data term proximal operator
        self._x_data_prox = self.task.proximal_operator(self._x - self._lagrange_mult / self._rho, self._rho)

        # Step 2: Apply regularization term proximal operator
        self._x = self._x_data_prox + self._lagrange_mult / self._rho
        self._Px = self.preconditioner(self._x)
        self.task.pre_denoising(self._Px)
        sig_denoise = self.sigma_denoise(self.current_iteration)
        self._Px = self.regularizer.denoise_operator(
            Pu=self._Px, sigma=sig_denoise, u=self._x, iter_num=self.current_iteration)
        self.task.post_denoising(self._Px)

        if self._current_iteration == self.max_iterations:
            raise AlgorithmStopped
        # '''
        self.preconditioner.update(self)
        # Theoretically, this step is part of the regularizer's proximal operator computation
        # but it is better to apply it after the preconditioner update.
        self._x = self.preconditioner.inverse(self._Px)

        # Step 3: Lagrangian update
        self._lagrange_mult += self._rho * (self._x_data_prox - self._x)
        '''
        self._x_data_prox = self.preconditioner(self._x_data_prox, use_optim_msg=False)
        self._lagrange_mult = self.preconditioner(self._lagrange_mult, use_optim_msg=False)
        self.preconditioner.update(self)
        self._x = self.preconditioner.inverse(self._Px)
        self._lagrange_mult += self._rho * (self._x_data_prox - self._Px)
        self._lagrange_mult = self.preconditioner.inverse(self._lagrange_mult)
        '''

        # Step 4: penalty parameter update
        self._rho *= self._alpha
