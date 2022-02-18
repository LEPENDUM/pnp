from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Union
if TYPE_CHECKING:
    from torch import Tensor

from abc import ABC, abstractmethod

from pnpcore import Task, Preconditioner, Regularizer, no_preco
from pnpcore import AlgorithmStopped, ConfigurationError
import torch.nn as nn
import torch


class Solver(ABC, nn.Module):
    """Abstract Base Solver class defining common structure for any solver."""

    task: Optional[Task]
    regularizer: Optional[Regularizer]
    preconditioner: Preconditioner
    _x: Optional[Tensor]
    _Px: Optional[Tensor]

    def __init__(self):
        super(Solver, self).__init__()
        self._x = None  # current estimate variable (with preconditioning)
        self._Px = None  # current estimate variable
        self._current_iteration = 0
        self.task = None
        self.regularizer = None
        self.preconditioner = no_preco
        self._max_iterations = None
        self.__device = None

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def params_strings(self) -> List[str]:
        pass

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @property
    def current_estimate(self) -> Tensor:
        return self._Px

    @abstractmethod
    def sigma_denoise(self, iteration: int) -> float:
        """
        standard deviation parameter used by the denoiser at a given iteration, for an algorithm based on a denoiser
        (with iteration numbers starting from 1).
        """
        pass

    @abstractmethod
    def prepare_solver_data(self) -> None:
        pass

    @abstractmethod
    def use_task_gradients(self) -> bool:
        pass

    @abstractmethod
    def use_task_proximal(self) -> bool:
        pass

    @abstractmethod
    def iterate(self) -> None:
        """
        Performs an iteration of the solver. The method may raise an AlgorithmStopped exception either to stop the
        algorithm before the maximum number of iterations, or to prevent the execution of unnecessary steps of the
        last iteration that do not change the estimate directly.
        """
        pass

    @property
    def device(self) -> Optional[Union[int, torch.device]]:
        return self.__device

    def to(self, *args, **kwargs):
        super(Solver, self).to(*args, **kwargs)
        self.__device = torch._C._nn._parse_to(*args, **kwargs)[0]

    def cuda(self, device=None):
        super(Solver, self).cuda(device)
        self.__device = device

    def cpu(self):
        super(Solver, self).cpu()
        self.__device = torch.device('cpu')


    def on_preco_change(self) -> None:
        if self.use_task_proximal():
            self.task.update_prox_on_preco_change()
        if self.use_task_gradients():
            self.task.update_grad_on_preco_change()
        self.regularizer.on_preco_change()

    def check_configured(self) -> None:
        missing_elements_msgs = []
        _add_missing_config_msgs(missing_elements_msgs, self.preconditioner, Preconditioner, 'preconditioner')
        _add_missing_config_msgs(missing_elements_msgs, self.task, Task, 'task')
        _add_missing_config_msgs(missing_elements_msgs, self.regularizer, Regularizer, 'regularizer')

        if len(missing_elements_msgs):
            raise ConfigurationError(f'{len(missing_elements_msgs)} missing/incorrect element(s) in solver configuration:'
                                     + ''.join(['\n- '+msg for msg in missing_elements_msgs]))

    def forward(self, inputs: dict, initialization: Tensor, max_iterations: int) -> Tensor:
        self.check_configured()
        # Iteration number 0 is used for initialisation phase only.
        # Otherwise iteration numbers start at 1 for first iteration.
        self._current_iteration = 0
        if max_iterations < 0:
            raise ValueError('The number of iterations should be a positive integer.')

        # Prepare all solver data from inputs
        self._max_iterations = max_iterations
        self.regularizer.set_preconditioner(self.preconditioner)
        self.task.set_preconditioner(self.preconditioner)
        self.task.set_inputs(inputs)
        self.task.to(self.device)  # Makes sure the input data given to the task is on the same device as the solver.
        # TODO: check initialization shape w.r.t. inputs
        self._Px = initialization.to(self.device)
        self.prepare_solver_data()

        if max_iterations > 0:  # no pre-computations if there is 0 iteration (returns post-processed intialization).
            self.preconditioner.create(self)
            self._x = self.preconditioner.inverse(self._Px)  # compute the initial preconditioned variable.

        # Run the iteative algorithm
        for self._current_iteration in range(1, 1 + max_iterations):
            try:
                self.iterate()
                yield self._Px
            except AlgorithmStopped:
                break

        self.task.post_process(self._Px)
        yield self._Px


def _add_missing_config_msgs(msgs, element_object, element_type, element_name):
    if element_object is None:
        msgs.append(element_name + ' not specified')
    elif not isinstance(element_object, element_type):
        msgs.append(f'the {element_name} must be an instance of \'{element_type.__name__}\', '
                    f'got \'{element_object.__class__.__name__}\' instead.')
    else:
        msgs.extend(element_object.missing_config_elements_msgs())
