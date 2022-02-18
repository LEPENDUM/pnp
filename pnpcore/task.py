from __future__ import annotations
from typing import TYPE_CHECKING, List, Type, Optional, Union
if TYPE_CHECKING:
    from torch import Tensor
    from pnpcore import Solver, Preconditioner

from abc import ABC, abstractmethod
from pnpcore.meta.meta import SingletonABCMeta
from six import string_types

from pnpcore import PSDMatrixPreconditioner, PrecoCreatorForTask, MathematicalIncompatibility, ConfigurationError
from torch import nn
import torch


class Task(ABC, nn.Module):

    def __init__(self):
        super(Task, self).__init__()
        self.__preconditioner = None
        self.__valid_proximal_operator = False
        self.__input_defaults = {}

    def get_preconditioner(self) -> Preconditioner:
        return self.__preconditioner

    def set_preconditioner(self, p: Optional[Preconditioner]) -> None:
        # setter implemented explicitly as a function
        # -> python's @property.setter style does not seem to work within nn.Module class for a nn.Module property.
        if p is None:
            pass
        elif p.get_preco_creator() is None:
            raise ConfigurationError(
                f'Error when assigning a preconditioner for the Task.\n'
                f'The preconditioner must be previously configured with a preconditioner creator object '
                f'(instance of PrecoCreatorForTask).')
        elif not isinstance(self, p.get_preco_creator().task_type()):
            raise ConfigurationError(
                f'The given preconditioner is configured with a preconditioner creator of type '
                f'{p.get_preco_creator().__class__.__name__} that is only compatible with task instances of '
                f'{p.get_preco_creator().task_type().__name__}.\n'
                f'The task type {self.__class__.__name__} is not compatible.')
        self.__valid_proximal_operator = self.check_proximal_compatibility(p)
        self.__preconditioner = p

    def check_proximal_compatibility(self, p: Preconditioner) -> bool:
        """
        Checks mathematical compatibility between the task's proximal operator and the preconditioner.
        Do not override!
        """
        return p.__class__ == IdentityPreconditioner or len(
            [c for c in self.preco_types_for_prox() if isinstance(p, c)])

    # TODO: prevent accidental override
    def proximal_operator(self, u: Tensor, weight: float) -> Tensor:
        """
        Proximal operator (including preconditioning) of the task data term multiplied by a weight.
        Do not override!
        """
        if not self.__valid_proximal_operator:
            raise MathematicalIncompatibility(
                f'The preconditioners of type {self.__preconditioner.__class__.__name__} are not declared as '
                f'mathematically compatible for the task\'s proximal operator.\n'
                f'To declare compatible preconditioners for the task, override it\'s method preco_types_for_prox.\n'
                f'Currently, the types (or base types) declared as compatible in addition to the no_preco object are:\n'
                f'{[c.__name__ for c in self.preco_types_for_prox()]}'
            )
        return self._proximal_operator(u, weight)

    def save_inputs(self, directory: str, name_prefixes: Union[str, List[str]], extension: str) -> None:
        """Saves the input data stored in the Task object. Do not Override"""
        if isinstance(name_prefixes, string_types):
            name_prefixes = [name_prefixes]
        for i in range(len(name_prefixes)):
            try:
                self.save_inputs_at_batch_id(directory, name_prefixes[i], extension, i)
            except IndexError as err:
                raise IndexError(
                    'The batch size of the task\'s inputs is smaller '
                    'than the given list of file names to save to.') from err

    def load_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        """
        Loads input data from files. Do not override!
        The inputs are only returned and not set in the task's data.
        """
        inputs = self.generate_optional_inputs()
        inputs.update(self.load_required_inputs(directory, name_prefix, extension))
        return inputs

    def set_inputs(self, inputs: dict) -> None:
        """Set the input data for the task. Do not override!"""
        self.check_inputs(inputs)
        self.set_optional_inputs(inputs)
        self.set_required_inputs(inputs)

    def generate_inputs(self, ground_truth: Tensor) -> dict:
        """
        Generates and returns all inputs from ground truth data according to the Task's degradation model.
        Do not override!

        * The optional inputs (e.g. parameters of the degradation model) are taken from the tasks attributes directly.
          (they should not appear in the dict if their value is equal to the default).

        * The method should not modify attributes of the Task object!
        """
        inputs = self.generate_optional_inputs()
        inputs.update(self.generate_required_inputs(ground_truth))
        return inputs

    def register_optional_input_attr(self, attr_name: str, input_key: str, default_value, init_value) -> None:
        """
        Register an attribute associated to an optional task input with a default value.
        :param attr_name: name of the class attribute associated to the task input.
        :param input_key: name of the task input in the inputs dictionary.
        :param default_value: default value of the task input.
        :param init_value: Initialisation value for the attribute. If None, the attribute is set to the default value.
        """
        if attr_name in self.__dict__:
            raise ValueError(f'An attribute with the name \'{attr_name}\' already exists!')
        if input_key in self.__input_defaults:
            raise ValueError(f'The input key \'{input_key}\' is already registered as an '
                             f'optional input (associated to the attribute \'{attr_name}\')!')
        if input_key in self.required_inputs_keys():
            raise ValueError(f'The input key \'{input_key}\' can\'t be added as an optional '
                             f'input because it is already used as a required input!')
        setattr(self, attr_name, default_value if init_value is None else init_value)
        self.__input_defaults[input_key] = (default_value, attr_name)

    def generate_optional_inputs(self) -> dict:
        """
        Generate an inputs dictionary containing only the optional inputs that are different from their default value.
        """
        return {k: self.__dict__[v[1]] for k, v in self.__input_defaults.items() if self.__dict__[v[1]] != v[0]}

    def set_optional_inputs(self, inputs):
        for k, v in self.__input_defaults.items():
            setattr(self, v[1], inputs.get(k, v[0]))

    # TODO: how to check the inputs also in the initializer?
    #  (-> call check_inputs in the run script before calling initializer?)
    def check_inputs(self, inputs: dict) -> None:
        """Checks the input data for the task."""
        missing_inputs_keys = [key for key in self.required_inputs_keys() if key not in inputs]
        if len(missing_inputs_keys) > 0:
            raise KeyError(f'The follwong required inputs are missing from the '
                           f'given inputs dictionary: {missing_inputs_keys}')


    def missing_config_elements_msgs(self) -> List[str]:
        return []

#############################################################################
#               Methods to be implemented by each task
# (at least the abstract methods are needed for a minimal implementation)
#############################################################################

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def params_strings(self) -> List[str]:
        pass

    def pre_post_denoise_strings(self) -> List[str]:
        return []

    @abstractmethod
    def _proximal_operator(self, u: Tensor, weight: float) -> Tensor:
        """
        Proximal operator (including preconditioning) of the task data term multiplied by a weight.
        (must be defined for each task).
        """
        pass

    @property
    @abstractmethod
    def data_term_inverse_weight(self) -> float:
        """Inverse of the weight of the Task's data term in the optimization."""
        pass

    @abstractmethod
    def set_required_inputs(self, inputs: dict) -> None:
        """
        Set the tasks data according to the required inputs.
        The attributes registered to optional inputs are already set before the call to this function.
        """
        pass

    @abstractmethod
    def required_inputs_keys(self) -> List:
        """Returns the list of required keys in the inputs dicitonary."""
        pass

    def generate_required_inputs(self, ground_truth: Tensor) -> dict:
        """
        Generates and returns the required inputs from ground truth data according to the Task's degradation model.

        The method should not store the generated inputs in the Task object!
        """
        raise NotImplementedError()

    def save_inputs_at_batch_id(self, directory: str, name_prefix: str, extension: str, batch_id: int) -> None:
        """Saves the input data stored in the Task object only for the given index in the batch."""
        raise NotImplementedError()

    def load_required_inputs(self, directory: str, name_prefix: str, extension: str) -> dict:
        """
        Loads the required input data from files. The method may also inlcude optional inputs
        if they need to be loaded from a file (otherwise, the optional inputs are taken from
        the tasks data and they don't need to be loaded with this method).
        """
        raise NotImplementedError()

    def reformat_ground_truth(self, ground_truth: Tensor) -> Tensor:
        """
        Reformat ground truth data to be compatible with the task's degradation model
        (e.g. crop borders if the there are restrictions on the ground truth data dimensions).
        """
        raise NotImplementedError(
            f'Please implement the Task\'s {self.reformat_ground_truth.__name__} '
            f'method to enable simulations with known ground truth data.'
        )

    def preco_types_for_prox(self) -> List[Type]:
        """
        Defines the types (or base types) of preconditioners mathematically compatible with the task's
        proximal operator at the exception of the Identity preconditioner (no_preco) that should always be compatible.
        """
        return []

    ####################
    # Callback methods
    ####################

    def pre_denoising(self, current_estimate: Tensor) -> None:
        """
        Method automatically called by the solver before denoising (if the solver uses denoising).
        The current estimate is given as argument. If it needs to be modified, the method should modify it
        directly without returning it.
        """
        pass

    def post_denoising(self, current_estimate: Tensor) -> None:
        """
        Method automatically called by the solver after denoising (if the solver uses denoising).
        The current estimate is given as argument. If it needs to be modified, the method should modify it
        directly without returning it.
        """
        pass

    def post_process(self, current_estimate: Tensor) -> None:
        """
        Method automatically called by the solver as the last processing step.
        The method must modify the input tensor directly without returning it.
        """
        pass

    def update_prox_on_preco_change(self) -> None:
        """
        Method automatically called by the solver when the preconditioner is changed and the tasks proximal
        operator will be used.
        This method must be implemented by the task if it's proximal operator uses pre-computed data for efficieny,
        and if this data must be updated when the preconditioner changes.
        """
        pass

    def update_grad_on_preco_change(self) -> None:
        """
        Method automatically called by the solver when the preconditioner is changed and the tasks gradient
        computations will be used.
        This method must be implemented by the task if it's gradient computations use pre-computed data for efficiency,
        and if this data must be updated when the preconditioner changes.
        """
        pass


class Initializer(ABC, nn.Module):
    """Base class for task initialization."""


    @abstractmethod
    def required_inputs_keys(self) -> List:
        """Returns the list of required keys in the inputs dicitonary."""
        pass

    def check_task_compatibility(self, task):
        pass


########################################################################################################################
# IdentityPreconditioner and IdentityPrecoCreator are defined in task module only to avoid cyclic dependency...
########################################################################################################################


class IdentityPreconditioner(PSDMatrixPreconditioner, metaclass=SingletonABCMeta):
    """Identity preconditioner (i.e. does not do any preconditioning)."""

    def __init__(self):
        super(IdentityPreconditioner, self).__init__()
        self.register_buffer('_P', torch.tensor([1.0]), persistent=False)
        self._preco_creator = IdentityPrecoCreator()

    def name(self) -> str:
        return 'no-preco'

    def params_strings(self) -> List[str]:
        return []

    def orthogonal_transform(self, M):
        return M

    def orthogonal_inverse_transform(self, M):
        return M

    def multiplier(self):
        return self._P

    def forward(self, M, use_optim_msg=True):
        return M

    def inverse(self, M):
        return M

    def backward(self, M):
        return M


class IdentityPrecoCreator(PrecoCreatorForTask):
    """Creator class for the Identity preconditioner (i.e. no_preco object) --> suitable for all Tasks."""

    def name(self) -> str:
        return ''

    def params_strings(self) -> List[str]:
        return []

    def create_preco_data(self, solver: Solver, preco: IdentityPreconditioner) -> None:
        # convert tensor to have the same number of dimensions as the processed data (needed for broadcast).
        preco._P = preco._P.reshape([1]*solver.current_estimate.ndim)

    def update_preco_data(self, solver: Solver, preco: IdentityPreconditioner) -> None:
        preco.skip_update_callbacks()  # avoids unnecessary callback to the solver since no update was performed.

    @staticmethod
    def task_type() -> Type[Task]:
        return Task

    @staticmethod
    def preconditioner_type() -> Type[Preconditioner]:
        return IdentityPreconditioner
