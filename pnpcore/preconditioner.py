from __future__ import annotations
from typing import TYPE_CHECKING, Type, Optional, List, Tuple
if TYPE_CHECKING:
    from pnpcore import Solver, Task
    from torch import Tensor

from abc import ABC, abstractmethod
from pnpcore import ConfigurationError
import torch.nn as nn
import torch.fft
from enum import Enum, auto


class CodeOptimPrecoMsg(Enum):
    IMAGE_FOURIER_DOMAIN_PX = auto()


class Preconditioner(ABC, nn.Module):
    """Base abstract class for Preconditioners."""

    _preco_creator: Optional[PrecoCreatorForTask]
    _code_optim_settings: List[CodeOptimPrecoMsg]

    def __init__(self):
        super(Preconditioner, self).__init__()
        self._preco_creator = None
        self.__skip_update_callbacks = False
        self._code_optim_settings = []

    def name(self) -> str:
        return self._preco_creator.name()

    def params_strings(self) -> List[str]:
        return self._preco_creator.params_strings()

    def setup_code_optim(self, list_msgs: List[CodeOptimPrecoMsg]) -> None:
        """
        Takes code optimization messages (from the regularizer) and keeps as a preconditioner attribute, the list of
        messages that are accepted by the preconditioner (e.g. FourierPreconditioner can use the IMAGE_FOURIER_DOMAIN_PX
        message to skip the inverse Fourier transform in the forward method. A regularizer that sends this message can
        directly process the Px value in the Fourier domain without performing Fourier transform).

        A preconditioner must overwrite this method in order to take code optimization messages into account.
        """
        pass

    def code_optim_settings(self) -> List[CodeOptimPrecoMsg]:
        return self._code_optim_settings

    def get_preco_creator(self):
        return self._preco_creator

    def set_preco_creator(self, pc: Optional[PrecoCreatorForTask]):
        # Allows either None object or compatible preconditioner creator object for this preconditioner type.
        if not (pc is None or isinstance(self, pc.preconditioner_type())):
            raise ConfigurationError(
                f'Preconditioner creators of type {pc.__class__.__name__} must be used by '
                f'instances of {pc.preconditioner_type().__name__}.\n'
                f'The preconditioner type {self.__class__.__name__} is not compatible.')
        self._preco_creator = pc

    def create(self, solver: Solver) -> None:
        """Create the preconditioning matrix P from the solver data. Do not override!!"""
        self._preco_creator.create_preco_data(solver, self)
        solver.on_preco_change()

    def update(self, solver: Solver) -> None:
        """Update the preconditioning matrix P from the solver data. Do not override!!"""
        self._preco_creator.update_preco_data(solver, self)
        if self.__skip_update_callbacks:
            self.__skip_update_callbacks = False
        else:
            solver.on_preco_change()

    def skip_update_callbacks(self) -> None:
        """Prevents the preconditioning update from calling back the solver.

        Must be called by preconditioner creator object's update_preco_data method when no update is performed to avoid
        unnecessary update computations by the solver."""
        self.__skip_update_callbacks = True

    def missing_config_elements_msgs(self) -> List[str]:
        if self._preco_creator is None:
            return [f'preconditioner creator not specified for the preconditioner {self.__class__.__name__}.']
        return []

########################################################################################################################

    @abstractmethod
    def forward(self, M: Tensor, use_optim_msg=True) -> Tensor:
        """
        Apply the preconditioning
        (e.g. for a linear preconditioner: matrix multiplication P*M with preconditioning matrix P).
        """
        pass

    @abstractmethod
    def inverse(self, M: Tensor) -> Tensor:
        """
        Apply the inverse preconditioning (e.g. for a linear preconditioner:
        matrix multiplicaton P^(-1)*M by the inverse of the preconditioning matrix: P^(-1)).
        """
        pass

    @abstractmethod
    def backward(self, M: Tensor) -> Tensor:
        """
        Apply the backward preconditioning: i.e. matrix multiplication J^t*M, where J^t is the transposed Jacobian
        matrix of the preconditioner.
        (e.g. for a linear preconditioner: matrix multiplication P^t*M with transposed preconditioning matrix P^t).
        """
        pass


    # def apply_inverse(self, M: Tensor) -> Tensor:
    #    """
    #    Apply the inverse preconditioning
    #    (e.g. for a linear preconditioner: matrix multiplication P^(-1)*M with inverse preconditioning matrix P^(-1)).
    #    """
    #    raise NotImplementedError()

    '''
    def orthogonal_transform_right(self, M: Tensor) -> Tensor:
        """
        Orthogonal transform representing the matrix multiplication V^t * M, where V^t are the right singular
        vectors of the preconditioning matrix P in the Singular Value Decomposition: P=U*Sigma*V^t.
        """
        raise NotImplementedError()

    def orthogonal_transform_left(self, M: Tensor) -> Tensor:
        """
        Orthogonal transform representing the matrix multiplication U * M, where U are the left singular
        vectors of the preconditioning matrix P in the Singular Value Decomposition: P=U*Sigma*V^t.
        """
        raise NotImplementedError()

    def multiplier(self) -> Tensor:
        """
        Returns singular values Sigma of the preconditioning matrix P
        in the Singular Value Decomposition: P=U*Sigma*V^t.
        """
        raise NotImplementedError()
    
    def multiply(self, M: Tensor) -> Tensor:
        """
        Matrix multiplicaton by the preconditioning matrix: P*M.
        (May need override for efficiency).
        """
        return self.orthogonal_transform_left(self.orthogonal_transform_right(M) * self.multiplier)

    def inverse_multiply(self, M: Tensor) -> Tensor:
        """
        Matrix multiplicaton by the inverse of the preconditioning matrix: P^(-1)*M.
        (May need override for efficiency).
        """
        return self.orthogonal_transform_left(self.orthogonal_transform_right(M) / self.multiplier)

    # TODO: Will be wrong with complex multipler => complex conjugate of multiplier is needed here!
    def transpose_multiply(self, M: Tensor) -> Tensor:
        """
        Matrix multiplicaton by the transposed preconditioning matrix: P^t * M.
        (May need override for efficiency).
        """
        return self.orthogonal_transform_right(self.orthogonal_transform_left(M) * self.multiplier)
    '''


class PSDMatrixPreconditioner(Preconditioner):
    """
    Base class for Positive Semi-Definite Preconditioning matrices, represented in the form:
    P=T^(-1)*D*T, where T is an orthogonal transform matrix, T^(-1) it's inverse and D a diagonal matrix.

    Note: the code of the base PSDMatrixPreconditioner class cannot guarantee that particular implementations
    correspond to true positive semi-definite matrices.
    """

    @abstractmethod
    def orthogonal_transform(self, M: Tensor) -> Tensor:
        """
        Orthogonal transform T in the matrix decomposition P=T^(-1)*D*T.
        """
        pass

    @abstractmethod
    def orthogonal_inverse_transform(self, M: Tensor) -> Tensor:
        """
        Orthogonal Inverse transform T^(-1) in the matrix decomposition P=T^(-1)*D*T.
        """
        pass

    @abstractmethod
    def multiplier(self) -> Tensor:
        """
        Returns the diagonal elements of the diagonal matrix D in the matrix decomposition P=T^(-1)*D*T.
        """
        pass

    def forward(self, M: Tensor, use_optim_msg=True) -> Tensor:
        # Implemented in the abstract class, but may need to be override by subclasses for faster computations.
        return self.orthogonal_inverse_transform(self.orthogonal_transform(M) * self.multiplier())

    def inverse(self, M: Tensor) -> Tensor:
        # Implemented in the abstract class, but may need to be override by subclasses for faster computations.
        return self.orthogonal_inverse_transform(self.orthogonal_transform(M) / self.multiplier())

    def backward(self, M: Tensor) -> Tensor:
        # Implemented in the abstract class, but may need to be override by subclasses for faster computations.
        return self.orthogonal_inverse_transform(self.orthogonal_transform(M) * self.multiplier())


class DiagonalPreconditioner(PSDMatrixPreconditioner):
    """Diagonal matrix Preconditioner."""

    def __init__(self):
        super(DiagonalPreconditioner, self).__init__()
        self.register_buffer('_P', None, persistent=False)

    def orthogonal_transform(self, M: Tensor) -> Tensor:
        return M

    def orthogonal_inverse_transform(self, M: Tensor) -> Tensor:
        return M

    def multiplier(self) -> Tensor:
        return self._P

    def forward(self, M: Tensor, use_optim_msg=True) -> Tensor:
        return self._P * M

    def inverse(self, M: Tensor) -> Tensor:
        return M / self._P

    def backward(self, M: Tensor) -> Tensor:
        return self._P * M


class FourierPreconditioner(PSDMatrixPreconditioner):
    """Preconditioning matrix of the form Phi^T*D*Phi with Phi the Fourier transform and D a diagonal matrix."""

    def __init__(self, dim: Tuple[int] = (-2, -1)):
        super(FourierPreconditioner, self).__init__()
        self.register_buffer('_P', None, persistent=False)
        self.dim = dim

    def absolute_dim_index(self, dim):
        return dim if dim >= 0 else self._P.ndim + dim

    def is_fft_dim(self, dim: int):
        for d in self.dim:
            if self.absolute_dim_index(d) == self.absolute_dim_index(dim):
                return True
        return False

    def orthogonal_transform(self, M: Tensor) -> Tensor:
        return torch.fft.fftn(M, dim=self.dim)

    def orthogonal_inverse_transform(self, M: Tensor) -> Tensor:
        return torch.fft.irfftn(M, [M.shape[i] for i in self.dim], dim=self.dim)

    def multiplier(self) -> Tensor:
        return self._P

    def forward(self, M: Tensor, use_optim_msg=True) -> Tensor:
        Px = self.orthogonal_transform(M) * self.multiplier()
        if CodeOptimPrecoMsg.IMAGE_FOURIER_DOMAIN_PX in self._code_optim_settings and use_optim_msg:
            # image spatial dimensions are at indices 2 and 3 (dimension indices 0 and 1 are for batch and channel).
            if len(self.dim) == 2 and self.is_fft_dim(2) and self.is_fft_dim(3):
                # skip inverse Fourier transform -> the regularizer sending the
                # IMAGE_FOURIER_DOMAIN_PX message will use Px in the Fourier domain directly.
                return Px
            # remove the IMAGE_FOURIER_DOMAIN_PX setting if the preconditioner can't handle it
            # (ie. if fft is not performed in the image dimensions).
            self._code_optim_settings = []
        return self.orthogonal_inverse_transform(Px)

    def setup_code_optim(self, list_msgs: List[CodeOptimPrecoMsg]) -> None:
        if CodeOptimPrecoMsg.IMAGE_FOURIER_DOMAIN_PX in list_msgs:
            self._code_optim_settings = [CodeOptimPrecoMsg.IMAGE_FOURIER_DOMAIN_PX]



class PrecoCreatorForTask(ABC):
    """
    Base class of preconditioner creators that create and update the internal data of a given type of preconditioner,
    for a given task.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def params_strings(self) -> List[str]:
        pass

    @abstractmethod
    def create_preco_data(self, solver: Solver, preco: Preconditioner) -> None:
        """Create preconditioner data of a given preconditioner type for a given task type."""
        pass

    @abstractmethod
    def update_preco_data(self, solver: Solver, preco: Preconditioner) -> None:
        """Update preconditioner data of a given preconditioner type for a given task type."""
        pass

    @staticmethod
    @abstractmethod
    def task_type() -> Type[Task]:
        """
        Returns the type (or base type) of the tasks for which the preconditioner creator can create and update
        preconditioner data.
        """
        pass

    @staticmethod
    @abstractmethod
    def preconditioner_type() -> Type[Preconditioner]:
        """
        Returns the type (or base type) of the preconditioners that can be created and updated by the
        preconditioner creator object.
        """
        pass
