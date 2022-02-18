from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Type
if TYPE_CHECKING:
    from torch import Tensor
    from pnpcore import CodeOptimPrecoMsg

from pnpcore import Preconditioner, no_preco, ConfigurationError
from abc import ABC, abstractmethod

import torch.nn as nn
import math


class Regularizer(ABC, nn.Module):

    def __init__(self):
        super(Regularizer, self).__init__()
        self.__preconditioner = None

    @abstractmethod
    def name(self) -> str:
        pass

    def get_preconditioner(self):
        return self.__preconditioner

    # def set_preconditioner(self, preco: Optional[Preconditioner]) -> None:
    #    self.__preconditioner = preco

    def set_preconditioner(self, p: Optional[Preconditioner]) -> None:
        # setter implemented explicitly as a function
        # -> python's @property.setter style does not seem to work within nn.Module class for a nn.Module property.
        if p is None:
            pass
        elif not (p is no_preco or len([c for c in self.compatible_preco_types() if isinstance(p, c)])):
            #checks if p is either the no_preco object, or a preconditioner object of another compatible type.
            raise ConfigurationError(
                f'The preconditioners of type {p.__class__.__name__} are not compatible with the regularizer.\n'
                f'The compatible types (or base types) in addition to the no_preco object are:\n'
                f'{[c.__name__ for c in self.compatible_preco_types()]}'
            )
        self.__preconditioner = p
        self.__preconditioner.setup_code_optim(self.code_optim_preco_msg())

    def code_optim_preco_msg(self) -> List[CodeOptimPrecoMsg]:
        return []

    def on_preco_change(self) -> None:
        """Callback function automatically called by the solver when the preconditioner is changed."""
        pass

    @abstractmethod
    def denoise_operator(self, Pu: Tensor, sigma: float, u: Tensor = None, iter_num: int = 0) -> Tensor:
        """
        Denoising operator of the regularizer function assuming noise covariance matrix (sigma * P)^2,
        where P is the preconditioning matrix.

        Pu is the Tensor to denoise (i.e. with preconditioning applied).
        For some regularizers the Tensor u (before applying preconditioning) may also be needed.
        """
        pass

    def proximal_operator(self, u: Tensor, weight: float, iter_num: int = 0) -> Tensor:
        """
        Proximal operator (including preconditioning) of the regularizer function multiplied by a weight.
        (May need override for efficiency).
        """
        return self.__preconditioner.inverse(
            self.denoise_operator(Pu=self.__preconditioner(u), sigma=math.sqrt(weight), u=u, iter_num=iter_num))

    def missing_config_elements_msgs(self) -> List[str]:
        return []


    def compatible_preco_types(self) -> List[Type]:
        return [Preconditioner]

