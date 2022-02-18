from __future__ import annotations
from typing import List, Union

import os
from six import string_types

from pnpcore import Solver


def join_names(*names: Union[str, List[str]], sep: str = '_') -> str:
    names_flat = []
    for elt in names:
        names_flat += [elt] if isinstance(elt, string_types) else elt
    return sep.join(filter(None, names_flat))


class DirectoryStructure:
    """Class defining a directory structure for saving/loading inputs and result files for the configured solver."""

    def __init__(self, root_dir: str, solver: Solver, dataset_name: str, init_name: str = ''):
        self.solver = solver
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.init_name = init_name

    def _task_data_path(self):
        return os.path.join(
            self.root_dir,
            self.solver.task.name(),
            join_names(self.dataset_name, self.solver.task.params_strings())
        )

    def input_directory(self) -> str:
        self.solver.check_configured()
        return os.path.join(self._task_data_path(), 'inputs')

    def output_directory(self, max_iterations: int) -> str:
        self.solver.check_configured()
        solver_dir = join_names(self.solver.name(), self.solver.params_strings())
        preco_str = self.solver.preconditioner.name()
        preco_params_str = join_names(self.solver.preconditioner.params_strings(), sep=',')
        preco_str += f'({preco_params_str})' if preco_params_str else ''
        reg_preco_dir = join_names(self.solver.regularizer.name(), preco_str)
        out_dir = os.path.join(self._task_data_path(), solver_dir, reg_preco_dir)
        init_str = f'(init-{self.init_name})' if len(self.init_name) else ''
        return os.path.join(out_dir, f'{max_iterations}-it' + init_str)

    def init_directory(self) -> str:
        return os.path.join(
            self._task_data_path(),
            join_names('init', self.init_name, sep='-')
        )
