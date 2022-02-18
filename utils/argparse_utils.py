from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from argparse import ArgumentParser

import argparse
import os

from dataclasses import dataclass
import warnings


@dataclass
class Argument:
    name: str
    short_name: Optional[str] = None
    type: Any = None
    action: Any = None
    default: Any = None
    required: bool = False
    help: Optional[str] = None

    def add_to_parser(self, parser: ArgumentParser):
        flags = ([] if self.short_name is None else ['-' + self.short_name]) + ['--' + self.name]
        kwargs = {
            'required': self.required,
            'help': self.help,
            'action': self.action
        }
        if self.default is not None:
            kwargs.update(default=self.default)
        if self.type is not None:
            kwargs.update(type=self.type)
        parser.add_argument(*flags, **kwargs)


def check_args_list(args_list: List[Argument]):
    names = [arg.name for arg in args_list]
    short_names = [arg.short_name for arg in args_list]
    doubles = [i for i in range(len(args_list)) if names.count(names[i]) > 1
               or (short_names[i] is not None and short_names.count(short_names[i]) > 1)]
    if len(doubles):
        doubles_strings = [('' if short_names[i] is None else '-' + short_names[i] + '/')
                           + '--' + names[i] for i in doubles]
        raise Exception(f'Several arguments have either the same name or the same short name appearing several times '
                        f'in the argument\'s list: \n{doubles_strings}')


def get_name_from_list(args_list: List[Argument], arg_name: str):
    idx = [i for i in range(len(args_list)) if args_list[i].name == arg_name]
    if len(idx) > 1:
        raise Exception(f'Duplicate argument "{arg_name}" in the list of arguments.')
    elif len(idx) == 0:
        warnings.warn(f'Warning: no argument "{arg_name}" was found in the list of arguments.')
        return None
    else:
        return args_list[idx[0]]


def overwrite_default(args_list: List[Argument], **arg_defaults):
    args = [get_name_from_list(args_list, name) for name in arg_defaults]
    for arg in args:
        if arg is None:
            continue
        arg.default = arg_defaults[arg.name]
        arg.required = False


def add_argument_list(parser: ArgumentParser, args_list: List[Argument]):
    for i in range(len(args_list)):
        args_list[i].add_to_parser(parser)


def make_parser(args_list: List[Argument], *args, **kwargs):
    check_args_list(args_list)
    parser = argparse.ArgumentParser(*args, **kwargs)
    add_argument_list(parser, args_list)
    return parser



########################################       Parameters Check Functions       ########################################


# binary value
def check_binary(value):
    try:
        ivalue = float(value)
        if ivalue != 0 and ivalue != 1:
            raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a binary value (0->False, 1->True).')
    return ivalue == 1


# strictly positive numeric value
def check_pos_numeric(value):
    try:
        ivalue = float(value)
        if ivalue <= 0:
            raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a strictly positive numeric value.')
    return ivalue


# non-negative numeric value
def check_nneg_numeric(value):
    try:
        fvalue = float(value)
        if fvalue < 0:
            raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a positive (or null) numeric value.')
    return fvalue


# numeric value, negative values are replaced by infinity
def check_numeric_neginf(value):
    try:
        fvalue = float(value)
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a numeric value.')
    if fvalue < 0:
        return float('inf')
    else:
        return fvalue


# strictly integer
def check_pos_int(value):
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a strictly positive int value.')
    return ivalue


# non-negative integer
def check_nneg_int(value):
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a positive (or null) int value.')
    return ivalue


# non-negative integer, and zero is replaced by infinity
def check_nneg_int_zinf(value):
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a positive (or null) int value.')
    if ivalue == 0:
        return float('inf')
    else:
        return ivalue


# name of an existing file
def check_isfile(value):
    realpath = os.path.realpath(value)
    if not os.path.isfile(realpath):
        raise argparse.ArgumentTypeError(f'Can\'t find the file: \'{realpath}\'')
    return realpath


# name of an existing directory
def check_isdir(value):
    realpath = os.path.realpath(value)
    if not os.path.isdir(realpath):
        raise argparse.ArgumentTypeError(f'Can\'t find the directory: \'{realpath}\'')
    return realpath
