import torch
import numpy as np

import os, time, re
import subprocess, shutil
import argparse
import inspect
from os.path import *
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any


def all_to_dict(module, exclude=[]):
    module_all = module.__all__
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if x in module_all
                 and x not in exclude
                 and getattr(module, x) not in exclude])


def add_arguments_for_function(parser, module, argument_for_func, default, skip_params=[], parameter_defaults={}):
    argument_group = parser.add_argument_group(argument_for_func.capitalize())

    module_dict = all_to_dict(module)
    argument_group.add_argument('--' + argument_for_func, type=str, default=default, choices=list(module_dict.keys()))

    args, unknown_args = parser.parse_known_args()
    func_obj = module_dict[vars(args)[argument_for_func]]

    argspec = inspect.getfullargspec(func_obj)

    defaults = argspec.defaults[::-1] if argspec.defaults else None

    args = argspec.args[::-1]
    for i, arg in enumerate(args):
        cmd_arg = '{}_{}'.format(argument_for_func, arg)
        if arg not in skip_params + ['device']:
            if arg in list(parameter_defaults.keys()):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(parameter_defaults[arg]),
                                            default=parameter_defaults[arg])
            elif (defaults is not None and i < len(defaults)):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(defaults[i]), default=defaults[i])
            else:
                print(("[Warning]: non-default argument '{}' detected on class '{}'. "
                       "This argument cannot be modified via the command line"
                       .format(arg, module.__class__.__name__)))


def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if inspect.isclass(getattr(module, x))
                 and x not in exclude
                 and getattr(module, x) not in exclude])


def add_arguments_for_module(parser, module, argument_for_class, default, skip_params=[], exception=[],
                             parameter_defaults={}):
    argument_group = parser.add_argument_group(argument_for_class.capitalize())

    module_dict = module_to_dict(module)
    argument_group.add_argument('--' + argument_for_class, type=str, default=default, choices=list(module_dict.keys()))

    if default is None:
        return None

    args, unknown_args = parser.parse_known_args()
    class_obj = module_dict[vars(args)[argument_for_class]]

    argspec = inspect.getfullargspec(class_obj.__init__)

    defaults = argspec.defaults[::-1] if argspec.defaults else None

    args = argspec.args[::-1]
    for i, arg in enumerate(args):
        cmd_arg = '{}_{}'.format(argument_for_class, arg)

        if arg not in skip_params + ['self', 'args']:
            if not np.array([bool(re.search(x, arg)) for x in exception]).any():
                if arg in list(parameter_defaults.keys()):
                    extra_args = args_exception(parameter_defaults[arg])
                    argument_group.add_argument('--{}'.format(cmd_arg), **extra_args)

                elif (defaults is not None and i < len(defaults)):
                    extra_args= args_exception(defaults[i])
                    argument_group.add_argument('--{}'.format(cmd_arg), **extra_args)

                else:
                    print(("[Warning]: non-default argument '{}' detected on class '{}'. "
						   "This argument cannot be modified via the command line"
						   .format(arg, module.__class__.__name__)))


def args_exception(args_val):
    extra_args = {'type': type(args_val), 'default': args_val}

    if isinstance(args_val, (tuple, list)):
        extra_args.update({'nargs': '+', 'type': type(args_val[0])})
    elif isinstance(args_val, bool):
        extra_args.update({'nargs': '?', 'type': _str2bool, 'const': not args_val, 'default': args_val})

    return extra_args


def _str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_instance(module: object, config_params: dict, **kwargs) -> Any:
    config_params['args'].update(kwargs)
    instance = getattr(module, config_params['module'])(**config_params['args'])

    # print(f'create_instanced module {module.__name__}')
    return instance


def create_function(fcn, config_params=None, **kwargs) -> Any:
    if config_params is None:
        config_params = {}

    config_params.update(kwargs)
    instance = fcn(**config_params)

    return instance


def format_dictionary_of_losses(labels, values):
    try:
        string = ', '.join([('{}: {:' + ('.3f' if value >= 0.001 else '.1e') +'}').format(name, value) for name, value in zip(labels, values)])
    except (TypeError, ValueError) as e:
        print((list(zip(labels, values))))
        string = '[Log Error] ' + str(e)

    return string


class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print(("  [{:.3f}{}] {}".format(duration, units, string)))

    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n" % string)
        fid.close()


class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = next(self.iterator)
        self.last_duration = (time.time() - start)
        return n

    next = __next__


def kwargs_from_args(args, argument_for_class):
    argument_for_class = argument_for_class + '_'
    return {key[len(argument_for_class):]: value for key, value in list(vars(args).items())
            if argument_for_class in key and key not in [argument_for_class + 'class', argument_for_class + 'func']}


def save_checkpoint(state, is_best, path, prefix, filename=None):
    if filename is None:
        filename = 'checkpoint.pth.tar'

    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')
