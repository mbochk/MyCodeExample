# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:28:36 2016

@author: mbochk
"""

import sys


class ParamDescription(object):
    pass


def generate_param_description(option_name, option_type, option_required):
    param_description = ParamDescription()
    param_description.name = option_name
    param_description.type = option_type
    param_description.required = option_required
    return param_description

# define parameters here

g_params = [["int-option", int, True],
            ["bool-option", bool, True],
            ["string-option", str, True],
            ["float-option", float, False]]

g_params = [generate_param_description(*item) for item in g_params]


####

def print_help(func_name):
    print "usage: {}".format(func_name),
    for param in g_params:
        param_string = param.name
        if param.type is not bool:
            param_string += " {}".format(param.type.__name__)
        if not param.required:
            param_string = "[{}]".format(param_string)
        print param_string,


def parse_equality_args(arg):
    if "=" in arg:
        return arg.split("=", 1)
    else:
        return arg


def getopt():
    func_name = sys.argv[0]
    argv_list = sys.argv[1:]

    argv_list = [parse_equality_args(arg) for arg in argv_list]
    if "--help" in argv_list:
        print_help(func_name)
        sys.exit()

    argv_gen = iter(argv_list)
    param_dict = {"--" + param.name: param for param in g_params}
    parsed_param = {}

    # parse given parameters one-by-one
    for arg in argv_gen:
        if arg in param_dict:
            param = param_dict[arg]
            if param.type is bool:
                parsed_param[param.name] = True
            else:
                val = next(argv_gen)
                parsed_param[param.name] = val
        else:
            raise ValueError("Unknown parameter '{}'".format(arg))

    # check required parameters
    for param in g_params:
        if param.required and param.name not in parsed_param:
            if param.type is bool:
                parsed_param[param.name] = False
            else:
                raise ValueError("Parameter {} is required".fromat(param.name))

    # do type conversion
    for param in g_params:
        if param.name in parsed_param:
            try:
                parsed_param[param.name] = param.type(parsed_param[param.name])
            except ValueError:
                raise ValueError("Parameter {} should be {}".format(
                    param.name, param.type.__name__))

    return parsed_param


def main():
    try:
        params = getopt()
        print params
    except SystemExit:
        return

if __name__ == "__main__":
    main()
