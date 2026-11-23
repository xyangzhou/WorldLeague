"""Resolve WIA-owned converter modules.

Used by worldleague scene --help.
"""
from importlib import import_module
import re

_PATTERN = re.compile(r'worldleague\.adapters\.[a-z][a-z0-9_]*')


def is_owned_module(name):
    return isinstance(name, str) and _PATTERN.fullmatch(name) is not None


def canonical_module_name(name):
    if not is_owned_module(name):
        raise ValueError('Use a canonical WIA category adapter.')
    return name


def import_owned_module(name):
    return import_module(canonical_module_name(name))


def module_identity(value):
    return value
