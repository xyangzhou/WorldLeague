"""Structured, public observation claims.

Used by worldleague scene --help.
"""
from copy import deepcopy


def validate_declarations(value):
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError('observed_objects must be a list with at most 32 entries.')
    for row in value:
        if (not isinstance(row, dict) or set(row) != {'label', 'camera'}
                or any(not isinstance(row[k], str) or not row[k].strip() or len(row[k]) > 160
                       for k in ('label', 'camera'))):
            raise ValueError('Each observed_objects entry requires only nonempty label and camera strings.')
    return deepcopy(value)
