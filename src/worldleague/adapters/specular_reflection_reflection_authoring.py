"""Convert Reflection Authoring to source-bound optical observation and answers.

Used by worldleague scene --help.
"""
from __future__ import annotations


from worldleague.tasks.background_policy import legal_background_contact


import numpy as np

from worldleague.tasks.interaction_setup import array as native_array


def _array(value):
    return native_array(value)


def _vector(value, size):
    result = _array(value)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Missing or invalid source/native geometry.")
    return result


def _under(body, root):
    return str(body) == str(root) or str(body).startswith(str(root) + '/')


def _native_grasp(robot, arm):
    reading = int(robot.is_grasping(arm))
    held = robot._ag_obj_in_hand[arm]
    return {'holding': {-1: False, 1: True}.get(reading),
        'object_name': held.name if held is not None else None,
        'constraint_active': robot._ag_obj_constraints[arm] is not None}


def _contacts(obj, support_paths, *, allowed_contact=None, session=None):
    supports, forbidden, allowed = [], [], []
    for contact in obj.contact_list():
        if session is not None and legal_background_contact(session, contact.body0, contact.body1):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        owned = [body for body in bodies if _under(body, obj.prim_path)]
        external = [body for body in bodies if not _under(body, obj.prim_path)]
        if not external:
            continue
        row = {'bodies': bodies, 'position': _vector(contact.position, 3).tolist(),
            'normal': _vector(contact.normal, 3).tolist()}
        if owned and all(any(_under(body, path) for path in support_paths) for body in external):
            supports.append(row)
        elif owned and allowed_contact is not None and allowed_contact(owned, external):
            allowed.append(row)
        else:
            forbidden.append(row)
    return supports, forbidden, allowed
