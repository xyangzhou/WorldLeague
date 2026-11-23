"""Native evidence for action-only evaluations.

Used by worldleague continuous --help.
"""


def freeze_native_boundary(native, observation):
    """Use the actual final physical boundary, never an invented DONE action."""
    freeze = getattr(native, 'freeze_done_boundary', None)
    if callable(freeze):
        freeze(observation)
