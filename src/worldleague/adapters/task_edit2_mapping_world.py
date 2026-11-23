"""Compose Mapping goals in the full native scene.

Used by worldleague scene --help.
"""
from __future__ import annotations

from copy import deepcopy

from worldleague.adapters.composite_task import CompositeEvaluator

REVISION = 'task_edit2_mapping_world_20261018_v1'


def configure_scene(config, parameters):
    """Request the complete source building without replacing or adding assets."""
    native = config.setdefault('scene', {})
    removed = {}
    for key in ('load_room_instances', 'load_room_types', 'load_object_categories',
                'not_load_object_categories'):
        if key in native:
            removed[key] = native.pop(key)
    return {'full_building_requested': True, 'cleared_filters': removed,
            'assets_added': []}


class MappingJointWorldEvaluator(CompositeEvaluator):
    """Apply the joint scene contract to each child evaluator."""
    def __init__(self, session, controller, config, output):
        runtime = deepcopy(config)
        # Only this private runtime copy uses CompositeEvaluator's explicit
        # compatibility extension. Persisted source configs remain byte-for-byte.
        for member in runtime['parameters']['members']:
            member['evaluation']['parameters']['mapping_composition_contract'] = {
                'joint_contract': config['parameters']['mapping_joint_world_contract']}
        super().__init__(session, controller, runtime, output)




def create_evaluator(session, controller, config, output):
    return MappingJointWorldEvaluator(session, controller, config, output)
