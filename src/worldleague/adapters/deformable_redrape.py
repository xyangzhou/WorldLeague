"""Initialize the authored floor layout using a fully draped native cloth.

Used by worldleague scene --help.
"""
from __future__ import annotations

import numpy as np

from worldleague.adapters.physical_structure_deformable import _array, _mesh


def configure_scene(config, parameters):
    """Set cloth scale before native cloth cooking establishes spring rest lengths."""
    from omnigibson.utils.constants import PrimType
    item, cloth = parameters['item_asset'], parameters['cloth_asset']
    x, y = parameters['source_item_xy']
    floor = parameters['floor_height_m']
    config.setdefault('objects', []).extend([
        {'type': 'DatasetObject', **item, 'position': [x, y, floor + .15],
         'orientation': [0., 0., 0., 1.], 'fixed_base': False, 'visual_only': False},
        {'type': 'DatasetObject', **cloth, 'position': [1000., 1000., floor + .8],
         'orientation': [0., 0., 0., 1.], 'fixed_base': False, 'visual_only': False,
         'prim_type': PrimType.CLOTH, 'abilities': {'cloth': {}},
         'scale': [parameters['cloth_scale_multiplier']] * 3,
         'load_config': {'default_point_configuration': 'default'}}])
    return {'cloth_scale_set_before_native_cooking': True}


def postprocess_joint_world(session, parameters):
    """The authored inventory replaces source postprocessing for this revision."""
    return {'joint_world_postprocess': True, 'native_cloth_loaded_at_authored_scale': True}


def projected_cover(triangles, low, high, resolution=17):
    """Measure vertical ray coverage of an inset target footprint, not AABB overlap."""
    low, high = np.asarray(low), np.asarray(high)
    inset = np.minimum((high - low)[:2] * .02, .001)
    grid = np.array([[x, y] for x in np.linspace(low[0] + inset[0], high[0] - inset[0], resolution)
                     for y in np.linspace(low[1] + inset[1], high[1] - inset[1], resolution)])
    # A face must intersect the upward ray above the target, not merely have one high vertex.
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    determinant = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    valid = abs(determinant) > 1e-12
    hit = np.zeros(len(grid), dtype=bool)
    for index, point in enumerate(grid):
        delta = point - a[:, :2]
        u = np.divide(delta[:, 0] * ac[:, 1] - delta[:, 1] * ac[:, 0], determinant,
                      out=np.zeros_like(determinant), where=valid)
        v = np.divide(ab[:, 0] * delta[:, 1] - ab[:, 1] * delta[:, 0], determinant,
                      out=np.zeros_like(determinant), where=valid)
        height = a[:, 2] + u * ab[:, 2] + v * ac[:, 2]
        hit[index] = np.any(valid & (u >= -1e-6) & (v >= -1e-6) &
                            (u + v <= 1. + 1e-6) & (height >= low[2] + .001))
    return {'sample_count': len(grid), 'covered_samples': int(hit.sum()),
            'coverage_fraction': float(hit.mean()), 'all_sampled_top_rays_covered': bool(hit.all()),
            'method': 'barycentric_mesh_projection_above_target_base',
            'side_view_visibility_verified': False}


def initialize_scene(session, parameters):
    """Use the asset's default cloth configuration and let native physics drape it."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Cloth redraping is initialization-only.')
    import torch
    import omnigibson as og
    cloth = session.env.scene.object_registry('name', parameters['cloth_name'])
    items = [session.env.scene.object_registry('name', name) for name in parameters['item_names']]
    session._redesigned_cloth_inventory = parameters
    if cloth is None or any(item is None for item in items):
        raise ValueError('The explicitly declared cover inventory is unavailable.')
    robot_before = _array(session.robot.get_position_orientation()[0]).tolist()
    if parameters.get('robot_clearance_xy_m'):
        from worldleague.tasks.interaction_setup import _place_robot
        _place_robot(session, session.robot, parameters['robot_clearance_xy_m'],
            [*parameters['source_item_xy'], parameters['floor_height_m']],
            parameters['floor_height_m'])
    cloth.root_link.mass = float(parameters['cloth_mass_kg'])
    for _ in range(90):
        og.sim.step()
    configurations = list(cloth.root_link.get_available_configurations())
    if 'default' not in configurations:
        raise ValueError('This cloth has no native default configuration for redraping.')
    scale = float(parameters.get('cloth_scale_multiplier', 1.))
    if not np.allclose(_array(cloth.scale), scale):
        raise ValueError('Cloth must be loaded at the authored scale before native cooking.')
    cloth.root_link.reset_points_to_configuration('default')
    lows, highs = zip(*[tuple(_array(v) for v in item.aabb) for item in items])
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    centre = (low + high) / 2
    centre[:2] += np.asarray(parameters.get('cloth_drop_xy_offset_m', [0., 0.]))
    clearance = float(parameters.get('cloth_drop_clearance_m', .035))
    cloth.set_position_orientation(
        position=torch.tensor([centre[0], centre[1], high[2] + clearance], dtype=torch.float32),
        orientation=torch.tensor([0., 0., 0., 1.], dtype=torch.float32))
    # Asset origins need not coincide with the flat mesh centre. Translate the
    # complete native configuration rigidly; never reshape individual particles.
    points, _ = _mesh(cloth)
    mesh_low, mesh_high = points.min(axis=0), points.max(axis=0)
    delta = np.array([centre[0] - (mesh_low[0] + mesh_high[0]) / 2,
                      centre[1] - (mesh_low[1] + mesh_high[1]) / 2,
                      high[2] + clearance - mesh_low[2]])
    cloth.root_link.set_particle_positions(torch.tensor(points + delta, dtype=torch.float32))
    # Low initial energy avoids the original fast downward throw folding an edge.
    cloth.root_link.set_linear_velocity(torch.zeros(3))
    for _ in range(int(parameters.get('settle_steps', 180))):
        og.sim.step()
    _, triangles = _mesh(cloth)
    receipts = {item.name: projected_cover(triangles, *[_array(v) for v in item.aabb])
                for item in items}
    result = {'revision': 'native_cloth_redrape_20261020_v1', 'configuration': 'default',
              'available_configurations': configurations, 'cloth_scale_multiplier': scale,
              'flat_mesh_extent_m': (mesh_high - mesh_low).tolist(),
              'cloth_drop_xy_offset_m': parameters.get('cloth_drop_xy_offset_m', [0., 0.]),
              'cloth_drop_clearance_m': clearance,
              'robot_position_before_clearance_m': robot_before,
              'robot_position_during_drop_m': _array(session.robot.get_position_orientation()[0]).tolist(),
              'initial_mesh_centre_translation_m': delta.tolist(),
              'items': receipts, 'accepted': all(v['all_sampled_top_rays_covered'] for v in receipts.values()),
              'validation_scope': 'Sampled top mesh coverage; mounted and side views need separate validation.'}
    session._redesigned_cloth_cover = result
    # Retain failed layouts as diagnostic captures rather than claiming coverage.
    return result


def on_interaction_initialized(session, evidence):
    """Attach final mounted-view coverage measurements to the capture receipt."""
    for sensor in session.robot.sensors.values():
        if 'rgb' in sensor.modalities and 'seg_instance' not in sensor.modalities:
            sensor.add_modality('seg_instance')
def prepare_interaction(session, bindings, policy):
    """Keep the native redrape layout instead of searching for a rigid table."""
    receipt = session._redesigned_cloth_cover
    inventory = session._redesigned_cloth_inventory
    if bindings['cloth'].name != inventory['cloth_name'] or bindings['item'].name not in inventory['item_names']:
        raise ValueError('The redrape inventory does not match the task bindings.')
    result = {'accepted': receipt.get('accepted') is True, 'relocated': False,
        'layout_owner': 'native_cloth_redrape_20261020_v1',
        'native_redrape_initialization': receipt,
        'group_translation_m': [0., 0., 0.],
        'reason': None if receipt.get('accepted') is True else 'native_redrape_cover_incomplete'}
    session.category_interaction_setup = result
    session._category_interaction_initialized = True
    return result


def _legacy_config(config):
    from worldleague.adapters import physical_structure_deformable as legacy
    return {**config, 'module': legacy.MODULE, 'operation': legacy.OPERATION,
        'bindings': {key: config['bindings'][key] for key in ('item', 'cloth', 'floor')},
        'binding_keys': ['item', 'cloth', 'floor']}


class UncoverSelectPlaceEvaluator:
    """Reuse real-cloth reveal evidence and require delivery of the revealed item."""
    def __init__(self, session, controller, config, output):
        from worldleague.adapters.physical_structure_deformable import SweepRetrieveEvaluator
        from worldleague.adapters.temporal_understanding_agent_observation import NativePlacementProgress
        from pathlib import Path
        self.session, self.controller, self.config = session, controller, config
        self.base = SweepRetrieveEvaluator(session, controller, _legacy_config(config), Path(output) / 'uncover')
        self.placement = NativePlacementProgress(config['parameters'])
        self.retrieved = False
        self.other_preserved = True

    def _other_positions(self):
        return {key: _array(self.session.env.scene.object_registry(
            'name', self.config['bindings'][key]).get_position_orientation()[0])
            for key in self.config['parameters'].get('protected_extra_keys', [])}

    def begin_episode(self, observation):
        self.initial_others = self._other_positions()
        self.base.begin_episode(observation)
        return self.record(observation, action=None)

    def _sample(self, count_step=False):
        from worldleague.adapters.temporal_understanding_agent_observation import native_temporal_placement
        if self.base.initial is None:
            return False, False
        self.base._update()
        self.retrieved |= bool(self.base.criteria and all(self.base.criteria.values()))
        self.other_preserved &= all(np.linalg.norm(point - self.initial_others[key]) <= .03
            for key, point in self._other_positions().items())
        measured = native_temporal_placement(self.session, self.controller, self.config,
            'item', self.config['parameters']['destination_key'])
        return self.placement.update(measured, self.config['bindings']['item'], count_step=count_step)

    def sample_step(self):
        self._sample(True)

    def record(self, observation, *, action=None):
        from worldleague.authoring.io import write_json
        from worldleague.tasks.completion import progress
        available, placed = self._sample()
        reveal_available = getattr(self.base, 'measurement_error', None) is None
        available = available and reveal_available
        items = [
            {'id': 'uncover_then_retrieve', 'kind': 'checkpoint', 'completed': self.retrieved, 'evaluation_validity': 'valid' if self.retrieved or reveal_available else 'unavailable'},
            {'id': 'selected_target_delivered', 'kind': 'terminal_condition', 'completed': placed,
             'evaluation_validity': 'valid' if available else 'unavailable'}]
        value = {'schema': 'cloth_uncover_select_place_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'completion_items': items,
            'task_success': bool(available and self.retrieved and placed and self.other_preserved and not self.base.invalid),
            'constraints_satisfied': bool(self.other_preserved and not self.base.invalid),
            'evaluation_validity': 'complete' if available else 'unavailable',
            'native_placement': self.placement.last,
            'measurement_error': getattr(self.base, 'measurement_error', None)}
        value.update(progress(value))
        write_json(self.base.output.parent / 'state.json', value)
        return value


def create_evaluator(session, controller, config, output):
    return UncoverSelectPlaceEvaluator(session, controller, config, output)
