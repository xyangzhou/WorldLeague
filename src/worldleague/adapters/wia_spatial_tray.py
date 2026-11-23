"""Read native retrieve-and-arrange tray evidence; place only at initialization.

Used by worldleague scene --help.
"""
from __future__ import annotations



from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from worldleague.adapters import active_layout
from worldleague.adapters.rigid_partial_insertion import _contact_bodies, _surface_points, _under
from worldleague.adapters.spatial_relations_physical_contact import _contact_geometry, _snapshot
from worldleague.adapters.wia_dimensional import _control, _grasp
from worldleague.authoring.io import write_json
from worldleague.tasks.interaction_setup import array
from worldleague.tasks.wia_spatial_tray_scoring import FINAL_STATE_REVISION, SpatialTrayProgress, tray_reference_bounds

MODULE = 'worldleague.adapters.wia_spatial_tray'
OPERATION = 'retrieve_arrange_in_tray'
ERRORS = (AttributeError, ValueError, RuntimeError, TypeError, KeyError, IndexError, AssertionError, ImportError)


def configure_scene(config, parameters):
    return active_layout.configure_scene(config, parameters)


def postprocess_joint_world(session, parameters):
    return active_layout.postprocess_joint_world(session, parameters)


def initialize_scene(session, parameters):
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Spatial-tray positioning is initialization-only.')
    return active_layout.initialize_scene(session, parameters)


def _configuration(session, config, bound=None):
    p = config['parameters']
    if bound is None:
        keys = [p['tray']['deck_key'], *p['targets']]
        bound = {key: session.env.scene.object_registry('name', config['bindings'][key]) for key in keys}
    return p, bound


def _native_boolean(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('An actual native boolean is required.')
    return bool(value)


def _fixture_snapshot(obj):
    """Read actual fixture geometry and native flags without requesting velocity."""
    position, quaternion = (array(value) for value in obj.get_position_orientation())
    low, high = (array(value) for value in obj.aabb)
    if (any(value.shape != (3,) for value in (position, low, high))
            or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0 or np.any(high <= low)):
        raise ValueError('Invalid measured native fixture geometry.')
    model = getattr(obj, 'model', None)
    return {'name': obj.name, 'category': obj.category, 'model': model, 'model_available': model is not None,
        'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
        'visual_only': _native_boolean(obj.visual_only), 'fixed_base': _native_boolean(obj.fixed_base)}


def _pair(first, second, tolerance):
    """Never substitute overlap or Touching alone for pair-specific PhysX data."""
    from omnigibson.object_states import Touching
    first_low, first_high = (array(value) for value in first.aabb)
    second_low, second_high = (array(value) for value in second.aabb)
    widths = np.minimum(first_high, second_high) - np.maximum(first_low, second_low)
    gap = float(np.linalg.norm(np.maximum(-widths, 0.)))
    touching = _native_boolean(first.states[Touching].get_value(second))
    return _contact_geometry({'obj1': first, 'obj2': second}, touching, gap, tolerance=tolerance)


def _deck_pair(target, deck, tolerance, *, on_top):
    """Retain PhysX evidence, or bound penetration at a fixed Cube's top face.

    Static links are absent from RigidContactAPI's dynamic column filters.
    For a horizontal boundingCube and a target strictly inside its XY bounds,
    only the top face can intersect a target extending above the deck. Every
    target collision point is at least target_min_z, so max(0, deck_max_z -
    target_min_z) bounds penetration into that face. A larger bound is
    inconclusive, never evidence of penetration. Native Touching AND OnTop
    remain required; the bound itself supplies neither contact nor support.
    """
    try:
        native = _pair(target, deck, tolerance)
    except ERRORS as error:
        return {'available': False, 'relation': 'unresolved', 'penetrating': None,
                'reason': f'{type(error).__name__}: {error}'}
    if native.get('available') is True:
        return native
    evidence = {'available': False, 'method': 'native_fixed_cube_top_bound'}
    try:
        from omnigibson.object_states import Touching
        if (not _native_boolean(on_top)
                or not _native_boolean(target.states[Touching].get_value(deck))):
            raise ValueError('Actual native Touching and OnTop must both be true.')
        if (not _native_boolean(deck.fixed_base) or _native_boolean(deck.visual_only)
                or getattr(deck, '_primitive_type', None) != 'Cube'):
            raise ValueError('The fallback requires the actual fixed collidable Cube deck.')
        if _native_boolean(target.fixed_base) or _native_boolean(target.visual_only):
            raise ValueError('The target must remain dynamic and collidable.')
        links = [link for link in deck.links.values() if not _native_boolean(link.is_meta_link)]
        if len(links) != 1 or len(links[0].collision_meshes) != 1:
            raise ValueError('The Cube deck must have exactly one physical collision mesh.')
        mesh = next(iter(links[0].collision_meshes.values()))
        if (not _native_boolean(mesh.collision_enabled)
                or mesh.get_collision_approximation() != 'boundingCube'):
            raise ValueError('An enabled native boundingCube collision shape is required.')
        _, quaternion = (array(value) for value in mesh.get_position_orientation())
        # Exact horizontal orientation, not a relaxed fixture-angle tolerance.
        if (quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0
                or np.any(quaternion[:2] != 0.)):
            raise ValueError('The measured Cube collision top must be horizontal.')
        low, high = (array(value) for value in target.aabb)
        deck_low, deck_high = (array(value) for value in mesh.aabb)
        if (any(value.shape != (3,) for value in (low, high, deck_low, deck_high))
                or np.any(high <= low) or np.any(deck_high <= deck_low)):
            raise ValueError('Current collision bounds must be finite and nondegenerate.')
        if (not np.all(low[:2] > deck_low[:2]) or not np.all(high[:2] < deck_high[:2])
                or high[2] <= deck_high[2]):
            raise ValueError('The complete target footprint must be inside the deck and extend above its top.')
        gap = float(low[2] - deck_high[2])
        depth_bound = max(0., -gap)
        evidence.update(target_collision_aabb_m=[low.tolist(), high.tolist()],
            deck_collision_aabb_m=[deck_low.tolist(), deck_high.tolist()],
            deck_collision_prim_path=str(mesh.prim_path),
            deck_collision_quaternion_xyzw=quaternion.tolist(),
            vertical_gap_m=gap, penetration_depth_upper_bound_m=depth_bound)
        if (isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tolerance)
                or tolerance <= 0 or depth_bound > tolerance):
            raise ValueError('The conservative depth bound does not establish the declared penetration tolerance.')
        evidence['available'] = True
        return {'available': True, 'relation': 'touching', 'penetrating': False,
            'separation_source': 'native_fixed_cube_top_bound',
            'penetration_depth_upper_bound_m': depth_bound,
            'penetration_tolerance_m': tolerance, 'native_touching': True, 'native_on_top': True,
            'fixed_deck_bound': evidence, 'native_pair_contact_geometry': deepcopy(native)}
    except ERRORS as error:
        evidence['reason'] = f'{type(error).__name__}: {error}'
        return {**native, 'fixed_deck_bound': evidence}


def _scale(obj):
    """Read the native scale rather than inferring it from a rotated AABB."""
    scale = array(obj.scale)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError('Native object scale must have three positive finite components.')
    return scale.tolist()


def _measure(session, controller, config, bound, baseline=None):
    from omnigibson.object_states import OnTop
    p, errors = config['parameters'], []
    tray, scoring = p['tray'], p['scoring']
    final_state = scoring.get('manipulation_revision') == FINAL_STATE_REVISION
    grasp, raw_grasp = _grasp(session, controller)
    control_valid, control = _control(session, controller)
    if control_valid and control.get('available') is not True:
        control_valid = None
    sample = {'native_time_s': None, 'control_valid': control_valid, 'control_evidence': control,
              'fixtures_valid': None, 'tray_valid': None,
              'grasp': grasp, 'native_grasp_diagnostics': raw_grasp, 'targets': {},
              'fixtures': {}, 'measurement_errors': errors}
    try:
        value = session.pipeline.og.sim.current_time
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError('Native simulation time is unavailable.')
        sample['native_time_s'] = float(value)
    except ERRORS as error:
        errors.append(f'clock: {type(error).__name__}: {error}')
    deck_key = tray['deck_key']
    deck = bound[deck_key]
    rotated = False
    try:
        row = _fixture_snapshot(deck)
        sample['fixtures'][deck_key] = row
        sample['fixtures_valid'] = not row['visual_only']
        # The authored usable rectangle is expressed in the deck's initial frame.
        # The reducer maps each target into that frame at the current deck pose.
        row['scale'] = _scale(deck)
        reference = (baseline or sample)['fixtures'][deck_key]
        sample['fixtures_valid'] &= bool(np.allclose(row['scale'], reference['scale'], rtol=1e-6, atol=1e-8))
        from scipy.spatial.transform import Rotation
        relative = (Rotation.from_quat(reference['quaternion_xyzw'])
                    * Rotation.from_quat(row['quaternion_xyzw']).inv()).as_matrix()
        # Axis permutations preserve a world AABB. Other rotations need collision vertices.
        rotated = not np.allclose(abs(relative), np.rint(abs(relative)), atol=1e-7, rtol=0)
        bounds = array(tray['usable_world_xy_m'])
        low, high = array(reference['aabb_min']), array(reference['aabb_max'])
        sample['tray_valid'] = bool(np.all(bounds[0] >= low[:2]) and np.all(bounds[1] <= high[:2])
            and abs(high[2] - tray['deck_top_world_z_m']) <= scoring['deck_height_tolerance_m'])
    except ERRORS as error:
        errors.append(f'tray: {type(error).__name__}: {error}')
    for key, name in p['targets'].items():
        row = {'name': name, 'dynamic_collidable': None, 'robot_contact': None,
               'on_top_deck': None, 'deck_contact_geometry': {'available': False}, 'measurement_errors': []}
        obj = bound[key]
        try:
            row.update(_snapshot(obj))
            if rotated:
                bounds = tray_reference_bounds(row, sample['fixtures'][deck_key], reference,
                                               points=array(_surface_points(obj)))
                row['tray_frame_bounds_m'] = [bounds['aabb_min'], bounds['aabb_max']]
            if final_state:
                row['scale'] = _scale(obj)
            row['dynamic_collidable'] = not row['visual_only'] and not row['fixed_base']
        except ERRORS as error:
            if rotated:
                row['tray_frame_bounds_m'] = None
            row['measurement_errors'].append(f'geometry: {type(error).__name__}: {error}')
        try:
            contacts = list(obj.contact_list())
            row['robot_contact'] = any(_under(body, session.robot.prim_path)
                for contact in contacts for body in _contact_bodies(contact))
        except ERRORS as error:
            row['measurement_errors'].append(f'robot_contact: {type(error).__name__}: {error}')
        try:
            row['on_top_deck'] = _native_boolean(obj.states[OnTop].get_value(deck))
            row['deck_contact_geometry'] = _deck_pair(obj, deck, scoring['penetration_tolerance_m'],
                                                      on_top=row['on_top_deck'])
        except ERRORS as error:
            row['measurement_errors'].append(f'deck_contact: {type(error).__name__}: {error}')
        sample['targets'][key] = row
    if p['goal_relation']['kind'] == 'contact':
        first, second = p['goal_relation']['pair']
        try:
            sample['contact_geometry'] = _pair(bound[first], bound[second], scoring['penetration_tolerance_m'])
        except ERRORS as error:
            sample['contact_geometry'] = {'available': False, 'relation': 'unresolved',
                                         'reason': f'{type(error).__name__}: {error}'}
    return sample


class SpatialTrayEvaluator:
    """Record passive evidence; only native sample_step calls advance the reducer."""

    def __init__(self, session, controller, config, output):
        self.p, self.bound = _configuration(session, config)
        self.session, self.controller = session, getattr(controller, 'controller', controller)
        self.config, self.output = deepcopy(config), Path(output)
        self.scorer, self.initial, self.last_sample = None, None, None

    def begin_episode(self, observation):
        if self.scorer is not None:
            raise RuntimeError('The spatial-tray episode has already begun.')
        self.initial = _measure(self.session, self.controller, self.config, self.bound)
        self.last_sample = deepcopy(self.initial)

        self.scorer = SpatialTrayProgress(self.p)
        self.scorer.begin(self.initial)
        return self.record(observation)

    def sample_step(self):
        if self.scorer is not None:
            self.last_sample = _measure(self.session, self.controller, self.config, self.bound, self.initial)
            self.scorer.update(self.last_sample)

    def record(self, observation, *, action=None):
        if self.scorer is None:
            return None
        result = self.scorer.result()
        result.update(snapshot_id=observation.snapshot_id, action_step=observation.action_step, action=action,
                      native_measurements=self.last_sample)
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
        return result


def create_evaluator(session, controller, config, output):
    return SpatialTrayEvaluator(session, controller, config, output)
