"""Convert original cloth covers into guarded sweep-and-retrieve operations.

Run CPU batching from the repository root using the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
Native qualification/evaluation only read state; they never execute robot actions.
"""
from __future__ import annotations


from worldleague.tasks.background_policy import legal_background_grasp


import math
from pathlib import Path

import numpy as np


MODULE = 'worldleague.adapters.physical_structure_deformable'
TASK_REVISION = 'task_edit_20261018_v1'
OPERATION = 'sweep_cloth_retrieve_item'
from worldleague.tasks.deformable_support import REVISION as DEFORMABLE_REVISION
LIFT_M = .05
CLOTH_SHIFT_M = .10
CLEARANCE_M = .02
HOLD_SECONDS = .25


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite source or native measurement.')
    return result


def _mesh(cloth):
    points = _array(cloth.root_link.compute_particle_positions())
    faces = _array(cloth.root_link.faces).astype(int)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('Native cloth must expose a nonempty triangular particle mesh.')
    if np.min(faces) < 0 or np.max(faces) >= len(points):
        raise ValueError('Invalid native cloth face indices.')
    return points, points[faces]


def _overlap_count(triangles, low, high, margin=0.):
    """Exact 2D triangle/rectangle SAT, including degenerate edge projections.

    All deformed faces are inspected. A cloth AABB can overlap while the actual
    mesh is clear; neither a source label nor a particle-only test is sufficient.
    """
    xy = triangles[:, :, :2]
    low, high = _array(low)[:2] - margin, _array(high)[:2] + margin
    centre, half = (low + high) / 2, (high - low) / 2
    overlap = np.all(xy.max(axis=1) >= low, axis=1) & np.all(xy.min(axis=1) <= high, axis=1)
    edges = np.roll(xy, -1, axis=1) - xy
    for edge in range(3):
        normal = np.stack((-edges[:, edge, 1], edges[:, edge, 0]), axis=1)
        projections = np.einsum('fvi,fi->fv', xy - centre, normal)
        extent = np.sum(np.abs(normal) * half, axis=1)
        overlap &= (projections.min(axis=1) <= extent) & (projections.max(axis=1) >= -extent)
    return int(np.count_nonzero(overlap))


def _state(session, controller, objects, footprint=None, *, full_cloth_contacts=False):
    from omnigibson.object_states import ContactBodies, Touching
    from worldleague.tasks.interaction_setup import touching_task_support
    from omnigibson.macros import gm
    import omnigibson as og
    item, cloth, floor = [objects[k] for k in ('item', 'cloth', 'floor')]
    low, high = [_array(v) for v in item.aabb]
    position, quaternion = [_array(v) for v in item.get_position_orientation()]
    if abs(np.linalg.norm(quaternion) - 1.) > .002:
        raise ValueError('Native target quaternion is not normalized.')
    points, triangles = _mesh(cloth)
    robot = session.robot
    hand = {robot.eef_links[controller.working_arm], *robot.finger_links[controller.working_arm]}
    other_robot_links = set(robot.links.values()) - hand
    if full_cloth_contacts:
        from worldleague.tasks.deformable_support import native_cloth_contact_paths
        from worldleague.tasks.support_sampling import same_body
        paths = native_cloth_contact_paths(cloth, [item, *robot.links.values()], points=points)
        contacts = {link for link in robot.links.values()
            if any(same_body(link.prim_path, path) for path in paths)}
        item_touching_cloth = any(same_body(item.prim_path, path) for path in paths)
    else:
        contacts = set(cloth.states[ContactBodies].get_value())
        item_touching_cloth = bool(cloth.states[Touching].get_value(item))
    if footprint is None:
        footprint = (low, high)
    floor_top = float(_array(floor.aabb[1])[2])
    footprint_low, footprint_high = [_array(v) for v in robot.links[robot.base_footprint_link_name].aabb]
    rooms = [session.env.scene.seg_map.get_room_instance_by_point(np.array([x, y]))
             for x in (footprint_low[0], footprint_high[0]) for y in (footprint_low[1], footprint_high[1])]
    return {'time_s': float(og.sim.current_time), 'sim_step_dt_s': float(og.sim.get_sim_step_dt()),
            'item_position': position.tolist(),
            'item_aabb_min': low.tolist(), 'item_aabb_max': high.tolist(), 'item_scale': _array(item.scale).tolist(),
            'cloth_scale': _array(cloth.scale).tolist(), 'cloth_centroid': points.mean(axis=0).tolist(),
            'cloth_point_count': len(points), 'cloth_face_count': len(triangles),
            'cloth_min_z': float(points[:, 2].min()),
            'initial_region_overlap_faces': _overlap_count(triangles, *footprint, margin=CLEARANCE_M),
            'above_item_overlap_faces': _overlap_count(triangles[triangles[:, :, 2].max(axis=1) >= low[2]], low, high),
            'working_hand_cloth_contacts': sorted(link.prim_path for link in contacts & hand),
            'other_robot_cloth_contacts': sorted(link.prim_path for link in contacts & other_robot_links),
            'item_touching_cloth': item_touching_cloth,
            'full_cloth_particle_contact_query': full_cloth_contacts,
            'item_touching_floor': bool(item.states[Touching].get_value(floor)),
            'item_touching_task_support': touching_task_support(session, item, floor),
            'item_linear_speed_m_s': float(np.linalg.norm(_array(item.get_linear_velocity()))),
            'floor_top_m': floor_top, 'robot_in_original_room': all(r == session.room_name for r in rooms),
            'rigid_assisted_grasp_mode': robot.grasping_mode == 'assisted' and gm.AG_CLOTH is False,
            'gripper': controller.debug_gripper_state()}




class SweepRetrieveEvaluator:
    """Track hand-driven reveal before a real target grasp and sustained lift."""
    def __init__(self, session, controller, config, output):
        self.parameters = config['parameters']
        self.session, self.controller, self.config = session, controller, config
        self.objects = {key: session.env.scene.object_registry('name', name)
                        for key, name in config['bindings'].items()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.initial = 0, None

    def begin_episode(self, observation):
        initial = _state(self.session, self.controller, self.objects,
            full_cloth_contacts=self.parameters.get('cover_revision') == 'full_cover_redrape_v1')
        self.episode += 1
        self.initial = initial
        self.hand_contact_seen, self.revealed_before_grasp, self.invalid = False, False, False
        self.measurement_error = None
        self.last_time = self.initial['time_s']
        self.hold_start, self.hold_samples, self.criteria = None, 0, {}
        self.record(observation, action=None)

    def _update(self):
        contact_options = {'full_cloth_contacts': True} if self.parameters.get('cover_revision') == 'full_cover_redrape_v1' else {}
        row = _state(self.session, self.controller, self.objects,
                     (self.initial['item_aabb_min'], self.initial['item_aabb_max']), **contact_options)
        g, initial = row['gripper'], self.initial
        held = (g.get('holding') is True and g.get('constraint_active') is True and
                g.get('object_name') == self.objects['item'].name and g.get('release_pending') is False)
        now = row['time_s']
        if not math.isfinite(now) or now < self.last_time - 1e-8:
            self.invalid = True
        new_sample = now > self.last_time + 1e-8
        dt = row['sim_step_dt_s']
        if not math.isfinite(dt) or dt <= 0:
            self.invalid = True
        consecutive = new_sample and 0 < now - self.last_time <= dt * 1.5 + 1e-7
        shift = float(np.linalg.norm(_array(row['cloth_centroid'])[:2] - initial['cloth_centroid'][:2]))
        lift = min(float(row['item_aabb_min'][2] - initial['item_aabb_min'][2]),
                   float(row['item_position'][2] - initial['item_position'][2]))
        item_still = (np.linalg.norm(_array(row['item_position'])[:2] - initial['item_position'][:2]) <= .03 and
                      abs(row['item_aabb_min'][2] - initial['item_aabb_min'][2]) <= .015 and row['item_touching_task_support'])
        cloth_clear = row['initial_region_overlap_faces'] == 0 and shift >= CLOTH_SHIFT_M
        if new_sample:
            self.hand_contact_seen |= bool(row['working_hand_cloth_contacts'])
            if held and not self.revealed_before_grasp:
                self.invalid = True
            if self.hand_contact_seen and cloth_clear and item_still and g.get('object_name') is None:
                self.revealed_before_grasp = True
            if (g.get('object_name') not in (None, self.objects['item'].name)
                    and not legal_background_grasp(self.session, g.get('object_name'))):
                self.invalid = True
        unchanged = (row['item_scale'] == initial['item_scale'] and row['cloth_scale'] == initial['cloth_scale'] and
                     row['cloth_point_count'] == initial['cloth_point_count'] and
                     row['cloth_face_count'] == initial['cloth_face_count'] and
                     abs(row['floor_top_m'] - initial['floor_top_m']) <= .001 and
                     all(not self.objects[key].visual_only and not self.objects[key].fixed_base for key in ('item', 'cloth')))
        self.criteria = {'working_hand_contact_seen': self.hand_contact_seen,
                         'native_cloth_moved_10cm_and_mesh_clear': cloth_clear,
                         'reveal_preceded_target_grasp': self.revealed_before_grasp,
                         'real_target_grasp_constraint_active': held, 'target_lift_5cm': lift >= LIFT_M,
                         'target_clear_of_cloth_and_task_surface': not row['item_touching_cloth'] and not row['item_touching_task_support'],
                         'original_physics_geometry_and_native_mode': unchanged and row['rigid_assisted_grasp_mode'],
                         'robot_remains_in_original_room': row['robot_in_original_room'],
                         'valid_operation_history': not self.invalid}
        valid = all(self.criteria.values())
        if not valid:
            self.hold_start, self.hold_samples = None, 0
        elif new_sample:
            if not consecutive:
                self.hold_start, self.hold_samples = None, 0
            if self.hold_start is None:
                self.hold_start = now
            self.hold_samples += 1
        self.last_time = max(now, self.last_time)
        self.criteria['sustained_native_hold'] = bool(valid and self.hold_start is not None and
            now - self.hold_start >= HOLD_SECONDS and self.hold_samples >= 3)
        return row, lift, shift

    def sample_step(self):
        if self.initial is not None:
            try:
                self._update()
            except Exception as error:
                self.invalid = True
                self.measurement_error = f'{type(error).__name__}: {error}'
                self.hold_start, self.hold_samples = None, 0

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        from worldleague.authoring.io import write_json
        try:
            row, lift, shift = self._update()
            value = {'schema': 'private_deformable_sweep_retrieve_v1', 'episode': self.episode,
                     'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                     'action': action, 'criteria': self.criteria, 'task_success': bool(all(self.criteria.values())),
                     'target_lift_m': lift, 'cloth_centroid_xy_shift_m': shift,
                     'hold_native_samples': self.hold_samples, 'initial_native_state': self.initial,
                     'native_state': row, 'configuration': self.config}
            value.update(evaluation_validity='unavailable' if self.measurement_error else 'complete',
                measurement_error=self.measurement_error, task_revision=DEFORMABLE_REVISION if
                    self.parameters.get('task_revision') == DEFORMABLE_REVISION else TASK_REVISION,
                completion_items=[{'id': 'native_cloth_reveal', 'completed': self.revealed_before_grasp,
                    'kind': 'checkpoint', 'evaluation_validity': 'valid' if self.revealed_before_grasp or not self.measurement_error else 'unavailable'},
                    {'id': 'native_retrieval', 'completed': bool(self.criteria['sustained_native_hold']),
                     'kind': 'terminal_condition', 'evaluation_validity': 'unavailable' if self.measurement_error else 'valid'}])
            if self.measurement_error:
                value['task_success'] = False
        except Exception as error:
            self.invalid = True
            self.measurement_error = f'{type(error).__name__}: {error}'
            self.hold_start, self.hold_samples = None, 0
            value = {'schema': 'private_deformable_sweep_retrieve_v1', 'episode': self.episode,
                     'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                     'action': action, 'task_success': False, 'measurement_unavailable': True,
                     'evaluation_validity': 'unavailable', 'measurement_error': str(error),
                     'exception_summary': f'{type(error).__name__}: {error}'}
        write_json(self.output / 'state.json', value)
        import json
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return SweepRetrieveEvaluator(session, controller, config, output)
