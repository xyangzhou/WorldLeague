"""Measure versioned Rigid partial insertion without changing world state.

Used by worldleague scene --help.
"""
from __future__ import annotations


from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from worldleague.authoring.io import write_json
from worldleague.tasks.interaction_setup import array
from worldleague.tasks.rigid_insertion_scoring import RigidInsertionProgress

REVISION = 'wia_rigid_partial_insertion_current_pose_v2'
CURRENT_POSE_POLICY = 'current_pose_v1'
_EMPTY_PAIR_CONTACT_REASON = 'ValueError: Native contact has no pair-specific separation samples.'
_RETAINED_CONTACT_SOURCE = 'retained_native_contact_for_unchanged_sleeping_pair'


@dataclass(frozen=True)
class _NativeContactWitness:
    episode: int
    native_time_s: float
    geometry_stamp: str
    contact_geometry: dict


def _native_bool(value, label):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(label + ' must be a measured native boolean.')
    return bool(value)


def _native_identity(prim):
    path = str(prim.prim_path)
    if not path.startswith('/'):
        raise ValueError('Native prim identity must be an absolute prim path.')
    return {'prim_path': path, 'wrapper_identity': id(prim),
            'class': type(prim).__module__ + '.' + type(prim).__qualname__}


def _native_pose(prim):
    position, quaternion = (array(value) for value in prim.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0:
        raise ValueError('Exact contact geometry requires a finite native world pose.')
    # No normalization or quaternion-sign equivalence: equality is exact.
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _native_scale(prim):
    try:
        value = prim.scale
    except AttributeError:
        if hasattr(type(prim), 'scale') or 'scale' in getattr(prim, '__dict__', {}):
            raise
        return {'exposed': False}
    scale = array(value)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError('Exact contact geometry requires positive finite native scales.')
    return {'exposed': True, 'value': scale.tolist()}


def _shape_array_stamp(value, *, points=False):
    if value is None:
        raise ValueError('Authored native collision shape data is unavailable.')
    values = np.asarray(value)
    if points:
        if values.ndim != 2 or values.shape[1] != 3 or not len(values) or values.dtype.kind not in 'fiu':
            raise ValueError('Native collision mesh points must be numeric nonempty N x 3 values.')
    elif values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu':
        raise ValueError('Native collision topology must be a nonempty integer array.')
    if not np.isfinite(values).all():
        raise ValueError('Native collision shape measurements must be finite.')
    values = np.ascontiguousarray(values)
    return {'shape': list(values.shape), 'dtype': values.dtype.str,
            'values': values.tolist()}


def _native_shape_stamp(mesh):
    """Read authored USD values rather than GeomPrim's cached points/faces."""
    prim = mesh.prim
    kind = str(prim.GetTypeName())
    if kind == 'Mesh':
        points = prim.GetAttribute('points').Get()
        counts = prim.GetAttribute('faceVertexCounts').Get()
        indices = prim.GetAttribute('faceVertexIndices').Get()
        shape = {'type': kind, 'points': _shape_array_stamp(points, points=True),
                 'faceVertexCounts': _shape_array_stamp(counts),
                 'faceVertexIndices': _shape_array_stamp(indices)}
        if (np.any(np.asarray(counts) < 3) or int(np.sum(counts)) != len(indices)
                or np.any(np.asarray(indices) < 0) or np.any(np.asarray(indices) >= len(points))):
            raise ValueError('Native collision mesh topology is inconsistent with its points.')
        approximation = mesh.get_collision_approximation()
        if not isinstance(approximation, str) or not approximation:
            raise ValueError('Native collision mesh approximation is unavailable.')
        shape['collision_approximation'] = approximation
        return shape
    attributes = {'Sphere': ('radius',), 'Cube': ('size',),
                  'Cylinder': ('radius', 'height', 'axis'), 'Cone': ('radius', 'height', 'axis'),
                  'Capsule': ('radius', 'height', 'axis')}.get(kind)
    if attributes is None:
        raise ValueError('Unsupported native collision shape for exact retention: ' + kind)
    shape = {'type': kind}
    for name in attributes:
        value = prim.GetAttribute(name).Get()
        if name == 'axis':
            if value not in ('X', 'Y', 'Z'):
                raise ValueError('Native primitive axis is unavailable.')
            shape[name] = str(value)
        else:
            if isinstance(value, (bool, np.bool_)) or value is None:
                raise ValueError('Native primitive dimensions are unavailable.')
            dimension = float(value)
            if not np.isfinite(dimension) or dimension <= 0:
                raise ValueError('Native primitive dimensions must be positive and finite.')
            shape[name] = dimension
    return shape


def _pair_geometry_stamp(target, container):
    """Capture immutable exact geometry plus separately measured sleep flags."""
    try:
        from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
        objects, sleeping = [], []
        for role, obj in (('target', target), ('container', container)):
            links = []
            for name, link in sorted(obj.links.items()):
                if _native_bool(link.is_meta_link, 'Native meta-link flag'):
                    continue
                meshes = link.collision_meshes
                if not meshes:
                    continue
                identity = _native_identity(link)
                dynamic = isinstance(link, RigidDynamicPrim)
                if dynamic:
                    dynamic = (not _native_bool(link.get_attribute('physics:kinematicEnabled'), 'Native kinematic flag')
                               and _native_bool(link.get_attribute('physics:rigidBodyEnabled'), 'Native rigid-body flag'))
                asleep = _native_bool(link.is_asleep, 'Native sleeping flag') if dynamic else None
                sleeping.append({'object_prim_path': str(obj.prim_path), 'link_prim_path': str(link.prim_path),
                                 'dynamic': dynamic, 'is_asleep': asleep})
                enabled_meshes = []
                for mesh_name, mesh in sorted(meshes.items()):
                    enabled = _native_bool(mesh.collision_enabled, 'Native collision-enabled flag')
                    entry = {'name': mesh_name, **_native_identity(mesh), 'enabled': enabled}
                    if enabled:
                        pose = _native_pose(mesh)
                        scale = _native_scale(mesh)
                        entry.update(world_pose=pose, scale=scale, shape=_native_shape_stamp(mesh))
                    enabled_meshes.append(entry)
                pose = _native_pose(link)
                links.append({'name': name, **identity, 'world_pose': pose,
                              'scale': _native_scale(link), 'collision_meshes': enabled_meshes})
            if not links:
                raise ValueError('Each object needs measured non-meta collision-bearing links.')
            if not any(mesh['enabled'] for link in links for mesh in link['collision_meshes']):
                raise ValueError('Each object needs at least one measured enabled collision mesh.')
            identity = _native_identity(obj)
            pose = _native_pose(obj)
            objects.append({**identity, 'world_pose': pose, 'scale': _native_scale(obj), 'links': links})
        return {'available': True, 'geometry_stamp': json.dumps(objects, sort_keys=True, allow_nan=False),
                'all_dynamic': all(item['dynamic'] for item in sleeping),
                'all_asleep': all(item['is_asleep'] is True for item in sleeping)}
    except (AttributeError, ImportError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
        return {'available': False, 'reason': f'{type(error).__name__}: {error}'}


def _under(path, parent):
    return str(path) == str(parent) or str(path).startswith(str(parent) + '/')


def _contact_bodies(contact):
    bodies = tuple(getattr(contact, key) for key in ('body0', 'body1'))
    if any(body is None or not str(body).startswith('/') for body in bodies):
        raise ValueError('Native contact body identities are unavailable.')
    return tuple(str(body) for body in bodies)


def _bounds_gap(first, second):
    lo, hi = (array(v) for v in first)
    other_lo, other_hi = (array(v) for v in second)
    if (any(value.shape != (3,) for value in (lo, hi, other_lo, other_hi))
            or np.any(hi < lo) or np.any(other_hi < other_lo)):
        raise ValueError('Native bounding boxes must have ordered finite 3D corners.')
    return float(np.linalg.norm(np.maximum(np.maximum(lo - other_hi, other_lo - hi), 0.)))


def _box_gap(first, second):
    return _bounds_gap(first.aabb, second.aabb)


def _aggregate_grasp(robot, controller):
    """Read every arm; the public working hand must be open and all hands empty.

    Inactive finger apertures are diagnostic: the public controller cannot open
    those grippers, so their measured half-open initial posture is not a failure.
    Native attachment, pending release and target contact remain required for
    every arm independently of aperture. Availability describes all-arm
    measurement; pickup_eligible describes a unique constrained working-hand
    pickup. holding/constraint_active aggregate all arms, while object_name
    retains the working-hand identity and held_names lists all identities.
    """
    arms, errors = {}, []
    try:
        arm_names = tuple(robot.arm_names)
        working_arm = controller.working_arm
        if not arm_names or working_arm not in arm_names:
            raise ValueError('The working arm must belong to the native robot arm list.')
        positions = array(robot.get_joint_positions())
        lower, upper = (array(value) for value in robot.control_limits['position'])
        tolerance = float(controller.gripper_config.joint_tolerance_m)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Measured gripper readback tolerance is unavailable.')
    except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return {'available': False, 'holding': None, 'object_name': None,
                'constraint_active': None, 'release_pending': None, 'gripper_open': None,
                'arms': arms, 'measurement_errors': [f'{type(error).__name__}: {error}']}
    for arm in arm_names:
        try:
            obj = robot._ag_obj_in_hand[arm]
            name = obj.name if obj is not None else None
            constraint = robot._ag_obj_constraints[arm] is not None
            pending = robot._ag_release_counter[arm] is not None
            reading = int(robot.is_grasping(arm))
            holding = {-1: False, 1: True}.get(reading)
            if holding is None:
                raise ValueError('Native is_grasping reports an unknown state.')
            dofs = array(robot.controllers[f'gripper_{arm}'].dof_idx).astype(int)
            fingers = robot.finger_joint_names[arm]
            if (len(dofs) != 2 or len(fingers) != 2 or any('prismatic' not in
                    str(robot.joints[name].joint_type).lower() for name in fingers)):
                raise ValueError('Both native grippers require two measured prismatic finger joints.')
            measured, closed, opened = positions[dofs], lower[dofs], upper[dofs]
            if np.any(opened <= closed):
                raise ValueError('Native open gripper limits must exceed closed limits.')
            if name is not None and (not isinstance(name, str) or not name):
                raise ValueError('Native held-object identity is unavailable.')
            if holding and name is None:
                raise ValueError('Native grasp has no measured held-object identity.')
            if constraint and (name is None or not holding):
                raise ValueError('Native attachment and grasp identity are inconsistent.')
            if name is not None and not holding and not pending:
                raise ValueError('Native held-object identity has no grasp or release transition.')
            arms[arm] = {'available': True, 'object_name': name, 'holding': holding,
                         'constraint_active': constraint, 'release_pending': pending,
                         'gripper_open': bool(np.max(np.abs(measured - opened)) <= tolerance),
                         'joint_positions_m': measured.tolist(), 'open_joint_limits_m': opened.tolist(),
                         'closed_joint_limits_m': closed.tolist(), 'is_grasping_state': reading}
        except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as error:
            reason = f'{type(error).__name__}: {error}'
            arms[arm] = {'available': False, 'reason': reason}
            errors.append(f'{arm}: {reason}')
    if errors:
        return {'available': False, 'holding': None, 'object_name': None,
                'constraint_active': None, 'release_pending': None, 'gripper_open': None,
                'working_arm': working_arm, 'arms': arms, 'measurement_errors': errors}
    identities = {row['object_name'] for row in arms.values() if row['object_name'] is not None}
    working = arms[working_arm]
    holding = any(row['holding'] for row in arms.values())
    constraint = any(row['constraint_active'] for row in arms.values())
    # These are observed states, not missing measurements. The legacy pickup
    # contract may reject them while final-state scorers still know that the
    # hands are not released. Keep qualification separate from availability.
    pickup_reasons = []
    if len(identities) > 1:
        pickup_reasons.append('Multiple native held identities cannot fit one working-object sample.')
    if (holding or constraint) and working['object_name'] is None:
        pickup_reasons.append('A nonworking-arm attachment cannot supply working-arm pickup evidence.')
    return {'available': True, 'object_name': working['object_name'],
            'held_names': sorted(identities), 'pickup_eligible': bool(not pickup_reasons
                and working['holding'] and working['constraint_active']),
            'pickup_ineligibility': pickup_reasons,
            'holding': holding, 'constraint_active': constraint,
            'release_pending': any(row['release_pending'] for row in arms.values()),
            'gripper_open': working['gripper_open'],
            'all_grippers_open': all(row['gripper_open'] for row in arms.values()),
            'working_arm': working_arm, 'arms': arms, 'measurement_errors': errors}


def _surface_points(target):
    """Use actual native collision vertices and triangle centres, not box corners."""
    import torch
    points = []
    for link in target.links.values():
        if link.is_meta_link:
            continue
        for mesh in link.collision_meshes.values():
            if not mesh.collision_enabled:
                continue
            local = mesh.points
            if local is not None and len(local):
                points.append(mesh.transform_local_points_to_world(local))
            faces = mesh.faces
            if faces is not None and len(faces):
                points.append(mesh.transform_local_points_to_world(mesh.mesh_face_centroids))
    if not points:
        raise ValueError('Native target collision surface points are unavailable.')
    result = torch.cat(points, dim=0)
    if result.ndim != 2 or result.shape[1] != 3 or not torch.isfinite(result).all():
        raise ValueError('Native target surface samples must be finite 3D world points.')
    return result


def partial_interior_evidence(target, container, margin, *, surface_points=None):
    """Find an inset target-surface witness in a real container cavity.

    Positive evidence uses container metadata meshes independently of Inside,
    then excludes the container's enabled collision solids. A negative finite
    surface sample is inconclusive unless the complete target AABB is disjoint
    from every volume AABB; this avoids certifying impossibility from sampling.
    """
    import torch
    from omnigibson.macros import macros
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('The native interior margin must be finite and positive.')
    types = macros.object_states.contains.CONTAINER_META_LINK_TYPES
    volumes = [link for link in container.links.values()
               if link.is_meta_link and link.meta_link_type in types and link.visual_meshes]
    if not volumes:
        return {'interior_valid': None, 'partial_inside': None,
                'reason': 'Native container interior volume meshes are unavailable.'}
    points = _surface_points(target) if surface_points is None else surface_points
    offsets = torch.tensor([[0., 0., 0.], [margin, 0., 0.], [-margin, 0., 0.],
                            [0., margin, 0.], [0., -margin, 0.],
                            [0., 0., margin], [0., 0., -margin]],
                           dtype=points.dtype, device=points.device)
    query = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    witnesses = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for volume in volumes:
        inside = volume.check_points_in_volume(query)
        witnesses |= inside.reshape(len(points), len(offsets)).all(dim=1)
    solids = [mesh for link in container.links.values() if not link.is_meta_link
              for mesh in link.collision_meshes.values() if mesh.collision_enabled]
    if not solids:
        return {'interior_valid': None, 'partial_inside': None,
                'reason': 'Container collision solids are unavailable.'}
    in_solid = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for mesh in solids:
        in_solid |= mesh.check_points_in_volume(points)
    witnesses &= ~in_solid
    indices = torch.where(witnesses)[0]
    # Native meta links often have no collision meshes: RigidPrim.aabb then
    # degenerates to the link position. Their volume uses visual meshes, whose
    # actual native visual_aabb is the corresponding conservative bound.
    disjoint = bool(not len(indices) and
                   all(_bounds_gap(target.aabb, volume.visual_aabb) > 0. for volume in volumes))
    return {'interior_valid': True, 'partial_inside': True if len(indices) else False if disjoint else None,
            'method': 'native_surface_in_inset_container_volume_excluding_collision_solids',
            'surface_sample_count': len(points), 'positive_witness_count': len(indices),
            'witness_world_m': array(points[indices[:8]]).tolist(),
            'interior_margin_m': margin, 'volume_prim_paths': [str(v.prim_path) for v in volumes],
            'reason': None if len(indices) or disjoint else 'Finite sampling found no conclusive interior witness.'}


class RigidPartialInsertionEvaluator:
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.p, self.output = config['parameters'], Path(output)
        self.objects = {key: session.env.scene.object_registry('name', name)
                        for key, name in config['bindings'].items()}
        self.targets = {key: config['bindings'][key] for key in self.p['targets']}
        self.scorer = RigidInsertionProgress(self.targets, self.p['containers'],
            **{key: self.p[key] for key in ('minimum_lift_m', 'stability_duration_s',
                'maximum_linear_speed_m_s', 'maximum_angular_speed_rad_s', 'max_sample_gap_s')},
            expected_containers=self.p['expected_containers'],
            manipulation_revision=self.p.get('manipulation_revision'))
        self.active = False
        self.answer, self.initial, self.measurement_error = {}, {}, None

        self.last_sample = None
        self._contact_witnesses = {}
        self._contact_witness_episode = 0
        self._contact_witness_time_s = None

    def begin_episode(self, observation):
        self._contact_witnesses.clear()
        self._contact_witness_episode += 1
        self._contact_witness_time_s = None

        self.initial = {}
        for key in dict.fromkeys([*self.p['targets'], *self.p['containers']]):
            try:
                obj = self.objects[key]
                scale = array(obj.scale)
                if scale.shape != (3,) or np.any(scale <= 0):
                    raise ValueError('Native object scale must have three positive components.')
                reference = {'scale': scale.tolist()}
                self.initial[key] = reference
            except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
                self.initial[key] = {'measurement_error': f'{type(error).__name__}: {error}'}
        try:
            sample = self._measure()
            self.measurement_error = None
        except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
            self.measurement_error = f'{type(error).__name__}: {error}'
            sample = self._unavailable_sample(self.measurement_error)
        self.scorer.begin(sample)
        self.last_sample = sample
        self.active = True
        return self.record(observation)

    def _protected_geometry(self):
        """Measure the explicit asset-scale constraint independently of container pose."""
        rows, errors, violations = {}, [], []
        for key in dict.fromkeys([*self.p['targets'], *self.p['containers']]):
            row = {}
            try:
                initial, obj = self.initial[key], self.objects[key]
                if initial.get('measurement_error'):
                    raise ValueError('Initial reference is unavailable: ' + initial['measurement_error'])
                baseline, current = array(initial['scale']), array(obj.scale)
                if baseline.shape != (3,) or current.shape != (3,) or np.any(baseline <= 0) or np.any(current <= 0):
                    raise ValueError('Native scale comparison requires positive three-component scales.')
                relative = np.abs(current / baseline - 1.)
                scale_valid = bool(np.all(relative <= self.p['scale_relative_tolerance']))
                row.update(baseline_scale=baseline.tolist(), current_scale=current.tolist(),
                           scale_relative_change=relative.tolist(), scale_unchanged=scale_valid)
                if not scale_valid:
                    violations.append(key + ':scale_changed')
                row['available'] = True
            except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
                reason = f'{type(error).__name__}: {error}'
                row.update(available=False, reason=reason)
                errors.append(key + ': ' + reason)
            rows[key] = row
        return {'available': not errors, 'unchanged': False if violations else None if errors else True,
                'objects': rows, 'violations': violations, 'measurement_errors': errors,
                'object_pose_policy': CURRENT_POSE_POLICY,
                'scale_relative_tolerance': self.p['scale_relative_tolerance']}

    def _unavailable_sample(self, reason):
        self._contact_witnesses.clear()
        try:
            now = float(self.session.pipeline.og.sim.current_time)
            now = now if np.isfinite(now) else None
        except (AttributeError, ValueError, RuntimeError, TypeError):
            now = None
        sample = {'native_time_s': now, 'measurement_errors': [reason]}
        # A failed geometry read must not erase an independently observed
        # prohibited setter or protected geometry mutation from the reducer's latch.
        try:
            protection = self._protected_geometry()
            sample.update(protected_geometry=protection,
                          control_valid=False if protection['unchanged'] is False else None)
            control = self.controller.geometric_rearrangement_evidence()
            setter = control.get('state_setter_used')
            public_valid = (not setter if control.get('available') is True and type(setter) is bool else None)
            controls = (public_valid, protection['unchanged'])
            sample.update(control_evidence=control,
                          control_valid=False if any(value is False for value in controls) else
                          None if any(value is None for value in controls) else True)
        except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
            sample['measurement_errors'].append(f'{type(error).__name__}: {error}')
        return sample

    def _observe_contact_time(self, native_time_s):
        try:
            if isinstance(native_time_s, (bool, np.bool_)) or native_time_s is None:
                raise ValueError('Native contact sample time is unavailable.')
            now = float(native_time_s)
            if not np.isfinite(now) or (self._contact_witness_time_s is not None and
                                       now < self._contact_witness_time_s):
                raise ValueError('Native contact sample time is unavailable or regressed.')
            self._contact_witness_time_s = now
            return True
        except (ValueError, TypeError, OverflowError):
            self._contact_witnesses.clear()
            return False

    def _contact_geometry_with_sleeping_retention(self, target_key, container_key, touching,
                                                  current, native_time_s, protection):
        """Retain actual separation only for an exactly unchanged sleeping pair."""
        pair = (target_key, container_key)
        if not self._observe_contact_time(native_time_s):
            return current
        previous = self._contact_witnesses.pop(pair, None)
        if touching is not True or not protection.get('available') or protection.get('unchanged') is not True:
            return current
        native_sample = (current.get('available') is True and
                         current.get('separation_source') == 'native_physx_contact_data')
        sleeping_read = (previous is not None and current.get('available') is False and
                         current.get('reason') == _EMPTY_PAIR_CONTACT_REASON)
        if not native_sample and not sleeping_read:
            return current
        measured = _pair_geometry_stamp(self.objects[target_key], self.objects[container_key])
        if measured.get('available') is not True or measured.get('all_dynamic') is not True:
            return current
        now = float(native_time_s)
        if native_sample:
            separations = current.get('separations_m')
            if separations is not None and len(separations) and np.isfinite(separations).all() and type(current.get('penetrating')) is bool:
                self._contact_witnesses[pair] = _NativeContactWitness(
                    self._contact_witness_episode, now, measured['geometry_stamp'], deepcopy(current))
            return current
        if (previous.episode != self._contact_witness_episode or now < previous.native_time_s
                or measured['all_asleep'] is not True or measured['geometry_stamp'] != previous.geometry_stamp):
            return current
        self._contact_witnesses[pair] = previous
        return {**previous.contact_geometry, 'separation_source': _RETAINED_CONTACT_SOURCE,
                'native_sample_time_s': previous.native_time_s, 'current_native_time_s': now}

    def _measure(self):
        from omnigibson.object_states import Touching
        from worldleague.adapters.spatial_relations_physical_contact import _contact_geometry
        native_time = float(self.session.pipeline.og.sim.current_time)
        if not np.isfinite(native_time):
            raise ValueError('Native rigid contact sample time must be finite.')
        self._observe_contact_time(native_time)
        control = self.controller.geometric_rearrangement_evidence()
        setter_used = control.get('state_setter_used')
        public_control_valid = (not setter_used if control.get('available') is True and
                                type(setter_used) is bool else None)
        protection = self._protected_geometry()
        controls = (public_control_valid, protection['unchanged'])
        control_valid = (False if any(value is False for value in controls) else
                         None if any(value is None for value in controls) else True)
        grasp = _aggregate_grasp(self.session.robot, self.controller)
        hand_links = [self.session.robot.eef_links[arm] for arm in self.session.robot.arm_names]
        hand_links += [link for arm in self.session.robot.arm_names for link in self.session.robot.finger_links[arm]]
        withdrawn = all(_box_gap(hand, self.objects[key]) >= self.p['withdrawal_clearance_m']
                        for hand in hand_links for key in self.p['targets'])
        rows = {}
        for key in self.p['targets']:
            target = self.objects[key]
            contacts = target.contact_list()
            robot_contact = any(_under(body, self.session.robot.prim_path)
                                for row in contacts for body in _contact_bodies(row))
            relations = {}
            for container_key in self.p['containers']:
                container = self.objects[container_key]
                try:
                    relation = partial_interior_evidence(target, container, self.p['interior_margin_m'])
                except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
                    relation = {'interior_valid': None, 'partial_inside': None,
                                'reason': f'{type(error).__name__}: {error}'}
                try:
                    native_touching = target.states[Touching].get_value(container)
                    if native_touching is None:
                        raise ValueError('Native target-container Touching evidence is unavailable.')
                    touching = bool(native_touching)
                    contact = _contact_geometry({'obj1': target, 'obj2': container}, touching,
                                                _box_gap(target, container), self.p['penetration_tolerance_m'])
                    contact = self._contact_geometry_with_sleeping_retention(
                        key, container_key, touching, contact, native_time, protection)
                    relation.update(touching=touching,
                        penetration_free=not contact['penetrating'] if contact['available'] else None,
                        contact_geometry=contact)
                except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
                    pair = (key, container_key)
                    self._contact_witnesses.pop(pair, None)
                    relation.update(touching=None, penetration_free=None,
                                    contact_error=f'{type(error).__name__}: {error}')
                relations[container_key] = relation
            rows[key] = {'object_name': target.name, 'lowest_z_m': float(array(target.aabb[0])[2]),
                         'linear_speed_m_s': float(np.linalg.norm(array(target.get_linear_velocity()))),
                         'angular_speed_rad_s': float(np.linalg.norm(array(target.get_angular_velocity()))),
                         'dynamic_collidable': not target.fixed_base and not target.visual_only,
                         'robot_contact': robot_contact, 'containers': relations}
        return {'native_time_s': native_time,
                'control_valid': control_valid, 'control_evidence': control,
                'protected_geometry': protection, 'grasp': grasp,
                'hand_clear': withdrawn, 'targets': rows}

    def sample_step(self):
        if self.active:
            try:
                self.last_sample = self._measure()
                self.scorer.update(self.last_sample)
                self.measurement_error = None
            except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
                self.measurement_error = f'{type(error).__name__}: {error}'
                self.last_sample = self._unavailable_sample(self.measurement_error)
                self.scorer.update(self.last_sample)

    def report_answer(self, answer, observation):
        if len(self.targets) == 1:
            self.answer = {next(iter(self.targets)): str(answer).strip()}
        else:
            value = json.loads(answer)
            if not isinstance(value, dict):
                raise ValueError('Multi-target answers require the declared label-to-container JSON mapping.')
            self.answer = {key: value.get(label, '') for key, label in self.p['target_labels'].items()}

    def record(self, observation, *, action=None):
        self.sample_step()
        measured = self.scorer.result()
        normalize = lambda text: ' '.join(str(text).lower().replace('_', ' ').split())
        answers = {key: normalize(self.answer.get(key, '')) == normalize(expected)
                   for key, expected in self.p['expected_answers'].items()}
        items = deepcopy(measured['completion_items'])
        items += [{'id': 'answer:' + key, 'kind': 'answer_requirement',
                   'completed': correct, 'evaluation_validity': 'valid'} for key, correct in answers.items()]
        result = {**measured, 'schema': REVISION, 'snapshot_id': observation.snapshot_id,
                  'action_step': observation.action_step, 'completion_items': items,
                  'evaluation_validity': measured['evaluation_validity'],
                  'physical_success': bool(measured['physical_success']),
                  'answer_correct': all(answers.values()),
                  'task_success': bool(measured['physical_success'] and all(answers.values())),

                  'diagnostic_reducer_physical_success': measured['physical_success'],
                  'physical_scorer_schema': measured['schema'],
                  'physical_scorer_version': measured['scoring_version'],
                  'reported_answers': self.answer, 'native_measurements': self.last_sample,
                  'measurement_error': self.measurement_error, 'scorer_version': REVISION,
                  'object_pose_policy': CURRENT_POSE_POLICY,
                  'evidence_scope': 'Native physical scoring; QA labels preserved separately.'}
        from worldleague.tasks.completion import progress
        result.update(progress(result))
        if result['evaluation_validity'] != 'complete':
            result['task_success'] = False
            result['physical_success'] = False
        write_json(self.output / 'state.json', result)
        return result


def create_evaluator(session, controller, config, output):
    return RigidPartialInsertionEvaluator(session, controller, config, output)
