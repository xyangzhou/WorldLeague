"""Restore reference optical identity using reset-only PandaOmron world XY.

Used by worldleague continuous --help.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

PROTOCOL = 'pandaomron_reference_eye_xy_v1'
NEARBY_XY_SEARCH_CONFIG = {'protocol': 'pandaomron_reference_eye_xy_search_v1',
                           'radius_m': 0.5, 'step_m': 0.1, 'max_candidates': 81}
POSITION_PRECISION_M = 1e-5
POSTURE_TOLERANCE = 1e-5





def _nearby_xy_offsets():
    """Return all 81 grid offsets, nearest first with deterministic integer ties."""
    indices = [(ix, iy) for ix in range(-5, 6) for iy in range(-5, 6)
               if ix * ix + iy * iy <= 25]
    indices.sort(key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, *pair))
    return [[ix * 0.1, iy * 0.1] for ix, iy in indices]


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Native measurement is nonfinite.')
    return result


def _configuration(session):
    task = session.loaded_task
    private = task.definition.private
    config = deepcopy(private.evaluation.model_dump(mode='json'))
    config.update(scene=task.scene_name, room=task.room_name)
    config['bindings'] = {key: private.bindings[key].name for key in config['binding_keys']}
    bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
    if any(obj is None or obj.name != config['bindings'][key] for key, obj in bound.items()):
        raise ValueError('Exact current native task bindings are unavailable.')
    return config, bound


def _floor(session):
    declarations = [item['parameters']['layout']['floor_name']
        for item in session.scene_adapter._realized
        if item.get('operation') == 'category_initialization'
        and item.get('module') == 'worldleague.adapters.wia_mirror']
    if not declarations or len(set(declarations)) != 1:
        raise ValueError('Realized mirror initializers must agree on one explicit floor_name.')
    floor = session.env.scene.object_registry('name', declarations[0])
    if (floor is None or floor.name != declarations[0] or floor.category not in ('floor', 'floors')
            or floor.visual_only):
        raise ValueError('The authored floor must resolve to the actual loaded collidable floor.')
    return floor, float(_array(floor.aabb[1])[2])


def _snapshot(robot, bound):
    pose = lambda obj, **kwargs: [_array(value).tolist() for value in obj.get_position_orientation(**kwargs)]
    joints = _array(robot.get_joint_positions())
    indices = _array(robot.controllers['base'].dof_idx).astype(int)
    mask = np.ones(len(joints), dtype=bool)
    mask[indices] = False
    return {'robot_pose': pose(robot), 'joint_positions': joints.tolist(),
            'posture_indices': np.flatnonzero(mask).tolist(),
            'objects': {key: pose(obj) for key, obj in bound.items()},
            'camera_mounts': {name: pose(sensor, frame='parent') for name, sensor in robot.sensors.items()}}


def _angle(a, b):
    return float((Rotation.from_quat(a).inv() * Rotation.from_quat(b)).magnitude())


def _invariants(before, after, config, settings):
    scoring = config['parameters']['scoring']
    pt, qt = scoring['fixture_position_tolerance_m'], scoring['fixture_angle_tolerance_rad']
    objects = {key: {'position_error_m': float(np.linalg.norm(_array(a[0]) - _array(after['objects'][key][0]))),
                      'orientation_error_rad': _angle(a[1], after['objects'][key][1])}
               for key, a in before['objects'].items()}
    for row in objects.values():
        row['accepted'] = row['position_error_m'] <= pt and row['orientation_error_rad'] <= qt
    calibration = settings.get('camera_calibration', {})
    mpt = min(float(calibration.get('mount_position_tolerance_m', 1e-6)), POSITION_PRECISION_M)
    mqt = float(calibration.get('mount_quaternion_abs_dot_tolerance', 1e-6))
    if not np.isfinite([mpt, mqt]).all() or min(mpt, mqt) < 0:
        raise ValueError('Camera mount profile tolerances must be finite and nonnegative.')
    mounts = {}
    for name, a in before['camera_mounts'].items():
        b = after['camera_mounts'][name]
        aq, bq = _array(a[1]), _array(b[1])
        pos = float(np.max(np.abs(_array(a[0]) - _array(b[0]))))
        dot = abs(1 - abs(float(np.dot(aq / np.linalg.norm(aq), bq / np.linalg.norm(bq)))))
        mounts[name] = {'position_error_m': pos, 'quaternion_abs_dot_error': dot,
                        'accepted': pos <= mpt and dot <= mqt}
    indices = before['posture_indices']
    joint_error = float(np.max(np.abs(_array(before['joint_positions'])[indices]
                                    - _array(after['joint_positions'])[indices])))
    orientation_error = _angle(before['robot_pose'][1], after['robot_pose'][1])
    checks = {'object_poses_preserved': all(row['accepted'] for row in objects.values()),
              'camera_parent_mounts_preserved': all(row['accepted'] for row in mounts.values()),
              'joint_posture_preserved': joint_error <= POSTURE_TOLERANCE,
              'root_orientation_preserved': orientation_error <= POSTURE_TOLERANCE}
    return {**checks, 'accepted': all(checks.values()), 'objects': objects, 'camera_mounts': mounts,
            'joint_posture_max_error': joint_error, 'root_orientation_error_rad': orientation_error,
            'tolerances': {'object_position_m': pt, 'object_orientation_rad': qt,
                          'joint_posture': POSTURE_TOLERANCE, 'root_orientation_rad': POSTURE_TOLERANCE,
                          'mount_position_m': mpt, 'mount_quaternion_abs_dot': mqt}}


def _settle_and_measure(session, robot, bound, config, settings, floor, floor_z, stage, receipt):
    """Run the existing initialization settling boundary and measure its effects."""
    import worldleague.adapters.wia_mirror as mirror
    from worldleague.tasks.interaction_setup import _robot_contact_check
    robot.keep_still()
    for _ in range(30):
        session.pipeline.og.sim.step()
        stage['physics_steps'] += 1
        receipt['physics_steps'] += 1
    for _ in range(4):
        session.pipeline.og.sim.render()
        stage['render_updates'] += 1
        receipt['render_updates'] += 1
    stage['after'] = _snapshot(robot, bound)
    stage['invariants'] = _invariants(stage['before'], stage['after'], config, settings)
    stage['native_checks'] = _robot_contact_check(robot, [str(floor.prim_path)], floor_z)
    stage['geometry_after'] = mirror.capture_initial_geometry(session, config, bound)


def _native_stage_accepted(stage):
    return (stage['invariants']['accepted'] and stage['geometry_after']['available'] is True
        and all(stage['native_checks'].get(key) is True for key in ('floor_supported', 'upright', 'collision_free')))


def _initialize_nearby_xy(session, reference, settings, robot, bound, config, floor, floor_z, receipt):
    """Try independent native states from the same grounded simulator baseline."""
    from worldleague.environments.pandaomron_initialization import (
        CandidateFloorSupportError, capture_initial_base_anchors, place_initial_robot_on_floor,
        restore_initial_base_anchors)
    import worldleague.adapters.wia_mirror as mirror
    from worldleague.tasks.interaction_setup import _robot_contact_check

    sim = session.pipeline.og.sim
    baseline_state = deepcopy(sim.dump_state(serialized=False))
    baseline_anchors = capture_initial_base_anchors(session)

    offsets = _nearby_xy_offsets()
    search = receipt['nearby_xy_search'] = {
        'protocol': settings['nearby_xy_search']['protocol'],
        'config': deepcopy(settings['nearby_xy_search']),
        'order': {'sort_keys': ['ix_squared_plus_iy_squared', 'ix', 'iy'],
                  'requested_offsets_m': offsets},
        'attempts': [], 'selected_index': None, 'restorations': []}
    before, geometry_before = receipt['before'], receipt['geometry_before']
    eye = _array(geometry_before['camera']['position_m'])
    reference_xy = _array(reference['frozen_eye_m'])[:2]

    def restore(reason, index=None):
        row = {'reason': reason, 'candidate_index': index, 'accepted': False,
               'simulator_state_restored': False, 'anchors_restored': False}
        search['restorations'].append(row)
        try:
            sim.load_state(deepcopy(baseline_state), serialized=False)
            row['simulator_state_restored'] = True
            row['anchor_receipts'] = restore_initial_base_anchors(session, baseline_anchors)
            row['anchors_restored'] = row['accepted'] = True
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            raise

    try:
        for index, offset in enumerate(offsets):
            restore('before_candidate', index)
            requested_eye_xy = reference_xy + _array(offset)
            delta = requested_eye_xy - eye[:2]
            requested_robot_xy = _array(before['robot_pose'][0])[:2] + delta
            stage = {'index': index, 'accepted': False, 'before': before,
                     'geometry_before': geometry_before, 'physics_steps': 0, 'render_updates': 0,
                     'requested_offset_from_reference_m': offset,
                     'requested': {'robot_world_xy_m': requested_robot_xy.tolist(),
                         'camera_world_xy_m': requested_eye_xy.tolist(),
                         'delta_world_xy_m': delta.tolist(), 'orientation_xyzw': before['robot_pose'][1]}}
            search['attempts'].append(stage)
            receipt['xy_adaptation'] = stage
            receipt['requested'] = stage['requested']
            # Remove legacy fields from the previous attempt until this attempt measures them.
            for key in ('placement', 'after', 'invariants', 'native_checks', 'geometry_after',
                        'camera_xy_error_m', 'camera_xy_matches_reference',
                        'camera_xy_requested_error_m', 'camera_xy_matches_requested',
                        'camera_distance_from_reference_m', 'camera_within_search_radius',
                        'camera_height_change_m', 'camera_height_preserved', 'optical_identity_preserved'):
                receipt.pop(key, None)
            try:
                stage['placement'] = place_initial_robot_on_floor(
                    session, robot, requested_robot_xy, before['robot_pose'][1], floor_z)
                _settle_and_measure(session, robot, bound, config, settings, floor, floor_z, stage, receipt)
            except CandidateFloorSupportError as error:
                stage['floor_support_error'] = f'{type(error).__name__}: {error}'
                stage['after'] = _snapshot(robot, bound)
                stage['invariants'] = _invariants(before, stage['after'], config, settings)
                stage['native_checks'] = _robot_contact_check(robot, [str(floor.prim_path)], floor_z)
                stage['geometry_after'] = mirror.capture_initial_geometry(session, config, bound)
            for key in ('placement', 'after', 'invariants', 'native_checks', 'geometry_after'):
                if key in stage:
                    receipt[key] = stage[key]
            camera = stage['geometry_after'].get('camera', {})
            actual = camera.get('position_m')
            checks = {'native_qualification': bool(_native_stage_accepted(stage)),
                      'floor_support_placement': 'floor_support_error' not in stage}
            if actual is not None:
                actual = _array(actual)
                stage['camera_xy_error_m'] = float(np.max(np.abs(actual[:2] - reference_xy)))
                stage['camera_xy_requested_error_m'] = float(np.max(np.abs(actual[:2] - requested_eye_xy)))
                stage['camera_distance_from_reference_m'] = float(np.linalg.norm(actual[:2] - reference_xy))
                stage['camera_height_change_m'] = float(actual[2] - eye[2])
                stage['camera_xy_matches_reference'] = stage['camera_xy_error_m'] <= POSITION_PRECISION_M
                stage['camera_xy_matches_requested'] = stage['camera_xy_requested_error_m'] <= POSITION_PRECISION_M
                stage['camera_within_search_radius'] = stage['camera_distance_from_reference_m'] <= 0.5
                stage['camera_height_preserved'] = abs(stage['camera_height_change_m']) <= POSITION_PRECISION_M
            else:
                for key in ('camera_xy_matches_reference', 'camera_xy_matches_requested',
                            'camera_within_search_radius', 'camera_height_preserved'):
                    stage[key] = False
            optics = stage['geometry_after'].get('optical_selection', {})
            passing = [key for key, row in optics.get('candidates', {}).items() if row.get('accepted') is True]
            stage['optical_identity_preserved'] = bool(optics.get('available') is True
                and optics.get('accepted') is True and optics.get('selected_candidate') == reference['selected_candidate']
                and passing == [reference['selected_candidate']])
            for key in ('camera_xy_matches_requested', 'camera_within_search_radius',
                        'camera_height_preserved', 'optical_identity_preserved'):
                checks[key] = stage[key]
            for key, value in stage.items():
                if key.startswith('camera_') or key == 'optical_identity_preserved':
                    receipt[key] = value
            stage['qualification'] = checks
            stage['rejection_reasons'] = [key for key, accepted in checks.items() if not accepted]
            if all(checks.values()):
                stage['accepted'] = receipt['accepted'] = True
                search['selected_index'] = index
                return receipt
        raise ValueError('Nearby reference XY search exhausted native geometry, posture, camera, support/contact or unique optical identity qualification.')
    except Exception as error:
        search['abort_error'] = f'{type(error).__name__}: {error}'
        if search['attempts']:
            search['attempts'][-1]['error'] = search['abort_error']
        raise
    finally:
        if not receipt['accepted']:
            restore('failed_search_rollback')


def initialize_optical_viewpoint(session, reference):
    """Ground the approved base, then adapt initial XY and verify optical identity."""
    receipt = {'schema': PROTOCOL, 'accepted': False, 'private_only': True,
               'reference': deepcopy(reference), 'errors': [], 'physics_steps': 0, 'render_updates': 0,
               'object_setters': 0, 'camera_setters': 0, 'selected_candidate_override': False}
    session.initial_viewpoint_receipt = receipt
    try:
        from worldleague.environments.pandaomron_initialization import (
            PROTOCOL as BASE_PROTOCOL, initialization_active, place_initial_robot_on_floor)
        robot = session.robot
        settings = getattr(session, 'initial_viewpoint_adaptation', None)
        if (not isinstance(settings, dict) or settings.get('protocol') != PROTOCOL
                or settings.get('reference') != reference
                or getattr(session, 'embodiment_config', {}).get('model') != 'pandaomron'
                or getattr(session, 'initial_base_adaptation', None) != BASE_PROTOCOL
                or not initialization_active(session, robot)
                or getattr(session, '_observation_source', None) is not None
                or getattr(session, 'episode_id', None) or getattr(session, 'task_evaluator', None) is not None
                or getattr(session, '_physical_step_generation', 0) != 0):
            raise RuntimeError('Reference viewpoint adaptation requires explicit opted-in PandaOmron reset before observations or evaluation.')
        config, bound = _configuration(session)
        floor, floor_z = _floor(session)
        receipt['floor'] = {'name': floor.name, 'prim_path': str(floor.prim_path), 'measured_aabb_top_z_m': floor_z}
        from worldleague.environments.robot_config import prime_cameras
        calibration = getattr(session, 'initial_camera_calibration_settings', None)
        receipt['camera_priming_render_updates'] = prime_cameras(
            robot, session.pipeline.og.sim, session.embodiment_config['eval']['camera_sensor_names'], calibration)
        receipt['render_updates'] += receipt['camera_priming_render_updates']
        settings = {**settings, 'camera_calibration': calibration}
        import worldleague.adapters.wia_mirror as mirror
        grounding = stage = receipt['grounding'] = {
            'adaptation': BASE_PROTOCOL, 'accepted': False, 'physics_steps': 0, 'render_updates': 0,
            'geometry_before': mirror.capture_initial_geometry(session, config, bound)}
        if grounding['geometry_before']['available'] is not True:
            raise ValueError('Complete native geometry is required before floor alignment.')
        grounding['before'] = _snapshot(robot, bound)
        root_position, root_orientation = grounding['before']['robot_pose']
        grounding['requested'] = {'robot_world_xy_m': root_position[:2], 'orientation_xyzw': root_orientation}
        grounding['placement'] = place_initial_robot_on_floor(
            session, robot, root_position[:2], root_orientation, floor_z)
        _settle_and_measure(session, robot, bound, config, settings, floor, floor_z, grounding, receipt)
        ground_eye_before = _array(grounding['geometry_before']['camera']['position_m'])
        ground_eye_after = _array(grounding['geometry_after']['camera']['position_m'])
        ground_root_after = _array(grounding['after']['robot_pose'][0])
        root_change = ground_root_after - _array(root_position)
        camera_change = ground_eye_after - ground_eye_before
        grounding['root_position_change_m'] = root_change.tolist()
        grounding['camera_position_change_m'] = camera_change.tolist()
        grounding['robot_xy_preserved'] = bool(np.max(np.abs(root_change[:2])) <= POSITION_PRECISION_M)
        grounding['camera_xy_preserved'] = bool(np.max(np.abs(camera_change[:2])) <= POSITION_PRECISION_M)
        grounding['camera_height_tracks_base_grounding'] = bool(
            abs(camera_change[2] - root_change[2]) <= POSITION_PRECISION_M)
        grounding['accepted'] = bool(_native_stage_accepted(grounding)
            and grounding['robot_xy_preserved'] and grounding['camera_xy_preserved']
            and grounding['camera_height_tracks_base_grounding'])
        if not grounding['accepted']:
            raise ValueError('Initial floor alignment failed native geometry, posture, camera or support/contact qualification.')

        # Height invariance applies after the separately authorized floor correction.
        before = receipt['before'] = grounding['after']
        receipt['geometry_before'] = grounding['geometry_after']
        if settings.get('nearby_xy_search') is not None:
            return _initialize_nearby_xy(
                session, reference, settings, robot, bound, config, floor, floor_z, receipt)
        stage = receipt['xy_adaptation'] = {'accepted': False, 'before': before,
            'geometry_before': receipt['geometry_before'], 'physics_steps': 0, 'render_updates': 0}
        eye = _array(receipt['geometry_before']['camera']['position_m'])
        delta = _array(reference['frozen_eye_m'])[:2] - eye[:2]
        requested_xy = _array(before['robot_pose'][0])[:2] + delta
        receipt['requested'] = stage['requested'] = {'robot_world_xy_m': requested_xy.tolist(),
                                'camera_world_xy_m': list(reference['frozen_eye_m'][:2]),
                                'delta_world_xy_m': delta.tolist(), 'orientation_xyzw': before['robot_pose'][1]}
        receipt['placement'] = stage['placement'] = place_initial_robot_on_floor(
            session, robot, requested_xy, before['robot_pose'][1], floor_z)
        _settle_and_measure(session, robot, bound, config, settings, floor, floor_z, stage, receipt)
        for key in ('after', 'invariants', 'native_checks', 'geometry_after'):
            receipt[key] = stage[key]
        camera = receipt['geometry_after']['camera']
        receipt['camera_xy_error_m'] = float(np.max(np.abs(_array(camera['position_m'])[:2]
                                                          - _array(reference['frozen_eye_m'])[:2])))
        receipt['camera_height_change_m'] = float(camera['position_m'][2] - eye[2])
        receipt['camera_xy_matches_reference'] = receipt['camera_xy_error_m'] <= POSITION_PRECISION_M
        receipt['camera_height_preserved'] = abs(receipt['camera_height_change_m']) <= POSITION_PRECISION_M
        optics = receipt['geometry_after']['optical_selection']
        passing = [key for key, row in optics.get('candidates', {}).items() if row.get('accepted') is True]
        receipt['optical_identity_preserved'] = bool(optics.get('available') is True and optics.get('accepted') is True
            and optics.get('selected_candidate') == reference['selected_candidate']
            and passing == [reference['selected_candidate']])
        accepted = (_native_stage_accepted(stage)
            and receipt['camera_xy_matches_reference'] and receipt['camera_height_preserved']
            and receipt['optical_identity_preserved'])
        if not accepted:
            raise ValueError('Reference XY placement failed native geometry, posture, camera, support/contact or unique optical identity qualification.')
        stage['accepted'] = receipt['accepted'] = True
        return receipt
    except Exception as error:
        receipt['errors'].append(f'{type(error).__name__}: {error}')
        if 'stage' in locals() and 'before' in stage and 'after' not in stage:
            try:
                stage['after'] = _snapshot(robot, bound)
                stage['geometry_after'] = mirror.capture_initial_geometry(session, config, bound)
            except Exception as read_error:
                receipt['errors'].append(f'Failure readback: {type(read_error).__name__}: {read_error}')
        raise
