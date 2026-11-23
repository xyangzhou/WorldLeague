"""Initialize explicitly opted-in anchored PandaOmron bases against native floors.

Used by worldleague continuous --help.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

PROTOCOL = 'pandaomron_anchored_floor_v1'
TOLERANCE_M = .005


class CandidateFloorSupportError(ValueError):
    """Reject a candidate whose measured footprint lacks geometric floor support."""


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    if raw.dtype.names:
        if set(raw.dtype.names) != {'x', 'y', 'z'}:
            raise ValueError('Unsupported structured native floor vector.')
        raw = np.stack([raw[axis] for axis in ('x', 'y', 'z')], axis=-1)
    value = np.asarray(raw, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Non-finite native measurement.')
    return value


def _under(body, root):
    return body == root or body.startswith(root.rstrip('/') + '/')


def opted_in(robot):
    return getattr(robot, '_pandaomron_floor_support_protocol', None) == PROTOCOL


def initialization_active(session, robot):
    return (opted_in(robot) and getattr(session, '_pandaomron_initialization_active', False)
            and getattr(robot, '_pandaomron_initialization_session', None) is session
            and getattr(session, '_observation_source', None) is None
            and getattr(session, 'episode_id', None) is None)


def _source_clearance(robot):
    """Read the source box, rather than copying a task's initial root height."""
    xml = ET.parse(robot._source_mjcf_path).getroot()
    body = xml.find(".//body[@name='mobilebase0_wheeled_base']")
    geom = body.find("geom[@name='mobilebase0_pedestal_feet_col']") if body is not None else None
    if geom is None or geom.get('type') != 'box' or geom.get('quat') is not None or geom.get('euler') is not None:
        raise ValueError('Verified source chassis box geometry is required.')
    size = _array([float(v) for v in geom.get('size', '').split()])
    offset = _array([float(v) for v in geom.get('pos', '0 0 0').split()])
    bodies = {row['name']: row for row in robot._source_manifest['bodies']}
    chassis, base = bodies['mobilebase0_wheeled_base'], bodies['mobilebase0_base']
    for row in (chassis, base):
        if not np.allclose(row['quaternion_world_wxyz_at_reset'], [1, 0, 0, 0], atol=1e-7):
            raise ValueError('Source clearance requires the verified upright source frames.')
    clearance = float(chassis['position_world_at_reset'][2] + offset[2] - size[2]
                      - base['position_world_at_reset'][2])
    if size.shape != (3,) or offset.shape != (3,):
        raise ValueError('Source chassis box dimensions are unavailable.')
    if not 0 <= clearance <= TOLERANCE_M:
        raise ValueError('Source chassis clearance is outside the existing support bound.')
    return {'clearance_m': clearance, 'source_mjcf_path': str(robot._source_mjcf_path),
            'source_geometry': geom.get('name'), 'source_box_half_extents_m': size.tolist()}


def begin_initialization(session):
    """Bind the scope immediately after environment construction, if selected."""
    if getattr(session, 'initial_base_adaptation', None) != PROTOCOL:
        return
    if getattr(session, 'embodiment_config', {}).get('model') != 'pandaomron':
        raise ValueError('Anchored floor initialization requires explicit PandaOmron configuration.')
    session._pandaomron_initialization_active = True
    session.initial_base_adaptation_receipt = {'adaptation': PROTOCOL, 'placements': []}
    for robot in session.env.robots:
        if getattr(robot, 'model_name', None) != 'pandaomron':
            raise ValueError('Anchored floor initialization cannot apply to another robot.')
        robot._pandaomron_source_floor_clearance = _source_clearance(robot)
        robot._pandaomron_floor_support_protocol = PROTOCOL
        robot._pandaomron_initialization_session = session


def end_initialization(session):
    session._pandaomron_initialization_active = False
    for robot in getattr(getattr(session, 'env', None), 'robots', ()):
        if getattr(robot, '_pandaomron_initialization_session', None) is session:
            robot._pandaomron_initialization_session = None


def native_vertical_ray(start, end):
    from omnigibson.utils.sampling_utils import raytest
    return raytest(start, end, only_closest=False)


def _scene_objects(robot):
    objects = robot.scene.objects
    return list(objects.values()) if isinstance(objects, dict) else list(objects)


def _floors(robot, floor_paths):
    objects = _scene_objects(robot)
    result = []
    for path in floor_paths:
        matches = [obj for obj in objects if str(obj.prim_path) == str(path)]
        if len(matches) != 1 or matches[0].category not in ('floor', 'floors') or matches[0].visual_only:
            raise ValueError('Floor roots must identify actual loaded collidable floor objects.')
        result.append(str(path))
    if not result:
        raise ValueError('No loaded floor roots are available.')
    return result


def _footprint(robot, requested_position=None, requested_orientation=None):
    link = robot.links['mobilebase0_wheeled_base']
    low, high = (_array(value) for value in link.aabb)
    centre = (low + high) / 2
    # Use the native chassis frame so a prior yaw does not enlarge the sampled
    # footprint a second time when generating another initial stance.
    link_position, link_orientation = (_array(value) for value in link.get_position_orientation())
    half = _array(robot._pandaomron_source_floor_clearance['source_box_half_extents_m'])[:2] * .8
    local = np.asarray([[0., 0., 0.], *[[half[0] * x, half[1] * y, 0.]
                                     for x in (-1, 1) for y in (-1, 1)]])
    samples = (Rotation.from_quat(link_orientation).apply(local) + link_position)[:, :2]
    if requested_position is not None:
        position, orientation = (_array(value) for value in robot.get_position_orientation())
        delta = Rotation.from_quat(requested_orientation) * Rotation.from_quat(orientation).inv()
        offsets = np.column_stack((samples - position[:2], np.zeros(len(samples))))
        samples = delta.apply(offsets)[:, :2] + requested_position[:2]
    return samples, float(low[2])


def _floor_rays(robot, floor_paths, samples, floor_z):
    roots = _floors(robot, floor_paths)
    rows = []
    for xy in samples:
        floor_z = float(_array(floor_z))
        start, end = [*xy, floor_z + .25], [*xy, floor_z - .25]
        row = {'ray_start_m': start, 'ray_end_m': end, 'hits': [], 'surface_z_m': None}
        candidates = []
        for hit in native_vertical_ray(start, end):
            if not hit.get('hit'):
                continue
            position, normal = _array(hit['position']), _array(hit['normal'])
            body = str(hit['rigidBody'])
            valid = any(_under(body, root) for root in roots) and normal.shape == (3,) and normal[2] >= .95
            row['hits'].append({'body': body, 'position_m': position.tolist(), 'normal': normal.tolist(),
                                'loaded_upward_floor': bool(valid)})
            if (valid and position.shape == (3,) and np.linalg.norm(position[:2] - xy) <= 1e-4
                    and end[2] <= position[2] <= start[2]):
                candidates.append(float(position[2]))
        if candidates:
            row['surface_z_m'] = max(candidates)
        rows.append(row)
    return rows


def measured_anchored_floor_support(robot, floor_paths, floor_z):
    """Measure the anchor and floor; never synthesize physical contact positions."""
    result = {'support_mechanism': 'world_anchor_and_measured_floor', 'floor_supported': False,
              'maximum_chassis_floor_gap_m': TOLERANCE_M, 'floor_rays': []}
    try:
        if not opted_in(robot):
            raise ValueError('Anchored support was not explicitly selected.')
        anchor = robot.native_world_anchor_measurement()
        result['world_anchor'] = anchor
        samples, bottom = _footprint(robot)
        rays = _floor_rays(robot, floor_paths, samples, floor_z)
        result.update(chassis_bottom_z_m=bottom, floor_rays=rays,
                      source_clearance=robot._pandaomron_source_floor_clearance)
        valid = bool(anchor['valid'])
        for row in rays:
            gap = None if row['surface_z_m'] is None else bottom - row['surface_z_m']
            row['chassis_floor_gap_m'] = gap
            row['nonpenetrating_within_support_bound'] = gap is not None and 0 <= gap <= TOLERANCE_M
            valid = valid and row['nonpenetrating_within_support_bound']
        result['floor_supported'] = bool(valid)
    except (ValueError, TypeError, RuntimeError, AttributeError, KeyError, IndexError) as error:
        result['measurement_error'] = f'{type(error).__name__}: {error}'
    return result


def _initialize_requested_pose(session, robot, position, orientation, receipt, *, floor_z=None):
    """Normalize a reset-only world pose to measured floor/source clearance."""
    if not initialization_active(session, robot):
        raise RuntimeError('PandaOmron floor placement is authorized only during reset initialization.')
    position, orientation = _array(position), _array(orientation)
    if position.shape != (3,) or orientation.shape != (4,):
        raise ValueError('A complete world pose is required.')
    paths = [str(obj.prim_path) for obj in _scene_objects(robot)
             if obj.category in ('floor', 'floors') and not obj.visual_only]
    if floor_z is None:
        floor_z = float(robot.scene.trav_map.floor_heights[0])
    samples, bottom = _footprint(robot, position, orientation)
    rays = _floor_rays(robot, paths, samples, floor_z)
    receipt['floor_rays_before'] = rays
    if any(row['surface_z_m'] is None for row in rays):
        raise CandidateFloorSupportError('Every initial chassis footprint sample requires a measured loaded floor hit.')
    clearance = robot._pandaomron_source_floor_clearance['clearance_m']
    surface = max(row['surface_z_m'] for row in rays)
    if any(surface + clearance - row['surface_z_m'] > TOLERANCE_M for row in rays):
        raise CandidateFloorSupportError('Measured floor variation exceeds the source chassis support bound.')
    old_position, old_orientation = (_array(value) for value in robot.get_position_orientation())
    effective = position.copy()
    effective[2] = old_position[2] + surface + clearance - bottom
    receipt.update(effective_world_position_m=effective.tolist(),
        source_clearance=robot._pandaomron_source_floor_clearance,
        floor_rays_before=rays, root_orientation_preserved=True)
    receipt['root_alignment'] = robot._set_initial_root_height(float(effective[2] - old_position[2]))
    # Existing planar kinematics implement the source pivot and x/y/yaw contract.
    robot._set_planar_position_orientation(effective, orientation, preserve_controllers=True)
    receipt['measured_world_position_m'], receipt['measured_world_orientation_xyzw'] = (
        _array(value).tolist() for value in robot.get_position_orientation())
    receipt['support_after_placement'] = measured_anchored_floor_support(robot, paths, floor_z)
    measured_position = _array(receipt['measured_world_position_m'])
    measured_orientation = _array(receipt['measured_world_orientation_xyzw'])
    receipt['position_error_m'] = float(np.linalg.norm(measured_position - effective))
    receipt['orientation_dot_error'] = float(1 - abs(np.dot(
        measured_orientation / np.linalg.norm(measured_orientation), orientation / np.linalg.norm(orientation))))
    receipt['requested_planar_pose_retained'] = (receipt['position_error_m'] <= 1e-5
                                               and receipt['orientation_dot_error'] <= 1e-6)
    if not receipt['requested_planar_pose_retained']:
        raise ValueError('Initial base placement changed the requested planar pose.')
    if not receipt['support_after_placement']['floor_supported']:
        raise ValueError('Initial anchored base placement failed measured floor/anchor qualification.')
    receipt['accepted'] = True
    return receipt


def initialize_requested_pose(session, robot, position, orientation, *, floor_z=None):
    """Record success or failure measurements for a reset-only requested pose."""
    if not initialization_active(session, robot):
        raise RuntimeError('PandaOmron floor placement is authorized only during reset initialization.')
    receipt = {'adaptation': PROTOCOL, 'accepted': False,
        'requested_world_position_m': _array(position).tolist(),
        'requested_world_orientation_xyzw': _array(orientation).tolist()}
    session.initial_base_adaptation_receipt['placements'].append(receipt)
    try:
        return _initialize_requested_pose(session, robot, position, orientation, receipt, floor_z=floor_z)
    except Exception as error:
        receipt['placement_error'] = f'{type(error).__name__}: {error}'
        raise


def capture_initial_base_anchors(session):
    """Capture USD anchors omitted by ordinary simulator dump_state."""
    result = []
    for robot in session.env.robots:
        if opted_in(robot):
            if not initialization_active(session, robot):
                raise RuntimeError('Initial anchor capture is authorized only during reset.')
            snapshot = robot.native_world_anchor_measurement()
            if not snapshot['valid']:
                raise ValueError('Cannot capture a baseline with an invalid world fixed-joint anchor.')
            result.append((robot, snapshot))
    return result


def restore_initial_base_anchors(session, snapshots):
    """Restore exact saved anchors immediately after initialization load_state."""
    receipts = []
    for robot, snapshot in snapshots:
        if not initialization_active(session, robot):
            raise RuntimeError('Initial anchor restoration is authorized only during reset.')
        receipts.append(robot._restore_initial_world_anchor(snapshot))
    if receipts:
        session.initial_base_adaptation_receipt.setdefault('baseline_anchor_restores', []).append(receipts)
    return receipts


def finish_initialization(session):
    """Qualify constructor-only starts before reset publishes any observation.

    Already qualified starts remain read only. A required source-floor alignment
    retains current x/y and orientation, followed by native settling and checks.
    """
    if getattr(session, 'initial_base_adaptation', None) != PROTOCOL:
        return None
    from worldleague.tasks.interaction_setup import _robot_contact_check

    receipt = {'adaptation': PROTOCOL, 'accepted': False, 'robots': [],
               'physics_steps': 0, 'render_updates': 0}
    session.initial_base_adaptation_receipt['final_qualification'] = receipt
    try:
        for robot in session.env.robots:
            row = {'robot_prim_path': str(robot.prim_path), 'accepted': False,
                   'vertical_alignment_applied': False, 'actual_vertical_correction_m': 0.,
                   'physics_steps': 0, 'render_updates': 0}
            receipt['robots'].append(row)
            if not initialization_active(session, robot):
                raise RuntimeError('Final base qualification is authorized only during reset initialization.')
            position, orientation = (_array(value) for value in robot.get_position_orientation())
            row.update(position_before_m=position.tolist(), orientation_before_xyzw=orientation.tolist())
            paths = [str(obj.prim_path) for obj in _scene_objects(robot)
                     if obj.category in ('floor', 'floors') and not obj.visual_only]
            floor_z = float(_array(robot.scene.trav_map.floor_heights[0]))
            row['nominal_floor_z_m'] = floor_z
            checks = _robot_contact_check(robot, paths, floor_z)
            row['checks_before'] = checks
            if not checks['upright']:
                raise ValueError('Final initial base qualification requires the retained robot to be upright.')
            if not checks['floor_supported']:
                row['placement'] = place_initial_robot_on_floor(session, robot, position[:2], orientation, floor_z)
                row['vertical_alignment_applied'] = True
                corrected = _array(robot.get_position_orientation()[0])
                row['actual_vertical_correction_m'] = float(corrected[2] - position[2])
            if not all(checks[key] for key in ('floor_supported', 'collision_free', 'upright')):
                robot.keep_still()
                for _ in range(30):
                    session.pipeline.og.sim.step()
                    row['physics_steps'] += 1
                    receipt['physics_steps'] += 1
                for _ in range(4):
                    session.pipeline.og.sim.render()
                    row['render_updates'] += 1
                    receipt['render_updates'] += 1
                checks = _robot_contact_check(robot, paths, floor_z)
            measured_position, measured_orientation = (_array(value) for value in robot.get_position_orientation())
            row.update(position_after_m=measured_position.tolist(), orientation_after_xyzw=measured_orientation.tolist(),
                       checks_after=checks)
            row['actual_vertical_correction_m'] = float(measured_position[2] - position[2])
            row['accepted'] = all(checks[key] for key in ('floor_supported', 'collision_free', 'upright'))
            if not row['accepted']:
                raise ValueError('Final initial base floor, collision, and upright qualification failed after settling.')
        receipt['accepted'] = bool(receipt['robots']) and all(row['accepted'] for row in receipt['robots'])
        if not receipt['accepted']:
            raise ValueError('Final base qualification requires the explicitly selected native robot.')
        return receipt
    except Exception as error:
        receipt['qualification_error'] = f'{type(error).__name__}: {error}'
        raise


def place_initial_robot_on_floor(session, robot, xy, orientation, floor_z):
    """Place an opted-in PandaOmron during reset, without stepping or observations."""
    position = _array(robot.get_position_orientation()[0]).copy()
    position[:2] = _array(xy)
    position[2] = float(floor_z)
    return initialize_requested_pose(session, robot, position, orientation, floor_z=floor_z)
