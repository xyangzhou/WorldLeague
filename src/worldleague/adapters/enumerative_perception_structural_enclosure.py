"""Convert original enclosures into private measured door-operation tasks.

Run on CPU with the behavior Python environment from the repository root:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
Only initialization hooks restore original mechanisms. Qualification and
evaluation read native state; all actions use the existing guarded controls.
"""
from __future__ import annotations




import math

import numpy as np

from worldleague.tasks.background_policy import legal_background_contact

MODULE = 'worldleague.adapters.enumerative_perception_structural_enclosure'
OPERATIONS = ('inspect_all_enclosures', 'inspect_any_enclosure')
MINIMUM_ANGLE_RAD = math.radians(30.)
MINIMUM_SLIDE_M = .06
CLOSED_ANGLE_RAD = math.radians(5.)
CLOSED_SLIDE_M = .005
LAYOUT_TOLERANCE_M = .05


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Source/native measurements must be finite.')
    return value


def _vector(value, size):
    value = _array(value)
    if value.shape != (size,):
        raise ValueError('Source/native vector shape is unresolved.')
    return value


def _angle(first, second):
    first, second = _vector(first, 4), _vector(second, 4)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-8:
        raise ValueError('Source/native orientation is unresolved.')
    return math.degrees(2 * math.acos(min(1., abs(float(np.dot(first, second) / denominator)))))


def _near(row, original, tolerance=LAYOUT_TOLERANCE_M):
    return math.dist(row['position'], original['position']) <= tolerance and _angle(
        row['quaternion_xyzw'], original['quaternion_xyzw']) <= 5.


def _snapshot(obj):
    position, quaternion = obj.get_position_orientation()
    low, high = [_array(value) for value in obj.aabb]
    return {'position': _array(position).tolist(), 'quaternion_xyzw': _array(quaternion).tolist(),
            'bbox': [low.tolist(), high.tolist()], 'visual_only': bool(obj.visual_only),
            'fixed_base': bool(obj.fixed_base)}


def _door_snapshot(obj, hand_paths):
    """Read Open's actual annotated joints and child-link contacts, not source flags."""
    from omnigibson.object_states import Open
    from omnigibson.utils.constants import JointType
    state = obj.states[Open]
    both_sides, joints, directions = state.relevant_joints_info
    rows = []
    for joint, direction in zip(joints, directions):
        if joint.joint_type == JointType.JOINT_REVOLUTE:
            kind, minimum, tolerance = 'revolute', MINIMUM_ANGLE_RAD, CLOSED_ANGLE_RAD
        elif joint.joint_type == JointType.JOINT_PRISMATIC:
            kind, minimum, tolerance = 'prismatic', MINIMUM_SLIDE_M, CLOSED_SLIDE_M
        else:
            raise ValueError('Native Open references an unsupported joint type.')
        link = next(link for link in obj.links.values() if link.prim_path == joint.body1)
        position, velocity = joint.get_state()[:2]
        if not math.isfinite(float(joint.lower_limit)) or not math.isfinite(float(joint.upper_limit)):
            raise ValueError('Native door limits must be measured and finite.')
        contacts = [{'body0': str(c.body0), 'body1': str(c.body1)} for c in link.contact_list()]
        handles = []
        for mesh in link.collision_meshes.values():
            name = str(mesh.prim.GetName())
            if 'handle' in name.lower() or 'handle' in str(link.prim_path).lower():
                handles.append({'mesh': str(mesh.prim.GetPath()),
                    'bbox': [_array(value).tolist() for value in mesh.aabb]})
        rows.append({'name': joint.name, 'position': float(_array(position).reshape(-1)[0]),
                     'velocity': float(_array(velocity).reshape(-1)[0]), 'kind': kind,
                     'lower_limit': float(joint.lower_limit), 'upper_limit': float(joint.upper_limit),
                     'open_direction': int(direction), 'minimum_travel': minimum,
                     'closed_tolerance': tolerance, 'driven': bool(joint.driven),
                     'door_link': link.prim_path, 'visible': bool(link.visible),
                     'collision_mesh_count': len(link.collision_meshes),
                     'contact_reporting': bool(link.contact_reporting_enabled), 'contacts': contacts,
                     'door_link_bbox': [_array(value).tolist() for value in link.aabb],
                     'handle_geometry': handles,
                     'hand_contact': any(c[field] in hand_paths for c in contacts for field in ('body0', 'body1'))})
    return {**_snapshot(obj), 'open': bool(state.get_value()), 'both_sides': bool(both_sides),
            'root_collision_mesh_count': len(obj.root_link.collision_meshes), 'joints': rows}


def _measure(session, controller, config, objects):
    from omnigibson.object_states import Inside
    robot, p = session.robot, config['parameters']
    arm = controller.working_arm
    hand_paths = {robot.eef_links[arm].prim_path, *(link.prim_path for link in robot.finger_links[arm])}
    doors = {key: _door_snapshot(objects[key], hand_paths) for key in p['container_keys']}
    contents = {key: {**_snapshot(objects[key]),
                     'inside_original_container': bool(objects[key].states[Inside].get_value(objects[p['relations'][key]]))}
                for key in p['content_keys']}
    position = _array(robot.get_position_orientation()[0])
    low, high = [_array(value) for value in robot.links[robot.base_footprint_link_name].aabb]
    regions = [session.env.scene.seg_map.get_room_instance_by_point(np.array([x, y]))
               for x in (low[0], (low[0] + high[0]) / 2, high[0])
               for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    floor_paths = {objects['floor'].prim_path}
    contacts = []
    for contact in robot.contact_list():
        if legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        for body in (str(contact.body0), str(contact.body1)):
            if not (body == robot.prim_path or body.startswith(robot.prim_path + '/')):
                contacts.append(body)
    on_floor = any(any(body.startswith(path + '/') for path in floor_paths) for body in contacts)
    obstacles = [body for body in contacts if not any(body.startswith(path + '/') for path in floor_paths)]
    return doors, contents, controller.debug_gripper_state(), {'position': position.tolist(),
        'in_original_room': all(region == session.room_name for region in regions),
        'supported_without_obstacle_contact': bool(on_floor and not obstacles), 'external_contacts': contacts}


# Explicit TaskEdit2 additions; historical converters and the current pool stay intact.
TASK_EDIT2_CURATION_REVISION = 'task_edit2_enclosure_additions_20261018_v1'
TASK_EDIT2_ENCLOSURE_GATES = ('actual_containment', 'independent_interior_geometry',
    'public_control_contact_opening', 'post_open_informative_visibility',
    'access_to_every_content', 'stable_robot_start')
