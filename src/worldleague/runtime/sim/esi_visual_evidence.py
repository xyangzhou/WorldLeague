"""Read synchronized private visual evidence without recognizing from mask size.

Used by worldleague scene --help.
"""
from __future__ import annotations

from copy import deepcopy
import math
import threading

import numpy as np

COLLECTOR_SCHEMA = 'native_visual_evidence_v1'
FEATURE_AUDIT_SCHEMA = 'native_visual_feature_audit_v1'
LOCATION_AUDIT_SCHEMA = 'native_visual_location_audit_v1'
FROZEN_INPUT_AUDIT_SCHEMA = 'native_frozen_initial_input_audit_v1'
FROZEN_GLOBAL = 'initial_overview_rgb'


def _owner():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Native visual evidence must be read on the simulator owner/main thread.')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose(obj):
    position, quaternion = [_array(value).astype(float) for value in obj.get_position_orientation()]
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(np.r_[position, quaternion]).all():
        raise ValueError('Native object/sensor pose is unavailable.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _identity(obj):
    scale = _array(obj.scale).astype(float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Native asset scale is unavailable.')
    return {'name': obj.name, 'prim_path': str(obj.prim_path), 'category': obj.category,
        'model': getattr(obj, 'model', None), 'scale': scale.tolist()}


def _native_context(session, observation):
    from worldleague.tasks.loader import LoadedTask
    adapter_loaded = getattr(getattr(session, 'scene_adapter', None), 'task', None)
    explicit_task = getattr(session, 'loaded_task', None)
    task = explicit_task if isinstance(explicit_task, LoadedTask) else (
        adapter_loaded if isinstance(adapter_loaded, LoadedTask) else None)
    scene = None
    owner = task if task is not None else scene
    definition = getattr(owner, 'definition', None)
    source = getattr(definition, 'source', None)
    identity = source.get('json') if isinstance(source, dict) else getattr(source, 'json_path', None)
    clock = float(session.pipeline.og.sim.current_time)
    if not math.isfinite(clock):
        raise ValueError('Native physics clock is unavailable.')
    reset = getattr(session, 'native_reset_id', None)
    return {'task_id': getattr(getattr(task, 'definition', None), 'id', None),
        'scene_id': getattr(getattr(scene, 'definition', None), 'id', None),
        'provenance_role': 'task_and_scene' if task is not None and scene is not None else
            'task' if task is not None else 'scene_only' if scene is not None else 'unavailable',
        'source_identity': identity,
        'reset_id': reset, 'episode_id': str(getattr(session, 'episode_id', 'reset_' + str(reset))),
        'seed': getattr(owner, 'seed', None),
        'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
        'captured_at': getattr(observation, 'captured_at', None), 'physics_time_s': clock,
        'phase': getattr(getattr(session, 'temporal_phase_controller', None), 'phase', None)}


def _context_complete(context):
    required = ('task_id', 'reset_id',
        'seed', 'snapshot_id', 'action_step', 'captured_at', 'physics_time_s')
    return all(context.get(key) is not None for key in required)


def _retained_capture_matches(session, observation):
    source = getattr(session, '_observation_source', None)
    snapshot = getattr(source, 'latest_snapshot', None)
    latest = getattr(snapshot, 'observation', None)
    return bool(latest is not None and all(getattr(latest, key, None) == getattr(observation, key, None)
        for key in ('snapshot_id', 'action_step', 'captured_at')))


def _labels(info, modality):
    labels = info.get(modality)
    if isinstance(labels, dict):
        labels = labels.get('idToLabels', labels.get('id_to_labels', labels))
    if not isinstance(labels, dict):
        raise ValueError('Native instance-label mapping is unavailable.')
    result = {}
    for key, value in labels.items():
        try:
            result[int(key)] = value
        except (ValueError, TypeError):
            continue
    if not result:
        raise ValueError('Native instance-label mapping is empty.')
    return result


def _label_matches(label, obj):
    if isinstance(label, str):
        return label == obj.name or label == str(obj.prim_path) or label.startswith(str(obj.prim_path) + '/')
    if isinstance(label, dict):
        # A semantic category such as {'class': 'bowl'} is not instance identity.
        return any(_label_matches(label.get(key), obj) for key in (
            'name', 'object_name', 'prim_path', 'primPath', 'path', 'label'))
    if isinstance(label, (tuple, list)):
        return any(_label_matches(value, obj) for value in label)
    return False


def _mask_statistics(segmentation, labels, obj):
    ids = sorted(key for key, label in labels.items() if _label_matches(label, obj))
    mask = np.isin(segmentation, ids)
    ys, xs = np.nonzero(mask)
    height, width = segmentation.shape
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    normalized = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height] if bbox else None
    return {'instance_ids': ids, 'visible_pixels': int(len(xs)),
        'mask_fraction': float(len(xs) / (width * height)), 'bbox_xyxy': bbox,
        'bbox_normalized': normalized,
        'minimum_extent_px': min(bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else 0}, mask


def _capture_metadata(info, modality):
    """Check channel frame stamps when the native annotator exposes them."""
    metadata = info.get('capture_metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    rgb, segmentation = metadata.get('rgb', {}), metadata.get(modality, {})
    for key in ('frame_id', 'frameId', 'timestamp'):
        first = rgb.get(key) if isinstance(rgb, dict) else None
        second = segmentation.get(key) if isinstance(segmentation, dict) else None
        if (first is None) != (second is None) or first is not None and first != second:
            raise ValueError('Native RGB/segmentation capture frame metadata differs or is incomplete.')
    def primitive(value):
        value = value.item() if isinstance(value, np.generic) else value
        return value if value is None or isinstance(value, (str, int, float, bool)) else None
    safe = {name: {key: primitive(value.get(key)) for key in ('frame_id', 'frameId', 'timestamp') if key in value}
        for name, value in (("rgb", rgb), (modality, segmentation)) if isinstance(value, dict)}
    return {'channel_metadata': safe,
        'native_frame_id': primitive(info.get('frame_id', info.get('frameId'))),
        'method': 'single_native_get_obs_exact_RGB_current_retained_snapshot_stable_physics_clock'}


def collect_native_visual_evidence(session, observation, bindings, parameters=None, *, controller=None):
    """Measure synchronized instance masks for the delivered native camera views."""
    _owner()
    delivered = observation.rgb_images()
    registry = getattr(session, 'robot_camera_registry', {})
    errors, measurements, cameras, identities, objects = {}, {}, {}, {}, {}
    base_context = {}
    try:
        base_context = _native_context(session, observation)
        retained = _retained_capture_matches(session, observation)
        robot_pose = _pose(session.robot)
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
        errors['native_capture_context'] = f'{type(error).__name__}: {error}'
        retained, robot_pose = False, None
    for key, binding in bindings.items():
        name = binding if isinstance(binding, str) else binding.get('name') if isinstance(binding, dict) else getattr(binding, 'name', None)
        try:
            obj = session.env.scene.object_registry('name', name)
            if obj is None or obj.name != name:
                raise ValueError('Bound native object identity is unavailable.')
            objects[key], identities[key] = obj, _identity(obj)
            measurements[key] = {'native_identity': identities[key], 'native_pose': _pose(obj), 'views': {}}
            segmentation = getattr(session.env.scene, 'seg_map', None)
            query = getattr(segmentation, 'get_room_instance_by_point', None)
            if callable(query):
                measured_room = query(np.asarray(measurements[key]['native_pose']['position_m'])[:2])
                measurements[key]['measured_room_label'] = str(measured_room) if measured_room is not None else None
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
            errors['binding:' + key] = f'{type(error).__name__}: {error}'
    context_complete = _context_complete(base_context) and retained
    if not retained:
        errors['retained_capture'] = 'Observation is not the current retained native capture.'
    if not _context_complete(base_context):
        errors['audit_context'] = 'Task/source/reset/seed/snapshot provenance is incomplete.'
    rows, available_views, audited_views, crop_count = [], [], set(), 0
    for view, delivered_rgb in delivered.items():
        if view == FROZEN_GLOBAL or view not in registry:
            continue
        try:
            sensor_name = registry[view]
            sensor = session.robot.sensors[sensor_name]
            if not str(sensor.prim_path).startswith(str(session.robot.prim_path) + '/'):
                raise ValueError('Registered camera is not mounted on the current native robot.')
            before = float(session.pipeline.og.sim.current_time)
            native, info = sensor.get_obs()
            actual = _array(native['rgb'])
            rgb = np.asarray(delivered_rgb)
            if (actual.dtype != np.uint8 or rgb.dtype != np.uint8 or actual.ndim != 3 or
                    actual.shape[-1] not in (3, 4) or not np.array_equal(actual[..., :3], rgb)):
                raise ValueError('Native RGB is not exactly equal to the delivered RGB.')
            modality = next((name for name in ('seg_instance', 'seg_instance_id') if name in native and name in info), None)
            if modality is None:
                raise ValueError('Private native instance segmentation modality/labels are unavailable.')
            segmentation = _array(native[modality])
            if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
                segmentation = segmentation[..., 0]
            if segmentation.shape != rgb.shape[:2] or not np.issubdtype(segmentation.dtype, np.integer):
                raise ValueError('Native instance mask dimensions/type do not match delivered RGB.')
            labels = _labels(info, modality)
            stamp = _capture_metadata(info, modality)
            after = float(session.pipeline.og.sim.current_time)
            if not math.isfinite(before) or before != after or before != base_context.get('physics_time_s'):
                raise ValueError('Native physics advanced between the capture context and mask read.')
            cam = {'camera': view, 'sensor_name': sensor_name, 'sensor_prim_path': str(sensor.prim_path),
                'segmentation_modality': modality,
                'image_size': [rgb.shape[1], rgb.shape[0]], 'camera_pose': _pose(sensor),
                'capture_synchronization': stamp, 'native_rgb_matches_delivered': True,
                'retained_capture_verified': retained}
            cameras[view] = cam
            available_views.append(view)
            masks, stats = {}, {}
            for key, obj in objects.items():
                stats[key], masks[key] = _mask_statistics(segmentation, labels, obj)
            for key in objects:
                context = {**base_context, 'camera': view, 'binding': key, 'native_identity': identities[key],
                           'native_pose': measurements[key]['native_pose'], 'robot_pose': robot_pose}
                row = {**stats[key], 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                       'camera': view, 'native_name': identities[key]['name'], 'native_rgb_matches_delivered': True,
                       'identity_synchronized': bool(context_complete and stats[key]['visible_pixels']),
                       'measurement_available': True, 'audit_context': context, 'audited': False,
                       'informative': False, 'informative_feature_evidence': {}, 'visit_verified': False,
                       'visit_evidence': {}, 'identity_evidence_source': 'native_instance_mask' if stats[key]['visible_pixels'] else 'unavailable'}
                measurements[key]['views'][view] = row
                rows.append((key, row))
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError, OSError) as error:
            errors[view] = f'{type(error).__name__}: {error}'
    selected = {}
    for key in bindings:
        candidates = [row for binding, row in rows if binding == key]
        if candidates:
            selected[key] = max(candidates, key=lambda row: (row['visit_verified'], row['informative'], row['visible_pixels']))
        else:
            selected[key] = {'native_name': identities.get(key, {}).get('name'), 'snapshot_id': observation.snapshot_id,
                'informative': False, 'informative_feature_evidence': {}, 'measurement_available': False,
                'visit_verified': False, 'visit_evidence': {}, 'information_unavailable_reason': 'no_synchronized_native_identity_view'}
    control = {'available': False, 'reason': 'native_public_control_context_unavailable'}
    provider = getattr(controller, 'geometric_rearrangement_evidence', None)
    if callable(provider):
        try:
            value = provider()
            fields = ('available', 'sample_id', 'control_sequence', 'control_active', 'state_setter_used',
                'controlled_object_names', 'grasped_object_names', 'contact_object_names', 'reason')
            control = {key: deepcopy(value[key]) for key in fields if key in value}
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
            control = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    information_available = any(row.get('informative') for row in selected.values())
    public_history = []
    history = getattr(observation, "action_history", ())
    if history:
        from worldharness.core.action_units import ROBOT_ACTIONS, TERMINAL_ATOMS
        from worldharness.core.observations import EXECUTION_STATUSES
        public_history = [{'action': row.action, 'result_summary': row.result_summary}
            for row in history[-32:]
            if row.action in ROBOT_ACTIONS + TERMINAL_ATOMS and row.result_summary in EXECUTION_STATUSES]
    return {'schema': COLLECTOR_SCHEMA, 'available': information_available,
        'informative_available': information_available, 'measurement_available': bool(available_views),
        'objects': selected, 'measurements': measurements, 'cameras': cameras,
        'view_evidence': [{'binding': key, **row} for key, row in rows if row['informative']],
        'available_views': available_views, 'audited_camera_names': sorted(audited_views),
        'all_delivered_views_audited': bool(delivered) and set(delivered) <= audited_views,
        'frozen_views': {FROZEN_GLOBAL: {'live': False}} if FROZEN_GLOBAL in delivered else {}, 'measurement_errors': errors,
        'native_capture_context': base_context, 'native_public_control_context': control,
        'public_action_history': public_history, 'crop_count': crop_count,
        'recognition_inferred_from_mask_size': False,
        'information_unavailable_reason': None if information_available else 'no_exact_audited_native_semantic_feature_receipt'}


def install_visual_evidence(session, controller=None):
    """Install native visibility measurement for the evaluator."""
    def provider(observation, bindings, parameters):
        return collect_native_visual_evidence(session, observation, bindings, parameters, controller=controller)
    session.task_visual_evidence = provider
    return provider
