"""Native occupancy and explicit Agent arrival reports for Cognitive Mapping.

Used by worldleague scene --help.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
import re

import numpy as np


def native_array(value):
    """Normalize ordinary tensors and PhysX structured x/y/z contact coordinates."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.names and all(axis in array.dtype.names for axis in ('x', 'y', 'z')):
        array = np.stack([array[axis] for axis in ('x', 'y', 'z')], axis=-1)
    return np.asarray(array, dtype=float)


def settled_native_state(session, measured, *, minimum_up_dot=.95):
    """Measure stopped room occupancy independently of generic robot Safety."""
    robot = session.robot
    chassis = robot.links[robot.base_footprint_link_name]
    linear, angular = (native_array(value) for value in (
        chassis.get_linear_velocity(), chassis.get_angular_velocity()))
    quaternion = native_array(measured['quaternion_xyzw'])
    if (linear.shape != (3,) or angular.shape != (3,) or not np.isfinite([linear, angular]).all()
            or quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) <= 0):
        raise ValueError('native_chassis_velocity_or_orientation_unavailable')
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    up_dot = float(1 - 2 * (x*x + y*y))
    return {**measured, 'linear_speed_m_s': float(np.linalg.norm(linear)),
        'angular_speed_rad_s': float(np.linalg.norm(angular)), 'up_dot': up_dot,
        'state_valid': bool(measured['chassis_region'] in measured['known_regions']
                            and measured.get('floor_layer_known', True)),
        'occupancy_valid': bool(measured['chassis_region'] in measured['known_regions']
                                and measured.get('floor_layer_known', True))}


class NativeVisitReportEvaluator:
    """Read native samples and validate same-boundary Agent-authored claims.

    Subclasses implement _measure_native, _qualify_start, and optionally
    _additional_checks. Every return value and file here is private evaluation.
    No method creates Agent text, advances simulation, or moves any native body.
    """
    schema = 'private_cognitive_mapping_visit_reports_v2'

    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        p = self.parameters
        required = p.get('required_regions')
        answer_only = (p.get('actual_capability') == 'direct_adjacency_and_unreachability_answers'
                       and p.get('source_graph_reachable') is False)
        if not isinstance(required, list) or (not required and not answer_only) or not all(isinstance(r, str) and r for r in required):
            raise ValueError('native_required_room_set_missing')
        if p.get('arrival_report_required') is not (False if answer_only else True):
            raise ValueError('explicit_agent_arrival_reports_required')
        self.required = set(required)
        self.required_order = list(p.get('required_order', []))
        if self.required_order:
            evidence, question = p.get('public_route_evidence', []), p.get('source_public_question')
            if (p.get('public_route_required') is not True or not isinstance(question, str)
                    or not evidence or [e.get('region') for e in evidence] != self.required_order
                    or not set(self.required_order).issubset(self.required)
                    or any(not e.get('source_phrase') or e['source_phrase'] not in question for e in evidence)):
                raise ValueError('ordered_route_requires_exact_public_question_evidence')
        elif p.get('public_route_required'):
            raise ValueError('public_route_flag_without_order')
        self.stability_steps = int(p.get('stability_steps', p.get('stop_samples', 3)))
        if self.stability_steps < 1:
            raise ValueError('positive_native_stability_samples_required')
        self.maximum_speed = float(p.get('maximum_stop_speed_m_s', p.get('maximum_terminal_speed_m_s', .02)))
        self.maximum_angular_speed = float(p.get('maximum_stop_angular_speed_rad_s',
            p.get('maximum_terminal_angular_speed_rad_s', .1)))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _time(self):
        value = float(self.session.pipeline.og.sim.current_time)
        if not math.isfinite(value):
            raise ValueError('native_simulation_clock_unavailable')
        return value

    def _additional_checks(self, measured):
        return {}

    def _task_faults(self):
        # Contact/pose safety history remains visible but does not erase factual
        # room visits. Measurement, continuity and task-invariant faults remain.
        return self.faults - {'native_floor_contact_or_pose_failed', 'native_nonfloor_contact'}

    def _occupancy_valid(self, measured):
        return measured.get('occupancy_valid', measured['state_valid'])

    def _settled(self, measured):
        return (measured['linear_speed_m_s'] <= self.maximum_speed
                and measured['angular_speed_rad_s'] <= self.maximum_angular_speed)

    def _append(self, filename, row):
        with (self.output / filename).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def begin_episode(self, observation):
        baseline = getattr(self, '_begin_baseline', None)
        if callable(baseline):
            baseline()
        self.episode += 1
        self.active = True
        self.visited, self.reported, self.visits, self.arrival_reports = set(), set(), [], []
        self.order_stage, self.samples, self.consecutive = 0, 0, 0
        self.completed_events, self.measurement_errors = {}, []
        self.done_boundary_credit = None
        self.boundary_signature, self.boundary_stable_samples = None, 0
        self.stable_region, self.sample_region = None, None
        self.faults, self.travel_m = set(), 0.
        measured = self._measure_native()
        self.previous = np.asarray(measured['position'], dtype=float)
        self.previous_time = self._time()
        return self.record(observation, action=None)

    def _read_boundary(self):
        measured = self._measure_native()
        displacement = float(np.linalg.norm(np.asarray(measured['position']) - self.previous))
        if displacement > self.controller.config.translation_tolerance_m:
            self.faults.add('unsampled_native_motion')
        checks = self._additional_checks(measured)
        if not all(checks.values()):
            self.faults.add('native_category_invariant_failed')
        if (not self._occupancy_valid(measured) or not self._settled(measured)
                or measured['chassis_region'] != self.sample_region):
            self.consecutive, self.stable_region = 0, None
        return measured, checks

    def sample_step(self):
        if not self.active:
            return
        try:
            measured, now = self._measure_native(), self._time()
            extra_checks = self._additional_checks(measured)
        except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
            self.faults.add('native_control_measurement_unavailable')
            self.measurement_errors.append({'stage': 'control_sample', 'reason': f'{type(error).__name__}: {error}'})
            self.consecutive, self.stable_region = 0, None
            self.boundary_signature, self.boundary_stable_samples = None, 0
            return
        point = np.asarray(measured['position'], dtype=float)
        elapsed = now - self.previous_time
        distance = float(np.linalg.norm(point - self.previous))
        cfg = self.controller.config
        limit = cfg.max_translation_speed_m_s * max(elapsed, 0.) + cfg.translation_tolerance_m
        if 'maximum_step_displacement_m' in self.parameters:
            limit = min(limit, self.parameters['maximum_step_displacement_m'])
        if elapsed <= 0 or distance > limit:
            self.faults.add('motion_clock_or_continuity_failed')
        if not all(extra_checks.values()):
            self.faults.add('native_category_invariant_failed')
        self.samples += 1
        self.travel_m += distance
        signature = tuple(measured.get('footprint_regions', ()))
        boundary_eligible = (len(signature) == 9 and measured.get('floor_layer_known', False)
                             and self._settled(measured) and not self._task_faults())
        self.boundary_stable_samples = (self.boundary_stable_samples + 1
            if boundary_eligible and signature == self.boundary_signature
            else 1 if boundary_eligible else 0)
        self.boundary_signature = signature
        region = measured['chassis_region']
        eligible = (self._occupancy_valid(measured) and self._settled(measured)
                    and region is not None and not self._task_faults())
        if eligible:
            self.consecutive = self.consecutive + 1 if region == self.sample_region else 1
        else:
            self.consecutive = 0
        self.sample_region = region
        if self.consecutive >= self.stability_steps:
            if self.stable_region != region:
                self.visited.add(region)
                self.visits.append({'region': region, 'sample': self.samples,
                    'position': measured['position'], 'simulation_time_s': now})
            self.stable_region = region
        else:
            self.stable_region = None
        self.previous, self.previous_time = point, now
        self._append('physical_trajectory.jsonl', {'event': 'native_control_sample',
            'sample': self.samples, 'simulation_time_s': now, 'native_state': measured,
            'visited_regions': sorted(self.visited), 'order_stage': self.order_stage,
            'motion_faults': sorted(self.faults)})

    def report_arrival(self, room_name: str, observation):
        """Privately accept a claim only at its current native stable occupancy."""
        if not self.active:
            return {'accepted': False, 'reason': 'episode_not_active', 'claimed_room': room_name}
        measured, measurement_error = None, None
        try:
            measured, _ = self._read_boundary()
        except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
            self.faults.add('native_arrival_measurement_unavailable')
            measurement_error = f'{type(error).__name__}: {error}'
            self.measurement_errors.append({'stage': 'arrival_report', 'reason': measurement_error})
        if measurement_error is not None:
            reason = 'native_arrival_measurement_unavailable'
        elif not isinstance(room_name, str) or not room_name:
            reason = 'invalid_agent_room_claim'
        elif self._task_faults() or not self._occupancy_valid(measured):
            reason = 'native_trajectory_or_current_state_invalid'
        elif measured['chassis_region'] != room_name:
            reason = 'claim_does_not_match_current_native_room'
        elif room_name not in self.visited:
            reason = 'room_not_natively_visited'
        elif self.stable_region != room_name or self.consecutive < self.stability_steps or not self._settled(measured):
            reason = 'current_native_occupancy_not_stable'
        else:
            reason = None
            self.reported.add(room_name)
        row = {'event': 'agent_arrival_report', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'claimed_room': room_name, 'accepted': reason is None, 'reason': reason,
            'native_control_sample': self.samples}
        row['order_advanced'] = False
        if reason is None:
            if self.required_order:
                if (self.order_stage < len(self.required_order)
                        and room_name == self.required_order[self.order_stage]):
                    item_id = f'step:{self.order_stage}:{room_name}'
                    self.order_stage += 1
                    row['order_advanced'] = True
                    row['completion_item_id'] = item_id
                    self.completed_events[item_id] = dict(row)
            elif room_name in self.required:
                item_id = f'room:{room_name}'
                row['completion_item_id'] = item_id
                self.completed_events.setdefault(item_id, dict(row))
        if measurement_error is not None:
            row['exception_summary'] = measurement_error
        self.arrival_reports.append(row)
        self._append('trajectory.jsonl', {**row, 'native_state': measured})
        return row

    def freeze_done_boundary(self, observation):
        """Freeze current boundary credit once; no history maximum or physics step."""
        if self.done_boundary_credit is not None:
            return self.record(observation)
        measured, additional = self._read_boundary()
        signature = tuple(measured.get('footprint_regions', ()))
        stable = (len(signature) == 9 and signature == self.boundary_signature
                  and self.boundary_stable_samples >= self.stability_steps
                  and measured.get('floor_layer_known', False) and self._settled(measured)
                  and not self._task_faults() and all(additional.values()))
        # Claims at older action boundaries cannot turn an incidental terminal
        # overlap into an arrival. DONE may carry these declarations itself.
        claimed = {r['claimed_room'] for r in self.arrival_reports
                   if r['action_step'] == observation.action_step
                   and isinstance(r.get('claimed_room'), str)}
        self.done_boundary_credit = {'schema': 'done_room_boundary_9point_v1',
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'stable': bool(stable), 'stable_samples': self.boundary_stable_samples,
            'footprint_regions': list(signature), 'declared_rooms': sorted(claimed),
            'fractions': {room: signature.count(room) / 9 for room in self.required
                          if stable and room in claimed and room in signature}}
        return self.record(observation)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        try:
            measured, additional = self._read_boundary()
        except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
            self.faults.add('native_record_measurement_unavailable')
            self.measurement_errors.append({'stage': 'record', 'reason': f'{type(error).__name__}: {error}'})
            measured, additional = None, {}
        checks = {
            'required_native_room_set_visited': self.required.issubset(self.visited),
            'required_agent_room_set_reported': self.required.issubset(self.reported),
            'source_requested_order_proven': self.order_stage == len(self.required_order),
            'task_trajectory_evidence_valid': not self._task_faults(), **additional}
        definitions = ([(f'step:{index}:{room}', room) for index, room in enumerate(self.required_order)]
                       if self.required_order else [(f'room:{room}', room) for room in sorted(self.required)])
        items = [{'id': item_id, 'room': room, 'completed': item_id in self.completed_events,
                  'evaluation_validity': 'valid' if not self.measurement_errors or item_id in self.completed_events else 'unavailable',
                  'kind': 'historical_checkpoint',
                  'evidence': self.completed_events.get(item_id)} for item_id, room in definitions]
        for index, item in enumerate(items):
            item['credit'] = float(item['completed'])
            if (not item['completed'] and self.done_boundary_credit is not None
                    and (not self.required_order or index == self.order_stage)):
                item['credit'] = self.done_boundary_credit['fractions'].get(item['room'], 0.)
                item['completed'] = item['credit'] == 1
                if item['credit']:
                    item['evidence'] = dict(self.done_boundary_credit)
                    item['kind'] = 'terminal_condition'
        completed = sum(item['completed'] for item in items)
        validity = 'unavailable' if self.measurement_errors else 'complete'
        earned = sum(item['credit'] for item in items)
        score = earned / len(items) if items and validity == 'complete' else None
        constraints_satisfied = bool(not self._task_faults() and measured is not None and all(additional.values()))
        row = {'schema': self.schema, 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'operation': self.config['operation'],
            'task_success': bool(all(checks.values())),
            'checks': checks, 'native_state': measured,
            'scoring_version': 'task_completion_v2', 'evaluation_validity': validity,
            'completion_items': items, 'completed_goals': completed, 'total_goals': len(items),
            'completion_score': score, 'score_percent': None if score is None else 100. * score,
            'diagnostic_completion_lower_bound': earned / len(items) if items else None,
            'room_boundary_scoring': 'done_room_boundary_9point_v1',
            'done_boundary_credit': deepcopy(self.done_boundary_credit),
            'constraints_satisfied': constraints_satisfied,
            'continuous_constraints_satisfied': constraints_satisfied,
            'task_constraints_satisfied': bool(not self._task_faults() and all(additional.values())),
            'measurement_errors': list(self.measurement_errors),
            'required_regions': sorted(self.required), 'visited_regions': sorted(self.visited),
            'reported_regions': sorted(self.reported), 'required_order': self.required_order,
            'order_stage': self.order_stage, 'measured_visits': self.visits,
            'arrival_reports': self.arrival_reports, 'native_control_samples': self.samples,
            'sampled_travel_m': self.travel_m, 'motion_faults': sorted(self._task_faults()),
            'proof_requirements': {'stable_full_chassis_control_samples': self.stability_steps,
                'explicit_agent_arrival_reports': self.parameters['arrival_report_required'],
                'required_room_set': bool(self.required),
                'ordered_route': bool(self.required_order), 'terminal_room_required': False,
                'minimum_travel_required': False, 'source_route_evidence': self.parameters.get('public_route_evidence', [])},
            'configuration': self.config}
        (self.output / 'state.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        self._append('trajectory.jsonl', row)
        return row


REVISED_MAPPING_OPERATIONS = {
    'Connectivity': 'answer_direct_adjacency_with_exploration',
    'Long-Term Navigation': 'execute_supplied_ordered_route',
    'Regional Boundary': 'answer_observed_anchor_region_relation',
    'Traversable Passage': 'answer_shortest_route_via_region',
}


def legacy_mapping_config(config):
    legacy = deepcopy(config)
    legacy['operation'] = config['parameters']['legacy_operation']
    return legacy


def queried_portal_crossing(session, portal):
    """Validate current-episode evidence for an explicitly identified opening."""
    events = getattr(session, 'portal_crossings', None)
    if not isinstance(events, list):
        return {'completed': False, 'evaluation_validity': 'unavailable', 'evidence': None}
    episode_id = getattr(session, 'episode_id', None)
    for event in events:
        if (episode_id is None or event.get('episode_id') != episode_id
                or event.get('portal_id') != portal['portal_id']
                or (event.get('from_region'), event.get('to_region')) !=
                   (portal['from_region'], portal['to_region'])
                or event.get('through_opening_verified') is not True
                or event.get('public_controls_only') is not True):
            continue
        before, after = (native_array(event.get(key)) for key in ('before_position_m', 'after_position_m'))
        time = event.get('simulation_time_s')
        if (before.shape == (3,) and after.shape == (3,) and np.isfinite([before, after]).all()
                and isinstance(time, (int, float)) and math.isfinite(time)
                and np.linalg.norm(after - before) > 0):
            return {'completed': True, 'evaluation_validity': 'valid', 'evidence': deepcopy(event)}
    return {'completed': False, 'evaluation_validity': 'valid', 'evidence': None}


class SourceAnswerMappingEvaluator:
    """Combine existing native visits with semantic outputs and episode evidence."""

    def __init__(self, session, controller, config, output, native):
        self.session, self.controller, self.config = session, controller, config
        self.parameters, self.native, self.output = config['parameters'], native, Path(output)
        self.observation_declarations_available = self.parameters['actual_capability'] == 'observed_anchor_region_relation'
        if self.observation_declarations_available:
            from worldleague.native.mapping_observations import MappingObservationEvidence
            self.observation_evidence = MappingObservationEvidence(session, config)

    def capture_observation(self, observation):
        if self.observation_declarations_available:
            self.observation_evidence.capture_observation(observation)

    def report_observed_objects(self, objects, delivered_views, observation):
        if not self.observation_declarations_available:
            raise ValueError('This task has no anchor declaration contract.')
        self.observation_evidence.report(objects, delivered_views, observation)

    def begin_episode(self, observation):
        self.answer, self.route_stage = None, 0
        self.reachability_response = None
        self.anchor_evidence, self.route_events = {}, []
        if self.observation_declarations_available:
            self.observation_evidence.begin_episode()
        self.native.begin_episode(observation)
        return self.record(observation)

    def sample_step(self):
        self.native.sample_step()

    def freeze_done_boundary(self, observation):
        self.native.freeze_done_boundary(observation)
        return self.record(observation)

    def report_answer(self, answer, observation):
        if self.parameters.get('reachability_answer'):
            text = str(answer).strip()
            text = re.sub(r'^ANSWER:\s*', '', text, flags=re.IGNORECASE)
            self.answer, self.reachability_response = None, None
            for field, value in re.findall(r'(?:^|;)\s*(adjacency|reachability)\s*=\s*([^;]+)', text, re.IGNORECASE):
                if field.casefold() == 'adjacency':
                    self.answer = value.strip()
                else:
                    self.reachability_response = value.strip()
        else:
            self.answer = str(answer)

    def report_arrival(self, room_name, observation):
        row = self.native.report_arrival(room_name, observation)
        route = self.parameters.get('route_answer_order', [])
        if row.get('accepted') and self.route_stage < len(route) and room_name == route[self.route_stage]:
            self.route_events.append(deepcopy(row))
            self.route_stage += 1
        return row

    def _observed_anchors(self):
        for row in self.observation_evidence.receipts:
            self.anchor_evidence.setdefault(row['binding'], deepcopy(row))
        return self.observation_evidence.available()

    def record(self, observation, *, action=None):
        row = self.native.record(observation, action=action)
        p = self.parameters
        valid = row['evaluation_validity'] == 'complete'
        correct = self.answer is not None and ' '.join(self.answer.casefold().split()) == ' '.join(str(p['source_qa']['answer']).casefold().split())
        if p['actual_capability'] == 'observed_anchor_region_relation':
            valid = self._observed_anchors() and valid
            items = [{'id': 'anchor:' + key, 'kind': 'historical_checkpoint',
                'completed': key in self.anchor_evidence, 'evaluation_validity': 'valid',
                'evidence': self.anchor_evidence.get(key)} for key in p['required_anchor_bindings']]
        elif p.get('route_answer_order'):
            items = [{'id': f'route_step:{index}:{room}', 'kind': 'historical_checkpoint',
                'completed': index < self.route_stage, 'evaluation_validity': 'valid'}
                for index, room in enumerate(p['route_answer_order'])]
            terminal = row.get('done_boundary_credit') or {}
            if self.route_stage < len(items):
                pending = items[self.route_stage]
                pending['credit'] = terminal.get('fractions', {}).get(
                    p['route_answer_order'][self.route_stage], 0.)
                if pending['credit']:
                    pending['kind'] = 'terminal_condition'
                    pending['evidence'] = deepcopy(terminal)
        else:
            items = deepcopy(row['completion_items'])
        if p['source_qa']['scored']:
            items.append({'id': 'source_relation_answer', 'kind': 'answer_requirement',
                'completed': bool(correct), 'evaluation_validity': 'valid'})
        if p.get('reachability_answer'):
            reachable_correct = (self.reachability_response is not None
                and self.reachability_response.casefold() == p['reachability_answer']['answer'])
            items.append({'id': 'global_reachability_answer', 'kind': 'answer_requirement',
                'completed': reachable_correct, 'evaluation_validity': 'valid'})
        if p.get('queried_portal'):
            crossing = queried_portal_crossing(self.session, p['queried_portal'])
            items.append({'id': 'queried_portal_crossing', 'kind': 'historical_checkpoint', **crossing})
            valid = valid and crossing['evaluation_validity'] == 'valid'
        if not valid:
            for item in items:
                item['evaluation_validity'] = 'unavailable'
        if p.get('reachability_answer'):
            row['reachability_response'] = self.reachability_response
            row['reachability_answer_correct'] = reachable_correct
        row.update(schema='private_revised_mapping_v1', operation=self.config['operation'],
            native_visit_task_success=row['task_success'], configuration=self.config,
            completion_items=items, evaluation_validity='complete' if valid else 'unavailable',
            answer=self.answer, answer_correct=bool(correct),
            observed_anchor_evidence=deepcopy(self.anchor_evidence), route_answer_stage=self.route_stage,
            anchor_declarations=deepcopy(self.observation_evidence.claims) if self.observation_declarations_available else [],
            anchor_capture_evidence=deepcopy(self.observation_evidence.capture) if self.observation_declarations_available else None,
            native_anchor_relation=deepcopy(getattr(self.native, 'relation_evidence', None)),
            task_success=bool(valid and row['task_constraints_satisfied'] and all(item['completed'] for item in items)))
        from worldleague.tasks.completion import progress
        row.update(progress(row))
        (self.output / 'state.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        return row
