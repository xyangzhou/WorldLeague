"""Reduce native spatial-tray evidence without advancing the simulator.

Used by worldleague scene --help.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from numbers import Real

import numpy as np

FINAL_STATE_REVISION = 'final_state_v1'
MOVABLE_FIXTURE_SCORING_VERSION = 'wia_spatial_tray_current_tray_progress_v2'


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    return float(value) if np.isfinite(value) else None


def _vector(value, shape):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.shape != shape:
            return None
        result = raw.astype(float, copy=True)
        return result if np.isfinite(result).all() else None
    except (TypeError, ValueError, OverflowError):
        return None


def tray_reference_bounds(row, current_deck, initial_deck, *, points=None):
    """Express measured world geometry in the initial tray frame.

    p_reference = R_initial R_current^-1 (p_world - t_current) + t_initial.
    Use native collision points for an oblique deck rotation and box corners for axis permutations.
    """
    from scipy.spatial.transform import Rotation
    low, high = (_vector(row.get(key), (3,)) for key in ('aabb_min', 'aabb_max'))
    poses = [(_vector(deck.get('position'), (3,)), _vector(deck.get('quaternion_xyzw'), (4,)))
             for deck in (current_deck, initial_deck)]
    if (low is None or high is None or np.any(high <= low)
            or any(p is None or q is None or np.linalg.norm(q) == 0 for p, q in poses)):
        raise ValueError('Current and initial tray poses and target bounds are required.')
    (position, quaternion), (initial_position, initial_quaternion) = poses
    if points is None:
        points = np.array(list(product(*zip(low, high))))
    local = Rotation.from_quat(quaternion).inv().apply(points - position)
    reference = Rotation.from_quat(initial_quaternion).apply(local) + initial_position
    return {**row, 'aabb_min': reference.min(axis=0).tolist(),
            'aabb_max': reference.max(axis=0).tolist()}


class SpatialTrayProgress:
    """Reduce the declared manipulation contract and simultaneous terminal dwell."""

    def __init__(self, parameters):
        self.p = deepcopy(parameters)
        self.scoring = self.p['scoring']
        self.final_state = self.scoring.get('manipulation_revision') == FINAL_STATE_REVISION
        self._begun = False

    def begin(self, sample):
        self._begun = True
        self._initial = deepcopy(sample)
        self._initial_errors = []
        self._violations = set()
        self._states = {key: {'acquired': False, 'moved': False, 'maximum_held_lift_m': None,
                              'maximum_displacement_m': None} for key in self.p['targets']}
        self._last_time = self._high_time = self._dwell_start = None
        self._stable_duration = 0.
        self._current = {}
        result = self._consume(sample, initial=True)
        self._initial_errors = list(result['measurement_errors'])
        if self._initial_errors:
            self._current['available'] = False
        return self.result()

    def _inside(self, row):
        low, high = _vector(row.get('aabb_min'), (3,)), _vector(row.get('aabb_max'), (3,))
        bounds = np.asarray(self.p['tray']['usable_world_xy_m'])
        return bool(low is not None and high is not None and np.all(high > low)
                    and np.all(low[:2] >= bounds[0] - 1e-9) and np.all(high[:2] <= bounds[1] + 1e-9))

    def _relation(self, sample, rows):
        goal = self.p['goal_relation']
        if goal['kind'] == 'shape':
            from worldleague.adapters.spatial_relations_geometric_configuration import shape_geometry
            centres = [(np.asarray(row['aabb_min'])[:2] + np.asarray(row['aabb_max'])[:2]) / 2
                       for row in rows.values()]
            return shape_geometry(centres, goal['shape'])
        geometry = deepcopy(sample.get('contact_geometry', {}))
        geometry['satisfied'] = bool(geometry.get('available') is True
            and geometry.get('relation') == goal['relation'] and geometry.get('penetrating') is False
            and (goal['relation'] != 'separated'
                 or (_number(geometry.get('minimum_separation_m')) is not None
                     and geometry['minimum_separation_m'] >= self.scoring['minimum_separation_m'])))
        return geometry

    def update(self, sample):
        if not self._begun:
            raise RuntimeError('Begin the spatial-tray episode before sampling.')
        return self._consume(sample, initial=False)

    def _consume(self, sample, *, initial):
        errors, rows = [], {}
        now = _number(sample.get('native_time_s'))
        gap = bool(now is not None and self._last_time is not None
                   and now - self._last_time > self.scoring['max_sample_gap_s'])
        if now is None:
            errors.append('native_time_unavailable')
        elif self._high_time is not None and now < self._high_time:
            self._violations.add('native_time_regressed')
        if now is not None:
            self._high_time = now if self._high_time is None else max(now, self._high_time)
        self._last_time = now
        for flag in ('control_valid', 'fixtures_valid', 'tray_valid'):
            if type(sample.get(flag)) is not bool:
                errors.append(flag + '_unavailable')
            elif sample[flag] is False:
                self._violations.add(flag + '_false')
        grasp = sample.get('grasp', {})
        held, released = grasp.get('held_names'), grasp.get('released')
        grasp_valid = (grasp.get('available') is True and isinstance(held, list)
                       and all(isinstance(name, str) and name for name in held)
                       and len(set(held)) == len(held) and type(released) is bool
                       and not (released and held))
        if not grasp_valid:
            errors.append('aggregate_grasp_unavailable_or_inconsistent')
            held, released = [], False
        checks = {}
        for key, name in self.p['targets'].items():
            row = deepcopy(sample.get('targets', {}).get(key, {}))
            rows[key] = row
            low, high = _vector(row.get('aabb_min'), (3,)), _vector(row.get('aabb_max'), (3,))
            position = _vector(row.get('position'), (3,))
            if (row.get('name') != name or low is None or high is None or position is None
                    or np.any(high <= low)):
                errors.append(key + ':native_identity_or_geometry_unavailable')
            scale_valid = True
            if self.final_state:
                scale = _vector(row.get('scale'), (3,))
                initial_scale = _vector(self._initial.get('targets', {}).get(key, {}).get('scale'), (3,))
                if scale is None or np.any(scale <= 0):
                    errors.append(key + ':native_scale_unavailable')
                    scale_valid = False
                elif initial_scale is None or np.any(initial_scale <= 0):
                    errors.append(key + ':initial_native_scale_unavailable')
                    scale_valid = False
                elif not np.array_equal(scale, initial_scale):
                    self._violations.add(key + ':asset_scale_changed')
                    scale_valid = False
            for field in ('linear_speed_m_s', 'angular_speed_rad_s'):
                if _number(row.get(field)) is None or row[field] < 0:
                    errors.append(key + ':' + field + '_unavailable')
            for field in ('dynamic_collidable', 'robot_contact', 'on_top_deck'):
                if type(row.get(field)) is not bool:
                    errors.append(key + ':' + field + '_unavailable')
            deck = row.get('deck_contact_geometry', {})
            if deck.get('available') is not True:
                errors.append(key + ':native_deck_contact_unavailable')
            if row.get('dynamic_collidable') is False:
                self._violations.add(key + ':target_not_dynamic_collidable')
            initial_row = self._initial.get('targets', {}).get(key, {})
            initial_position = _vector(initial_row.get('position'), (3,))
            initial_low = _vector(initial_row.get('aabb_min'), (3,))
            if not initial and position is not None and initial_position is not None:
                distance = float(np.linalg.norm(position - initial_position))
                state = self._states[key]
                previous = state['maximum_displacement_m']
                state['maximum_displacement_m'] = distance if previous is None else max(previous, distance)
                if not self.final_state:
                    state['moved'] |= distance >= self.scoring['minimum_displacement_m']
                if grasp_valid and name in held and low is not None and initial_low is not None:
                    lift = float(low[2] - initial_low[2])
                    previous = state['maximum_held_lift_m']
                    state['maximum_held_lift_m'] = lift if previous is None else max(previous, lift)
                    if not self.final_state:
                        state['acquired'] |= lift >= self.scoring['minimum_lift_m']
            deck_key = self.p['tray']['deck_key']
            try:
                if 'tray_frame_bounds_m' in row:
                    bounds = row['tray_frame_bounds_m']
                    if bounds is None:
                        raise ValueError('Target collision bounds in the tray frame are unavailable.')
                    tray_row = dict(row, aabb_min=bounds[0], aabb_max=bounds[1])
                else:
                    tray_row = tray_reference_bounds(row, sample.get('fixtures', {}).get(deck_key, {}),
                        self._initial.get('fixtures', {}).get(deck_key, {}))
                rows[key] = tray_row
            except ValueError as error:
                errors.append(key + ':current_tray_geometry_unavailable:' + str(error))
                tray_row = {}
            tray_low = _vector(tray_row.get('aabb_min'), (3,))
            gap_z = None if tray_low is None else float(tray_low[2] - self.p['tray']['deck_top_world_z_m'])
            checks[key] = {
                'wholly_inside_usable_xy': self._inside(tray_row),
                'actual_nonpenetrating_deck_contact': bool(deck.get('available') is True
                    and deck.get('relation') == 'touching' and deck.get('penetrating') is False),
                'on_top_deck': row.get('on_top_deck') is True,
                'bottom_at_deck': gap_z is not None and abs(gap_z) <= self.scoring['maximum_bottom_gap_m'],
                'no_robot_support': row.get('robot_contact') is False,
                'dynamic_collidable': row.get('dynamic_collidable') is True,
                'stationary': bool(_number(row.get('linear_speed_m_s')) is not None
                    and _number(row.get('angular_speed_rad_s')) is not None
                    and row['linear_speed_m_s'] <= self.scoring['maximum_linear_speed_m_s']
                    and row['angular_speed_rad_s'] <= self.scoring['maximum_angular_speed_rad_s']),
            }
            if self.final_state:
                checks[key]['asset_scale_preserved'] = scale_valid
                checks[key]['original_identity_preserved'] = row.get('name') == name
        relation = {'available': False, 'satisfied': False}
        try:
            relation = self._relation(sample, rows)
            if self.p['goal_relation']['kind'] == 'contact' and relation.get('available') is not True:
                errors.append('native_pair_contact_unavailable')
        except (ValueError, KeyError, TypeError) as error:
            errors.append('declared_relation_unavailable:' + str(error))
        if gap:
            errors.append('native_sample_gap_exceeded')
        available = bool(not errors and not self._initial_errors)
        terminal = bool(available and not self._violations and released and not held
            and relation.get('satisfied') is True and all(all(value.values()) for value in checks.values())
            and (self.final_state or all(state['acquired'] and state['moved'] for state in self._states.values())))
        if not terminal or initial:
            self._dwell_start, self._stable_duration = None, 0.
        else:
            if self._dwell_start is None:
                self._dwell_start = now
            self._stable_duration = now - self._dwell_start
        self._current = {'available': available, 'measurement_errors': errors, 'native_time_s': now,
                         'terminal_checks': checks, 'declared_relation': relation,
                         'released_empty_hands': bool(grasp_valid and released and not held),
                         'terminal_valid': terminal, 'sample_gap_exceeded': gap}
        return self.result()

    def result(self):
        current = self._current
        stable = bool(current.get('terminal_valid') and self._stable_duration + 1e-9 >= self.scoring['stability_duration_s'])
        first = ('targets_on_tray', bool(current.get('terminal_checks'))
                 and all(all(value for key, value in checks.items() if key != 'stationary')
                         for checks in current['terminal_checks'].values())) if self.final_state else (
                 'retrieve_all', all(state['acquired'] and state['moved'] for state in self._states.values()))
        checkpoints = [first,
                       ('declared_relation', current.get('declared_relation', {}).get('satisfied') is True),
                       ('stable_released_tray_placement', stable)]
        validity = 'invalid' if self._violations else 'complete' if current.get('available') else 'unavailable'
        score = sum(done for _, done in checkpoints) / len(checkpoints) if validity == 'complete' else None
        success = bool(validity == 'complete' and all(done for _, done in checkpoints))
        version = MOVABLE_FIXTURE_SCORING_VERSION
        revision = {'manipulation_revision': FINAL_STATE_REVISION} if self.final_state else {}
        from .tray_evidence import resolve_known_failure
        return resolve_known_failure({'schema': version, 'scoring_version': version, **revision,
            'evaluation_validity': validity, 'task_success': success, 'physical_success': success,
            'completion_items': [{'id': key, 'completed': bool(done),
                'kind': 'historical_checkpoint' if key == 'retrieve_all' else 'terminal_condition'}
                for key, done in checkpoints], 'completed_goals': sum(done for _, done in checkpoints),
            'total_goals': len(checkpoints), 'completion_score': score,
            'score_percent': None if score is None else 100 * score,
            'per_target': self._states, 'stable_duration_s': self._stable_duration,
            'required_stability_duration_s': self.scoring['stability_duration_s'],
            'initial_measurement_errors': self._initial_errors,
            'constraint_violations': sorted(self._violations), **current})
