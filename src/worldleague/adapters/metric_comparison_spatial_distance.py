"""Preserve Spatial Distance scenes and add private selected-object lift goals.

Run CategoryConverter().convert_all(sources, ConversionContext(output_root))
from the repository root, or use the shared category campaign CPU runner.
"""
from __future__ import annotations

import math

import numpy as np


def _normalize(value):
    return str(value or '').strip().lower().replace(' ', '_')


def resolve_fixed_distance_reference(session, source_evidence, *, position_tolerance_m=.02,
                                     heading_tolerance_deg=3., bounds_tolerance_m=.02):
    """Resolve one unchanged native anchor from private source geometry, never QA.

    Return ``(native_object_or_None, JSON_evidence)``. Missing, moved or ambiguous
    anchors fail closed. The caller must separately check candidate XY ordering
    before/after a common Z-only support lift; this function moves nothing and
    awards no physical goal credit.
    """
    evidence = {'accepted': False, 'candidate_count': 0,
                'reason': 'fixed_distance_reference_unresolved'}
    try:
        from worldleague.tasks.interaction_setup import array as native_array
        metadata = source_evidence['reference_metadata']
        reconstructed = source_evidence['reference_aabb_xy_reconstruction']
        original_position = np.asarray(metadata['position'], dtype=float)
        original_quaternion = np.asarray(metadata['quaternion_xyzw'], dtype=float)
        centre = np.asarray(reconstructed['centre_xy'], dtype=float)
        half = np.asarray(reconstructed['half_extents_xy'], dtype=float)
        if (original_position.shape != (3,) or original_quaternion.shape != (4,)
                or centre.shape != (2,) or half.shape != (2,) or np.any(half <= 0)
                or not all(np.isfinite(value).all() for value in (
                    original_position, original_quaternion, centre, half))
                or np.linalg.norm(original_quaternion) == 0
                or not all(math.isfinite(value) and value >= 0 for value in (
                    position_tolerance_m, heading_tolerance_deg, bounds_tolerance_m))):
            raise ValueError('Invalid source anchor geometry or resolution tolerance.')
        category = metadata.get('category') or source_evidence['reference_category']
        name = metadata.get('name') or source_evidence.get('reference_native_name')
        values = session.env.scene.objects
        objects = list(values.values()) if isinstance(values, dict) else list(values)
        matches, rejections = [], []
        for obj in objects:
            if _normalize(obj.category) != _normalize(category) or (name and obj.name != name):
                continue
            position, quaternion = (native_array(value) for value in obj.get_position_orientation())
            low, high = (native_array(value) for value in obj.aabb)
            if (position.shape != (3,) or quaternion.shape != (4,) or low.shape != (3,)
                    or high.shape != (3,) or np.any(high <= low)
                    or not all(np.isfinite(value).all() for value in (position, quaternion, low, high))
                    or np.linalg.norm(quaternion) == 0):
                rejections.append({'name': obj.name, 'reason': 'invalid_native_anchor_geometry'})
                continue
            cosine = abs(float(np.dot(quaternion / np.linalg.norm(quaternion),
                                      original_quaternion / np.linalg.norm(original_quaternion))))
            angle = math.degrees(2 * math.acos(float(np.clip(cosine, 0., 1.))))
            position_error = float(np.linalg.norm(position - original_position))
            bounds_error = float(max(np.max(np.abs(low[:2] - (centre - half))),
                                     np.max(np.abs(high[:2] - (centre + half)))))
            row = {'name': obj.name, 'position_m': position.tolist(),
                   'quaternion_xyzw': quaternion.tolist(), 'bounds_m': [low.tolist(), high.tolist()],
                   'source_position_error_m': position_error, 'source_heading_error_deg': angle,
                   'source_xy_bounds_error_m': bounds_error}
            if (position_error <= position_tolerance_m and angle <= heading_tolerance_deg
                    and bounds_error <= bounds_tolerance_m and not obj.visual_only):
                matches.append((obj, row))
            else:
                rejections.append({**row, 'reason': 'native_anchor_differs_from_source'})
        evidence.update(candidate_count=len(matches), rejected_candidates=rejections,
                        source_reference_category=category)
        if len(matches) != 1:
            evidence['reason'] = ('fixed_distance_reference_ambiguous' if matches
                                  else 'fixed_distance_reference_unresolved')
            return None, evidence
        obj, row = matches[0]
        evidence.update(accepted=True, reason=None, reference=row)
        return obj, evidence
    except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, OverflowError) as error:
        evidence.update(reason='fixed_distance_reference_measurement_unavailable',
                        exception_summary=f'{type(error).__name__}: {error}')
        return None, evidence
