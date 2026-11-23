"""Select a measured local surface during scene placement.

Used by worldleague scene.
"""
from __future__ import annotations


def local_support_surface(floor_path, hits):
    """Select the nearest upward-facing hit belonging to a bound support."""
    floor_hits = [hit for hit in hits if hit.get("hit")
                  and (hit["rigidBody"] == floor_path or hit["rigidBody"].startswith(floor_path + "/"))
                  and float(hit["normal"][2]) > .5]
    if not floor_hits:
        raise ValueError("No local upward-facing support surface was found.")
    return min(floor_hits, key=lambda hit: float(hit["distance"]))
