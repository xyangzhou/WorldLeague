"""Restore compiled-source topological collision exclusions for imported robots.

Used by worldleague continuous --help.
"""
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET


def source_topology_collision_pairs(mjcf_path, link_names=None):
    """Return body pairs excluded by the assembled MJCF's rigid-group topology."""
    root = ET.parse(Path(mjcf_path)).getroot()
    if root.findall("include") or root.findall("contact/pair"):
        raise ValueError("Use an assembled source without explicit geom-pair overrides.")
    groups, weld_parents = {"world": "world"}, {"world": None}

    def visit(body, parent):
        name = body.attrib["name"]
        movable = body.find("joint") is not None or body.find("freejoint") is not None
        movable = movable or body.get("mocap", "false") == "true"
        groups[name] = name if movable else groups[parent]
        if movable:
            weld_parents[name] = groups[parent]
        for child in body.findall("body"):
            visit(child, name)

    for body in root.findall("worldbody/body"):
        visit(body, "world")
    flag = root.find("option/flag")
    filter_parent = flag is None or flag.get("filterparent", "enable") != "disable"
    selected = set(groups) - {"world"}
    if link_names is not None:
        selected.intersection_update(link_names)
    excluded = set()
    for left, right in combinations(sorted(selected), 2):
        a, b = groups[left], groups[right]
        same = a == b
        parent_child = (filter_parent and a != "world" and b != "world"
                        and (weld_parents[a] == b or weld_parents[b] == a))
        if same or parent_child:
            excluded.add((left, right))
    for row in root.findall("contact/exclude"):
        pair = tuple(sorted((row.attrib["body1"], row.attrib["body2"])))
        if set(pair).issubset(selected):
            excluded.add(pair)
    return [list(pair) for pair in sorted(excluded)]
