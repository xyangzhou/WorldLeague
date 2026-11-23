"""Apply the joint velocities specified by a benchmark scene before construction."""
import json
from worldleague.resources import resolve_resource_path

def configure_scene(config, parameters):
    state = json.loads(resolve_resource_path(parameters['source_scene_file']).read_text())
    registry = state['state']['registry']['object_registry']
    for name, velocities in parameters['overrides'].items():
        registry[name]['joint_vel'] = list(velocities)
    config['scene']['scene_file'] = state
    return {'revision': parameters['revision'], 'initialization_only': True}
