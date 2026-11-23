"""Load model-independent embodiments.

Used by worldleague continuous --help.
"""
from copy import deepcopy


ROBOT_TYPES = {"r1pro": "R1Pro", "pandaomron": "PandaOmron"}


def robot_type(config):
    return ROBOT_TYPES[config["model"]]


def native_robot_config(config, previous, *, legacy=True):
    """Translate SDK constructor spelling while retaining declared control semantics."""
    result = deepcopy(config)
    result.pop("eval", None)  # Camera-role metadata belongs to WIA, not the robot constructor.
    if legacy:
        result.pop("model")
        result["type"] = robot_type(config)
    # Source/variant placement is task-owned; never inherit R1 joints or controllers.
    for key in ("position", "orientation", "pose_frame"):
        if key in previous:
            result[key] = deepcopy(previous[key])
    return result


def apply_robot_settings(robot, simulator, config):
    """Apply YAML optics and link masses before scene settling on the installed SDK."""
    settings = config["eval"]
    masses = settings.get("link_masses", {})
    receipt = {"cameras": {}, "link_masses": {}}
    if masses:
        playing = simulator.is_playing()
        if playing:
            simulator.stop()
        try:
            for name, mass in masses.items():
                link = getattr(robot, name)
                link.mass = float(mass)
                receipt["link_masses"][name] = float(link.mass)
        finally:
            if playing:
                simulator.play()
    for role, values in settings.get("camera_settings", {}).items():
        sensor = robot.sensors[settings["camera_sensor_names"][role]]
        receipt["cameras"][role] = {}
        for key, value in values.items():
            if getattr(sensor, key) != value:
                setattr(sensor, key, value)
            actual = float(getattr(sensor, key)) if key in {"horizontal_aperture", "focal_length"} else int(getattr(sensor, key))
            receipt["cameras"][role][key] = actual
    return receipt


def prime_cameras(robot, simulator, roles, settings):
    """Initialize renderer camera annotators before collecting the policy observation."""
    cameras = [robot.sensors[name] for name in roles.values()]
    for camera in cameras:
        _ = camera.intrinsic_matrix
    updates = 0
    for _ in range(settings["max_attempts"]):
        for _ in range(settings["render_updates_per_attempt"]):
            simulator.render()
            updates += 1
        matrices = [camera.intrinsic_matrix for camera in cameras]
        if all(float(matrix[0, 0]) > 0 and float(matrix[1, 1]) > 0 for matrix in matrices):
            break
    return updates
