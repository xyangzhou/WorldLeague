"""Identify incidental contact while scoring. Used by worldleague eval and continuous."""


def _background_bodies(session):
    scene, task = session.env.scene, session.loaded_task
    cached = getattr(session, '_background_bodies', None)
    if cached is None or cached[0] is not scene or cached[1] is not task:
        bound = {role.name for role in task.definition.private.bindings.values()}
        objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
        movable = {obj.name: str(obj.prim_path) for obj in objects
                   if obj is not session.robot and obj.name not in bound
                   and not obj.fixed_base and not obj.visual_only}
        cached = (scene, task, frozenset(movable), frozenset(movable.values()))
        session._background_bodies = cached
    return cached[2], cached[3]


def legal_background_grasp(session, object_name):
    names, _ = _background_bodies(session)
    return object_name in names


def legal_background_contact(session, body0, body1):
    """Recognize an unbound movable body touching any link of the robot."""
    robot = str(session.robot.prim_path)
    first, second = str(body0), str(body1)
    own_first = first == robot or first.startswith(robot + '/')
    own_second = second == robot or second.startswith(robot + '/')
    if own_first == own_second:
        return False
    path = second if own_first else first
    _, roots = _background_bodies(session)
    while path:
        if path in roots:
            return True
        path = path.rpartition('/')[0]
    return False
