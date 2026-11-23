"""Apply the native contact sleep setting. Run with worldleague scene T24 or T25."""


def initialize_scene(session, parameters):
    links = []
    for name in parameters['object_names']:
        obj = session.env.scene.object_registry('name', name)
        links.extend(link for link in obj.links.values() if not link.is_meta_link and link.collision_meshes)
    for link in links:
        link.sleep_threshold = parameters['sleep_threshold']
        link.wake()
    return {'links_updated': len(links), 'sleep_threshold': parameters['sleep_threshold']}
