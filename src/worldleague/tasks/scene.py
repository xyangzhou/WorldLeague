"""Private minimal ESI patches and precondition checks. Use through run_esi_robot.py.

Used by worldleague scene --help.
"""
from __future__ import annotations


import copy
from pathlib import Path

import numpy as np

from .schema import ObjectPerturbationPatch, ObjectPosePatch, PrimitivePatch, RobotPosePatch


class ScenePatches:
    def __init__(self, task, bindings):
        self.task = task
        self.bindings = bindings
        self.variant = task.definition.scene.variants[task.variant]
        self.seed = task.seed
        self.applied = []
        self._before_applied = self._after_applied = False
        # Preserve the existing placement / robot pose / nuisance RNG streams.
        placement_rng = np.random.default_rng(self.seed)
        robot_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        nuisance_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2]))
        self._settings = {"camera_size": [1024, 1024]}
        self.restore_robot_head_mount = False
        self._realized = []
        shared_shift = None
        for patch in self.variant.patches:
            item = patch.model_dump(mode="json")
            if patch.operation == 'restore_robot_head_mount':
                self.restore_robot_head_mount = True
            elif isinstance(patch, RobotPosePatch):
                position = np.array(patch.position)
                yaw = patch.yaw_deg + float(robot_rng.choice(patch.heading_offsets_deg))
                position[:2] += robot_rng.uniform(-np.array(patch.jitter_xy_m), patch.jitter_xy_m)
                item.update(position=position.tolist(), yaw_deg=yaw)
                self._settings.update(robot_position=position.tolist(), robot_yaw_deg=yaw,
                                      head_pitch_deg=patch.head_pitch_deg)
            elif isinstance(patch, ObjectPosePatch):
                if patch.candidates:
                    index = int(placement_rng.integers(len(patch.candidates)))
                    candidate = patch.candidates[index]
                    position = np.array(candidate.position)
                    item.update(selected_index=index, selected_region=candidate.region)
                else:
                    position = np.array(patch.position)
                position[:2] += placement_rng.uniform(-np.array(patch.jitter_xy_m), patch.jitter_xy_m)
                item["position"] = position.tolist()
            elif isinstance(patch, ObjectPerturbationPatch):
                item["offsets_xy_m"] = {binding: nuisance_rng.uniform(-np.array(patch.jitter_xy_m),
                    patch.jitter_xy_m).tolist() for binding in patch.bindings}
            elif isinstance(patch, PrimitivePatch):
                if shared_shift is None:
                    shared_shift = float(placement_rng.uniform(-patch.shared_jitter_x_m, patch.shared_jitter_x_m))
                position = list(patch.position)
                position[0] += shared_shift
                item["position"] = position
            self._realized.append(item)

    def observation_settings(self):
        return copy.deepcopy(self._settings)

    def _binding(self, key):
        return self.bindings[key]

    def apply_to_env_config(self, config):
        if self._before_applied:
            raise RuntimeError("Scene patches are initialization-only.")
        objects = config.setdefault("objects", [])
        for item in self._realized:
            if item["phase"] != "before_construction":
                continue
            operation = item["operation"]
            if operation == 'category_initialization':
                from worldleague.native.esi_category_evaluation import owned_category_module
                module = owned_category_module(item['module'])
                hook = getattr(module, 'configure_scene', None)
                item['initialization_evidence'] = hook(config, copy.deepcopy(item['parameters']))
            elif operation == 'add_existing_r1':
                import omnigibson as og
                import yaml
                native = yaml.safe_load((Path(og.example_config_path) / 'r1_primitives.yaml').read_text())
                robots = native.get('robots', [])
                config['robots'] = copy.deepcopy(robots)
            elif operation == "robot_pose":
                robot = config["robots"][0]
                half = np.deg2rad(item["yaw_deg"]) / 2
                robot.update(position=item["position"], orientation=[0., 0., float(np.sin(half)), float(np.cos(half))],
                             pose_frame="world")
            elif operation in ("object_pose", "object_scale", "object_physics", "object_perturbation"):
                keys = [item["binding"]] if operation != "object_perturbation" else item["bindings"]
                for key in keys:
                    binding = self._binding(key)
                    obj = next((obj for obj in objects if obj.get("name") == binding.name), None)
                    if operation == "object_pose":
                        obj["position"] = item["position"]
                        if item["quaternion_xyzw"] is not None:
                            obj["orientation"] = item["quaternion_xyzw"]
                    elif operation == "object_scale":
                        scale = np.asarray(obj.get("scale", [1., 1., 1.]), dtype=float)
                        obj["scale"] = (np.broadcast_to(scale, (3,)) * item["multiplier"]).tolist()
                    elif operation == 'object_physics':
                        obj['visual_only'] = False
                        if item['fixed_base'] is not None:
                            obj['fixed_base'] = False
                    else:
                        position = np.array(obj["position"], dtype=float)
                        position[:2] += item["offsets_xy_m"][key]
                        obj["position"] = position.tolist()
            elif operation == "add_primitive":
                objects.append({"type": "PrimitiveObject", "name": item["name"], "primitive_type": "Cube",
                    "category": "object", "size": 1., "scale": item["dimensions_m"], "fixed_base": True,
                    "visual_only": False, "rgba": item["rgba"], "position": item["position"],
                    "orientation": item["quaternion_xyzw"]})
            elif operation in ("add_dataset_object", "add_usd_object"):
                obj = {"name": item["name"], "position": item["position"],
                       "orientation": item["quaternion_xyzw"], "visual_only": False}
                if operation == "add_dataset_object":
                    obj.update(type="DatasetObject", category=item["category"], model=item["model"],
                               scale=item["scale"], fixed_base=False)
                else:
                    from worldleague.resources import ROOT
                    asset = (ROOT / item["asset"]).resolve()
                    if not asset.is_relative_to(ROOT / "assets") or not asset.is_file():
                        raise ValueError("USD object asset must exist inside repository assets/.")
                    obj.update(type="USDObject", usd_path=str(asset), fixed_base=True,
                               kinematic_only=False, self_collisions=False)
                objects.append(obj)
            self.applied.append(copy.deepcopy(item))
        if any(item['operation'] == 'prepare_category_interaction' and item.get('allow_fallback_table')
               for item in self._realized):
            # Source count/inspection builders often filter the room to walls
            # and floors. Restore existing furniture before searching surfaces.
            native_scene = config.get('scene', {})
            if native_scene.get('load_object_categories'):
                support_categories = ('table', 'breakfast_table', 'coffee_table', 'dining_table',
                    'desk', 'countertop', 'counter', 'straight_chair', 'stool', 'bench', 'shelf')
                native_scene['load_object_categories'] = list(dict.fromkeys(
                    [*native_scene['load_object_categories'], *support_categories]))
            from .interaction_supports import preload_supports
            slots = max((len(item.get('placement_groups', ())) or 1 for item in self._realized
                         if item['operation'] == 'prepare_category_interaction'), default=1)
            preload_supports(config, slots)
        self._before_applied = True

    def source_postprocess_override(self, session):
        """Use an explicit owned joint-world initializer instead of one child hook.

        Ordinary source tasks retain their original postprocessor. A composed
        world may have exactly one category initializer that explicitly owns
        every source mechanism and returns its native construction receipt.
        """
        from worldleague.native.esi_category_evaluation import owned_category_module
        hooks = []
        for item in self._realized:
            if item['operation'] != 'category_initialization' or item['phase'] != 'before_construction':
                continue
            module = owned_category_module(item['module'])
            callback = getattr(module, 'postprocess_joint_world', None)
            if callable(callback):
                hooks.append((item, callback))
        if not hooks:
            return None
        if len(hooks) != 1:
            raise ValueError('A composed world requires one owner of source mechanism initialization.')
        item, callback = hooks[0]
        result = callback(session, copy.deepcopy(item['parameters']))
        self.applied.append({'operation': 'joint_world_postprocess', 'module': item['module'],
                             'initialization_evidence': copy.deepcopy(result)})
        return result

    def apply_after_postprocessing(self, session):
        if not self._before_applied or self._after_applied:
            raise RuntimeError("Postprocessing patches must run exactly once after construction.")
        session._release_edge_policy = bool(getattr(self.task, 'release_v1', False)
            or getattr(session, '_capture_release_edge_policy', False))
        for item in self._realized:
            if item['operation'] == 'prepare_category_interaction':
                from .interaction_setup import initialize_interaction
                from worldleague.native.esi_category_evaluation import owned_category_module
                owners = [getattr(owned_category_module(patch['module']), 'prepare_interaction', None)
                    for patch in self.applied if patch['operation'] == 'category_initialization'
                    and patch.get('phase') == 'after_postprocessing']
                owners = [hook for hook in owners if callable(hook)]
                if len(owners) > 1:
                    raise ValueError('An interaction layout requires a single native initializer owner.')
                item['initialization_evidence'] = (owners[0](session, self.bindings, item) if owners
                    else initialize_interaction(session, self.bindings, item))
                self.applied.append(copy.deepcopy(item))
                for initialized in self.applied:
                    if (initialized['operation'] == 'category_initialization'
                            and initialized.get('phase') == 'after_postprocessing'):
                        from worldleague.native.esi_category_evaluation import owned_category_module
                        callback = getattr(owned_category_module(initialized['module']), 'on_interaction_initialized', None)
                        if callable(callback):
                            callback(session, copy.deepcopy(item['initialization_evidence']))
                continue
            if item['operation'] == 'category_initialization':
                if item['phase'] == 'after_postprocessing':
                    from worldleague.native.esi_category_evaluation import owned_category_module
                    module = owned_category_module(item['module'])
                    hook = getattr(module, 'initialize_scene', None)
                    session._category_initialization_active = True
                    try:
                        item['initialization_evidence'] = hook(session, copy.deepcopy(item['parameters']))
                    finally:
                        session._category_initialization_active = False
                    self.applied.append(copy.deepcopy(item))
                continue
            if item['operation'] == 'restore_robot_head_mount':
                mounts = getattr(session, '_category_sensor_mounts', {})
                if not mounts or not session.env.robots:
                    raise ValueError('Original native robot head mount was not captured before source postprocessing.')
                for name, (position, quaternion) in mounts.items():
                    session.env.robots[0].sensors[name].set_position_orientation(
                        position=position, orientation=quaternion, frame='parent')
                self.applied.append({**copy.deepcopy(item), 'restored_sensors': sorted(mounts)})
                continue
            if item["operation"] == "add_usd_object":
                from isaacsim.core.api.materials import OmniGlass
                from omnigibson.utils.physx_utils import bind_material
                import omnigibson.lazy as lazy
                import torch
                obj = session.env.scene.object_registry("name", item["name"])
                meshes = {mesh.prim.GetName(): mesh for link in obj.links.values()
                          for mesh in link.visual_meshes.values()}
                if item["glass_mesh_names"]:
                    material = item["glass_material"]
                    material_path = obj.prim_path + "/Looks/OmniGlassClear"
                    OmniGlass(material_path, color=np.array(material["color"]), ior=material["ior"],
                              depth=material["depth_m"], thin_walled=material["thin_walled"])
                for name in item["glass_mesh_names"]:
                    bind_material(meshes[name].prim_path, material_path)
                    meshes[name].prim.CreateAttribute("primvars:doNotCastShadows",
                        lazy.pxr.Sdf.ValueTypeNames.Bool).Set(not material["cast_shadows"])
                from omnigibson.utils.constants import JointType
                for unit, kind in (("rad", JointType.JOINT_REVOLUTE), ("m", JointType.JOINT_PRISMATIC)):
                    for name, position in item[f"initial_joint_positions_{unit}"].items():
                        joint = obj.joints[name]
                        joint.set_pos(torch.tensor([position]), drive=False)
                        joint.set_vel(torch.zeros(1), drive=False)
            if item["phase"] != "after_postprocessing":
                continue
            binding = self._binding(item["binding"])
            obj = session.env.scene.object_registry("name", binding.name)
            if item['operation'] == 'object_physics':
                # The installed native setter enables collisions and gravity on
                # every owned link. Source postprocessing can otherwise reset it.
                # The native setter asserts stopped GPU physics even for a
                # redundant assignment. Keep already physical source objects
                # intact; an actual restoration still uses the native setter.
                if obj.visual_only:
                    obj.visual_only = False
                self.applied.append(copy.deepcopy(item))
                continue
            import torch
            values = {"position": torch.tensor(item["position"], dtype=torch.float32)}
            if item["quaternion_xyzw"] is not None:
                values["orientation"] = torch.tensor(item["quaternion_xyzw"], dtype=torch.float32)
            obj.set_position_orientation(**values)
            self.applied.append(copy.deepcopy(item))
        if session._release_edge_policy:
            from .pickup_edges import apply_edge_policy, refresh_edge_references, settle_edge_policy
            names = list(dict.fromkeys([binding.name for binding in self.bindings.values()]
                + list(getattr(session, 'task_object_names', ()))
                + list(getattr(session, 'task_state', {}).get('dynamic_object_names', ()))
                + list(getattr(session, 'postprocess', {}).get('dynamic_object_names', ()))))
            objects = [session.env.scene.object_registry('name', name) for name in names]
            session._category_initialization_active = True
            try:
                placements = apply_edge_policy(session, objects)
                settle_edge_policy(session, placements)
                session._edge_placement_bounds = {}
                refresh_edge_references(getattr(session, 'category_interaction_setup', {}), placements, session)
                for applied in self.applied:
                    refresh_edge_references(applied.get('initialization_evidence'), placements, session)
            finally:
                session._category_initialization_active = False
            self.applied.append({'operation': 'current_support_edge_policy',
                'initialization_evidence': {'schema': 'pickup_edge_policy_v2',
                    'margin_m': .03, 'placements': copy.deepcopy(session.pickup_edge_placements),
                    'physical_grasp_verified': False}})
        self._after_applied = True


class DeclarativeScene(ScenePatches):
    def install_visual_evidence(self, session, controller):
        from ..runtime.sim.esi_visual_evidence import install_visual_evidence
        return install_visual_evidence(session, controller)

    @property
    def requires_visual_discovery(self):
        parameters = self.task.definition.private.evaluation.parameters
        runtime = getattr(self.task.definition, "runtime", None)
        return (getattr(runtime, "requires_native_visual_discovery", False)
                or parameters.get("requires_native_visual_discovery", False)
                or parameters.get("actual_capability") == "observed_anchor_region_relation")

    @property
    def start_clock_at_first_frame(self):
        return self.task.definition.private.evaluation.parameters.get("prototype") == "M4"

    @property
    def reflection_sensor_profile(self):
        from ..runtime.sim.esi_reflection import profile_from_definition
        return profile_from_definition(self.task.definition)

    def preprocess_state(self, module, path, state):
        from ..runtime.sim.esi_loading import release_variant_preprocess
        return release_variant_preprocess(self.task, module.__name__, path, state)

    def initialize_robot(self, session):
        from ..runtime.sim.esi_initial_posture import initialize_controller_posture
        from .interaction_setup import initialize_loaded_room_start
        initialize_controller_posture(session, self)
        if not getattr(session, "reflection_sensor_profile", None) and any(
                item.get("operation") == "prepare_category_interaction" for item in self._realized):
            initialize_loaded_room_start(session)

    def __init__(self, task):
        super().__init__(task, task.definition.private.bindings)


    def create_evaluator(self, session, controller, output):
        from worldleague.native.esi_task_evaluation import GraspLiftEvaluator
        from worldleague.native.esi_glass_box import OpenExtractLiftEvaluator, SlideExtractLiftEvaluator
        # Evaluator selection depends on supported evaluator types, never task IDs.
        evaluators = {"grasp_lift": GraspLiftEvaluator, "open_extract_lift": OpenExtractLiftEvaluator,
                      "slide_extract_lift": SlideExtractLiftEvaluator}
        if self.task.definition.private.evaluation.type == 'stack_relations':
            from worldleague.native.esi_category_evaluation import StackRelationsEvaluator
            evaluators['stack_relations'] = StackRelationsEvaluator
        if self.task.definition.private.evaluation.type == 'extract_lift':
            from worldleague.native.esi_category_evaluation import ExtractLiftEvaluator
            evaluators['extract_lift'] = ExtractLiftEvaluator
        if self.task.definition.private.evaluation.type == 'category_operation':
            from worldleague.native.esi_category_evaluation import create_category_operation_evaluator
            evaluators['category_operation'] = create_category_operation_evaluator
        config = self.task.definition.private.evaluation.model_dump()
        config.update(scene=self.task.scene_name, room=self.task.room_name)
        if config['type'] == 'category_operation':
            config['bindings'] = {key: self._binding(key).name for key in config['binding_keys']}
        else:
            config['target'] = self._binding(config['target']).name
        if config["type"] in ('extract_lift', "open_extract_lift", "slide_extract_lift"):
            config["container"] = self._binding(config["container"]).name
        if config['type'] == 'stack_relations':
            config['relations'] = [{'upper': self._binding(r['upper']).name,
                                    'lower': self._binding(r['lower']).name} for r in config['relations']]
        try:
            evaluator = evaluators[config["type"]](session, controller, config, output)
        except Exception as error:
            if self.task.run_mode != 'diagnostic':
                raise
            from worldleague.native.esi_category_evaluation import UnavailableEvaluator
            evaluator = UnavailableEvaluator(config, 'evaluator_initialization', error)
        if self.task.definition.kind == 'category_robot_task':
            from worldleague.native.esi_category_evaluation import CategoryEvaluationEnvelope
            evaluator = CategoryEvaluationEnvelope(evaluator, self.task, output)
        if hasattr(evaluator, "sample_step"):
            previous_on_step = controller.on_step
            def sample_evaluation():
                if previous_on_step:
                    previous_on_step()
                evaluator.sample_step()
            controller.on_step = sample_evaluation
        return evaluator
