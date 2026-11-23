"""Closed task contracts. Validate through worldleague.tasks.loader.load_task(path)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Number = float
Positive = Number
Nonnegative = Number
Vector3 = tuple[Number, Number, Number]
Vector2 = tuple[Nonnegative, Nonnegative]
Quaternion = tuple[Number, Number, Number, Number]
Text = str
Capability = Literal["base_translation", "base_rotation", "ee_translation", "gripper", "ee_rotation", "trunk_height"]


class ClosedModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Source(ClosedModel):
    provider: Literal["esi"]
    json_path: Text = Field(alias="json")


class PublicTask(ClosedModel):
    objective: Text
    success_criteria: Text
    environment: Text
    constraints: tuple[Text, ...] = ()

    @field_validator("objective", "success_criteria")
    @classmethod
    def public_instructions(cls, text: str) -> str:
        """Present previously published tasks with the current instruction wording."""
        return (text
                .replace("Physical progress and physical success are scored separately from terminal QA accuracy. ", "")
                .replace("The task score allocates 20 points to QA accuracy and 80 points to physical-task completion. ", "")
                .replace("The physical result is frozen before the QA request. ", "")
                .replace("Operation credit also requires your active DONE command after observing the containment state.",
                         "After observing the containment state, issue DONE."))

    @field_validator("constraints")
    @classmethod
    def public_constraints(cls, constraints: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(text.removeprefix("A correct physical state alone earns no credit. ") for text in constraints)

    def task_text(self):
        parts = [self.objective.strip(), "Physical success criteria: " + self.success_criteria.strip()]
        if self.constraints:
            parts.append("Task constraints: " + "; ".join(self.constraints))
        return "\n\n".join(parts)


class Requirements(ClosedModel):
    capabilities: tuple[Capability, ...]


class Binding(ClosedModel):
    name: Text
    category: Text | None = None
    model: Text | None = None


class GraspLiftEvaluation(ClosedModel):
    type: Literal["grasp_lift"]
    target: Text
    minimum_lift_m: Positive


class ExtractLiftEvaluation(ClosedModel):
    type: Literal['extract_lift']
    target: Text
    container: Text
    minimum_lift_m: Positive = 0.05


class CategoryOperationEvaluation(ClosedModel):
    """An owned category adapter supplies private, read-only native evidence."""
    type: Literal['category_operation']
    module: Text
    operation: Text
    binding_keys: tuple[Text, ...] = ()
    parameters: dict = Field(default_factory=dict)


class StackRelation(ClosedModel):
    upper: Text
    lower: Text


class StackRelationsEvaluation(ClosedModel):
    type: Literal['stack_relations']
    target: Text
    relations: tuple[StackRelation, ...] = Field(min_length=1)
    stability_steps: int = 15
    minimum_lift_m: Positive = 0.05


class OpenExtractLiftEvaluation(ClosedModel):
    type: Literal["open_extract_lift"]
    target: Text
    container: Text
    hinge_joint: Text = "lid_hinge"
    minimum_open_angle_deg: Number = 60.
    rim_height_m: Positive
    extraction_clearance_m: Positive = 0.005
    minimum_lift_m: Positive = 0.05


class SlideExtractLiftEvaluation(ClosedModel):
    type: Literal["slide_extract_lift"]
    target: Text
    container: Text
    slide_joint: Text = "lid_slide"
    minimum_open_displacement_m: Positive = 0.20
    rim_height_m: Positive
    extraction_clearance_m: Positive = 0.005
    minimum_lift_m: Positive = 0.05


class PrivateTask(ClosedModel):
    bindings: dict[str, Binding]
    evaluation: Annotated[GraspLiftEvaluation | ExtractLiftEvaluation | CategoryOperationEvaluation | OpenExtractLiftEvaluation | SlideExtractLiftEvaluation | StackRelationsEvaluation,
                          Field(discriminator="type")]


class RobotPosePatch(ClosedModel):
    operation: Literal["robot_pose"]
    phase: Literal["before_construction"] = "before_construction"
    position: Vector3
    yaw_deg: Number
    heading_offsets_deg: tuple[Number, ...] = (0.,)
    jitter_xy_m: Vector2 = (0., 0.)
    head_pitch_deg: Number = 45.


class Placement(ClosedModel):
    region: Text
    position: Vector3


class ObjectPosePatch(ClosedModel):
    operation: Literal["object_pose"]
    phase: Literal["before_construction", "after_postprocessing"] = "before_construction"
    binding: Text
    position: Vector3 | None = None
    candidates: tuple[Placement, ...] = ()
    quaternion_xyzw: Quaternion | None = None
    jitter_xy_m: Vector2 = (0., 0.)


class RobotHeadMountPatch(ClosedModel):
    operation: Literal['restore_robot_head_mount']
    phase: Literal['after_postprocessing'] = 'after_postprocessing'


class ExistingRobotPatch(ClosedModel):
    """Add the installed R1 configuration to an originally robot-free builder."""
    operation: Literal['add_existing_r1']
    phase: Literal['before_construction'] = 'before_construction'


class CategoryInitializationPatch(ClosedModel):
    """An owned converter may restore original mechanisms during initialization."""
    operation: Literal['category_initialization']
    phase: Literal['before_construction', 'after_postprocessing'] = 'after_postprocessing'
    module: Text
    parameters: dict = Field(default_factory=dict)


class CategoryInteractionPatch(ClosedModel):
    """Screen native starts and declare reversible support placement rules."""
    operation: Literal['prepare_category_interaction']
    phase: Literal['after_postprocessing'] = 'after_postprocessing'
    mode: Literal['navigation', 'manipulation', 'articulation', 'inspection']
    group_bindings: tuple[Text, ...] = ()
    manipulation_bindings: tuple[Text, ...] = ()
    relocation_policy: Literal['preserve', 'rigid_group', 'comparison_pairs', 'relation_group', 'vertical_only', 'independent_groups', 'shared_tabletop', 'seeded_geometry'] = 'preserve'
    layout_parameters: dict = Field(default_factory=dict)
    controller_initial_trunk_joint_positions_rad: tuple[Number, Number, Number, Number] | None = None
    placement_groups: tuple[tuple[Text, ...], ...] = ()
    support_bindings: tuple[Text, ...] = ()
    support_preferences: dict[Text, Text] = Field(default_factory=dict)
    support_search: Literal['bounded_group', 'measured_surfaces'] = 'bounded_group'
    allow_fallback_table: bool = False
    anchor_policy: Literal['none', 'illumination_common_translation', 'liquid_volume_common_translation'] = 'none'
    source_camera_policy: Literal['translate_with_group', 'preserve'] = 'translate_with_group'
    fixed_reference_evidence: dict = Field(default_factory=dict)
    source_region: Text | None = None
    minimum_grasp_height_m: Number = .35
    maximum_grasp_height_m: Number = 1.10
    maximum_stance_distance_m: Number = .85
    warning_codes: tuple[Text, ...] = ()


class ObjectPhysicsPatch(ClosedModel):
    """Enable an existing task object for real manipulation without resizing it."""
    operation: Literal['object_physics']
    phase: Literal['before_construction', 'after_postprocessing'] = 'before_construction'
    binding: Text
    visual_only: Literal[False] = False
    fixed_base: Literal[False] | None = None


class ObjectScalePatch(ClosedModel):
    operation: Literal["object_scale"]
    phase: Literal["before_construction"] = "before_construction"
    binding: Text
    multiplier: Positive


class PrimitivePatch(ClosedModel):
    operation: Literal["add_primitive"]
    phase: Literal["before_construction"] = "before_construction"
    name: Text
    primitive_type: Literal["Cube"] = "Cube"
    position: Vector3
    dimensions_m: tuple[Positive, Positive, Positive]
    quaternion_xyzw: Quaternion = (0., 0., 0., 1.)
    rgba: tuple[Number, Number, Number, Number] = (0.25, 0.32, 0.40, 1.)
    # The existing Milestone 8 screens share one bounded X displacement.
    shared_jitter_x_m: Nonnegative = 0.


class ObjectPerturbationPatch(ClosedModel):
    operation: Literal["object_perturbation"]
    phase: Literal["before_construction"] = "before_construction"
    bindings: tuple[Text, ...]
    jitter_xy_m: Vector2


class DatasetObjectPatch(ClosedModel):
    operation: Literal["add_dataset_object"]
    phase: Literal["before_construction"] = "before_construction"
    name: Text
    category: Text
    model: Text
    position: Vector3
    scale: Positive | tuple[Positive, Positive, Positive] = 1.
    quaternion_xyzw: Quaternion = (0., 0., 0., 1.)


class GlassMaterial(ClosedModel):
    color: tuple[Number, Number, Number] = (1., 1., 1.)
    ior: Positive = 1.45
    depth_m: Positive = 0.1
    thin_walled: bool = True
    cast_shadows: bool = False


class USDObjectPatch(ClosedModel):
    operation: Literal["add_usd_object"]
    phase: Literal["before_construction"] = "before_construction"
    name: Text
    asset: Text
    position: Vector3
    quaternion_xyzw: Quaternion = (0., 0., 0., 1.)
    glass_mesh_names: tuple[Text, ...] = ()
    glass_material: GlassMaterial = Field(default_factory=GlassMaterial)
    initial_joint_positions_rad: dict[Text, Number] = Field(default_factory=dict)
    initial_joint_positions_m: dict[Text, Number] = Field(default_factory=dict)


Patch = Annotated[RobotPosePatch | RobotHeadMountPatch | ExistingRobotPatch | CategoryInitializationPatch | CategoryInteractionPatch | ObjectPhysicsPatch | ObjectPosePatch | ObjectScalePatch | PrimitivePatch | ObjectPerturbationPatch
                  | DatasetObjectPatch | USDObjectPatch,
                  Field(discriminator="operation")]
Visibility = Literal["none", "visible", "hidden_zero", "hidden_unrecognizable"]


class SceneVariant(ClosedModel):
    patches: tuple[Patch, ...]
    initial_visibility: Visibility | None = None


class Scene(ClosedModel):
    policy: Literal["native_then_minimal_patch_else_skip"]
    variants: dict[str, SceneVariant]


class Defaults(ClosedModel):
    seed: int = 7
    max_steps: int = 60
    survey_budget_fraction: float | None = None
    wall_timeout_s: Positive = 5400.
    world_advances_during_inference: bool = True
    observation_condition: Literal["current", "initial_view_only"] = "current"
    action_mode: Literal["fixed", "parameterized"] = "parameterized"


class TaskDefinition(ClosedModel):
    id: Text
    version: int
    source: Source
    public: PublicTask
    requirements: Requirements
    private: PrivateTask
    scene: Scene
    defaults: Defaults


class CategoryTaskDefinition(TaskDefinition):
    """Pending category-generated physical goals retain the native task contracts."""
    kind: Literal['category_robot_task']
    repair: dict = Field(default_factory=dict)
    status: Literal['pending', 'ready', 'discarded'] = 'pending'
