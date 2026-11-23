# Episode scoring

Both interfaces use the selected task's native measurements. A VLM supplies decision-boundary declarations on tasks that require them. Continuous policies use native control boundaries and the task's action-only declaration rules. The simulator scores each episode before terminal QA.

| Output | Rule |
| --- | --- |
| `interactive_score` | 100 times the sum of physical predicate credits divided by the number of physical predicates. A measured predicate contributes 0 or 1 unless the task defines fractional credit. Semantic answer items are excluded. |
| `physical_success` | All physical goals must be complete and all applicable task constraints, including terminal release or a required declaration, must hold. Verified historical progress survives a later terminal constraint failure. |
| `terminal_qa` | Percentage of the task's terminal questions answered correctly. Each answer is checked against its schema and reference. Missing or malformed answers on a measurable question earn zero. Missing reference evidence makes the task QA percentage unavailable. Continuous episodes omit QA. |

A Boolean physical success is reported in `result.json`. A missing measurement is represented by `null`. For example, where every goal is measured but a required constraint is unknown, the Interactive Score remains available and success can be unknown. Invalid or pending episodes with an unmeasurable score are excluded from the corresponding mean. `worldleague profile RESULTS` averages available episode values within each task, then weights tasks equally within each dimension, track and overall. A success rate is a fraction from 0 to 1. Each metric has its own task count.

## How physical events count

- **Room visits.** A visit requires the native task's room and stability evidence, plus the applicable arrival declaration. Ordered visits count after their predecessor. The final room-boundary tasks assign fractional credit using chassis sample points.
- **Rigid insertion (T22–T25).** Evaluate each object against its eligible container in the container's current pose. The public T23 contract allows sequential verification and removal after the dwell. Verified objects keep their credit. Full success still requires released hands at the end. Moving a box alone neither earns insertion credit nor cancels it.
- **Tray arrangements (T26, T28, T30, T32, T33).** Evaluate containment in the current tray frame, native support/contact, the declared relation, and stable released placement. Translating or rotating the tray and objects together preserves the relative placement. Moving the tray out from under the objects loses placement when support or containment is lost. The original table and source furniture do not enter the terminal placement predicate.
- **Task-specific references.** A mirror task can explicitly require a fixed optical cue or anchor. A ramp comparison can require an unchanged ramp and materials. A history task can require the contents of a particular box to be preserved. These requirements use their bound objects. Incidental handling of ordinary movable scenery does not impose an extra episode-wide penalty.
- **Terminal QA.** The public question gives its answer type and observation time. String comparisons use case folding and whitespace normalization after schema validation. Lists are order-insensitive only where the question specifies this. A VLM's terminal answers cannot change the already computed physical score.

Read `result.json` for the three top-level metrics. `physical_score.json` contains the physical item IDs, measured credit or completion, validity, denominator, constraint status and binary outcome. VLM runs also write question-level results to `terminal_qa.json`. Read the selected benchmark task's public success criteria for the applicable geometry, order, dwell and release requirements. Under instruction protocol `task_instructions_v3`, policies receive the physical task and terminal questions. The evaluator computes the physical and QA metrics after the interaction.
