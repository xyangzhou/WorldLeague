# Analyze evaluation runs

Give each system or experimental condition a name and its own result directory. Each episode needs its original task alias and seed in `result.json`. The default analysis uses seed 0.

```bash
worldleague analyze --system vlm=runs/vlm --system wam=runs/wam > comparison.json
worldleague analyze --system sol=runs/sol --seed 0 --compare-seed 1 > repeat.json
worldleague analyze --system visible=runs/visible --system masked=runs/masked --pair visible masked > feedback.json
```

| Output | Interpretation |
| --- | --- |
| `profile`, `common59_profile` | Average repeated episodes within task, then weight measured tasks equally. The common suite excludes T52. Success is a fraction. Each metric includes its measured task count. |
| `common59.coverage` | Best physical outcome across the supplied systems at the selected seed. A verified success settles a task. Without one, an unmeasured system leaves the best outcome unknown. Full progress with a failed constraint counts as progress without success. |
| `goal_items`, `qa_groups`, `full_qa_physical_scores` | Per-task predicate completions and fractional credits, initial/final/history QA with corresponding operation scores, and zero/partial/full physical score among fully correct QA episodes. |
| `decisions`, `termination`, `budget_use` | Decision outcome counts, stop reasons, next-command changes, and complete 1/3/5-decision windows within each episode. Rates use eligible windows or average the eligible episode rates. Termination is paired with the physical outcome. |
| `resources` | Task-weighted means overall and by dimension. The reference HTTP client records each interaction or terminal request, latency and provider-reported usage. Per-field coverage accompanies token and cost totals. Custom policy computation time is recorded separately from HTTP request time. |
| `paired_seed_tasks`, `paired_conditions` | Scores for the same task in both runs and the difference (second minus first). Budget use and termination for individual episodes are in each system's report. |

WorldHarness writes one row per budgeted turn to `decision_events.jsonl`, including invalid decisions, phase declarations, final DONE and the native motion status. A well-formed survey declaration rejected for incomplete inspection remains a declaration (`survey_rejected`); the next turn receives feedback. The raw decisions remain in `decisions.jsonl`. Wall time, action execution time and model request time can overlap when the simulator advances during inference. The provider may omit usage or cost. Older episodes can still contribute scores when detailed logs are unavailable. See the measured coverage before comparing models.

The `combined` section pools decision counts and QA outcomes across the supplied systems. For the post-stop analysis it reports both the fraction of eligible windows and the unweighted mean of eligible episode rates. An external harness can call `turn.record_request(elapsed_seconds, usage)` per provider attempt, including tool rounds. The runner adds the phase and action step. A policy object can receive the same callback via `set_request_recorder(callback)`. The optional usage mapping accepts `prompt_tokens`, `completion_tokens`, `total_tokens`, and `cost` in USD. Without receipts, only the runner's policy-call wall time is available.

## Termination and resource tables

`termination_table` gives one row per final termination reason per system and for the pooled episodes. Its disjoint counts are `zero`, `partial`, `full_without_success`, `success` and `unknown`. A score of 100 with physical failure belongs in `full_without_success`. Intermediate declarations stay in the decision log. Each row includes the episode mean Interactive Score and the median budget fraction among known failures with a recorded budget. Use `episode_endings` to select individual tasks. VLMs count decisions and continuous policies count executed controls.

`resource_episodes` gives the measurements and cost basis for each episode. `resources` gives task-weighted means and coverage. New runs record seconds on a monotonic wall clock.

| Field | Measurement |
| --- | --- |
| `scene_load_s` | Native environment reset call. This is part of setup. |
| `setup_s` | Evaluation launch through environment and evaluator readiness. |
| `episode_s` | Ready environment through physical scoring and applicable terminal QA, including observations and harness work. |
| `evaluation_total_s` | Setup plus episode. Shutdown is outside this interval. |
| `inference_s`, `interaction_inference_s`, `terminal_qa_inference_s` | Policy request durations with their recorded phases. |
| `action_execution_s` | Wall time spent executing actions. The simulator can also advance during inference. |
| `wall_s` | Original runner elapsed time, retained for reading older files. Historical setup and episode splits without measurements are null. |

`cost_reported_usd` sums provider charges, including explicit zeros. A historical tariff can be supplied with `--tariffs FILE`. Key each tariff by the exact system name used with `--system`. For example, `{"trial":{"rates":{"input":2,"output":10}}}` applies USD per million tokens. Optional rate keys include `cached_input`, `write_30m` and `write_12h`. Use `long_input_threshold` and `long_rates` for a separate tier. The report includes the selected tariff.

`tariff_cost_usd` prices every measurable request at the selected tariff and requires complete request coverage. `cost_estimated_usd` sums tariff amounts only for calls lacking provider charges. `cost_usd` uses the provider charge first and requires coverage of every logged attempt. `cost_known_usd` preserves the available subtotal. Reported, estimated and unknown call counts accompany these values. A missing receipt log leaves the charge unknown. Continuous policies have no API charge measurement in this protocol.
