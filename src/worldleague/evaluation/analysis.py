"""Summarize saved episode evidence with worldleague analyze --system NAME=DIRECTORY."""

import json
from collections import Counter, defaultdict
from functools import cache, lru_cache
from statistics import mean, median

from .costs import cost_totals
from worldleague.score import profile
from worldleague.tasks.loader import catalogue


def average(values):
    measured = [x for x in values if x is not None]
    return mean(measured) if measured else None


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def records(path):
    if path.is_file():
        with path.open() as lines:
            for line in lines:
                yield json.loads(line)


def physical_band(score, success):
    if success is True:
        return "success"
    if score is None or success is None:
        return "unknown"
    return "progress_without_success" if score > 0 else "zero"


def termination_table(episodes):
    """Count episodes by final reason, retaining unmet constraints at full credit."""
    groups = defaultdict(list)
    for row in episodes:
        groups[row["reason"]].append(row)
    table = {}
    for reason, rows in groups.items():
        counts = dict.fromkeys(("zero", "partial", "full_without_success", "success", "unknown"), 0)
        fractions = []
        for row in rows:
            score, ok = row["score"], row["success"]
            band = physical_band(score, ok)
            if band == "progress_without_success":
                band = "full_without_success" if score == 100 else "partial"
            counts[band] += 1
            if ok is False and row["budget_fraction"] is not None:
                fractions.append(row["budget_fraction"])
        table[reason] = {
            **counts, "episodes": len(rows),
            "mean_interactive_score": average(row["score"] for row in rows),
            "measured_scores": sum(row["score"] is not None for row in rows),
            "unsuccessful_budget_median": median(fractions) if fractions else None,
            "unsuccessful_measured_budgets": len(fractions),
        }
    return table


def stop_windows(events):
    counts = Counter(row["outcome"] for row in events)
    next_choices = Counter()
    stops = Counter()
    windows = {n: Counter() for n in (1, 3, 5)}
    for i, row in enumerate(events):
        if row["outcome"] != "stopped":
            continue
        stops[row["status"]] += 1
        command = row["command"]
        for n, tally in windows.items():
            after = events[i + 1 : i + 1 + n]
            if len(after) != n:
                continue
            first = after[0]
            if n == 1:
                choice = "invalid_or_declaration"
                if first["outcome"] in ("completed", "stopped"):
                    if first["command"] == command:
                        choice = "same_command"
                    elif first["command"]["action"] == command["action"]:
                        choice = "changed_parameters"
                    else:
                        choice = "changed_action"
                next_choices[choice] += 1
            completed = any(a["outcome"] == "completed" for a in after)
            repeated_stop = any(
                a["outcome"] == "stopped"
                and a["command"].get("action") == command.get("action")
                and a["status"] == row["status"]
                for a in after
            )
            recurred = any(
                a["outcome"] == "stopped" and a["status"] == row["status"]
                for a in after
            )
            tally.update(
                windows=1,
                completed=completed,
                recurred=recurred,
                same_action_stop=repeated_stop,
                both=completed and recurred,
            )
    return counts, stops, next_choices, windows


def resources(path, result, tariff=None):
    action_policy = "control_steps" in result or (path / "plan.json").is_file()
    continuous = (
        read(path / "summary.json")
        if action_policy
        else {}
    )
    timing = continuous.get("timings") or result.get("timings", {})
    values = {
        "wall_s": result.get(
            "wall_s",
            continuous.get("timings", {}).get("wall_s", result.get("runtime_s")),
        ),
        "scene_load_s": timing.get("scene_load_s"),
        "setup_s": timing.get("setup_s"),
        "episode_s": timing.get("episode_s"),
        "evaluation_total_s": timing.get("evaluation_total_s"),
        "action_execution_s": timing.get("action_execution_s"),
        "policy_decision_s": timing.get("decision_s"),
        "policy_qa_s": timing.get("qa_s"),
    }
    phases = defaultdict(list)
    for row in records(path / "requests.jsonl"):
        if "phase" in row:
            phases[row["phase"]].append(row)
        elif "model_call" in row:
            phases["interaction"].append(row)
    for phase, rows in phases.items():
        values[phase + "_calls"] = len(rows)
        values[phase + "_inference_s"] = sum(row["inference_s"] for row in rows)
    if phases:
        values["inference_s"] = sum(
            v for k, v in values.items() if k.endswith("_inference_s")
        )
        rows = [row for group in phases.values() for row in group]
        if not action_policy:
            values.update(cost_totals(rows, tariff))
            for key, label in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                receipts = [r.get("usage") or {} for r in rows]
                known = [usage.get(key, usage.get(label)) for usage in receipts]
                values[label + "_known_calls"] = sum(x is not None for x in known)
                values[label] = (
                    sum(known)
                    if all(isinstance(x, (int, float)) for x in known)
                    else None
                )
    elif continuous:
        values["inference_s"] = timing.get("inference_s")
    return values


def metrics(runs):
    return {
        k: average(row.get(k) for row in runs)
        for k in ("interactive_score", "physical_success", "terminal_qa")
    }


def summarize(episodes, catalogue_rows, tariff=None):
    groups = defaultdict(list)
    by_task = defaultdict(list)
    components = defaultdict(lambda: defaultdict(Counter))
    qa_groups = defaultdict(lambda: defaultdict(list))
    termination = defaultdict(Counter)
    qa_full = Counter()
    decisions, stops, choices = Counter(), Counter(), Counter()
    pooled = defaultdict(Counter)
    episode_rates = defaultdict(lambda: defaultdict(list))
    budgets = []
    endings = []
    resource_episodes = []
    measured_events = 0
    protocols = Counter()
    for path, row in episodes:
        task = row["task"]
        by_task[task].append(row)
        protocols[row.get("instruction_protocol") or "unrecorded"] += 1
        op, ok, qa = (
            row.get("interactive_score"),
            row.get("physical_success"),
            row.get("terminal_qa"),
        )
        band = physical_band(op, ok)
        termination[row.get("termination") or "unrecorded"][band] += 1
        if qa == 100:
            qa_full["episodes"] += 1
            qa_full[
                "unknown"
                if op is None
                else "zero"
                if op == 0
                else "full"
                if op == 100
                else "partial"
            ] += 1
        used = row.get("decisions", row.get("control_steps"))
        limit = row.get("decision_budget", row.get("control_budget"))
        endings.append({"task": task, "seed": row.get("seed"), "reason": row.get("termination") or "unrecorded",
                        "success": ok, "score": op,
                        "budget_fraction": used / limit if limit and used is not None else None})
        if limit and used is not None:
            budgets.append(
                {
                    "task": task,
                    "seed": row.get("seed"),
                    "used": used,
                    "budget": limit,
                    "fraction": used / limit,
                    "termination": row.get("termination"),
                }
            )
        for item in read(path / "physical_score.json").get("completion_items") or []:
            tally = components[task][item["id"]]
            if item.get("evaluation_validity", "valid") in ("valid", "complete"):
                tally["measured"] += 1
                tally["completed"] += item["completed"] is True
                tally["credit_sum"] += item.get(
                    "credit", int(item["completed"] is True)
                )
            else:
                tally["unknown"] += 1
        for group, qs in question_groups(path, task).items():
            qa_groups[group][task].append((average(qs), op))
        if (path / "decision_events.jsonl").is_file():
            measured_events += 1
            c, reasons, follow, windows = stop_windows(
                list(records(path / "decision_events.jsonl"))
            )
            decisions.update(c)
            stops.update(reasons)
            choices.update(follow)
            for n, counters in windows.items():
                pooled[n].update(counters)
                if counters["windows"]:
                    for key in ("completed", "recurred", "both", "same_action_stop"):
                        episode_rates[n][key].append(
                            counters[key] / counters["windows"]
                        )
        usage = resources(path, row, tariff)
        groups[task].append(usage)
        if usage.get("cost_usd") is not None:
            basis = "mixed" if usage["cost_estimated_calls"] and usage["cost_reported_calls"] else (
                "estimated" if usage["cost_estimated_calls"] else "reported")
        elif usage.get("cost_usd_known_calls"):
            basis = "partial"
        else:
            basis = "not_applicable" if "control_steps" in row or (path / "plan.json").is_file() else "unknown"
        resource_episodes.append({"task": task, "seed": row.get("seed"), "cost_basis": basis, **usage})

    def resource_table(tasks):
        keys = sorted({k for t in tasks for run in groups[t] for k in run})
        table = {}
        for k in keys:
            task_means = [average(run.get(k) for run in groups[t]) for t in tasks]
            table[k] = {
                "mean_per_task": average(task_means),
                "tasks": sum(v is not None for v in task_means),
            }
            if k.endswith(("_tokens", "_usd", "_calls")):
                measured = [
                    run[k] for t in tasks for run in groups[t] if run.get(k) is not None
                ]
                table[k].update(
                    measured_episode_sum=sum(measured) if measured else None,
                    measured_episodes=len(measured),
                )
        return table

    all_tasks = sorted(by_task)
    resource_groups = {"overall": resource_table(all_tasks)}
    for dimension in sorted({catalogue_rows[t]["dimension"] for t in all_tasks}):
        resource_groups[dimension] = resource_table(
            [t for t in all_tasks if catalogue_rows[t]["dimension"] == dimension]
        )
    paired_time = sum(
        r["inference_s"]
        for runs in groups.values()
        for r in runs
        if "terminal_qa_inference_s" in r and "interaction_inference_s" in r
    )
    qa_time = sum(
        r["terminal_qa_inference_s"]
        for runs in groups.values()
        for r in runs
        if "interaction_inference_s" in r and "terminal_qa_inference_s" in r
    )
    return {
        "episodes": len(episodes),
        "protocols": protocols,
        "profile": profile([r for _, r in episodes]),
        "tasks": {t: metrics(runs) for t, runs in by_task.items()},
        "goal_items": components,
        "qa_groups": {
            g: {
                "tasks": len(tasks),
                "qa_percent": average(
                    average(a for a, _ in pairs) for pairs in tasks.values()
                ),
                "interactive_score": average(
                    average(o for _, o in pairs) for pairs in tasks.values()
                ),
            }
            for g, tasks in qa_groups.items()
        },
        "full_qa_physical_scores": qa_full,
        "termination": termination,
        "termination_table": termination_table(endings),
        "episode_endings": endings,
        "budget_use": budgets,
        "decisions": {
            "episodes_with_events": measured_events,
            "outcomes": decisions,
            "stop_reasons": stops,
            "next_choice": choices,
            "followup": {
                n: {
                    "counts": c,
                    "pooled_rates": {
                        k: c[k] / c["windows"] if c["windows"] else None
                        for k in ("completed", "recurred", "both", "same_action_stop")
                    },
                    "episode_mean_rates": {
                        k: average(v) for k, v in episode_rates[n].items()
                    },
                    "episodes": len(episode_rates[n]["completed"]),
                }
                for n, c in pooled.items()
            },
        },
        "resources": resource_groups,
        "resource_tariff": tariff,
        "resource_episodes": resource_episodes,
        "terminal_qa_share_of_request_time": qa_time / paired_time
        if paired_time
        else None,
    }


def question_groups(path, task):
    items = read(path / "terminal_qa.json").get("items", [])
    groups = defaultdict(list)
    for item in items:
        when = item.get("time") or question_times(task).get(item["id"])
        group = {"operation_history": "history", "phase_1_to_phase_2": "history"}.get(
            when, when
        )
        if group:
            groups[group].append(
                None if item.get("correct") is None else 100 * item["correct"]
            )
    return {
        group: answers
        for group, answers in groups.items()
        if all(value is not None for value in answers)
    }


@cache
def question_times(task):
    from worldleague.tasks.loader import load_task

    spec = load_task(task).definition.private.evaluation.parameters["terminal_qa"]
    return {q["id"]: q["time"] for q in spec["questions"]}


def combined_outcomes(systems):
    """Pool counts across systems and weight each eligible episode once for the episode rate."""
    full_qa, outcomes, reasons, choices = Counter(), Counter(), Counter(), Counter()
    termination = defaultdict(Counter)
    windows = defaultdict(Counter)
    episode_sums = defaultdict(Counter)
    episodes = Counter()
    for system in systems.values():
        full_qa.update(system["full_qa_physical_scores"])
        for reason, values in system["termination"].items():
            termination[reason].update(values)
        decisions = system["decisions"]
        outcomes.update(decisions["outcomes"])
        reasons.update(decisions["stop_reasons"])
        choices.update(decisions["next_choice"])
        for n, data in decisions["followup"].items():
            windows[n].update(data["counts"])
            episodes[n] += data["episodes"]
            for key, rate in data["episode_mean_rates"].items():
                if rate is not None:
                    episode_sums[n][key] += rate * data["episodes"]
    followup = {}
    for n, counts in windows.items():
        followup[n] = {
            "counts": counts,
            "episodes": episodes[n],
            "pooled_rates": {
                key: counts[key] / counts["windows"]
                for key in counts
                if key != "windows" and counts["windows"]
            },
            "episode_mean_rates": {
                key: value / episodes[n]
                for key, value in episode_sums[n].items()
                if episodes[n]
            },
        }
    return {
        "full_qa_physical_scores": full_qa,
        "termination": termination,
        "termination_table": termination_table(row for system in systems.values() for row in system["episode_endings"]),
        "decisions": outcomes,
        "stop_reasons": reasons,
        "next_choice": choices,
        "followup": followup,
    }


def analyze(named_dirs, seed=0, compare_seed=None, tariffs=None):
    rows = {r["alias"]: r for r in catalogue()}
    systems, repeats, selected = {}, {}, {}
    for name, directory in named_dirs.items():
        all_runs = [(f.parent, read(f)) for f in sorted(directory.rglob("result.json"))]
        selected[name] = [(p, r) for p, r in all_runs if r.get("seed") == seed]
        tariff = (tariffs or {}).get(name)
        systems[name] = summarize(selected[name], rows, tariff)
        if compare_seed is not None:
            other = summarize(
                [(p, r) for p, r in all_runs if r.get("seed") == compare_seed], rows, tariff
            )
            repeats[name] = paired_tasks(systems[name]["tasks"], other["tasks"])
    shared = sorted(set(rows) - {"T52"})
    counts = defaultdict(Counter)
    coverage = {}
    for task in shared:
        observed = [s["tasks"].get(task, {}) for s in systems.values()]
        if any((r.get("physical_success") or 0) > 0 for r in observed):
            band = "success"
        elif any(
            r.get("interactive_score") is None or r.get("physical_success") is None
            for r in observed
        ):
            band = "unknown"
        else:
            band = (
                "progress_without_success"
                if any(r["interactive_score"] > 0 for r in observed)
                else "zero"
            )
        coverage[task] = band
        for label in ("overall", rows[task]["track"], rows[task]["dimension"]):
            counts[label][band] += 1
    for name, runs in selected.items():
        systems[name]["common59_profile"] = profile(
            [r for _, r in runs if r["task"] in shared]
        )
    return {
        "seed": seed,
        "systems": systems,
        "combined": combined_outcomes(systems),
        "common59": {"coverage": counts, "tasks": coverage},
        "compare_seed": compare_seed,
        "paired_seed_tasks": repeats,
    }


def paired_tasks(left, right):
    result = {}
    for task in sorted(left.keys() & right.keys()):
        a, b = left[task], right[task]
        result[task] = {
            "first": a,
            "second": b,
            "delta": {
                key: b[key] - a[key]
                if a[key] is not None and b[key] is not None
                else None
                for key in a
            },
        }
    return result
