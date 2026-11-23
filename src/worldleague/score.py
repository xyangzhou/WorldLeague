"""Summarize WorldLeague episode results. Run: worldleague profile RESULT_DIR."""

from collections import defaultdict
from statistics import mean

from .tasks.loader import catalogue


def profile(episodes):
    """Average episodes per task, then average the evaluated tasks within each group."""
    rows = {row["alias"]: row for row in catalogue()}
    tasks = defaultdict(list)
    for episode in episodes:
        tasks[episode["task"]].append(episode)
    groups = defaultdict(list)
    for alias, runs in tasks.items():
        row = rows[alias]
        scores = {}
        for key in ("interactive_score", "physical_success", "terminal_qa"):
            values = [run[key] for run in runs if run.get(key) is not None]
            scores[key] = mean(values) if values else None
        for label in (row["dimension"], row["track"], "overall"):
            groups[label].append(scores)
    summary = {}
    for label, scores in groups.items():
        group = {"tasks": len(scores)}
        for metric in ("interactive_score", "physical_success", "terminal_qa"):
            values = [row[metric] for row in scores if row[metric] is not None]
            group[metric] = mean(values) if values else None
            group[metric + "_tasks"] = len(values)
        summary[label] = group
    return summary
