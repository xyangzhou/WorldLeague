"""Pure delayed historical collection queries.

Used by worldleague scene --help.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import random

QUERY_RULE = 'earliest_per_category'


class IncompleteSurvey(ValueError):
    """The observed history does not yet permit the public delivery request."""


def category_capacities(parameters):
    minimum = parameters.get('query_min_category_instances', 2)
    reserve = int(parameters.get('require_history_subset', True))
    counts = Counter(parameters['target_categories'].values())
    return {name: n - reserve for name, n in counts.items() if n >= minimum and n > reserve}


def seeded_collection_query(frozen_history, parameters):
    """Return public requests and private selected keys from actual frozen rows.

    Rows contain ``key``, ``snapshot_id`` and ``action_step`` in observation
    order. First encounters sharing one snapshot have no internal chronology.
    Seeded counts depend only on inventory, never on encounter order; selection
    then takes each requested category's first N actual unique encounters.
    """
    capacities = category_capacities(parameters)
    rows = deepcopy(list(frozen_history))
    keys = [row['key'] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(parameters['target_keys']):
        raise ValueError('Frozen history must contain every declared instance exactly once.')
    if any('snapshot_id' not in row or 'action_step' not in row for row in rows):
        raise ValueError('Frozen encounters require actual observation boundaries.')
    if any(left['action_step'] > right['action_step'] for left, right in zip(rows, rows[1:])):
        raise ValueError('Frozen history must preserve observed chronology.')
    rng = random.Random(parameters['followup_seed'])
    remaining = rng.randint(parameters['query_min_total'], parameters['query_max_total'])
    categories = sorted(capacities)
    rng.shuffle(categories)
    requests = []
    for index, category in enumerate(categories):
        rest_capacity = sum(capacities[key] for key in categories[index + 1:])
        count = rng.randint(max(0, remaining - rest_capacity), min(capacities[category], remaining))
        if count:
            requests.append({'category': category, 'count': count})
        remaining -= count
    requests.sort(key=lambda request: request['category'])
    selected = []
    for request in requests:
        category = request['category']
        category_rows = [row for row in rows if parameters['target_categories'][row['key']] == category]
        boundaries = defaultdict(list)
        for row in category_rows:
            boundaries[row['snapshot_id']].append(row['key'])
        if any(len(group) > 1 for group in boundaries.values()):
            raise IncompleteSurvey('Ambiguous simultaneous first encounters in queried category: ' + category)
        selected.extend(row['key'] for row in category_rows[:request['count']])
    return {'rule': QUERY_RULE, 'requests': requests, 'selected_keys': selected,
            'require_history_subset': parameters.get('require_history_subset', True),
            'total_count': len(selected)}


def public_collection_query(query, destination_label):
    """Expose historical instructions without instance answers or native IDs."""
    requests = deepcopy(query['requests'])
    instructions = '; '.join(f"the first {row['count']} {row['category']} you inspected" for row in requests)
    return {'rule': QUERY_RULE, 'requests': requests, 'total_count': query['total_count'],
        'text': ('Survey complete. Retrieve ' + instructions + '. Use your actual first-inspection history '
                 'within each requested category. Place every requested item on ' + destination_label + '.')}
