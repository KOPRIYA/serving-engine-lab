"""Check recorded benchmark completeness and recompute metrics without loading a model."""
import argparse
import json
import math
from pathlib import Path

from .benchmark import metrics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(payload):
    """Raise ValueError on incomplete or inconsistent evidence; return comparison counts."""
    rounds, variants, workloads = payload['rounds'], payload['variants'], payload['workloads']
    require(type(rounds) is int and rounds > 0, 'invalid round count')
    require(len(variants) == len(set(variants)) and 'blocking' in variants,
            'invalid variants')
    require(len(variants) > 1 and bool(workloads), 'comparison workloads required')
    expected = {(r, w, v) for r in range(rounds) for w in workloads for v in variants}
    runs = {}
    for run in payload['runs']:
        key = (run['round'], run['workload'], run['label'])
        require(key in expected and key not in runs, f'duplicate or unexpected run: {key}')
        rows = run['requests']
        fixture = workloads[run['workload']]
        specs = {s['id']: s for s in fixture}
        require(bool(specs) and len(specs) == len(fixture), 'invalid workload request IDs')
        ids = [row['id'] for row in rows]
        require(len(ids) == len(set(ids)) and set(ids) == set(specs),
                f'missing or duplicate request: {key}')
        duration = run['metrics']['duration_seconds']
        require(math.isfinite(duration) and duration > 0, 'invalid duration')
        for row in rows:
            spec = specs[row['id']]
            times = row['token_times']
            require(len(times) == len(row['tokens']) == spec['max_tokens'],
                    'incomplete token sequence')
            require(row['finish_reason'] == 'length', 'unexpected finish reason')
            require(row['prompt_tokens'] == len(spec['prompt']), 'prompt length mismatch')
            require(math.isclose(row['arrival'], spec['arrival'], abs_tol=1e-7),
                    'scheduled arrival changed')
            timeline = [row['arrival'], row['admitted'], *times, row['finished'], duration]
            require(all(math.isfinite(t) and t >= 0 for t in timeline), 'invalid timestamp')
            require(all(a <= b for a, b in zip(timeline, timeline[1:])),
                    'timestamps out of order')
        calculated = metrics(rows, duration)
        for name, value in calculated.items():
            recorded = run['metrics'][name]
            if value is None:
                require(recorded is None, f'metric mismatch: {name}')
            else:
                require(isinstance(recorded, (int, float)) and math.isfinite(recorded)
                        and math.isclose(value, recorded, rel_tol=1e-8, abs_tol=1e-7),
                        f'metric mismatch: {name}')
        runs[key] = {row['id']: row['tokens'] for row in rows}
    require(set(runs) == expected, 'missing benchmark runs')
    matches = comparisons = 0
    failures = []
    for (round_id, workload, label), rows in runs.items():
        if label == 'blocking':
            continue
        baseline = runs[(round_id, workload, 'blocking')]
        for request_id, tokens in rows.items():
            comparisons += 1
            equal = tokens == baseline[request_id]
            matches += int(equal)
            if not equal:
                failures.append((round_id, label, request_id))
    recorded = payload['consistency']
    require(recorded['matches'] == matches and recorded['comparisons'] == comparisons,
            'consistency counts mismatch')
    recorded_failures = [(f['round'], f['variant'], f['request']) for f in recorded['failures']]
    require(sorted(recorded_failures) == sorted(failures), 'consistency failures mismatch')
    require(not failures, 'output sequences differ from blocking')
    return {'runs': len(runs), 'matches': matches, 'comparisons': comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(json.loads(args.raw.read_text()))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f'FAIL: {exc}\n')
    print(f"PASS: {result['runs']} runs; {result['matches']}/{result['comparisons']} "
          'output comparisons; recorded metrics match timestamps.')
    print('Internal consistency only; not proof of authenticity or current-source performance.')


if __name__ == '__main__':
    main()
