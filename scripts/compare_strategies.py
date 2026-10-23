"""Benchmark all three strategies against ONE user-owned model endpoint.

Uses only the core public EEGSession API. Neither Chainlit nor eeg-agent-ui is
required. Results are measured, not invented.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

from eeg_agent import EEGSession, STRATEGIES


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare Plan-Execute, ReAct and ReWOO')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('benchmarks/local_runs'))
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='qwen3-8b')
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy in STRATEGIES:
        directory = args.output_root / strategy
        start = time.perf_counter()
        report, error = None, ''
        try:
            session = EEGSession(data_root=args.data_root, output_root=directory,
                                 model=args.model, base_url=args.base_url,
                                 strategy=strategy)
            session.set_file(args.input)
            report = session.run_agent(args.task)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        elapsed = time.perf_counter() - start
        trace = report.get('trace', []) if report else []
        row = {
            'strategy': strategy,
            'passed': report is not None and report.get('status') == 'completed',
            'elapsed_seconds': round(elapsed, 2),
            'tool_calls': len(trace),
            'tool_failures': sum(t.get('status') == 'error' for t in trace),
            'artifact_manifest': report.get('manifest', '') if report else '',
            'runtime_error': error or (str(report.get('error') or '') if report else ''),
        }
        rows.append(row)
        print(strategy, row['passed'], row['elapsed_seconds'], 'seconds')
    dest = args.output_root / 'comparison.csv'
    with dest.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote', dest)


if __name__ == '__main__':
    main()
