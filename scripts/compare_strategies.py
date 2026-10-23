"""Benchmark all three strategies using the SAME local Qwen endpoint and EEG file.

Runs each in a separate agent invocation (different output directories), then
writes measured duration/tool count/success. No invented performance results.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--task',required=True)
p.add_argument('--output-root',type=Path,default=Path('benchmarks/local_runs'))
p.add_argument('--base-url',default='http://localhost:8000/v1')
p.add_argument('--model',default='Qwen/Qwen3-8B')
args=p.parse_args()
args.output_root.mkdir(parents=True,exist_ok=True)
rows=[]
for strategy in ['plan-execute','react','rewoo']:
    directory=args.output_root/strategy
    cmd=[sys.executable,'-m','eeg_agent.cli','--strategy',strategy,
         '--input',str(args.input),'--data-root',str(args.data_root),
         '--output-root',str(directory),'--task',args.task,
         '--base-url',args.base_url,'--model',args.model]
    start=time.perf_counter()
    process=subprocess.run(cmd,text=True,capture_output=True,timeout=7200)
    seconds=time.perf_counter()-start
    manifests=sorted(directory.glob('agent_run_*.json'),key=lambda f:f.stat().st_mtime)
    record=json.loads(manifests[-1].read_text('utf-8')) if manifests else {}
    trace=record.get('trace',[])
    rows.append({'strategy':strategy,'passed':record.get('status')=='completed' and process.returncode==0,
                 'elapsed_seconds':round(seconds,2),'tool_calls':len(trace),
                 'tool_failures':sum(t.get('status')=='error' for t in trace),
                 'artifact_manifest':str(manifests[-1]) if manifests else '',
                 'runtime_error':process.stderr[-1000:] if process.returncode else ''})
    print(strategy,rows[-1]['passed'],rows[-1]['elapsed_seconds'],'seconds')
with (args.output_root/'comparison.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Wrote',args.output_root/'comparison.csv')
