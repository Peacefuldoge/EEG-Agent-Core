"""python -m eeg_agent.cli --strategy react --input path.edf --task '...'"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sqlite3
import uuid


def main(argv=None):
    p=argparse.ArgumentParser(description='Local LangGraph/vLLM/Qwen EEG Agent')
    p.add_argument('--strategy',choices=['plan-execute','react','rewoo'],default='plan-execute')
    p.add_argument('--task',required=True,help='Research analysis goal written in natural language')
    p.add_argument('--input',type=Path,required=True,help='Input EEG file')
    p.add_argument('--data-root',type=Path,required=True,help='Allowlisted input directory')
    p.add_argument('--output-root',type=Path,default=Path('./agent_runs'))
    p.add_argument('--model',default=os.getenv('EEG_LLM_MODEL','Qwen/Qwen3-8B'))
    p.add_argument('--base-url',default=os.getenv('EEG_LLM_BASE_URL','http://localhost:8000/v1'))
    p.add_argument('--max-steps',type=int,default=12)
    p.add_argument('--max-replans',type=int,default=1)
    p.add_argument('--thread-id',default=None,help='Optional checkpoint thread identifier')
    p.add_argument('--no-checkpoint',action='store_true')
    args=p.parse_args(argv)
    if args.max_steps<1 or args.max_steps>30 or args.max_replans<0 or args.max_replans>3:
        p.error('--max-steps must be 1..30 and --max-replans must be 0..3')
    # Safety: fail immediately before sending a path to LLM.
    from eeg_agent.tools import EEGTools
    tools=EEGTools(args.data_root,args.output_root)
    resolved=tools._resolve(str(args.input))
    from eeg_agent.model import QwenModel
    from eeg_agent.strategies import AgentGraphs
    model=QwenModel(base_url=args.base_url,model=args.model)
    os.environ.setdefault('LANGGRAPH_STRICT_MSGPACK','true')
    conn=None;checkpointer=None
    if not args.no_checkpoint:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn=sqlite3.connect(str(tools.output_root/'agent_checkpoints.sqlite3'),check_same_thread=False)
        checkpointer=SqliteSaver(conn)
    graph=AgentGraphs(model,tools,max_steps=args.max_steps,max_replans=args.max_replans).build(
        args.strategy,checkpointer)
    thread_id=args.thread_id or uuid.uuid4().hex
    initial={'task':args.task,'input_path':str(resolved),'strategy':args.strategy,
             'cursor':0,'iterations':0,'replans':0,'steps':[],
             'trace':[],'evidence':{},'latest_npz':None,'error':None,'status':'running'}
    try:
        result=graph.invoke(initial,config={'configurable':{'thread_id':thread_id},'recursion_limit':max(70,args.max_steps*7)})
    finally:
        if conn is not None:conn.close()
    record={'strategy':args.strategy,'thread_id':thread_id,'model':args.model,
            'task':args.task,'input_path':str(resolved),'status':result.get('status'),
            'final_answer':result.get('final_answer'),
            'trace':result.get('trace',[]),'error':result.get('error'),
            'artifacts':result.get('evidence',{})}
    manifest=tools.output_root/f'agent_run_{thread_id}.json'
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'status':record['status'],'answer':record['final_answer'],
                      'trace_path':str(manifest),'thread_id':thread_id},ensure_ascii=False,indent=2))
    if record['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
