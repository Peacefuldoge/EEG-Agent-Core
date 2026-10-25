"""LLM-independent validation, strict $reference binding and timed tool execution."""
from __future__ import annotations
import re
import time
from datetime import datetime, timezone
from typing import Any
from .schema import Plan, Step
from .tools import EEGTools

REF=re.compile(r'^\$(input|latest_npz|(?:P[1-9][0-9]*:)?E[1-9][0-9]*)(?:\.([a-z][a-zA-Z0-9_]*))?$')


def bind_value(obj:Any, state:dict) -> Any:
    """Substitute only safe named references; never eval, format or expand paths."""
    if isinstance(obj,str) and obj.startswith('$'):
        m=REF.fullmatch(obj)
        if not m:
            raise ValueError(f'Unsupported reference syntax: {obj!r}')
        key,field=m.groups()
        if key=='input':
            value=state['input_path']
        elif key=='latest_npz':
            value=state.get('latest_npz')
        else:
            evidence=state.get('evidence',{})
            if key.startswith('P1:'):
                key=key.split(':',1)[1]
            if key not in evidence:
                raise ValueError(f'Unresolved dependency: {key}')
            value=evidence[key]
        if field:
            if not isinstance(value,dict) or field not in value:
                raise ValueError(f'Missing evidence key {key}.{field}')
            value=value[field]
        if value is None:
            raise ValueError(f'Reference {obj} has no value')
        return value
    if isinstance(obj,list):return [bind_value(v,state) for v in obj]
    if isinstance(obj,dict):return {k:bind_value(v,state) for k,v in obj.items()}
    return obj


def execute_step(step:Step, state:dict, tools:EEGTools) -> dict:
    evidence=dict(state.get('evidence',{}));trace=list(state.get('trace',[]))
    started=datetime.now(timezone.utc).isoformat();clock=time.perf_counter()
    attempt_id=f'{step.id}:attempt:{1+sum(t.get("step")==step.id for t in trace)}'
    entry={'step':step.id,'step_id':step.id,'attempt_id':attempt_id,'tool':step.tool,
           'started_at':started}
    try:
        for dep in step.depends_on:
            if dep not in evidence:raise ValueError(f'{step.id} dependency {dep} not complete')
        args=bind_value(step.args,state)
        entry['arguments']=args
        outcome=tools.invoke(step.tool,args)
        if not isinstance(outcome,dict):raise TypeError('Tool must return a result dictionary')
        if outcome.get('status') in ('failed','error','partial_failure'):
            raise RuntimeError(f'Tool returned {outcome["status"]}: '+
                               str(outcome.get('error') or f'{outcome.get("failed",0)} failed files'))
        evidence[step.id]=outcome
        latest=outcome.get('output_npz',state.get('latest_npz'))
        entry.update(status='ok',result=outcome)
        return {'evidence':evidence,'trace':trace+[finish_trace(entry,clock)],
                'latest_npz':latest,'error':None}
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        entry.update(status='error',error=error)
        return {'evidence':evidence,'trace':trace+[finish_trace(entry,clock)],'error':error}


def finish_trace(entry:dict, clock:float) -> dict:
    entry['finished_at']=datetime.now(timezone.utc).isoformat()
    entry['duration_ms']=round((time.perf_counter()-clock)*1000,3)
    return entry


def parse_plan(obj:dict, allowed_tools=None) -> list[dict]:
    """Validate a finite acyclic tool DAG and unique IDs before touching data."""
    ctx={'allowed_tools':allowed_tools} if allowed_tools is not None else None
    return [step.model_dump() for step in Plan.model_validate(obj,context=ctx).steps]


def validate_tool_args(step:Step,tools:EEGTools,state:dict) -> None:
    tools.registry[step.tool].args.model_validate(bind_value(step.args,state))
