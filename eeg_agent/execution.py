"""LLM-independent step validation, $reference binding and bounded tool execution."""
from __future__ import annotations
import re
from typing import Any
from .schema import Plan, Step
from .tools import EEGTools

REF=re.compile(r'^\$(input|latest_npz|E[1-9][0-9]*)(?:\.([a-z][a-zA-Z0-9_]*))?$')

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
    if isinstance(obj,list):
        return [bind_value(v,state) for v in obj]
    if isinstance(obj,dict):
        return {k:bind_value(v,state) for k,v in obj.items()}
    return obj

def execute_step(step:Step, state:dict, tools:EEGTools) -> dict:
    evidence=dict(state.get('evidence',{}))
    trace=list(state.get('trace',[]))
    for dep in step.depends_on:
        if dep not in evidence:
            raise ValueError(f'{step.id} dependency {dep} not complete')
    args=bind_value(step.args,state)
    try:
        outcome=tools.invoke(step.tool,args)
        evidence[step.id]=outcome
        if 'output_npz' in outcome:
            latest=outcome['output_npz']
        else:
            latest=state.get('latest_npz')
        trace.append({'step':step.id,'tool':step.tool,'status':'ok',
                      'arguments':args,'result':outcome})
        return {'evidence':evidence,'trace':trace,'latest_npz':latest,'error':None}
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        trace.append({'step':step.id,'tool':step.tool,'status':'error',
                      'arguments':args,'error':error})
        return {'evidence':evidence,'trace':trace,'error':error}

def parse_plan(obj:dict, allowed_tools=None) -> list[dict]:
    """Validate DAG ordering, type/schema, and unique IDs before touching any data."""
    return [step.model_dump() for step in Plan.model_validate(obj,context={"allowed_tools":allowed_tools} if allowed_tools is not None else None).steps]

def validate_tool_args(step:Step,tools:EEGTools,state:dict) -> None:
    """Validate arguments before executing, allowing model references to resolve."""
    args=bind_value(step.args,state)
    tools.registry[step.tool].args.model_validate(args)
