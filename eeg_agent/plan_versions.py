"""Plan-scoped IDs and immutable evidence across LLM replanning.

The first plan keeps E1.. keys for backwards compatibility. Subsequent plans
use P2:E1, P3:E1, etc. Explicit $P1:E1 resolves the legacy first-plan key.
Unqualified references in a new plan always refer to that new plan, never to
stale previous evidence. Cross-plan reuse must name its source explicitly.
"""
from __future__ import annotations
import re
from copy import deepcopy
from .schema import Plan

_LOCAL = re.compile(r'^\$E[1-9][0-9]*(?:\.[a-z][a-zA-Z0-9_]*)?$')
_EXPLICIT = re.compile(r'^\$P([1-9][0-9]*):E([1-9][0-9]*)(?:\.[a-z][a-zA-Z0-9_]*)?$')


def activate_plan(state: dict, steps: list[dict], *, allowed_tools=None) -> dict:
    """Validate and namespace the *entire* plan before changing graph state.

    A successful plan generation increments the version, not a failed attempt.
    Completed evidence from previous plans is immutable and remains accessible
    by an explicit qualified reference only.
    """
    number=int(state.get('plan_version',0))+1
    active=f'P{number}'
    old_evidence=state.get('evidence',{})
    ordered_ids=[step['id'] for step in steps]
    ids=set(ordered_ids)
    transformed=[]

    def rewrite(obj, *, dependencies):
        if isinstance(obj,str):
            if _LOCAL.fullmatch(obj):
                local=obj[1:].split('.',1)[0]
                if local not in ids:
                    raise ValueError(f'Unknown current-plan reference {obj}')
                dependencies.add(local)
                return '$'+(local if number==1 else f'{active}:{local}')+obj[1+len(local):]
            match=_EXPLICIT.fullmatch(obj)
            if match:
                version,step_num=map(int,match.groups())
                if version>=number:
                    raise ValueError(f'Cross-plan reference must point to an earlier plan: {obj}')
                key=(f'E{step_num}' if version==1 else f'P{version}:E{step_num}')
                if key not in old_evidence:
                    raise ValueError(f'Cross-plan evidence is not available: {obj}')
                # Explicit qualified refs can be kept; bind_value resolves P1 legacy IDs.
                return obj
            return obj
        if isinstance(obj,list):return [rewrite(v,dependencies=dependencies) for v in obj]
        if isinstance(obj,dict):return {k:rewrite(v,dependencies=dependencies) for k,v in obj.items()}
        return obj

    for raw in steps:
        step=deepcopy(raw)
        deps=set(step.get('depends_on') or ())
        args=rewrite(step['args'],dependencies=deps)
        local=step['id']
        step['id']= local if number==1 else f'{active}:{local}'
        step['args']=args
        # Preserve plan order for priority; scheduler resolves actual dependencies.
        step['depends_on']=[(dep if number==1 else f'{active}:{dep}')
                             for dep in ordered_ids if dep in deps]
        transformed.append(step)
    # Validates dependencies and future/self references against the rewritten plan.
    validated=Plan.model_validate({'steps':transformed},
                        context={'allowed_tools':allowed_tools} if allowed_tools is not None else None)
    transformed=[step.model_dump() for step in validated.steps]
    previous=list(state.get('plan_history',[]))
    previous.append({'plan_id':active,'steps':transformed})
    return {'steps':transformed,'cursor':0,'step_status':{},'error':None,'status':'running',
            'plan_version':number,'active_plan_id':active,'plan_history':previous,
            # Never implicitly point a revised plan at a stale previous-plan artifact.
            'latest_npz':None}
