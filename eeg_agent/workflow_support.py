"""Deterministic batching for dependency-independent, read-only plan steps."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from .schema import Step
from .execution import execute_step


def ready_batch(state, tools, cursor, *, max_parallel=3, forbidden=()):
    steps=state.get('steps',[])
    if cursor>=len(steps):return []
    context={'allowed_tools':set(tools.registry)}
    first=Step.model_validate(steps[cursor],context=context)
    if not tools.registry[first.tool].parallel_safe or first.tool in forbidden:return [first]
    chosen=[]; ids=set()
    for raw in steps[cursor:cursor+max_parallel]:
        step=Step.model_validate(raw,context=context)
        if (not tools.registry[step.tool].parallel_safe or step.tool in forbidden
            or any(dep in ids for dep in step.depends_on)
            or any(dep not in state.get('evidence',{}) for dep in step.depends_on)):
            break
        # Explicit references are checked; no dependence on a step executing concurrently.
        if any(f'$'+sid in str(step.args) for sid in ids):break
        chosen.append(step);ids.add(step.id)
    return chosen or [first]


def execute_ready(batch, state, tools):
    """Return merge-safe updates; no parallel side effects, ordering is deterministic."""
    if len(batch)==1:
        return execute_step(batch[0],state,tools)
    with ThreadPoolExecutor(max_workers=min(3,len(batch)),thread_name_prefix='eeg-analysis') as pool:
        updates=list(pool.map(lambda step:execute_step(step,state,tools),batch))
    evidence=dict(state.get('evidence',{})); trace=list(state.get('trace',[]));errors=[]
    for step, result in zip(batch,updates):
        evidence.update({step.id:result['evidence'][step.id]}) if step.id in result.get('evidence',{}) else None
        trace.extend(result.get('trace',[])[len(state.get('trace',[])):])
        if result.get('error'):errors.append(f"{step.id}: {result['error']}")
    return {'evidence':evidence,'trace':trace,'error':'; '.join(errors) if errors else None,
            'latest_npz':state.get('latest_npz')}
