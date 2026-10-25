"""Dependency-aware, deterministic DAG scheduling of eligible scientific steps.

LangGraph owns graph checkpoints; this scheduler only chooses tool DAG nodes.
Step IDs are plan-scoped. A completed dependency means verified evidence exists,
not merely that its index is smaller than a cursor.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from .schema import Step
from .execution import execute_step


def ready_batch(state, tools, cursor=0, *, max_parallel=3, forbidden=()):
    """Find globally ready pending nodes, even when interleaved with blocked nodes.

    The original cursor argument is accepted for older callers but is not an
    index into the plan. In-flight nodes are never reselected after checkpoints.
    Non-parallel/approval-sensitive tools always run alone.
    """
    del cursor
    registry=set(tools.registry)
    completed=state.get('evidence',{})
    statuses=state.get('step_status',{})
    pending=[]
    for raw in state.get('steps',[]):
        step=Step.model_validate(raw,context={'allowed_tools':registry})
        if statuses.get(step.id) in ('completed','running','failed','rejected') or step.id in completed:
            continue
        if all(dep in completed and statuses.get(dep) not in ('failed','rejected') for dep in step.depends_on):
            pending.append(step)
    if not pending:return []
    batch=[]
    for step in pending:
        spec=tools.registry[step.tool]
        isolated=not spec.parallel_safe or step.tool in forbidden
        # Global mutable $latest_npz is order-dependent and never batch-safe.
        if '$latest_npz' in str(step.args):isolated=True
        if isolated:
            if not batch:return [step]
            continue
        if len(batch)<max_parallel:batch.append(step)
        if len(batch)>=max_parallel:break
    return batch


def execute_ready(batch, state, tools, *, max_parallel=3):
    """Merge deterministic per-step outcomes, including failures and timestamps."""
    if not batch:raise ValueError('No runnable steps')
    if len(batch)==1:
        result=execute_step(batch[0],state,tools)
        return {**result,'step_status':{
            **state.get('step_status',{}),batch[0].id:'failed' if result.get('error') else 'completed'}}
    with ThreadPoolExecutor(max_workers=min(max_parallel,len(batch)),thread_name_prefix='eeg-analysis') as pool:
        updates=list(pool.map(lambda step:execute_step(step,state,tools),batch))
    evidence=dict(state.get('evidence',{})); trace=list(state.get('trace',[]))
    statuses=dict(state.get('step_status',{}));errors=[];latest=state.get('latest_npz')
    before=len(trace)
    for step,result in zip(batch,updates):
        unexpected_writer=(result.get('latest_npz') and
                           result['latest_npz']!=state.get('latest_npz'))
        failure=(f'{step.id}: parallel-safe tool attempted to update latest_npz'
                 if unexpected_writer else result.get('error'))
        entries=result.get('trace',[])[before:]
        if unexpected_writer:
            for entry in entries:
                if entry.get('step')==step.id:
                    entry={**entry,'status':'error','error':failure}
                trace.append(entry)
        else:
            trace.extend(entries)
            if step.id in result.get('evidence',{}):
                evidence[step.id]=result['evidence'][step.id]
        statuses[step.id]='failed' if failure else 'completed'
        if failure:errors.append(f'{step.id}: {failure}')
    return {'evidence':evidence,'trace':trace,'error':'; '.join(errors) if errors else None,
            'latest_npz':latest,'step_status':statuses}
