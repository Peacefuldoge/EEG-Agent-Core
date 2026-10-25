"""Three LangGraph graphs sharing the same tool registry and local vLLM Qwen client."""
from __future__ import annotations
from typing import Any,Annotated,TypedDict
from datetime import datetime, timezone
import time
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,ToolMessage
from .schema import Step
from .execution import execute_step,bind_value
from .model import SYSTEM
from .workflow_support import ready_batch,execute_ready
from .plan_versions import activate_plan
from .qc_policy import evaluate_tool
from .execution import finish_trace

class WorkflowState(TypedDict,total=False):
    task:str
    input_path:str
    strategy:str
    steps:list[dict]
    cursor:int
    iterations:int
    replans:int
    evidence:dict
    trace:list[dict]
    error:str|None
    latest_npz:str|None
    status:str
    final_answer:str
    messages:Annotated[list,add_messages]
    tool_cursor:int
    qc_summary:dict
    plan_version:int
    active_plan_id:str
    plan_history:list[dict]
    step_status:dict[str,str]
    tool_failures:int

class AgentGraphs:
    def __init__(self, model, tools, *, max_steps:int=12, max_replans:int=1, approval_tools=None):
        self.model=model
        self.tools=tools
        self.max_steps=max_steps
        self.max_replans=max_replans
        self.approval_tools=frozenset(approval_tools or ())

    def _approval(self, name, arguments, step_id=None):
        """HITL: optional parameter modification always triggers fresh approval.

        Interrupt calls are indexed deterministically on node replay; no tools
        are called until the FINAL approval has been obtained.
        """
        if name not in self.approval_tools:
            return arguments
        from langgraph.types import interrupt
        current=dict(arguments)
        for edit_round in range(3):
            response=interrupt({
                'type':'approval_required','tool':name,'step_id':step_id,
                'arguments':dict(current),'edit_round':edit_round,
                'message':'Review scientifically significant EEG parameters before execution.'
            })
            if not isinstance(response,dict):return None
            if 'arguments' in response:
                changes=response['arguments']
                if not isinstance(changes,dict):raise ValueError('Approval edits must be a JSON object')
                if 'source' in changes and changes['source']!=current.get('source'):
                    raise ValueError('Approval edits cannot silently change input source')
                current.update(changes)
                # No execution yet: strict schema and scientific constraints are
                # checked BEFORE a second explicit human confirmation is requested.
                current=self.tools.registry[name].args.model_validate(current).model_dump()
                if name in ('preprocess_eeg','batch_preprocess'):
                    from .science import review_preprocessing
                    from .schema import PreprocessArgs
                    if name=='batch_preprocess':
                        nested=PreprocessArgs.model_validate({'source':'placeholder.edf',
                                                            **current.get('preprocessing',{})})
                        report=review_preprocessing(nested.model_dump())
                    else:
                        report=review_preprocessing(current)
                    if report.errors:raise ValueError('; '.join(report.errors))
                continue
            return current if response.get('approved') is True else None
        raise ValueError('Too many approval edits; submit a new task')

    def _guard(self, name, args, state, step_id):
        """No LLM can bypass QC/science policy; human approval is explicit."""
        decision=evaluate_tool(name,args,state.get('qc_summary'))
        if decision.action=='reject':
            raise ValueError('Scientific policy rejected operation: '+decision.reason)
        if decision.action=='review' and name not in self.approval_tools:
            raise ValueError('QC approval required: '+decision.reason)
        return None  # caller invokes _approval outside error handling

    def _run_plan_steps(self,s):
        steps=s.get('steps',[])
        completed=sum(raw['id'] in s.get('evidence',{}) for raw in steps)
        if completed==len(steps):
            return {'cursor':completed,'step_status':s.get('step_status',{}),'error':None}
        batch=ready_batch(s,self.tools,s.get('cursor',0),max_parallel=3,forbidden=self.approval_tools)
        if not batch:
            return {'error':'DAG has unfinished steps but no eligible dependency-ready nodes',
                    'status':'failed'}
        if s.get('iterations',0)+len(batch)>self.max_steps:
            return {'error':'Step budget exhausted','status':'failed'}
        # All gated tools run singly. A gate can never approve several nodes
        # together, even when the registry labels them parallel safe.
        for step in batch:
            if evaluate_tool(step.tool,step.args,s.get('qc_summary')).action!='allow':
                batch=[step];break
        if len(batch)==1:
            step=batch[0]
            try:
                args=bind_value(step.args,s)
                self._guard(step.tool,args,s,step.id)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                entry={'step':step.id,'step_id':step.id,
                       'attempt_id':f'{step.id}:attempt:{1+sum(t.get("step")==step.id for t in s.get("trace",[]))}',
                       'tool':step.tool,'status':'error',
                       'error':error,'started_at':datetime.now(timezone.utc).isoformat(),
                       'finished_at':datetime.now(timezone.utc).isoformat(),'duration_ms':0.0}
                self._progress(step.tool,'failed',step_id=step.id,error=error)
                return {'status':'failed','error':error,
                        'step_status':{**s.get('step_status',{}),step.id:'failed'},
                        'trace':list(s.get('trace',[]))+[entry],
                        'tool_failures':s.get('tool_failures',0)+1}
            authorized=self._approval(step.tool,args,step.id)
            if authorized is True:authorized=args
            if authorized is False:authorized=None
            if authorized is None:
                error='User denied '+step.tool
                self._progress(step.tool,'rejected',step_id=step.id,error=error)
                return {'status':'rejected','error':error,
                        'step_status':{**s.get('step_status',{}),step.id:'rejected'},
                        'trace':list(s.get('trace',[]))+[{'step':step.id,'step_id':step.id,
                        'tool':step.tool,'status':'rejected','error':error,
                        'attempt_id':f'{step.id}:attempt:{1+sum(t.get("step")==step.id for t in s.get("trace",[]))}'}]}
            batch=[step.model_copy(update={'args':authorized})]
        before=len(s.get('trace',[]))
        for step in batch:self._progress(step.tool,'started',step_id=step.id)
        outcome=execute_ready(batch,s,self.tools)
        for item in outcome.get('trace',[])[before:]:
            self._progress(item['tool'],'completed' if item['status']=='ok' else 'failed',
                           step_id=item.get('step_id'),attempt_id=item.get('attempt_id'),
                           duration_ms=item.get('duration_ms'),error=item.get('error'))
        outcome.update(cursor=sum(raw['id'] in outcome.get('evidence',{}) for raw in steps),
                       iterations=s.get('iterations',0)+len(batch),
                       tool_failures=s.get('tool_failures',0)+sum(
                           item['status']=='error' for item in outcome.get('trace',[])[before:]))
        for step in batch:
            qc=outcome.get('evidence',{}).get(step.id,{})
            if step.tool=='assess_eeg_quality' and qc.get('status')=='requires_review':
                outcome['qc_summary']=qc
                outcome['error']='QC requires review: propose a revised safe plan and request approval.'
        return outcome

    def _qc_preflight(self,s):
        try:
            qc=self.tools.invoke('assess_eeg_quality',{'source':s['input_path'],'max_windows':16,'max_raw_seconds':20})
            return {'qc_summary':qc}
        except Exception as exc:
            return {'qc_summary':{'status':'unavailable','error':f'{type(exc).__name__}: {exc}'}}

    @staticmethod
    def _progress(tool, phase, **details):
        try:
            from langgraph.config import get_stream_writer
            get_stream_writer()({'kind':'tool_progress','tool':tool,'phase':phase,
                                 **{k:v for k,v in details.items() if v is not None}})
        except (ImportError,RuntimeError,LookupError):
            pass

    @staticmethod
    def _is_complete(s):
        return bool(s.get('steps')) and all(
            step['id'] in s.get('evidence',{}) for step in s.get('steps',[]))

    def _summary(self,state):
        try:
            answer=self.model.summarize(state)
        except Exception as exc:
            answer=f"Execution status: {state.get('status')}. Summary unavailable: {exc}. See tool trace and artifacts."
        return {'final_answer':answer}

    def plan_execute(self,checkpointer=None):
        g=StateGraph(WorkflowState)
        def plan(s):
            try:
                steps=self.model.plan(s,self.tools,rewoo=False,replanning=s.get('replans',0)>0)
                return activate_plan(s,steps,allowed_tools=set(self.tools.registry))
            except Exception as exc:
                return {'error':f'Planning failed: {exc}','status':'failed','steps':[]}
        def execute(s):
            if s.get('iterations',0)>=self.max_steps:
                return {'error':'Step budget exhausted','status':'failed'}
            return self._run_plan_steps(s)
        def route_plan(s):
            if s.get('status')=='rejected':return 'finish'
            if s.get('error'):
                return 'summary' if s.get('replans',0)>=self.max_replans or s.get('iterations',0)>=self.max_steps else 'replan'
            return 'finish' if self._is_complete(s) else ('execute' if s.get('iterations',0)<self.max_steps else 'finish')
        def replan(s):
            return {'replans':s.get('replans',0)+1}
        def finish(s):
            unfinished=not self._is_complete(s)
            error=s.get('error') or ('Step budget exhausted before plan completion' if unfinished else None)
            status=('rejected' if s.get('status')=='rejected' else 'failed' if error else
                    'partial_success' if s.get('tool_failures',0) else 'completed')
            return {'status':status,'error':error}
        g.add_node('qc',self._qc_preflight);g.add_node('plan',plan);g.add_node('execute',execute);g.add_node('replan',replan)
        g.add_node('finish',finish);g.add_node('summary',self._summary)
        g.add_edge(START,'qc');g.add_edge('qc','plan')
        g.add_conditional_edges('plan',lambda s:'summary' if s.get('error') else 'execute',{'summary':'finish','execute':'execute'})
        g.add_conditional_edges('execute',route_plan,{'summary':'finish','replan':'replan','execute':'execute','finish':'finish'})
        g.add_edge('replan','plan');g.add_edge('finish','summary');g.add_edge('summary',END)
        # NOTE: a planning failure routes directly to finish, then summary.
        return g.compile(checkpointer=checkpointer)

    def rewoo(self,checkpointer=None):
        g=StateGraph(WorkflowState)
        def planner(s):
            try:
                steps=self.model.plan(s,self.tools,rewoo=True,replanning=s.get('replans',0)>0)
                return activate_plan(s,steps,allowed_tools=set(self.tools.registry))
            except Exception as exc:
                return {'error':f'Planning failed: {exc}','status':'failed','steps':[]}
        def worker(s):
            if s.get('iterations',0)>=self.max_steps:
                return {'status':'failed','error':'Step budget exhausted'}
            return self._run_plan_steps(s)
        def replan(s):
            return {'replans':s.get('replans',0)+1}
        def solver(s):
            unfinished=not self._is_complete(s)
            error=s.get('error') or ('Step budget exhausted before plan completion' if unfinished else None)
            status=('rejected' if s.get('status')=='rejected' else 'failed' if error else
                    'partial_success' if s.get('tool_failures',0) else 'completed')
            return {'status':status,'error':error,**self._summary({**s,'status':status,'error':error})}
        g.add_node('qc',self._qc_preflight);g.add_node('planner',planner);g.add_node('worker',worker);g.add_node('replan',replan)
        g.add_node('solver',solver)
        g.add_edge(START,'qc');g.add_edge('qc','planner')
        g.add_conditional_edges('planner',lambda s:'solver' if s.get('error') else 'worker',
                                {'solver':'solver','worker':'worker'})
        def route_worker(s):
            if s.get('status')=='rejected':return 'solver'
            if s.get('error'):
                return 'replan' if s.get('replans',0)<self.max_replans else 'solver'
            return 'solver' if self._is_complete(s) or s.get('iterations',0)>=self.max_steps else 'worker'
        g.add_conditional_edges('worker',route_worker,
                                {'solver':'solver','worker':'worker','replan':'replan'})
        g.add_edge('replan','planner')
        g.add_edge('solver',END)
        return g.compile(checkpointer=checkpointer)

    def react(self,checkpointer=None):
        g=StateGraph(WorkflowState)
        def reason(s):
            if s.get('iterations',0)>=self.max_steps:
                return {'status':'failed','error':'Step budget exhausted'}
            messages=s.get('messages',[])
            if not messages:
                messages=[SystemMessage(content=SYSTEM+' The allowed source path is provided in the next message. Call tools, never invent computed values. All sources must be literal paths under the allowed data root or $input / $latest_npz.'),
                          HumanMessage(content=f"Task: {s['task']}\nInput: {s['input_path']}\nQC preflight: {str(s.get('qc_summary',{}))[:2000]}\nUse $input as the first source, $latest_npz after preprocessing.")]
            try:
                response=self.model.react(messages,self.tools.openai_schemas())
            except Exception as exc:
                return {'status':'failed','error':f'LLM request failed: {exc}','messages':messages if not s.get('messages') else []}
            return {'messages':(messages if not s.get('messages') else [])+[response],
                    'iterations':s.get('iterations',0)+1, 'tool_cursor':0}
        def route(s):
            if s.get('error'):return 'finish'
            last=s['messages'][-1]
            return 'tools' if getattr(last,'tool_calls',None) else 'finish'
        def apply_tools(s):
            # One tool per checkpointed node; never lose a failed call in trace.
            calls=[x for x in s['messages'] if isinstance(x,AIMessage) and getattr(x,'tool_calls',None)]
            if not calls:return {'error':'No pending tool calls','status':'failed'}
            batch=calls[-1].tool_calls;index=s.get('tool_cursor',0)
            if index>=len(batch):return {'error':'Tool cursor out of range','status':'failed'}
            call=batch[index];name=call['name'];kwargs=call.get('args',{})
            evidence=dict(s.get('evidence',{}));trace=list(s.get('trace',[]))
            latest=s.get('latest_npz');ctx={**s,'evidence':evidence,'latest_npz':latest}
            step_id=str(call['id']);started=datetime.now(timezone.utc).isoformat();clock=time.perf_counter()
            attempt_id=f'{step_id}:attempt:{1+sum(t.get("step_id")==step_id for t in trace)}'
            base={'step':step_id,'step_id':step_id,'attempt_id':attempt_id,'tool':name,'started_at':started}
            def failed(exc):
                err=f'{type(exc).__name__}: {exc}'
                event=finish_trace({**base,'status':'error','error':err},clock)
                self._progress(name,'failed',step_id=step_id,attempt_id=attempt_id,
                               duration_ms=event['duration_ms'],error=err)
                return {'messages':[ToolMessage(content=err[:10000],tool_call_id=step_id,name=name)],
                        'trace':trace+[event],'evidence':evidence,'latest_npz':latest,
                        'tool_cursor':index+1,'tool_failures':s.get('tool_failures',0)+1,
                        'error':err,'status':'failed'}
            try:
                if len(batch)>3:raise ValueError('Max 3 tool calls per model turn')
                args=bind_value(kwargs,ctx)
                self._guard(name,args,s,step_id)
            except (ValueError,TypeError,KeyError) as exc:
                return failed(exc)
            # LangGraph human approval interrupts MUST propagate unchanged.
            approved_args=self._approval(name,args,step_id)
            if approved_args is True:approved_args=args
            if approved_args is False:approved_args=None
            if approved_args is None:
                error='User denied '+name
                event=finish_trace({**base,'status':'rejected','error':error},clock)
                self._progress(name,'rejected',step_id=step_id,attempt_id=attempt_id,error=error)
                return {'status':'rejected','error':error,'trace':trace+[event],
                        'tool_cursor':index+1}
            try:
                self._progress(name,'started',step_id=step_id,attempt_id=attempt_id)
                result=self.tools.invoke(name,approved_args)
                if not isinstance(result,dict):raise TypeError('Tool must return a result dictionary')
                if result.get('status') in ('failed','error','partial_failure'):
                    raise RuntimeError(f'Tool returned {result["status"]}: '+
                                       str(result.get('error') or f'{result.get("failed",0)} failed files'))
                evidence[f'E{len(evidence)+1}']=result
                if 'output_npz' in result:latest=result['output_npz']
                event=finish_trace({**base,'arguments':approved_args,'status':'ok','result':result},clock)
                self._progress(name,'completed',step_id=step_id,attempt_id=attempt_id,
                               duration_ms=event['duration_ms'])
                update={'messages':[ToolMessage(content=str(result)[:10000],tool_call_id=step_id,name=name)],
                        'trace':trace+[event],'evidence':evidence,'latest_npz':latest,
                        'tool_cursor':index+1}
                if name=='assess_eeg_quality' and result.get('status')=='requires_review':
                    update['qc_summary']=result
                return update
            except Exception as exc:
                return failed(exc)
        def route_tools(s):
            if s.get('status')=='rejected' or s.get('error'):return 'finish'
            last=next((x for x in reversed(s.get('messages',[]))
                       if isinstance(x,AIMessage) and getattr(x,'tool_calls',None)),None)
            return 'tools' if last and s.get('tool_cursor',0)<len(last.tool_calls) else 'reason'
        def finish(s):
            # Don't invoke another LLM if it already gave its final ReAct answer.
            answer=''
            if s.get('messages'):
                tail=s['messages'][-1]
                if isinstance(tail,AIMessage) and not tail.tool_calls:
                    answer=str(tail.content)
            failures=s.get('tool_failures',0) or any(t.get('status')=='error' for t in s.get('trace',[]))
            status=('rejected' if s.get('status')=='rejected' else
                    'failed' if s.get('error') or (failures and not s.get('evidence')) else
                    'partial_success' if failures else 'completed')
            if status in ('failed','partial_success') and not s.get('error'):
                answer=f'Tool execution failed; inspect trace. '+answer
            return {'status':status,'final_answer':
                (f"Stopped: {s.get('error')}" if s.get('error') else answer or 'No final answer')}
        g.add_node('qc',self._qc_preflight);g.add_node('reason',reason);g.add_node('tools',apply_tools);g.add_node('finish',finish)
        g.add_edge(START,'qc');g.add_edge('qc','reason')
        g.add_conditional_edges('reason',route,{'finish':'finish','tools':'tools'})
        g.add_conditional_edges('tools',route_tools,{'reason':'reason','tools':'tools','finish':'finish'});g.add_edge('finish',END)
        return g.compile(checkpointer=checkpointer)

    def build(self,strategy:str,checkpointer=None):
        if strategy=='plan-execute':return self.plan_execute(checkpointer)
        if strategy=='react':return self.react(checkpointer)
        if strategy=='rewoo':return self.rewoo(checkpointer)
        raise ValueError('Choose plan-execute, react, rewoo')
