"""Three LangGraph graphs sharing the same tool registry and local vLLM Qwen client."""
from __future__ import annotations
from typing import Any,Annotated,TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,ToolMessage
from .schema import Step
from .execution import execute_step,bind_value
from .model import SYSTEM

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

class AgentGraphs:
    def __init__(self, model, tools, *, max_steps:int=12, max_replans:int=1):
        self.model=model
        self.tools=tools
        self.max_steps=max_steps
        self.max_replans=max_replans

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
                return {'steps':steps,'cursor':0,'error':None,'status':'running'}
            except Exception as exc:
                return {'error':f'Planning failed: {exc}','status':'failed','steps':[]}
        def execute(s):
            if s.get('cursor',0)>=len(s.get('steps',[])) or s.get('iterations',0)>=self.max_steps:
                return {'error':'Step budget exhausted','status':'failed'}
            step=Step.model_validate(s['steps'][s['cursor']])
            try:
                result=execute_step(step,s,self.tools)
            except Exception as exc:
                result={'error':f'{type(exc).__name__}: {exc}',
                        'trace':list(s.get('trace',[]))+[{'step':step.id,'tool':step.tool,'status':'error','error':str(exc)}]}
            return {**result,'cursor':s['cursor']+1,'iterations':s.get('iterations',0)+1}
        def route_plan(s):
            if s.get('error'):
                return 'summary' if s.get('replans',0)>=self.max_replans or s.get('iterations',0)>=self.max_steps else 'replan'
            return 'execute' if s.get('cursor',0)<len(s.get('steps',[])) and s.get('iterations',0)<self.max_steps else 'finish'
        def replan(s):
            return {'replans':s.get('replans',0)+1}
        def finish(s):
            unfinished=s.get('cursor',0)<len(s.get('steps',[]))
            error=s.get('error') or ('Step budget exhausted before plan completion' if unfinished else None)
            return {'status':'failed' if error else 'completed','error':error}
        g.add_node('plan',plan);g.add_node('execute',execute);g.add_node('replan',replan)
        g.add_node('finish',finish);g.add_node('summary',self._summary)
        g.add_edge(START,'plan')
        g.add_conditional_edges('plan',lambda s:'summary' if s.get('error') else 'execute',{'summary':'finish','execute':'execute'})
        g.add_conditional_edges('execute',route_plan,{'summary':'finish','replan':'replan','execute':'execute','finish':'finish'})
        g.add_edge('replan','plan');g.add_edge('finish','summary');g.add_edge('summary',END)
        # NOTE: a planning failure routes directly to finish, then summary.
        return g.compile(checkpointer=checkpointer)

    def rewoo(self,checkpointer=None):
        g=StateGraph(WorkflowState)
        def planner(s):
            try:
                steps=self.model.plan(s,self.tools,rewoo=True)
                return {'steps':steps,'cursor':0,'error':None,'status':'running'}
            except Exception as exc:
                return {'error':f'Planning failed: {exc}','status':'failed','steps':[]}
        def worker(s):
            if s.get('cursor',0)>=len(s.get('steps',[])) or s.get('iterations',0)>=self.max_steps:
                return {'status':'failed','error':'Step budget exhausted'}
            step=Step.model_validate(s['steps'][s['cursor']])
            try:
                result=execute_step(step,s,self.tools)
            except Exception as exc:
                result={'error':f'{type(exc).__name__}: {exc}',
                        'trace':list(s.get('trace',[]))+[{'step':step.id,'tool':step.tool,'status':'error','error':str(exc)}]}
            return {**result,'cursor':s['cursor']+1,'iterations':s.get('iterations',0)+1}
        def solver(s):
            unfinished=s.get('cursor',0)<len(s.get('steps',[]))
            error=s.get('error') or ('Step budget exhausted before plan completion' if unfinished else None)
            status='failed' if error else 'completed'
            return {'status':status,'error':error,**self._summary({**s,'status':status,'error':error})}
        g.add_node('planner',planner);g.add_node('worker',worker)
        g.add_node('solver',solver)
        g.add_edge(START,'planner')
        g.add_conditional_edges('planner',lambda s:'solver' if s.get('error') else 'worker',
                                {'solver':'solver','worker':'worker'})
        g.add_conditional_edges('worker',lambda s:'solver' if s.get('error') or s.get('cursor',0)>=len(s.get('steps',[])) or s.get('iterations',0)>=self.max_steps else 'worker',
                                {'solver':'solver','worker':'worker'})
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
                          HumanMessage(content=f"Task: {s['task']}\nInput: {s['input_path']}\nUse $input as the first source, $latest_npz after preprocessing.")]
            try:
                response=self.model.react(messages,self.tools.openai_schemas())
            except Exception as exc:
                return {'status':'failed','error':f'LLM request failed: {exc}','messages':messages if not s.get('messages') else []}
            return {'messages':(messages if not s.get('messages') else [])+[response],
                    'iterations':s.get('iterations',0)+1}
        def route(s):
            if s.get('error'):return 'finish'
            last=s['messages'][-1]
            return 'tools' if getattr(last,'tool_calls',None) else 'finish'
        def apply_tools(s):
            last=s['messages'][-1]
            replies=[];evidence=dict(s.get('evidence',{}));trace=list(s.get('trace',[]))
            latest=s.get('latest_npz')
            for call in last.tool_calls:
                name=call['name'];kwargs=call.get('args',{})
                ctx={**s,'evidence':evidence,'latest_npz':latest}
                try:
                    if len(replies)>=3:
                        raise ValueError('Max 3 parallel calls per model turn; request remaining in a later turn')
                    args=bind_value(kwargs,ctx)
                    result=self.tools.invoke(name,args)
                    evidence[f'E{len(evidence)+1}']=result
                    if 'output_npz' in result:latest=result['output_npz']
                    trace.append({'tool':name,'status':'ok','result':result})
                    text=str(result)
                except Exception as exc:
                    text=f'{type(exc).__name__}: {exc}'
                    trace.append({'tool':name,'status':'error','error':text})
                replies.append(ToolMessage(content=text[:10000],tool_call_id=call['id'],name=name))
            return {'messages':replies,'trace':trace,'evidence':evidence,'latest_npz':latest}
        def finish(s):
            # Don't invoke another LLM if it already gave its final ReAct answer.
            answer=''
            if s.get('messages'):
                tail=s['messages'][-1]
                if isinstance(tail,AIMessage) and not tail.tool_calls:
                    answer=str(tail.content)
            return {'status':'failed' if s.get('error') else 'completed',
                    'final_answer':answer or (f"Stopped: {s.get('error')}" if s.get('error') else 'No final answer')}
        g.add_node('reason',reason);g.add_node('tools',apply_tools);g.add_node('finish',finish)
        g.add_edge(START,'reason')
        g.add_conditional_edges('reason',route,{'finish':'finish','tools':'tools'})
        g.add_edge('tools','reason');g.add_edge('finish',END)
        return g.compile(checkpointer=checkpointer)

    def build(self,strategy:str,checkpointer=None):
        if strategy=='plan-execute':return self.plan_execute(checkpointer)
        if strategy=='react':return self.react(checkpointer)
        if strategy=='rewoo':return self.rewoo(checkpointer)
        raise ValueError('Choose plan-execute, react, rewoo')
