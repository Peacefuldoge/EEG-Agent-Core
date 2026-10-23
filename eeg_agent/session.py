"""Persistent, UI-free sessions for local EEG Agent workflows.

The public event stream uses plain dictionaries and works in an existing CLI,
Chainlit, FastAPI or a custom host. LangGraph state/checkpoints and scientific
artifacts are persisted separately from conversation messages.
"""
from __future__ import annotations
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator
from .persistence import SessionStore

STRATEGIES = ('plan-execute', 'react', 'rewoo')
DEFAULT_APPROVAL_TOOLS = frozenset({'preprocess_eeg'})


def _content(chunk):
    parts = getattr(chunk, 'content', '')
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        return ''.join(str(p.get('text', '')) for p in parts if isinstance(p,dict) and p.get('type')=='text')
    return ''


class EEGSession:
    def __init__(self, *, data_root: str | Path, output_root: str | Path,
                 model: str='qwen3-8b', base_url: str='http://127.0.0.1:8000/v1',
                 strategy: str='react', max_steps: int=12, max_replans: int=1,
                 checkpoint: bool=True, llm=None, session_id: str | None=None,
                 approval_tools=None):
        from .tools import EEGTools
        if strategy not in STRATEGIES:
            raise ValueError('Invalid strategy: '+strategy)
        self.tools=EEGTools(data_root,output_root)
        self.store=SessionStore(self.tools.output_root)
        self.session_id = session_id or self.store.create(strategy=strategy,model=model,base_url=base_url)
        if session_id:
            saved=self.store.get(session_id)
            strategy=saved['strategy'];model=saved['model'];base_url=saved['base_url']
        self.strategy=strategy
        self.model_name=model
        self.base_url=base_url
        self.max_steps=max_steps
        self.max_replans=max_replans
        self.checkpoint=checkpoint
        self.approval_tools=frozenset(DEFAULT_APPROVAL_TOOLS if approval_tools is None and checkpoint
                                       else approval_tools or ())
        if self.approval_tools and not self.checkpoint:
            raise ValueError('Human approval requires a durable checkpoint')
        self.current_file=None
        self.history=self.store.history(self.session_id)
        if session_id and saved['current_file']:
            try: self.set_file(saved['current_file'])
            except (ValueError,FileNotFoundError): pass
        self._model=None
        self._injected_llm=llm

    @property
    def llm(self):
        if self._model is None:
            from .model import QwenModel
            candidate=self._injected_llm
            self._model=(candidate if candidate is not None and all(callable(getattr(candidate,k,None))
                         for k in ('plan','react','summarize')) else
                         QwenModel(chat=candidate) if candidate is not None else
                         QwenModel(model=self.model_name,base_url=self.base_url))
        return self._model

    @property
    def data_root(self):return self.tools.data_root
    @property
    def output_root(self):return self.tools.output_root

    def set_strategy(self,strategy):
        if strategy not in STRATEGIES:raise ValueError('Choose from '+', '.join(STRATEGIES))
        self.strategy=strategy
        self.store.update_session(self.session_id,strategy=strategy)

    def set_file(self,path):
        self.current_file=str(self.tools._resolve(str(path)))
        self.store.update_session(self.session_id,current_file=self.current_file)
        return self.current_file

    def clear(self):
        self.history=[]
        self.store.clear_history(self.session_id)

    def list_sessions(self,limit=50):return self.store.list_sessions(limit)
    def get_history(self,limit=None):return self.store.messages(self.session_id,limit)
    def list_runs(self,limit=50):return self.store.list_runs(self.session_id,limit)
    def get_run(self,thread_id):
        run=self.store.get_run(thread_id)
        if run['session_id']!=self.session_id:
            raise ValueError('Task does not belong to this session')
        return run

    def _remember(self,user,assistant):
        self.store.append_turn(self.session_id,user,assistant[:6000])
        self.history.append((user,assistant[:6000]))
        self.history=self.history[-10:]

    def _messages(self,message):
        from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
        messages=[SystemMessage(content='You are a local EEG research assistant. Respond in the user language. Never invent medical findings or claim unexecuted tool results.')]
        for user,answer in self.history[-5:]:
            messages.extend([HumanMessage(content=user),AIMessage(content=answer)])
        messages.append(HumanMessage(content=message))
        return messages

    def stream_chat(self,message):
        """Yield {'type':'token','text':...} and final {'type':'completed'} events."""
        messages=self._messages(message)
        response=[]
        for chunk in self.llm.chat.stream(messages):
            content=_content(chunk)
            if content:
                response.append(content)
                yield {'type':'token','text':content,'scope':'chat','session_id':self.session_id}
        answer=''.join(response)
        self._remember(message,answer)
        yield {'type':'completed','answer':answer,'session_id':self.session_id}

    def chat(self,message):
        answer=''
        for item in self.stream_chat(message):
            if item['type']=='completed':answer=item['answer']
        return answer

    def _make_graph(self, strategy, saver):
        from .strategies import AgentGraphs
        return AgentGraphs(self.llm,self.tools,max_steps=self.max_steps,
                           max_replans=self.max_replans,
                           approval_tools=self.approval_tools).build(strategy,saver)

    def _checkpoint(self):
        if not self.checkpoint:
            return None,None
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn=sqlite3.connect(str(self.output_root/'agent_checkpoints.sqlite3'),check_same_thread=False,timeout=30)
        return SqliteSaver(conn),conn

    @staticmethod
    def _interrupts(snapshot):
        pending=[]
        for task in getattr(snapshot,'tasks',[]) or []:
            for intr in getattr(task,'interrupts',[]) or []:
                value=getattr(intr,'value',intr)
                pending.append(value if isinstance(value,dict) else {'message':str(value)})
        return pending

    def _stream_graph(self, *, run, initial=None, resume=None):
        thread_id=run['thread_id'];strategy=run['strategy']
        saver,conn=self._checkpoint()
        state=dict(initial or {})
        pending=[]
        try:
            graph=self._make_graph(strategy,saver)
            config={'configurable':{'thread_id':thread_id},
                    'recursion_limit':max(70,self.max_steps*8)}
            if resume is not None:
                from langgraph.types import Command
                input_value=Command(resume=resume)
            else:
                input_value=initial
            # debug emits node starts; updates emits node finishes; messages emits
            # user-facing generated text. Only minimal data escapes the Core.
            for mode,payload in graph.stream(input_value,config=config,
                                              stream_mode=['debug','updates','messages','custom'],
                                              durability='sync' if saver is not None else 'async'):
                if mode=='debug' and isinstance(payload,dict):
                    if payload.get('type')=='task':
                        p=payload.get('payload') or {}
                        node=p.get('name')
                        if node:
                            yield {'type':'node_started','node':node,'thread_id':thread_id}
                elif mode=='updates' and isinstance(payload,dict):
                    for node,update in payload.items():
                        if node=='__interrupt__':
                            continue
                        if isinstance(update,dict):state.update(update)
                        yield {'type':'node_completed','node':str(node),
                               'status':state.get('status','running'),'thread_id':thread_id}
                elif mode=='messages' and isinstance(payload,tuple) and len(payload)==2:
                    chunk,metadata=payload
                    node=metadata.get('langgraph_node','') if isinstance(metadata,dict) else ''
                    # Planner JSON and tool-call internals never go to user-facing text.
                    if node in ('reason','summary','solver'):
                        text=_content(chunk)
                        if text and not getattr(chunk,'tool_call_chunks',None):
                            yield {'type':'token','text':text,'node':node,'scope':'agent','thread_id':thread_id}
                elif mode=='custom' and isinstance(payload,dict):
                    yield {'type':'tool_progress','data':payload,'thread_id':thread_id}
            if saver is not None:
                snapshot=graph.get_state(config)
                state.update(dict(getattr(snapshot,'values',{}) or {}))
                pending=self._interrupts(snapshot)
                if getattr(snapshot,'next',()) and not pending:
                    # Running was interrupted by shutdown, not an HITL breakpoint.
                    status='interrupted'
                elif pending:
                    status='awaiting_approval'
                else:
                    status=state.get('status','completed')
            else:
                status=state.get('status','completed')
        except Exception as exc:
            self.store.update_run(thread_id,status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if conn is not None:conn.close()
        answer=str(state.get('final_answer') or state.get('error') or '')
        latest_npz=state.get('latest_npz')
        if latest_npz and Path(latest_npz).is_file():
            self.set_file(latest_npz)
        record={'status':status,'strategy':strategy,'model':run['model'],
                'session_id':self.session_id,'thread_id':thread_id,
                'input_file':run['input_file'],'task':run['task'],
                'answer':answer,'error':state.get('error'),
                'trace':state.get('trace',[]),'evidence':state.get('evidence',{}),
                'latest_npz':latest_npz,'pending':pending}
        manifest=self.output_root/f'agent_run_{thread_id}.json'
        manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        record['manifest']=str(manifest)
        self.store.update_run(thread_id,status=status,manifest=str(manifest),
                              error=record['error'],pending=pending or None)
        if pending:
            yield {'type':'approval_required','thread_id':thread_id,'requests':pending,
                   'session_id':self.session_id}
        if status not in ('awaiting_approval','interrupted'):
            self._remember(run['task'],answer[:6000])
        yield {'type':'run_finished','record':record,'thread_id':thread_id,'status':status}

    def stream_agent(self,message):
        if not self.current_file:
            raise ValueError('Please choose an EEG file first: /file /path/to/subject.edf')
        source=self.tools._resolve(self.current_file)
        brief='\n'.join(f'Previous user: {u[:400]}\nPrevious assistant: {a[:650]}'
                        for u,a in self.history[-3:])
        task=message if not brief else f'Conversation context (background only):\n{brief}\n\nCurrent EEG analysis request:\n{message}'
        thread_id=uuid.uuid4().hex
        initial={'task':task,'input_path':str(source),'strategy':self.strategy,
                 'cursor':0,'iterations':0,'replans':0,'tool_cursor':0,'steps':[],
                 'trace':[],'evidence':{},'latest_npz':None,'error':None,'status':'running'}
        self.store.create_run(session_id=self.session_id,thread_id=thread_id,
                              task=message,input_file=str(source),strategy=self.strategy,
                              model=self.model_name)
        run=self.get_run(thread_id)
        yield {'type':'run_started','thread_id':thread_id,'session_id':self.session_id,
               'strategy':self.strategy}
        yield from self._stream_graph(run=run,initial=initial)

    def resume_run(self,thread_id,*,approved:bool | None=None):
        """Resume a paused graph; 'approved' is REQUIRED for HITL requests.

        For a crashed/interrupted non-HITL graph, set approved=None; the last
        in-progress node may run again, so EEGTools must be idempotent at node
        boundaries or use per-file caching for large batches.
        """
        if not self.checkpoint:
            raise ValueError('Resume requires checkpoint=True')
        run=self.get_run(thread_id)
        if run['status']=='awaiting_approval':
            if not isinstance(approved,bool):
                raise ValueError('Approval decision required (True or False)')
            resume={'approved':approved}
        elif run['status'] in ('running','interrupted','failed'):
            if approved is not None:
                raise ValueError('No pending approval for this task')
            resume=None
        else:
            raise ValueError('Task is not resumable: '+run['status'])
        # For unfinished non-HITL work pass None as graph input to resume.
        # Verify an unfinished checkpoint exists before attempting replay.
        saver,conn=self._checkpoint()
        try:
            graph=self._make_graph(run['strategy'],saver)
            snapshot=graph.get_state({'configurable':{'thread_id':thread_id}})
            if not getattr(snapshot,'next',()):
                raise ValueError('No unfinished LangGraph checkpoint is available for this task')
        finally:
            if conn is not None:conn.close()
        self.store.update_run(thread_id,status='running')
        yield {'type':'run_resumed','thread_id':thread_id,'session_id':self.session_id}
        yield from self._stream_graph(run=run,resume=resume)

    def run_agent(self,message,on_node:Callable[[str,dict],Any] | None=None,on_event=None):
        result=None
        for event in self.stream_agent(message):
            if on_event:on_event(event)
            if event['type']=='node_completed' and on_node:
                on_node(event['node'],{'status':event.get('status')})
            if event['type']=='run_finished':result=event['record']
        return result
