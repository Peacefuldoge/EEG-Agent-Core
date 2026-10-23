"""vLLM OpenAI-compatible Qwen model adapter. No cloud/model token needed."""
from __future__ import annotations
import json
import os
from typing import Any
from pydantic import ValidationError
from .execution import parse_plan

SYSTEM = ('You are an EEG research workflow orchestrator. Do not diagnose disease. '
          'Scientific numbers must come from tools. Never invent channel data, '
          'never propose arbitrary code or shell commands. Use tools only. '
          'Prefer inspect_eeg, preprocess_eeg, analyze_spectrum, '
          'analyze_connectivity, visualize_eeg. '
          'When given an input path use the reference $input. '
          'After preprocessing, use $latest_npz or $E#.output_npz. '
          'If data is already a preprocessed 3D NPZ, skip preprocessing. '
          'Ask for human review when interpolation or ICA assumptions are unclear.')

def extract_json(text: str) -> dict:
    text=text.strip()
    if text.startswith('```'):
        lines=text.splitlines()
        if len(lines)>=3 and lines[-1].strip()=='```':
            text='\n'.join(lines[1:-1]).strip()
    try:
        obj=json.loads(text)
    except json.JSONDecodeError:
        start=text.find('{'); end=text.rfind('}')
        if start<0 or end<=start:
            raise ValueError('Model did not return a JSON object')
        obj=json.loads(text[start:end+1])
    if not isinstance(obj,dict):
        raise ValueError('Model returned non-object JSON')
    return obj

class QwenModel:
    def __init__(self, chat=None, *, base_url=None,model=None,api_key=None):
        if chat is None:
            from langchain_openai import ChatOpenAI
            self.chat=ChatOpenAI(
                model=model or os.getenv('EEG_LLM_MODEL','Qwen/Qwen3-8B'),
                base_url=base_url or os.getenv('EEG_LLM_BASE_URL','http://localhost:8000/v1'),
                api_key=api_key or os.getenv('EEG_LLM_API_KEY','EMPTY'),
                temperature=0,max_retries=1,timeout=180,
            )
        else:
            self.chat=chat

    def plan(self, state:dict, tools, *, rewoo:bool=False, replanning:bool=False):
        from langchain_core.messages import SystemMessage,HumanMessage
        catalog=[{'tool':s.name,'purpose':s.description,'input_schema':s.args.model_json_schema()}
                 for s in tools.registry.values()]
        extra='ReWOO: define a dependency DAG with depends_on; later steps can use $E1.output_npz.' if rewoo else 'Plan-and-Execute: list ordered steps, using $latest_npz after preprocessing.'
        if replanning:
            extra+=' Previous attempt failed. Produce a DIFFERENT safe plan; do not repeat the same failed step without changing parameters.'
        prompt={'task':state['task'],'input_file':state['input_path'],
                'available_tools':catalog,'completed_evidence':state.get('evidence',{}),
                'last_error':state.get('error'),
                'required_response':{'steps':[{'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]}]},
                'instructions':extra+' Output valid JSON only, 1-12 steps. No markdown. '
                          'Use $input or $latest_npz for sources; only refer to prior $E#.field keys.'}
        msg=self.chat.invoke([SystemMessage(content=SYSTEM),HumanMessage(content=json.dumps(prompt,ensure_ascii=False,default=str))])
        return parse_plan(extract_json(str(msg.content)))

    def react(self, messages:list, tool_schemas:list):
        # Tool-call parser happens in vLLM; LangChain passes OpenAI-compatible tools.
        return self.chat.bind_tools(tool_schemas).invoke(messages)

    def summarize(self, state:dict):
        from langchain_core.messages import SystemMessage,HumanMessage
        payload={'task':state['task'],'status':state.get('status','complete'),
                 'error':state.get('error'),'evidence':state.get('evidence',{}),
                 'trace':state.get('trace',[])[-8:]}
        msg=self.chat.invoke([SystemMessage(content=SYSTEM+' Summarize verified results with file locations; explicitly mention failures and uncertainties. Do not claim diagnosis.'),
                              HumanMessage(content=json.dumps(payload,ensure_ascii=False,default=str)[:25000])])
        return str(msg.content)
