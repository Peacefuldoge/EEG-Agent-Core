"""Check the real deployed Qwen/vLLM model can emit an OpenAI-compatible tool call."""
from __future__ import annotations
import argparse
from openai import OpenAI

p=argparse.ArgumentParser()
p.add_argument('--base-url',default='http://localhost:8000/v1')
p.add_argument('--model',default='qwen3-8b')
a=p.parse_args()
c=OpenAI(base_url=a.base_url,api_key='EMPTY',timeout=120)
answer=c.chat.completions.create(model=a.model,temperature=0,
    messages=[{'role':'system','content':'Use the provided tools when asked to inspect an EEG file.'},
              {'role':'user','content':'Call inspect_eeg with source "$input". Do not answer without calling the function.'}],
    tools=[{'type':'function','function':{'name':'inspect_eeg',
        'description':'Inspect EEG recording metadata',
        'parameters':{'type':'object','properties':{'source':{'type':'string'}},
                      'required':['source'],'additionalProperties':False}}}],
    tool_choice='auto')
msg=answer.choices[0].message
if not msg.tool_calls:
    raise SystemExit(f'FAIL: Qwen did not generate an OpenAI tool call; got: {msg.content!r}')
if msg.tool_calls[0].function.name!='inspect_eeg':
    raise SystemExit(f'FAIL: unexpected tool: {msg.tool_calls[0].function.name}')
print('PASS: actual Qwen/vLLM tool parser returned',msg.tool_calls[0].function.name)
print('Arguments:',msg.tool_calls[0].function.arguments)
