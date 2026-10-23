"""Non-mutating vLLM service health check (does not start/download a model)."""
import argparse
from openai import OpenAI

p=argparse.ArgumentParser()
p.add_argument('--base-url',default='http://localhost:8000/v1')
a=p.parse_args()
c=OpenAI(base_url=a.base_url,api_key='EMPTY',timeout=8)
try:
    print('Models available:',[x.id for x in c.models.list().data])
except Exception as exc:
    raise SystemExit(f'Cannot reach vLLM: {exc}')
