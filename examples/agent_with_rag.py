"""Usage: uv run python examples/agent_with_rag.py --data-root ... --eeg ..."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from rag_bridge import RAGClient, register_rag_tool
from eeg_agent import EEGSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--output-root', default='./outputs')
    ap.add_argument('--eeg', required=True, help='An existing EDF/FIF/NPZ file under data-root')
    ap.add_argument('--strategy', default='react', choices=('react','rewoo','plan-execute'))
    ap.add_argument('--question', default='Find literature evidence for EEG QC and analyze my file using available tools.')
    args = ap.parse_args()
    session = EEGSession(
        data_root=Path(args.data_root), output_root=Path(args.output_root),
        model=os.getenv('EEG_LLM_MODEL','qwen3-8b'),
        base_url=os.getenv('EEG_LLM_BASE_URL','http://127.0.0.1:8000/v1'),
        strategy=args.strategy, checkpoint=True,
    )
    client = RAGClient()
    try:
        register_rag_tool(session, client)
        session.set_file(args.eeg)
        for event in session.stream_agent(args.question):
            print(event)
            if event.get('type') == 'approval_required':
                print('Approval required; use session.resume_run(thread_id, approved=...) after inspection.')
                break
    finally:
        client.close()


if __name__ == '__main__':
    main()
