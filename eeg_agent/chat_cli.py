"""Codex-inspired REPL, backed by existing EEG-Agent LangGraph strategies."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from .session import EEGSession, STRATEGIES


def main(argv=None):
    parser = argparse.ArgumentParser(description='Interactive local Qwen / EEG LangGraph Agent')
    parser.add_argument('--data-root', type=Path, default=Path('examples/demo_npz'))
    parser.add_argument('--output-root', type=Path, default=Path('agent_runs'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--strategy', choices=STRATEGIES, default='react')
    parser.add_argument('--mode', choices=('chat', 'agent'), default='chat')
    parser.add_argument('--model', default=os.getenv('EEG_LLM_MODEL', 'qwen3-8b'))
    parser.add_argument('--base-url', default=os.getenv('EEG_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--no-checkpoint', action='store_true')
    args = parser.parse_args(argv)
    session = EEGSession(data_root=args.data_root, output_root=args.output_root,
                         strategy=args.strategy, model=args.model, base_url=args.base_url,
                         checkpoint=not args.no_checkpoint)
    if args.input:
        session.set_file(args.input)
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from rich.console import Console
        console = Console()
        history_file = session.tools.output_root / '.chat_prompt_history'
        prompt = PromptSession(history=FileHistory(str(history_file)))
        def show(s, style=None): console.print(s, style=style, markup=False)
        def ask(s): return prompt.prompt(s)
    except ImportError:
        def show(s, style=None): print(s)
        def ask(s): return input(s)
    mode = args.mode
    show('EEG-Agent | model=' + args.model + ' | mode=' + mode + ' | strategy=' + session.strategy)
    show('Commands: /chat /agent /file PATH /strategy react|plan-execute|rewoo /status /clear /help /exit')
    while True:
        try:
            msg = ask(f'[{mode}:{session.strategy}] > ').strip()
        except (KeyboardInterrupt, EOFError):
            show('\nGoodbye.'); break
        if not msg:
            continue
        cmd, _, value = msg.partition(' ')
        if cmd in ('/exit', '/quit'):
            break
        if cmd == '/help':
            show('Use /file /mnt/d/EEG/sub01.edf, then /agent and ask a task. /chat switches to conversation.\n/strategy selects the LangGraph graph. /clear forgets recent chat context. /status shows current state. /exit quits.')
            continue
        if cmd == '/chat': mode = 'chat'; show('Mode: normal Qwen conversation'); continue
        if cmd == '/agent': mode = 'agent'; show('Mode: EEG tool execution via LangGraph'); continue
        if cmd == '/strategy':
            try: session.set_strategy(value.strip()); show('Strategy: ' + session.strategy)
            except ValueError as e: show(f'Error: {e}')
            continue
        if cmd == '/file':
            try: show('Input: ' + session.set_file(value.strip().strip('"\'')))
            except Exception as e: show(f'Error: {e}')
            continue
        if cmd == '/clear': session.clear(); show('Context cleared.'); continue
        if cmd == '/status':
            show('mode=' + mode + ' | strategy=' + session.strategy + ' | file=' + str(session.current_file) + ' | endpoint=' + session.base_url)
            continue
        try:
            if mode == 'chat':
                show('\nQwen:\n' + session.chat(msg) + '\n')
            else:
                def on_node(node, update):
                    info = 'failed' if update.get('error') else update.get('status', '')
                    show(f'  → {node}' + (f' [{info}]' if info else ''))
                report = session.run_agent(msg, on_node=on_node)
                show('\nAgent (' + report['status'] + '):\n' + report['answer'])
                show('Trace: ' + report['manifest'] + '\n')
        except KeyboardInterrupt:
            show('Interrupted current task; use a new request.')
        except Exception as e:
            show(f'Execution error: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
