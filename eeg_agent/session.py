"""Shared, local-only chat and LangGraph execution session for CLI/Web UI.

Each EEG task gets a new LangGraph checkpoint thread. Conversation history is
maintained at the UI layer, preventing previous terminal graph state or ReAct
ToolMessages from leaking into a new task's state.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Callable, Any

STRATEGIES = ('plan-execute', 'react', 'rewoo')


class EEGSession:
    def __init__(self, *, data_root: str | Path, output_root: str | Path,
                 model: str = 'qwen3-8b', base_url: str = 'http://127.0.0.1:8000/v1',
                 strategy: str = 'react', max_steps: int = 12,
                 max_replans: int = 1, checkpoint: bool = True, llm=None):
        from .tools import EEGTools
        if strategy not in STRATEGIES:
            raise ValueError('Invalid strategy: ' + strategy)
        self.tools = EEGTools(data_root, output_root)
        self.strategy = strategy
        self.model_name = model
        self.base_url = base_url
        self.max_steps = max_steps
        self.max_replans = max_replans
        self.checkpoint = checkpoint
        self.current_file: str | None = None
        self.history: list[tuple[str, str]] = []
        self._model = None
        self._injected_llm = llm  # optional existing LangChain chat model or full adapter

    @property
    def llm(self):
        if self._model is None:
            from .model import QwenModel
            if self._injected_llm is not None:
                candidate = self._injected_llm
                self._model = (candidate if all(callable(getattr(candidate, k, None))
                                                 for k in ('plan', 'react', 'summarize'))
                               else QwenModel(chat=candidate))
            else:
                self._model = QwenModel(model=self.model_name, base_url=self.base_url)
        return self._model

    def set_strategy(self, strategy: str):
        if strategy not in STRATEGIES:
            raise ValueError(f'Choose from {", ".join(STRATEGIES)}')
        self.strategy = strategy

    def set_file(self, path: str | Path) -> str:
        resolved = self.tools._resolve(str(path))
        self.current_file = str(resolved)
        return self.current_file

    @property
    def data_root(self) -> Path:
        return self.tools.data_root

    @property
    def output_root(self) -> Path:
        return self.tools.output_root

    def clear(self):
        self.history.clear()

    def chat(self, message: str) -> str:
        """Normal conversational Qwen without EEG tool execution."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        messages = [SystemMessage(content='You are a local EEG research assistant. Respond in the same language as the user. Do not invent medical findings or claim that EEG tools were executed.')]
        for user, answer in self.history[-5:]:
            messages.extend([HumanMessage(content=user), AIMessage(content=answer)])
        messages.append(HumanMessage(content=message))
        result = self.llm.chat.invoke(messages)
        answer = str(result.content)
        self.history.append((message, answer[:6000]))
        self.history = self.history[-10:]
        return answer

    def run_agent(self, message: str, on_node: Callable[[str, dict], Any] | None = None) -> dict:
        if not self.current_file:
            raise ValueError('Please choose an EEG file first: /file /path/to/subject.edf')
        from .strategies import AgentGraphs
        from .model import QwenModel
        source = self.tools._resolve(self.current_file)
        brief = '\n'.join(f'Previous user: {u[:400]}\nPrevious assistant: {a[:650]}'
                          for u, a in self.history[-3:])
        task = message if not brief else f'Conversation context (background only):\n{brief}\n\nCurrent EEG analysis request:\n{message}'
        thread_id = uuid.uuid4().hex
        initial = {
            'task': task, 'input_path': str(source), 'strategy': self.strategy,
            'cursor': 0, 'iterations': 0, 'replans': 0, 'steps': [],
            'trace': [], 'evidence': {}, 'latest_npz': None,
            'error': None, 'status': 'running',
        }
        conn = None
        saver = None
        os.environ.setdefault('LANGGRAPH_STRICT_MSGPACK', 'true')
        if self.checkpoint:
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(str(self.tools.output_root / 'agent_checkpoints.sqlite3'), check_same_thread=False)
            saver = SqliteSaver(conn)
        state = dict(initial)
        try:
            graph = AgentGraphs(self.llm, self.tools, max_steps=self.max_steps,
                                max_replans=self.max_replans).build(self.strategy, saver)
            config = {'configurable': {'thread_id': thread_id},
                      'recursion_limit': max(70, self.max_steps * 7)}
            for event in graph.stream(initial, config=config, stream_mode='updates'):
                for node, update in event.items():
                    if isinstance(update, dict):
                        state.update(update)
                    if on_node:
                        on_node(str(node), update if isinstance(update, dict) else {})
        finally:
            if conn is not None:
                conn.close()
        status = state.get('status', 'failed')
        answer = str(state.get('final_answer') or state.get('error') or '(No answer)')
        latest_npz = state.get('latest_npz')
        if latest_npz and Path(latest_npz).is_file():
            # Enables next-turn questions to analyze processed windows directly.
            self.set_file(latest_npz)
        record = {
            'status': status, 'strategy': self.strategy, 'model': self.model_name,
            'thread_id': thread_id, 'input_file': str(source), 'task': message,
            'answer': answer, 'error': state.get('error'), 'trace': state.get('trace', []),
            'evidence': state.get('evidence', {}), 'latest_npz': latest_npz,
        }
        manifest = self.tools.output_root / f'agent_run_{thread_id}.json'
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        record['manifest'] = str(manifest)
        self.history.append((message, answer[:6000]))
        self.history = self.history[-10:]
        return record
