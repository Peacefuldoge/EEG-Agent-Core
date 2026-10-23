"""Chainlit Web UI for local Qwen chat + three LangGraph EEG workflows.
Run: chainlit run eeg_agent/web_app.py --host 127.0.0.1 --port 8002
"""
from __future__ import annotations
import asyncio
import os
import shutil
import uuid
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select
from eeg_agent.session import EEGSession, STRATEGIES
from eeg_agent.tools import SUPPORTED_RAW

DATA_ROOT = Path(os.getenv('EEG_DATA_ROOT', './examples/demo_npz')).resolve()
OUTPUT_ROOT = Path(os.getenv('EEG_OUTPUT_ROOT', './agent_runs')).resolve()
MODEL = os.getenv('EEG_LLM_MODEL', 'qwen3-8b')
BASE_URL = os.getenv('EEG_LLM_BASE_URL', 'http://127.0.0.1:8000/v1')


@cl.on_chat_start
async def start():
    session = EEGSession(data_root=DATA_ROOT, output_root=OUTPUT_ROOT,
                         model=MODEL, base_url=BASE_URL)
    cl.user_session.set('eeg_session', session)
    cl.user_session.set('mode', 'chat')
    await cl.ChatSettings([
        Select(id='Strategy', label='LangGraph execution strategy',
               values=list(STRATEGIES), initial_index=1),
        Select(id='Mode', label='Mode', values=['chat', 'agent'], initial_index=0),
    ]).send()
    await cl.Message(content=(
        '**本地 EEG-Agent 已连接会话。**\n\n'
        '先普通聊天可直接提问；做真实 EEG 分析请用 `/file /绝对路径/subject.edf` 选择文件，'
        '或用 `/upload` 上传单个文件，再用 `/agent` 进入工具执行模式。\n\n'
        f'模型：`{MODEL}` · vLLM：`{BASE_URL}` · 允许数据根目录：`{DATA_ROOT}`'
    )).send()


@cl.on_settings_update
async def settings_update(settings):
    s = cl.user_session.get('eeg_session')
    s.set_strategy(settings.get('Strategy', 'react'))
    cl.user_session.set('mode', settings.get('Mode', 'chat'))


@cl.on_message
async def on_message(msg: cl.Message):
    s: EEGSession = cl.user_session.get('eeg_session')
    text = msg.content.strip()
    cmd, _, tail = text.partition(' ')
    if cmd == '/file':
        try:
            path = s.set_file(tail.strip().strip('"\''))
            await cl.Message(content=f'已选择 EEG 文件：`{path}`').send()
        except Exception as exc:
            await cl.Message(content=f'文件选择失败：{exc}').send()
        return
    if cmd == '/upload':
        # Copy an uploaded recording into an allowlisted location. No path or
        # filename supplied by the browser is trusted for final output paths.
        files = await cl.AskFileMessage(
            content='选择一个 EEG 文件（最大 512 MB；vhdr 等多文件格式建议使用 /file）',
            accept={'application/octet-stream': sorted(SUPPORTED_RAW)},
            max_size_mb=512, max_files=1,
        ).send()
        if not files:
            await cl.Message(content='未收到文件。').send()
            return
        f = files[0]
        source = Path(f.path)
        extension = Path(f.name).suffix.lower()
        if extension not in SUPPORTED_RAW or not source.is_file() or source.stat().st_size > 512*1024*1024:
            await cl.Message(content='文件扩展名、大小或临时文件无效。').send()
            return
        folder = OUTPUT_ROOT / 'uploads'
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / (uuid.uuid4().hex + extension)
        await asyncio.to_thread(shutil.copyfile, source, dest)
        s.set_file(dest)
        await cl.Message(content=f'上传完成，已选择本地 EEG 文件：`{dest}`。输入 `/agent` 开始分析。').send()
        return
    if cmd == '/agent':
        cl.user_session.set('mode','agent')
        await cl.Message(content=f'已进入 EEG Agent 模式，策略：`{s.strategy}`').send()
        return
    if cmd == '/chat':
        cl.user_session.set('mode','chat')
        await cl.Message(content='已进入普通 Qwen 聊天模式。').send()
        return
    if cmd == '/strategy':
        try:
            s.set_strategy(tail.strip())
            await cl.Message(content=f'已切换 LangGraph 策略：`{s.strategy}`').send()
        except ValueError as exc:
            await cl.Message(content=str(exc)).send()
        return
    if cmd == '/clear':
        s.clear()
        await cl.Message(content='聊天上下文已清空。').send()
        return
    if cmd == '/status':
        await cl.Message(content=f'模式：`{cl.user_session.get("mode")}`，策略：`{s.strategy}`，EEG：`{s.current_file or "尚未选择"}`').send()
        return
    if not text:
        return
    mode = cl.user_session.get('mode', 'chat')
    try:
        if mode == 'chat':
            async with cl.Step(name='Qwen local chat', type='llm') as step:
                answer = await asyncio.to_thread(s.chat, text)
                step.output = 'Qwen replied via local vLLM'
            await cl.Message(content=answer).send()
            return
        if not s.current_file:
            await cl.Message(content='请先输入 `/file /mnt/d/EEG/subject.edf`，路径必须在 EEG_DATA_ROOT 中。').send()
            return
        nodes = []
        def on_node(node, update):
            nodes.append((node, update.get('status'), update.get('error')))
        async with cl.Step(name=f'LangGraph · {s.strategy}', type='tool') as step:
            step.input = text
            report = await asyncio.to_thread(s.run_agent, text, on_node)
            step.output = '\n'.join(f'{name}: {status or ("error" if err else "ok")}' for name,status,err in nodes)
        output = f'**状态：{report["status"]}**\n\n{report["answer"]}'
        if report.get('error'):
            output += f'\n\n**执行错误：** {report["error"]}'
        # Only expose run-generated artifacts under the configured output root.
        files = []
        for item in [report.get('manifest')]:
            if item:
                p = Path(item).resolve()
                if p.is_file() and p.is_relative_to(OUTPUT_ROOT):
                    files.append(cl.File(name=p.name, path=str(p)))
        seen = set()
        for entry in report.get('trace', []):
            result = entry.get('result', {})
            if not isinstance(result, dict):
                continue
            for key in ('report_html', 'summary_json'):
                item = result.get(key)
                if not item: continue
                p = Path(item).resolve()
                if p.is_file() and p.is_relative_to(OUTPUT_ROOT) and str(p) not in seen:
                    files.append(cl.File(name=p.name, path=str(p)))
                    seen.add(str(p))
        await cl.Message(content=output, elements=files).send()
    except Exception as exc:
        await cl.Message(content=f'执行失败：`{type(exc).__name__}: {exc}`').send()
