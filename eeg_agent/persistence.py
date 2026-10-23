"""Local SQLite session/run index; LangGraph checkpoint blobs live in a separate DB.

No model, UI or LangGraph import is needed for this storage module.
"""
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, directory: str | Path):
        self.path = Path(directory).resolve() / 'agent_sessions.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, strategy TEXT NOT NULL,
                current_file TEXT, model TEXT NOT NULL, base_url TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                thread_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
                task TEXT NOT NULL, input_file TEXT NOT NULL, strategy TEXT NOT NULL,
                model TEXT NOT NULL, status TEXT NOT NULL, manifest TEXT,
                error TEXT, pending_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS ix_runs_session ON runs(session_id, updated_at);
            CREATE INDEX IF NOT EXISTS ix_msg_session ON messages(session_id, id);
            ''')

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=30000')
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    def create(self, *, strategy: str, model: str, base_url: str, session_id=None):
        sid = session_id or uuid.uuid4().hex
        now = utcnow()
        with self.connect() as db:
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?,?)',
                       (sid, strategy, None, model, base_url, now, now))
        return sid

    def get(self, session_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
        if row is None:
            raise ValueError(f'Unknown session_id: {session_id}')
        return dict(row)

    def update_session(self, session_id, **fields):
        allowed = {'strategy', 'current_file', 'model', 'base_url'}
        if not set(fields) <= allowed:
            raise ValueError('Unsupported session fields')
        if not fields:
            return
        fields['updated_at'] = utcnow()
        with self.connect() as db:
            db.execute('UPDATE sessions SET '+', '.join(f'{k}=?' for k in fields)+' WHERE session_id=?',
                       (*fields.values(),session_id))

    def list_sessions(self, limit=50):
        with self.connect() as db:
            rows=db.execute('SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?', (min(max(int(limit),1),200),)).fetchall()
        return [dict(x) for x in rows]

    def append_turn(self, session_id, user, assistant):
        now=utcnow()
        with self.connect() as db:
            db.executemany('INSERT INTO messages(session_id,role,content,created_at) VALUES (?,?,?,?)',
                           [(session_id,'user',user,now),(session_id,'assistant',assistant,now)])
            db.execute('UPDATE sessions SET updated_at=? WHERE session_id=?',(now,session_id))

    def history(self, session_id, limit=10):
        with self.connect() as db:
            rows=db.execute('SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?',
                            (session_id,2*min(max(int(limit),1),100))).fetchall()
        rows=list(reversed(rows)); pairs=[]
        for i in range(0,len(rows)-1,2):
            if rows[i]['role']=='user' and rows[i+1]['role']=='assistant':
                pairs.append((rows[i]['content'],rows[i+1]['content']))
        return pairs

    def messages(self, session_id, limit=None):
        with self.connect() as db:
            if limit is None:
                rows=db.execute('SELECT role,content,created_at FROM messages WHERE session_id=? ORDER BY id',
                                (session_id,)).fetchall()
            else:
                rows=db.execute('SELECT role,content,created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?',
                                (session_id,min(max(int(limit),1),1000))).fetchall()
                rows=list(reversed(rows))
        return [dict(x) for x in rows]

    def clear_history(self, session_id):
        with self.connect() as db:
            db.execute('DELETE FROM messages WHERE session_id=?',(session_id,))

    def create_run(self, *, session_id, thread_id, task, input_file, strategy, model):
        now=utcnow()
        with self.connect() as db:
            db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (thread_id,session_id,task,input_file,strategy,model,'running',None,None,None,now,now))
        return thread_id

    def get_run(self, thread_id):
        with self.connect() as db:
            row=db.execute('SELECT * FROM runs WHERE thread_id=?',(thread_id,)).fetchone()
        if row is None:
            raise ValueError(f'Unknown task thread: {thread_id}')
        d=dict(row)
        d['pending'] = json.loads(d.pop('pending_json') or 'null')
        return d

    def list_runs(self, session_id=None, limit=50):
        with self.connect() as db:
            if session_id:
                rows=db.execute('SELECT * FROM runs WHERE session_id=? ORDER BY updated_at DESC LIMIT ?',
                                (session_id,min(max(int(limit),1),200))).fetchall()
            else:
                rows=db.execute('SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?',
                                (min(max(int(limit),1),200),)).fetchall()
        return [{**dict(r),'pending':json.loads(r['pending_json'] or 'null')} for r in rows]

    def update_run(self, thread_id, *, status, manifest=None, error=None, pending=None):
        with self.connect() as db:
            db.execute('UPDATE runs SET status=?,manifest=?,error=?,pending_json=?,updated_at=? WHERE thread_id=?',
                       (status,manifest,error,json.dumps(pending,ensure_ascii=False,default=str) if pending is not None else None,utcnow(),thread_id))
