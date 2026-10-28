"""Small real business tools confined to this application's agent-data directory.

No shell, network, mail, arbitrary paths, or permanent-delete tools are exposed.
Successful writes, revision history, idempotent results and audit summaries commit
in one SQLite transaction. Export files contain UTF-8 text with no generated code.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3


class AgentToolError(RuntimeError):
    def __init__(self, code, message, details=None, status=422, retryable=False):
        super().__init__(message)
        self.code, self.details = code, details or {}
        self.status, self.retryable = status, retryable

    def result(self):
        error = {'code': self.code, 'message': str(self), 'retryable': self.retryable}
        if self.details:
            error['details'] = self.details
        return {'ok': False, 'error': error}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class ToolBox:
    LIMITS = {'documents': 200, 'tasks': 200, 'title_characters': 120,
              'document_characters': 20000, 'task_detail_characters': 4000}
    MUTATING = frozenset(('document_save', 'document_export', 'task_add', 'task_update'))
    TOOLS = frozenset(('workspace_list', 'document_read')) | MUTATING

    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise AgentToolError('invalid_workspace', 'APP 根目录不存在。')
        self.data_dir = self.root / 'agent-data'
        self.exports_dir = self.data_dir / 'exports'
        self.db_path = self.data_dir / 'workspace.sqlite3'
        self._check_layout()
        self.data_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    revision INTEGER NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL,
                    saved_at TEXT NOT NULL, PRIMARY KEY(document_id, revision)
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, detail TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('todo','in_progress','done')),
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS exports (
                    token TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                    revision INTEGER NOT NULL, filename TEXT NOT NULL UNIQUE,
                    payload BLOB NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(document_id, revision)
                );
                CREATE TABLE IF NOT EXISTS operations (
                    operation_key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                    tool TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL UNIQUE,
                    tool TEXT NOT NULL, at TEXT NOT NULL, result_summary_json TEXT NOT NULL
                );
                PRAGMA user_version=1;
                COMMIT;
            ''')

    def _check_layout(self):
        # A pre-existing symlink/junction must never redirect this APP into
        # another user folder. There is no path supplied by a tool caller.
        for path in (self.data_dir, self.exports_dir, self.db_path):
            if path.resolve() != path:
                raise AgentToolError('unsafe_workspace', 'APP 数据路径不能是符号链接或目录联接。', status=409)

    def _connect(self):
        self._check_layout()
        connection = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA busy_timeout=10000')
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    @staticmethod
    def _id(value, kind):
        if not isinstance(value, str) or not re.fullmatch(kind + r'_[a-f0-9]{32}', value):
            raise AgentToolError('invalid_argument', '条目编号格式不正确。', {'field': 'id'})
        return value

    @staticmethod
    def _integer(value, field, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 2147483647:
            raise AgentToolError('invalid_argument', '版本号必须是有效整数。', {'field': field, 'minimum': minimum})
        return value

    @staticmethod
    def _text(value, field, maximum, title=False):
        if not isinstance(value, str) or len(value) > maximum:
            raise AgentToolError('invalid_argument', '文本字段格式或长度不正确。', {'field': field, 'maximum_characters': maximum})
        try:
            value.encode('utf-8')
        except UnicodeError:
            raise AgentToolError('invalid_argument', '文本包含无效 Unicode 字符。', {'field': field}) from None
        if '\x00' in value or (title and any(ord(ch) < 32 or ord(ch) == 127 for ch in value)):
            raise AgentToolError('invalid_argument', '文本包含不允许的控制字符。', {'field': field})
        if title:
            value = value.strip()
            if not value:
                raise AgentToolError('invalid_argument', '标题不能为空。', {'field': field})
        return value

    def _arguments(self, name, args):
        if not isinstance(args, dict) or any(not isinstance(key, str) for key in args):
            raise AgentToolError('invalid_argument', '工具参数必须是 JSON 对象。')
        allowed = {
            'workspace_list': set(), 'document_read': {'id', 'revision'},
            'document_save': {'id', 'title', 'content', 'expected_revision'},
            'document_export': {'id', 'revision'}, 'task_add': {'title', 'detail'},
            'task_update': {'id', 'status', 'expected_revision'},
        }[name]
        if set(args) - allowed:
            # Never echo unknown values (such as an accidentally supplied key).
            raise AgentToolError('invalid_argument', '工具收到未定义的参数字段。')
        result = {}
        if name == 'workspace_list':
            return result
        if name in ('document_read', 'document_export'):
            result['id'] = self._id(args.get('id'), 'doc')
            revision = args.get('revision')
            result['revision'] = self._integer(revision, 'revision') if revision is not None else None
        elif name == 'document_save':
            result['id'] = self._id(args['id'], 'doc') if args.get('id') is not None else None
            result['title'] = self._text(args.get('title'), 'title', 120, title=True)
            result['content'] = self._text(args.get('content'), 'content', 20000)
            expected = args.get('expected_revision')
            if result['id'] is None:
                if expected is not None:
                    self._integer(expected, 'expected_revision', minimum=0)
                if expected not in (None, 0):
                    raise AgentToolError('invalid_argument', '新建文档请省略版本号或使用 0。', {'field': 'expected_revision'})
                result['expected_revision'] = 0
            else:
                result['expected_revision'] = self._integer(expected, 'expected_revision')
        elif name == 'task_add':
            result['title'] = self._text(args.get('title'), 'title', 120, title=True)
            result['detail'] = self._text(args.get('detail', ''), 'detail', 4000)
        else:
            result['id'] = self._id(args.get('id'), 'task')
            if args.get('status') not in ('todo', 'in_progress', 'done'):
                raise AgentToolError('invalid_argument', '待办状态可选 todo、in_progress、done。', {'field': 'status'})
            result['status'] = args['status']
            result['expected_revision'] = self._integer(args.get('expected_revision'), 'expected_revision')
        return result

    @staticmethod
    def _operation_key(operation_id):
        if not isinstance(operation_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', operation_id):
            raise AgentToolError('invalid_operation_id', '写入需要 1–160 字符的有效操作编号。')
        # Do not persist the raw caller identifier. Audit and recovery correlate
        # its SHA256, even if a caller accidentally used sensitive text as an ID.
        return hashlib.sha256(operation_id.encode('ascii')).hexdigest()

    @staticmethod
    def _document_summary(row):
        return {'id': row['id'], 'title': row['title'], 'revision': row['revision'],
                'content_characters': len(row['content']), 'created_at': row['created_at'],
                'updated_at': row['updated_at']}

    @staticmethod
    def _task_summary(row, detail=False):
        result = {'id': row['id'], 'title': row['title'], 'status': row['status'],
                  'revision': row['revision'], 'detail_characters': len(row['detail']),
                  'created_at': row['created_at'], 'updated_at': row['updated_at']}
        if detail:
            result['detail'] = row['detail']
        return result

    def snapshot(self):
        with closing(self._connect()) as connection:
            # One read transaction gives summaries from the same committed view.
            connection.execute('BEGIN')
            documents = [self._document_summary(row) for row in connection.execute('SELECT * FROM documents ORDER BY updated_at DESC,id')]
            tasks = [self._task_summary(row) for row in connection.execute('SELECT * FROM tasks ORDER BY updated_at DESC,id')]
            export_count = connection.execute('SELECT COUNT(*) FROM exports').fetchone()[0]
            connection.commit()
        return {'documents': documents, 'tasks': tasks,
                'counts': {'documents': len(documents), 'tasks': len(tasks), 'exports': export_count},
                'limits': deepcopy(self.LIMITS)}

    def operation(self, operation_id):
        key = self._operation_key(operation_id)
        with closing(self._connect()) as connection:
            row = connection.execute('SELECT result_json FROM operations WHERE operation_key=?', (key,)).fetchone()
        if row is None:
            return None
        result = json.loads(row['result_json'])
        result['replayed'] = True
        return result

    def _read_document(self, connection, document_id, revision=None):
        row = connection.execute('SELECT * FROM documents WHERE id=?', (document_id,)).fetchone()
        if row is None:
            raise AgentToolError('not_found', '文档不存在。', {'id': document_id}, status=404)
        result = self._document_summary(row)
        result['current_revision'] = row['revision']
        if revision is None or revision == row['revision']:
            result['content'] = row['content']
        else:
            version = connection.execute('SELECT * FROM document_versions WHERE document_id=? AND revision=?', (document_id, revision)).fetchone()
            if version is None:
                raise AgentToolError('revision_not_found', '文档历史版本不存在。', {'id': document_id, 'revision': revision}, status=404)
            result.update(title=version['title'], content=version['content'], revision=revision,
                          content_characters=len(version['content']), updated_at=version['saved_at'])
        return result

    @staticmethod
    def _cas(row, expected):
        if row['revision'] != expected:
            raise AgentToolError('revision_conflict', '条目已被更新；请先重新读取当前版本，再决定是否修改。',
                {'id': row['id'], 'expected_revision': expected, 'current_revision': row['revision']}, status=409)

    def _mutate(self, connection, name, args, created_files):
        now = _now()
        if name == 'document_save':
            document_id = args['id']
            created = document_id is None
            if created:
                if connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] >= 200:
                    raise AgentToolError('capacity_reached', '文档数量已达到 200 条上限。', {'kind': 'documents', 'limit': 200}, status=409)
                document_id = 'doc_' + secrets.token_hex(16)
                connection.execute('INSERT INTO documents VALUES (?,?,?,?,?,?)',
                    (document_id, args['title'], args['content'], 1, now, now))
                revision = 1
            else:
                old = connection.execute('SELECT * FROM documents WHERE id=?', (document_id,)).fetchone()
                if old is None:
                    raise AgentToolError('not_found', '文档不存在。', {'id': document_id}, status=404)
                self._cas(old, args['expected_revision'])
                revision = old['revision'] + 1
                connection.execute('UPDATE documents SET title=?,content=?,revision=?,updated_at=? WHERE id=? AND revision=?',
                    (args['title'], args['content'], revision, now, document_id, old['revision']))
            connection.execute('INSERT INTO document_versions VALUES (?,?,?,?,?)',
                (document_id, revision, args['title'], args['content'], now))
            row = connection.execute('SELECT * FROM documents WHERE id=?', (document_id,)).fetchone()
            return {'ok': True, 'document': self._document_summary(row), 'created': created}, {'entity': 'document', 'id': document_id, 'revision': revision, 'created': created}
        if name == 'task_add':
            if connection.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] >= 200:
                raise AgentToolError('capacity_reached', '待办数量已达到 200 条上限。', {'kind': 'tasks', 'limit': 200}, status=409)
            task_id = 'task_' + secrets.token_hex(16)
            connection.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?)', (task_id, args['title'], args['detail'], 'todo', 1, now, now))
            row = connection.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            return {'ok': True, 'task': self._task_summary(row, True)}, {'entity': 'task', 'id': task_id, 'revision': 1}
        if name == 'task_update':
            row = connection.execute('SELECT * FROM tasks WHERE id=?', (args['id'],)).fetchone()
            if row is None:
                raise AgentToolError('not_found', '待办不存在。', {'id': args['id']}, status=404)
            self._cas(row, args['expected_revision'])
            revision = row['revision'] + 1
            connection.execute('UPDATE tasks SET status=?,revision=?,updated_at=? WHERE id=? AND revision=?',
                (args['status'], revision, now, args['id'], row['revision']))
            row = connection.execute('SELECT * FROM tasks WHERE id=?', (args['id'],)).fetchone()
            return {'ok': True, 'task': self._task_summary(row, True)}, {'entity': 'task', 'id': args['id'], 'revision': revision, 'status': args['status']}
        document = self._read_document(connection, args['id'], args['revision'])
        exported = connection.execute('SELECT * FROM exports WHERE document_id=? AND revision=?', (args['id'], document['revision'])).fetchone()
        if exported is None:
            token = secrets.token_hex(24)
            filename = f"document-{args['id'][4:]}-r{document['revision']}-{token[:12]}.txt"
            payload = document['content'].encode('utf-8')
            digest = hashlib.sha256(payload).hexdigest()
            self._check_layout()
            path = self.exports_dir / filename
            if path.resolve() != path:
                raise AgentToolError('unsafe_workspace', '导出路径校验失败。', status=409)
            # Token is not published before COMMIT. A file-write/commit failure
            # rolls back SQLite and removes only this freshly generated file.
            with path.open('xb') as handle:
                created_files.append(path)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            connection.execute('INSERT INTO exports VALUES (?,?,?,?,?,?,?)',
                (token, args['id'], document['revision'], filename, payload, digest, now))
            exported = connection.execute('SELECT * FROM exports WHERE token=?', (token,)).fetchone()
        result = {'token': exported['token'], 'filename': exported['filename'],
                  'mime_type': 'text/plain; charset=utf-8', 'size_bytes': len(exported['payload']),
                  'document_id': args['id'], 'revision': exported['revision'], 'sha256': exported['sha256'],
                  'download_url': '/api/agent/exports/' + exported['token']}
        return {'ok': True, 'export': result}, {'entity': 'export', 'document_id': args['id'], 'revision': exported['revision']}

    def invoke(self, name, args, operation_id=None):
        connection = None
        created_files = []
        committed = False
        try:
            if not isinstance(name, str) or name not in self.TOOLS:
                raise AgentToolError('unknown_tool', '该工具未开放。')
            if isinstance(args, dict):
                args = dict(args)
            values = self._arguments(name, args)
            if name == 'workspace_list':
                return {'ok': True, 'workspace': self.snapshot()}
            if name == 'document_read':
                with closing(self._connect()) as read_connection:
                    read_connection.execute('BEGIN')
                    document = self._read_document(read_connection, values['id'], values['revision'])
                    read_connection.commit()
                return {'ok': True, 'document': document}
            key = self._operation_key(operation_id)
            fingerprint = hashlib.sha256(_json({'tool': name, 'arguments': args}).encode('utf-8')).hexdigest()
            connection = self._connect()
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('SELECT fingerprint,result_json FROM operations WHERE operation_key=?', (key,)).fetchone()
            if previous is not None:
                if previous['fingerprint'] != fingerprint:
                    raise AgentToolError('operation_conflict', '同一操作编号已用于不同参数；请核对已提交结果。', status=409)
                result = json.loads(previous['result_json'])
                result['replayed'] = True
                connection.rollback()
                return result
            result, summary = self._mutate(connection, name, values, created_files)
            result['replayed'] = False
            now = _now()
            connection.execute('INSERT INTO operations VALUES (?,?,?,?,?)', (key, fingerprint, name, _json(result), now))
            connection.execute('INSERT INTO audit(operation_id,tool,at,result_summary_json) VALUES (?,?,?,?)',
                ('sha256:' + key, name, now, _json(summary)))
            connection.commit()
            committed = True
            return result
        except AgentToolError as error:
            return error.result()
        except (sqlite3.Error, OSError, UnicodeError, ValueError, TypeError):
            # Do not expose raw SQL, paths, arguments, user text or credentials.
            return AgentToolError('storage_unavailable', '本地工作区暂时不可用，本次操作未确认提交；可按操作编号查询结果后重试。', status=503, retryable=True).result()
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.rollback()
                connection.close()
            if not committed:
                for path in created_files:
                    try:
                        if path.parent == self.exports_dir and path.resolve() == path:
                            path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def get_export(self, token):
        if not isinstance(token, str) or not re.fullmatch(r'[a-f0-9]{48}', token):
            return None
        with closing(self._connect()) as connection:
            row = connection.execute('SELECT filename,payload,sha256 FROM exports WHERE token=?', (token,)).fetchone()
        if row is None:
            return None
        if not re.fullmatch(r'document-[a-f0-9]{32}-r[0-9]+-[a-f0-9]{12}\.txt', row['filename']):
            return None
        payload = bytes(row['payload'])
        if hashlib.sha256(payload).hexdigest() != row['sha256']:
            return None
        # Downloads use the transactionally saved bytes, never a caller path or
        # an externally modified file in exports/. That directory keeps a copy.
        return payload, row['filename']

    @staticmethod
    def schemas():
        def tool(name, description, properties, required=()):
            return {'type': 'function', 'function': {'name': name, 'description': description,
                'parameters': {'type': 'object', 'properties': properties,
                               'required': list(required), 'additionalProperties': False}}}
        identifier = {'type': 'string', 'description': '工作区返回的条目 id；不能是文件路径。'}
        revision = {'type': 'integer', 'minimum': 1, 'maximum': 2147483647}
        title = {'type': 'string', 'minLength': 1, 'maxLength': 120}
        return [
            tool('workspace_list', '列出本 APP 文档和待办摘要，不读取用户其他文件。', {}),
            tool('document_read', '读取一份文档，省略 revision 时读取当前版；指定 revision 可读取保留的历史版本。',
                 {'id': identifier, 'revision': revision}, ('id',)),
            tool('document_save', '新建或保存文档。新建省略 id 和 expected_revision（或版本0）；更新必须提供 id 和当前 expected_revision，冲突时先重读。旧版本永久保留。',
                 {'id': identifier, 'title': title, 'content': {'type': 'string', 'maxLength': 20000},
                  'expected_revision': {'type': 'integer', 'minimum': 0, 'maximum': 2147483647}}, ('title', 'content')),
            tool('document_export', '将当前或指定历史版本导出为 UTF-8 txt，返回下载 token 和安全文件名。不能指定路径。',
                 {'id': identifier, 'revision': revision}, ('id',)),
            tool('task_add', '在本 APP 新建待办，初始状态 todo。',
                 {'title': title, 'detail': {'type': 'string', 'maxLength': 4000}}, ('title',)),
            tool('task_update', '按当前版本修改待办状态，不永久删除。冲突时先读取 workspace_list 获取当前版本。',
                 {'id': identifier, 'status': {'type': 'string', 'enum': ['todo', 'in_progress', 'done']},
                  'expected_revision': revision}, ('id', 'status', 'expected_revision')),
        ]
