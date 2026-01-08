"""SQLite owns source provenance, full-text index and resumable embedding cache."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import numpy as np


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Store:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
          id TEXT PRIMARY KEY,root TEXT NOT NULL,fingerprint TEXT NOT NULL,content_hash TEXT NOT NULL,
          company TEXT NOT NULL,title TEXT NOT NULL,category TEXT NOT NULL,publication_date TEXT,
          publication_period TEXT,date_basis TEXT,scope TEXT,source_url TEXT,path TEXT,aliases TEXT);
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL,text TEXT NOT NULL,locator TEXT NOT NULL,text_hash TEXT NOT NULL,
          UNIQUE(document_id,ordinal));
        CREATE TABLE IF NOT EXISTS company_aliases(
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          alias TEXT NOT NULL, PRIMARY KEY(document_id,alias));
        CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(text_hash);
        CREATE INDEX IF NOT EXISTS documents_filter ON documents(company,publication_date,category);
        CREATE TABLE IF NOT EXISTS embeddings(text_hash TEXT PRIMARY KEY,dimension INTEGER NOT NULL,vector BLOB NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(title,text,tokenize='unicode61');
        CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
          DELETE FROM chunks_fts WHERE rowid=old.id;
        END;
        ''')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def stats(self):
        documents = self.db.execute('SELECT count(*) FROM documents').fetchone()[0]
        chunks = self.db.execute('SELECT count(*) FROM chunks').fetchone()[0]
        indexed = self.db.execute('SELECT count(*) FROM chunks JOIN embeddings USING(text_hash)').fetchone()[0]
        companies = [dict(r) for r in self.db.execute('''SELECT d.company,count(DISTINCT d.id) documents,
                      count(c.id) chunks FROM documents d LEFT JOIN chunks c ON d.id=c.document_id GROUP BY d.company''')]
        return dict(documents=documents, chunks=chunks, indexed_chunks=indexed, companies=companies,
                    embedding_identity=self.meta('embedding_identity'))

    def meta(self, key):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return row[0] if row else None

    def company_aliases(self):
        companies = {}
        for row in self.db.execute('''SELECT DISTINCT d.company,a.alias FROM documents d
                                     LEFT JOIN company_aliases a ON a.document_id=d.id'''):
            names = companies.setdefault(row['company'], [])
            if row['alias'] is not None:
                names.append(row['alias'])
        return companies

    def check_embedding_identity(self, gateway):
        identity = gateway.settings.base_url + '|' + gateway.settings.embedding_model
        if gateway.settings.openclaw_embedding_model:
            identity += '|' + gateway.settings.openclaw_embedding_model
        previous = self.meta('embedding_identity')
        populated = self.db.execute('SELECT 1 FROM embeddings LIMIT 1').fetchone()
        if previous and previous != identity and populated:
            raise ValueError('Embedding endpoint/model differs from this index. Use a new RAG_DB_PATH or rebuild vectors explicitly.')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('embedding_identity', identity))

    def unchanged(self, key, fingerprint):
        row = self.db.execute('SELECT fingerprint FROM documents WHERE id=?', (key,)).fetchone()
        return bool(row and row[0] == fingerprint)

    def put(self, source, root, fingerprint, content_hash, segments):
        values = (source.key, str(Path(root).resolve()), fingerprint, content_hash, source.company, source.title,
                  source.category, source.publication_date, source.publication_period, source.date_basis,
                  source.scope, source.source_url, str(source.path), json.dumps(source.aliases, ensure_ascii=False))
        with self.db:
            self.db.execute('DELETE FROM documents WHERE id=?', (source.key,))
            self.db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
            self.db.executemany('INSERT INTO company_aliases VALUES (?,?)',
                                [(source.key, name) for name in sorted(set(source.company_aliases))])
            for i, segment in enumerate(segments):
                row = self.db.execute('INSERT INTO chunks(document_id,ordinal,text,locator,text_hash) VALUES (?,?,?,?,?)',
                                      (source.key, i, segment.text, segment.locator, digest(segment.text)))
                self.db.execute('INSERT INTO chunks_fts(rowid,title,text) VALUES (?,?,?)',
                                (row.lastrowid, source.company + ' ' + source.title, segment.text))

    def embed_pending(self, gateway, batch_size=64, progress=None):
        self.check_embedding_identity(gateway)
        total = 0
        while True:
            rows = self.db.execute('''SELECT c.text_hash,c.text FROM chunks c LEFT JOIN embeddings e USING(text_hash)
                       WHERE e.text_hash IS NULL GROUP BY c.text_hash LIMIT ?''', (batch_size,)).fetchall()
            if not rows:
                return total
            vectors = gateway.embed([r['text'] for r in rows])
            dimension = self.meta('embedding_dimension')
            if dimension and int(dimension) != vectors.shape[1]:
                raise ValueError('Embedding dimension changed; use a new database or rebuild vectors.')
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                                ('embedding_dimension', str(vectors.shape[1])))
                self.db.executemany('INSERT OR IGNORE INTO embeddings VALUES (?,?,?)',
                    [(r['text_hash'], len(v), v.astype('<f4').tobytes()) for r, v in zip(rows, vectors)])
            total += len(rows)
            if progress:
                progress(total)

    def source(self, chunk_id):
        row = self.db.execute('''SELECT c.id,c.text,c.locator,c.text_hash,c.ordinal,d.id document_id,
                 d.company,d.title,d.category,d.publication_date,d.publication_period,d.date_basis,d.scope,
                 d.source_url,d.path,d.aliases FROM chunks c JOIN documents d ON c.document_id=d.id WHERE c.id=?''',
                              (chunk_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result['aliases'] = json.loads(result['aliases'])
        return result

    @staticmethod
    def filters(companies=None, date_from=None, date_to=None, categories=None):
        clauses, values = [], []
        for column, entries in [('company', companies), ('category', categories)]:
            if entries:
                clauses.append(f'd.{column} IN ({",".join("?" for _ in entries)})')
                values.extend(entries)
        for op, date in [('>=', date_from), ('<=', date_to)]:
            if date:
                clauses.append(f'd.publication_date {op} ?')
                values.append(date)
        return (' AND ' + ' AND '.join(clauses) if clauses else ''), values

    def lexical(self, query, limit=80, **filters):
        stop = {'the','a','an','of','in','to','for','and','or','is','was','were','what','how','did','does','by','with','on','as','its','it','at','from'}
        terms = list(dict.fromkeys(t.lower() for t in re.findall(r'\w+', query) if t.lower() not in stop))[:40]
        if not terms:
            return []
        match = ' OR '.join('"' + t + '"' for t in terms)
        where, values = self.filters(**filters)
        return [r[0] for r in self.db.execute('''SELECT c.id FROM chunks_fts f JOIN chunks c ON f.rowid=c.id
             JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ?''' + where +
             ' ORDER BY bm25(chunks_fts,2.0,1.0),c.id LIMIT ?', [match, *values, limit])]

    def dense(self, vector, limit=80, **filters):
        where, values = self.filters(**filters)
        dimension = self.meta('embedding_dimension')
        if dimension and len(vector) != int(dimension):
            raise ValueError('Query vector dimension does not match the index.')
        # ponytail: exact vector scan in bounded batches; move to ANN when measured latency warrants it.
        cursor = self.db.execute('''SELECT c.id,e.vector FROM chunks c JOIN documents d ON d.id=c.document_id
                          JOIN embeddings e USING(text_hash) WHERE 1=1''' + where, values)
        best = []
        while rows := cursor.fetchmany(2048):
            matrix = np.stack([np.frombuffer(r['vector'], dtype='<f4') for r in rows])
            scores = matrix @ vector
            selected = np.argsort(-scores)[:limit]
            best.extend((float(scores[i]), rows[i]['id']) for i in selected)
            best = sorted(best, key=lambda x: (-x[0], x[1]))[:limit]
        return [item[1] for item in best]
