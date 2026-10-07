"""SQLite owns provenance/FTS/cache; Qdrant owns the searchable vector index."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid
import numpy as np
from qdrant_client import QdrantClient, models


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Store:
    def __init__(self, path, qdrant_url='http://127.0.0.1:6333', *, vector_client=None):
        path = Path(path)
        self.path = path.resolve()
        self.qdrant_url = qdrant_url.rstrip('/')
        self._vectors = vector_client
        self._owns_vectors = vector_client is None
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
        CREATE TABLE IF NOT EXISTS vector_pending(chunk_id INTEGER PRIMARY KEY);
        CREATE TRIGGER IF NOT EXISTS chunks_vector_insert AFTER INSERT ON chunks BEGIN
          INSERT OR IGNORE INTO vector_pending VALUES (new.id);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_vector_delete AFTER DELETE ON chunks BEGIN
          INSERT OR IGNORE INTO vector_pending VALUES (old.id);
        END;
        ''')
        if not self.meta('vector_index_id'):
            with self.db:
                created = self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                                          ('vector_index_id', uuid.uuid4().hex)).rowcount
                if created:
                    self.db.execute('INSERT OR IGNORE INTO vector_pending SELECT id FROM chunks')
        # Copies at different paths and new databases at an old path get separate collections.
        self.collection = 'rag_' + digest(str(self.path) + '|' + self.meta('vector_index_id'))[:32]
        self.vector_target = self.qdrant_url + '|' + self.collection

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        try:
            if self._owns_vectors and self._vectors is not None:
                self._vectors.close()
        finally:
            self.db.close()

    @property
    def vectors(self):
        if self._vectors is None:
            self._vectors = QdrantClient(url=self.qdrant_url, timeout=60)
        return self._vectors

    def stats(self):
        documents = self.db.execute('SELECT count(*) FROM documents').fetchone()[0]
        chunks = self.db.execute('SELECT count(*) FROM chunks').fetchone()[0]
        embedded = self.db.execute('SELECT count(*) FROM chunks JOIN embeddings USING(text_hash)').fetchone()[0]
        pending = self.db.execute('SELECT count(*) FROM vector_pending').fetchone()[0]
        indexed = self.db.execute('''SELECT count(*) FROM chunks c JOIN embeddings USING(text_hash)
                         WHERE NOT EXISTS (SELECT 1 FROM vector_pending p WHERE p.chunk_id=c.id)''').fetchone()[0]
        if self.meta('qdrant_target') != self.vector_target:
            indexed = 0
        companies = [dict(r) for r in self.db.execute('''SELECT d.company,count(DISTINCT d.id) documents,
                      count(c.id) chunks FROM documents d LEFT JOIN chunks c ON d.id=c.document_id GROUP BY d.company''')]
        return dict(documents=documents, chunks=chunks, indexed_chunks=indexed, embedded_chunks=embedded,
                    pending_vector_changes=pending, vector_backend='qdrant', vector_collection=self.collection, companies=companies,
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
        if not 1 <= batch_size <= 256:
            raise ValueError('Embedding batch size must be 1–256.')
        self.check_embedding_identity(gateway)
        self.sync_vectors()
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
            self.sync_vectors()
            total += len(rows)
            if progress:
                progress(total)

    def sync_vectors(self, batch_size=128, progress=None):
        """Replay cached vectors/deletions, acknowledging each batch only after Qdrant."""
        if not 1 <= batch_size <= 256:
            raise ValueError('Vector batch size must be 1–256.')
        dimension = self.meta('embedding_dimension')
        if not dimension:
            return 0
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            exists = self.vectors.collection_exists(self.collection)
            if not exists or self.meta('qdrant_target') != self.vector_target:
                if exists:
                    self.vectors.delete(self.collection, models.FilterSelector(filter=models.Filter()), wait=True)
                self.db.execute('INSERT OR IGNORE INTO vector_pending SELECT id FROM chunks')
                self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('qdrant_target', self.vector_target))
            if not exists:
                self.vectors.create_collection(self.collection, vectors_config=models.VectorParams(
                    size=int(dimension), distance=models.Distance.COSINE, on_disk=True),
                    hnsw_config=models.HnswConfigDiff(on_disk=True))
            info = self.vectors.get_collection(self.collection)
            if (info.config.params.vectors.size != int(dimension)
                    or info.config.params.vectors.distance != models.Distance.COSINE):
                raise ValueError('Qdrant vector dimension/distance does not match this index.')
            for field, schema in [('company', models.PayloadSchemaType.KEYWORD),
                                  ('category', models.PayloadSchemaType.KEYWORD),
                                  ('publication_day', models.PayloadSchemaType.INTEGER)]:
                if field not in info.payload_schema:
                    self.vectors.create_payload_index(self.collection, field, schema, wait=True)
        total = 0
        repaired = False
        while True:
            # Hold the write lock through acknowledgement so a concurrent importer cannot
            # reuse a chunk ID between uploading its vector and clearing the pending row.
            with self.db:
                self.db.execute('BEGIN IMMEDIATE')
                rows = self.db.execute('''SELECT p.chunk_id,c.id,c.text_hash,e.dimension,e.vector,
                          d.company,d.category,d.publication_date FROM vector_pending p
                          LEFT JOIN chunks c ON c.id=p.chunk_id
                          LEFT JOIN documents d ON d.id=c.document_id
                          LEFT JOIN embeddings e ON e.text_hash=c.text_hash
                          WHERE c.id IS NULL OR e.text_hash IS NOT NULL ORDER BY p.chunk_id LIMIT ?''',
                                       (batch_size,)).fetchall()
                if not rows:
                    if self.db.execute('SELECT 1 FROM vector_pending LIMIT 1').fetchone():
                        return total  # Remaining chunks still need embeddings.
                    expected = self.db.execute('SELECT count(*) FROM chunks').fetchone()[0]
                    if self.vectors.count(self.collection, exact=True).count == expected:
                        return total
                    if repaired:
                        raise RuntimeError('Qdrant index remains incomplete after restoring cached vectors.')
                    # Reconcile after draining, including loss concurrent with pending changes.
                    self.vectors.delete(self.collection, models.FilterSelector(filter=models.Filter()), wait=True)
                    self.db.execute('INSERT OR IGNORE INTO vector_pending SELECT id FROM chunks')
                    repaired = True
                    continue
                points, deleted = [], []
                for row in rows:
                    if row['id'] is None:
                        deleted.append(row['chunk_id'])
                        continue
                    vector = np.frombuffer(row['vector'], dtype='<f4')
                    if (row['dimension'] != int(dimension) or len(vector) != int(dimension)
                            or not np.isfinite(vector).all() or not np.any(vector)):
                        raise ValueError('Cached vector dimension/value is invalid; rebuild the embedding cache.')
                    payload = {key: row[key] for key in ('company', 'category', 'text_hash')}
                    if row['publication_date']:
                        payload['publication_day'] = int(row['publication_date'].replace('-', ''))
                    points.append(models.PointStruct(id=row['id'], vector=vector.tolist(), payload=payload))
                if deleted:
                    self.vectors.delete(self.collection, models.PointIdsList(points=deleted), wait=True)
                if points:
                    self.vectors.upsert(self.collection, points, wait=True)
                self.db.executemany('DELETE FROM vector_pending WHERE chunk_id=?', [(r['chunk_id'],) for r in rows])
            total += len(rows)
            if progress:
                progress(total)

    def require_vector_index(self):
        stats = self.stats()
        if (stats['indexed_chunks'] != stats['chunks'] or stats['pending_vector_changes']
                or (stats['chunks'] and self.meta('qdrant_target') != self.vector_target)):
            raise ValueError('Embedding index incomplete. Run company-rag embed, or use lexical mode explicitly.')
        if not stats['chunks']:
            return stats
        if not self.vectors.collection_exists(self.collection):
            raise ValueError('Qdrant index is missing. Run company-rag sync-vectors to restore cached vectors.')
        if self.vectors.count(self.collection, exact=True).count != stats['chunks']:
            raise ValueError('Qdrant index is incomplete. Run company-rag sync-vectors to restore cached vectors.')
        return stats

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

    def dense(self, vector, limit=80, *, _index_checked=False, **filters):
        dimension = self.meta('embedding_dimension')
        if dimension and len(vector) != int(dimension):
            raise ValueError('Query vector dimension does not match the index.')
        if limit < 1:
            return []
        if not _index_checked:
            stats = self.require_vector_index()
            if not stats['chunks']:
                return []
        conditions = []
        for column, key in [('company', 'companies'), ('category', 'categories')]:
            if filters.get(key):
                conditions.append(models.FieldCondition(key=column, match=models.MatchAny(any=filters[key])))
        bounds = {op: int(filters[key].replace('-', '')) for op, key in [('gte', 'date_from'), ('lte', 'date_to')]
                  if filters.get(key)}
        if bounds:
            conditions.append(models.FieldCondition(key='publication_day', range=models.Range(**bounds)))
        hits = self.vectors.query_points(self.collection, query=np.asarray(vector).tolist(),
                    query_filter=models.Filter(must=conditions) if conditions else None,
                    limit=limit, with_payload=['text_hash'], with_vectors=False).points
        for hit in hits:
            row = self.db.execute('SELECT text_hash FROM chunks WHERE id=?', (hit.id,)).fetchone()
            if not row or row['text_hash'] != hit.payload.get('text_hash'):
                with self.db:
                    self.db.execute('INSERT OR IGNORE INTO vector_pending VALUES (?)', (hit.id,))
                raise ValueError('Qdrant source mismatch. Run company-rag sync-vectors before searching.')
        return [hit.id for hit in sorted(hits, key=lambda hit: (-hit.score, hit.id))]
