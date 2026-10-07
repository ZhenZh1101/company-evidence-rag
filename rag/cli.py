import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings
from qdrant_client.http.exceptions import ApiException
from .config import Settings
from .client import Gateway
from .store import Store, digest
from .pipeline import RAG


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def doctor(settings):
    from qdrant_client import QdrantClient
    report = dict(vector_backend='qdrant', errors={})
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=10)
        try:
            collections = client.get_collections().collections
            report['qdrant'] = dict(status='ok', collections=len(collections))
        finally:
            client.close()
    except Exception as exc:
        report['errors']['qdrant'] = f'{type(exc).__name__}: check RAG_QDRANT_URL and start Qdrant with docker compose up -d.'
    gateway = Gateway(settings)
    try:
        vectors = gateway.embed(['The company revenue increased ten percent.', 'Corporate sales grew by 10%.', 'Bananas are tropical fruit.'])
        report.update(embedding_dimension=vectors.shape[1], related_cosine=float(vectors[0]@vectors[1]),
                      unrelated_cosine=float(vectors[0]@vectors[2]))
    except (ValueError, RuntimeError, OSError) as exc:
        report['errors']['embedding'] = str(exc)
    try:
        prompt = 'Return JSON: {"status":"OK"}.' if settings.chat_response_format == 'json_object' else 'Reply with only OK.'
        report['chat'] = gateway.chat([{'role':'user','content':prompt}], max_tokens=16)
    except (ValueError, RuntimeError, OSError) as exc:
        report['errors']['chat'] = str(exc)
    output(report)
    return bool(report['errors'])


def ingest(args, store):
    from .ingest import discover, extract, chunk_segments
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), imported=0, unchanged=0, failures=[], discovery_errors=[], warnings=[], skipped=[], roots=[])
    parsed = {}
    for root_arg in args.paths:
        root = Path(root_arg).expanduser().resolve()
        sources, skipped = discover(root, args.company)
        report['skipped'].extend(skipped)
        report['discovery_errors'].extend(entry for entry in skipped if entry.get('kind') == 'error')
        report['roots'].append(str(root))
        if args.limit:
            sources = sources[:args.limit]
        for i, source in enumerate(sources, 1):
            try:
                with source.path.open('rb') as stream:
                    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                ocr = getattr(args, 'ocr', False)
                parser_version = 'html-v3|' if source.path.suffix.lower() in {'.html', '.htm'} else ('parser-v2-ocr|' if ocr else 'parser-v1|')
                fingerprint = digest(parser_version + sha + json.dumps(asdict(source), default=str, sort_keys=True))
                if store.unchanged(source.key, fingerprint):
                    report['unchanged'] += 1
                    continue
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if sha not in parsed:
                        parsed[sha] = chunk_segments(extract(source, ocr=ocr))
                    segments = parsed[sha]
                report['warnings'].extend(dict(path=str(source.path), message=str(w.message)) for w in caught)
                if not segments:
                    raise ValueError('No indexable text (scanned/image-only or unsupported content).')
                store.put(source, root, fingerprint, sha, segments)
                report['imported'] += 1
            except Exception as exc:
                report['failures'].append(dict(path=str(source.path), error=f'{type(exc).__name__}: {exc}'))
            if i % 25 == 0 or i == len(sources):
                print(f'{root.name}: {i}/{len(sources)} documents; imported={report["imported"]}, failed={len(report["failures"])}', file=sys.stderr, flush=True)
        discovery_errors = [entry for entry in skipped if entry.get('kind') == 'error']
        if args.prune and discovery_errors:
            report['warnings'].append(dict(path=str(root),message='Pruning disabled for undiscovered sources because discovery had errors; only explicitly excluded files may be removed.'))
        if args.prune and not args.limit:
            keep = {s.key for s in sources}
            excluded = {entry['path'] for entry in skipped if entry.get('kind') != 'error'}
            old = [r['id'] for r in store.db.execute('SELECT id,path FROM documents WHERE root=?', (str(root),))
                   if r['id'] not in keep and (not discovery_errors or r['path'] in excluded)]
            with store.db:
                store.db.executemany('DELETE FROM documents WHERE id=?', [(key,) for key in old])
            report.setdefault('pruned', 0)
            report['pruned'] += len(old)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['stats'] = store.stats()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    output({k: v if k not in ('failures','discovery_errors','warnings','skipped') else len(v) for k,v in report.items()})
    return bool(report['failures'] or report['discovery_errors'])


def main():
    parser = argparse.ArgumentParser(description='Evidence-grounded company RAG')
    parser.add_argument('--db', type=Path, help='Override RAG_DB_PATH')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Verify Qdrant, chat and embedding endpoints independently')
    sub.add_parser('stats')
    load = sub.add_parser('ingest', help='Import archive indexes, folders or individual supported files')
    load.add_argument('paths', nargs='+')
    load.add_argument('--company', help='Company ticker for generic files/folders')
    load.add_argument('--limit', type=int, help='Explicit partial import: first N selected documents per root')
    load.add_argument('--prune', action='store_true', help='Remove sources no longer present in these roots')
    load.add_argument('--ocr', action='store_true', help='Use local Poppler + Tesseract (English) for blank PDF pages')
    load.add_argument('--report', default='data/ingest-report.json')
    embed = sub.add_parser('embed', help='Resume embedding all pending chunks')
    embed.add_argument('--batch-size', type=int, default=64)
    sync = sub.add_parser('sync-vectors', help='Copy cached embeddings and pending changes to Qdrant without model calls')
    sync.add_argument('--batch-size', type=int, default=128)
    for name in ('search','ask'):
        p = sub.add_parser(name)
        p.add_argument('question')
        p.add_argument('--company', action='append', dest='companies')
        p.add_argument('--category', action='append', dest='categories')
        p.add_argument('--date-from')
        p.add_argument('--date-to')
        p.add_argument('--top-k', type=int, default=10)
        p.add_argument('--no-rewrite', action='store_true')
        p.add_argument('--mode', choices=['hybrid','lexical','dense'], default='hybrid')
        p.add_argument('--language', choices=['en', 'zh-CN'], default='en', help='Answer and warning language (default: en)')
    serve = sub.add_parser('serve', help='Serve local web UI and API')
    serve.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.db:
            settings = replace(settings, db_path=args.db)
        if args.command == 'serve':
            import uvicorn
            from .web import create_app
            uvicorn.run(create_app(settings), host='127.0.0.1', port=args.port)
            return
        if args.command == 'doctor':
            if doctor(settings):
                sys.exit(1)
            return
        with Store(settings.db_path, settings.qdrant_url) as store:
            if args.command == 'stats':
                output(store.stats())
            elif args.command == 'ingest':
                if args.limit is not None and args.limit < 1:
                    raise ValueError('--limit must be positive')
                if ingest(args, store):
                    sys.exit(2)
            elif args.command == 'embed':
                if not 1 <= args.batch_size <= 256:
                    raise ValueError('--batch-size must be 1–256')
                count = store.embed_pending(Gateway(settings), args.batch_size, lambda n: print(f'Embedded {n} new unique chunks', file=sys.stderr, flush=True))
                output(dict(new_vectors=count, **store.stats()))
            elif args.command == 'sync-vectors':
                if not 1 <= args.batch_size <= 256:
                    raise ValueError('--batch-size must be 1–256')
                count = store.sync_vectors(args.batch_size, lambda n: print(f'Synced {n} vector changes to Qdrant', file=sys.stderr, flush=True))
                output(dict(synced_vectors=count, **store.stats()))
            else:
                options = {k:getattr(args,k) for k in ('companies','categories','date_from','date_to','top_k','mode','language')}
                options['rewrite'] = not args.no_rewrite
                output(getattr(RAG(store,Gateway(settings)),args.command)(args.question, **options))
    except ApiException as exc:
        print(f'Error: Qdrant request failed ({type(exc).__name__}). Check RAG_QDRANT_URL and start Qdrant with docker compose up -d.', file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
