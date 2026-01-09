"""Three-company evidence smoke checks; numeric matches are not semantic accuracy."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from rag.client import Gateway
from rag.config import Settings
from rag.pipeline import RAG
from rag.store import Store


def compact(text):
    return re.sub(r"[\s,]", "", text)


def indicators(response, case):
    sources = response.get('sources', [])
    ranks = [next((i for i, source in enumerate(sources, 1)
                   if source['company'] == case['company'] and value in compact(source['text'])), None)
             for value in case['values']]
    violations = [s['id'] for s in sources if s['company'] != case['company']
                  or not s['publication_date']
                  or not case['date_from'] <= s['publication_date'] <= case['date_to']]
    return dict(numeric_evidence_ranks=ranks, numeric_evidence_pass=all(ranks),
                publication_filter_pass=bool(sources) and not violations, filter_violations=violations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, help='Database to check; defaults to RAG_DB_PATH')
    parser.add_argument('--output', type=Path, default=Path('data/evaluation-new-companies.json'))
    parser.add_argument('--generation', action='store_true', help='Also preserve three Chinese answers for review')
    parser.add_argument('--mode', choices=['hybrid','lexical','dense'], default='hybrid')
    args = parser.parse_args()
    settings = Settings.from_env()
    cases = json.loads((Path(__file__).resolve().parents[1] / 'tests/new_companies_golden.json').read_text())
    report = dict(note='Hand-selected smoke cases. Numeric substring matches after removing whitespace and commas '
                       'do not validate metric, period, units, semantics, or general accuracy. Review full responses.',
                  created_at=datetime.now(timezone.utc).isoformat(), db=str((args.db or settings.db_path).resolve()),
                  cases=cases, retrieval=[], generation=[])
    with Store(args.db or settings.db_path, settings.qdrant_url) as store:
        rag = RAG(store, Gateway(settings))
        report['stats'] = store.stats()
        for case in cases:
            for kind in ('retrieval', 'generation') if args.generation else ('retrieval',):
                options = dict(companies=[case['company']], date_from=case['date_from'], date_to=case['date_to'],
                               mode=args.mode, rewrite=kind == 'generation', top_k=10)
                if kind == 'generation':
                    options['language'] = 'zh-CN'
                question = case['question'] if kind == 'retrieval' else case['chinese_question']
                result = dict(id=case['id'], question=question, options=options)
                try:
                    response = (rag.search if kind == 'retrieval' else rag.ask)(question, **options)
                    result.update(indicators(response, case), response=response)
                    if kind == 'generation':
                        result['answer_contains_expected_numbers'] = all(v in compact(response['answer']) for v in case['values'])
                        result['has_verified_citations'] = bool(response['citations'])
                        result['insufficient_evidence'] = response['insufficient_evidence']
                except Exception as exc:
                    result['error'] = f'{type(exc).__name__}: {exc}'
                report[kind].append(result)
                print(kind, case['id'], {k: v for k, v in result.items() if k.endswith('_pass') or k == 'error'}, flush=True)
        try:
            rag.search('net sales', companies=['__UNKNOWN_SMOKE_COMPANY__'], mode='lexical', rewrite=False)
        except ValueError as exc:
            report['unknown_company_filter'] = dict(passed='Unknown company filter' in str(exc), error=str(exc))
        else:
            report['unknown_company_filter'] = dict(passed=False, error='Unknown company filter was accepted')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Full responses saved to {args.output.resolve()}', flush=True)
    assert report['unknown_company_filter']['passed'], 'Unknown company filter must raise ValueError'
    return int(any('error' in r or not r['numeric_evidence_pass'] or not r['publication_filter_pass']
                   for r in report['retrieval'] + report['generation']))


if __name__ == '__main__':
    raise SystemExit(main())
