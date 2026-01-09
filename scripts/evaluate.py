"""Reproducible smoke evaluation, not a statistical financial-QA accuracy claim."""
import argparse
import json
from pathlib import Path
import time
from rag.config import Settings
from rag.client import Gateway
from rag.store import Store
from rag.pipeline import RAG, normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/evaluation.json')
    parser.add_argument('--generation', action='store_true', help='Also run generation checks from the configured golden dataset')
    parser.add_argument('--generation-golden', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'tests/generation_golden.json',
                        help='JSON file containing generation evaluation cases')
    args = parser.parse_args()
    settings = Settings.from_env()
    cases = json.loads((Path(__file__).resolve().parents[1] / 'tests/golden.json').read_text())
    results = []
    with Store(settings.db_path, settings.qdrant_url) as store:
        rag = RAG(store, Gateway(settings))
        for mode in ('lexical', 'dense', 'hybrid'):
            for case in cases:
                start = time.monotonic()
                response = rag.search(case['question'], companies=case['companies'], date_to=case.get('date_to'),
                                      mode=mode, rewrite=False, top_k=10)
                sources = response['sources']
                ranks = []
                for target in case['targets']:
                    rank = next((i for i, s in enumerate(sources, 1) if s['company'] == target['company']
                                 and all(term in normalized(s['text']) for term in target['text'])), None)
                    ranks.append(rank)
                violations = [s['id'] for s in sources if s['company'] not in case['companies']
                              or (case.get('date_to') and (not s['publication_date'] or s['publication_date'] > case['date_to']))]
                results.append(dict(id=case['id'], mode=mode, evidence_ranks=ranks, hit_all=all(ranks),
                                    reciprocal_rank=sum(1/r if r else 0 for r in ranks)/len(ranks),
                                    filter_violations=violations, seconds=round(time.monotonic()-start,3),
                                    sources=[{'id':s['id'],'path':s['path'],'locator':s['locator']} for s in sources]))
                print(mode,case['id'],ranks,flush=True)
        generations = []
        if args.generation:
            questions = json.loads(args.generation_golden.read_text(encoding='utf-8'))
            for case in questions:
                response = rag.ask(case['question'], companies=case['companies'], language=case.get('language', 'zh-CN'))
                generations.append(dict(id=case['id'],question=case['question'],**response))
                print('generation',case['id'],'insufficient=',response['insufficient_evidence'],flush=True)
        summary = {}
        for mode in ('lexical','dense','hybrid'):
            subset = [r for r in results if r['mode']==mode]
            summary[mode] = dict(cases=len(subset), hit_all_at_10=sum(r['hit_all'] for r in subset)/len(subset),
                                 mean_reciprocal_rank=sum(r['reciprocal_rank'] for r in subset)/len(subset),
                                 filter_violations=sum(len(r['filter_violations']) for r in subset))
        report = dict(note='Small, hand-selected smoke set from two earnings releases. Matching numeric evidence is not semantic answer validation or proof of generalization. Generation responses require human review.',
                      stats=store.stats(),summary=summary,retrieval=results,generation=generations)
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
