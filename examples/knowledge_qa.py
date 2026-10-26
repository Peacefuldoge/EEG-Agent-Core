"""Standalone no-EEG-file knowledge Q&A: uses same Qwen3/vLLM through RAG API."""
import argparse
from rag_bridge import RAGClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('question')
    args = ap.parse_args()
    client = RAGClient()
    try:
        result = client.answer(args.question)
        print(result.get('answer', ''))
        for hit in result.get('sources', []):
            print('[%s] %s | DOI=%s | page=%s' % (
                hit.get('citation', '?'), hit.get('title',''), hit.get('doi',''), hit.get('page',0)))
        if result.get('citation_warning'):
            print('WARNING:', result['citation_warning'])
    finally:
        client.close()


if __name__ == '__main__':
    main()
