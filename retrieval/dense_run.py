"""
Dense retrieval with all-MiniLM-L6-v2, written out as a TREC run file.

Same setup as app.py, so offline scores match what the app does:
    - model sentence-transformers/all-MiniLM-L6-v2
    - passage text is "title. text" when a title exists
    - embeddings normalised, so the score is cosine similarity (-1 to 1)

The run holds the full ranking with no similarity floor. Declining a question
is scored separately in ndcg_eval.py with --abstain-below, so the threshold
never distorts NDCG.

Usage (Walert now):
    python dense_run.py \
        --corpus  walert/quantitative_eval/data/collection.jsonl \
        --topics  walert/quantitative_eval/data/topics.csv \
        --output  runs/walert.minilm.txt

Usage (our corpus, once Kripa's file lands):
    python dense_run.py \
        --corpus  ../data/processed/passages.jsonl \
        --topics  ../data/questions.jsonl \
        --output  runs/fwo.minilm.txt

Deduplication experiment: --max-per-source N keeps at most N passages from
each source page (source_id), walking down the ranking until k are kept.
0 (the default) means no limit. The top passage is never removed, so the
abstention threshold behaves the same with or without dedup.
    python dense_run.py ... --max-per-source 1 --output runs/fwo.minilm.dedup1.txt

First run downloads the model (about 90 MB); after that it runs offline.
"""

import argparse
from pathlib import Path

import numpy as np

from data_io import load_corpus, load_sources, load_topics

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self, docs, model_name=MODEL_NAME, batch_size=64, device=None, encoder=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name, device=device)
        self.encoder = encoder
        self.batch_size = batch_size
        self.doc_ids = [d for d, _ in docs]
        self.vectors = self._encode([t for _, t in docs])

    def _encode(self, texts):
        vecs = self.encoder.encode(texts, batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def search_many(self, queries, k=100):
        """Return one [(docid, cosine), ...] list per query, best first.
        Ties broken by docid so runs are reproducible."""
        sims = self._encode(queries) @ self.vectors.T
        results = []
        for row in sims:
            ranked = sorted(zip(self.doc_ids, row.tolist()), key=lambda x: (-x[1], x[0]))
            results.append(ranked[:k])
        return results


def dedup_by_source(hits, sources, max_per_source, k):
    """Keep at most max_per_source hits per source page, in rank order."""
    if max_per_source <= 0:
        return hits[:k]
    counts, kept = {}, []
    for docid, score in hits:
        src = sources.get(docid, docid)
        if counts.get(src, 0) < max_per_source:
            counts[src] = counts.get(src, 0) + 1
            kept.append((docid, score))
            if len(kept) == k:
                break
    return kept


def write_trec_run(results, path, tag):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, hits in results.items():
            for rank, (docid, score) in enumerate(hits, 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=100, help="hits per query")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cpu, cuda or mps; default auto")
    ap.add_argument("--tag", default="minilm")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="keep at most N passages per source page (0 = no limit)")
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    topics = load_topics(args.topics)
    retriever = DenseRetriever(docs, args.model, args.batch_size, args.device)

    # rank everything, then trim, so dedup can reach past the first k
    hits = retriever.search_many([t["question"] for t in topics], len(docs))
    sources = load_sources(args.corpus)
    results = {t["qid"]: dedup_by_source(h, sources, args.max_per_source, args.k)
               for t, h in zip(topics, hits)}
    write_trec_run(results, args.output, args.tag)
    note = f" (max {args.max_per_source} per source)" if args.max_per_source > 0 else ""
    print(f"{len(docs)} passages, {len(topics)} queries -> {args.output}{note}")


if __name__ == "__main__":
    main()
