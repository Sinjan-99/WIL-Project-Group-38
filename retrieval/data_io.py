"""
Loaders that accept both the Walert collection and our own project files,
so switching datasets is a path change, not a code change.

Corpus (JSONL), either schema:
    Walert:   {"id": "P01", "contents": "..."}
    Project:  {"passage_id": "P0007", "title": "...", "url": "...", "passage": "...", ...}
              (older {"id", "text"} files still load)

Topics, either format:
    Walert:   topics.csv  (topic_id, topic, question_id, question)
    Project:  questions.jsonl  {"qid", "question", "gold_ids", "type"}

Qrels:
    TREC format "qid 0 docid rel" (Walert qrels.txt, graded 1/2), or derived
    from questions.jsonl gold_ids (binary, rel = 1).
"""

import csv
import json
from pathlib import Path

# Walert topic split, copied from walert/quantitative_eval/src/retrieval/eval.py.
# Topics with no qrels (W33-W38, W40-W43) are out of scope.
WALERT_KNOWN = {f"W{n:02d}" for n in range(1, 21)} | {"W39"}
WALERT_INFERRED = {f"W{n:02d}" for n in range(21, 33)}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path):
    """Return a list of (doc_id, text). Title is prepended when present,
    matching build_index() in app.py."""
    docs = []
    for n, rec in enumerate(read_jsonl(path), 1):
        doc_id = rec.get("passage_id", rec.get("id"))
        if doc_id is None:
            raise KeyError(f"{path} line {n}: no 'passage_id' or 'id' field")
        if "contents" in rec:                      # Walert
            text = rec["contents"]
        else:                                      # project: 'passage' (current) or 'text' (old)
            body = rec.get("passage", rec.get("text"))
            if body is None:
                raise KeyError(f"{path} line {n}: no 'passage' or 'text' field")
            title = str(rec.get("title", "")).strip()
            text = f"{title}. {body}" if title else body
        docs.append((str(doc_id), text))
    ids = [d for d, _ in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate passage ids in {path}")
    return docs


def load_sources(path):
    """Return {doc_id: source_id}. Passages with no source_id (Walert) count
    as their own source, so dedup and sources@k are no-ops there."""
    out = {}
    for rec in read_jsonl(path):
        doc_id = str(rec.get("passage_id", rec.get("id")))
        out[doc_id] = str(rec.get("source_id", doc_id))
    return out


def load_topics(path):
    """Return a list of dicts: {qid, question, type}.

    type is known / inferred / out_of_scope. For Walert it is derived from the
    topic id using the split above; for questions.jsonl it is read from the file.
    """
    path = Path(path)
    if path.suffix == ".csv":
        topics = []
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                tid = row["topic_id"]
                qtype = ("known" if tid in WALERT_KNOWN
                         else "inferred" if tid in WALERT_INFERRED
                         else "out_of_scope")
                topics.append({"qid": row["question_id"],
                               "question": row["question"],
                               "type": qtype})
        return topics
    return [{"qid": str(r["qid"]),
             "question": r["question"],
             "type": r.get("type", "unspecified")}
            for r in read_jsonl(path)]


def load_qrels(path):
    """Return {qid: {docid: rel}} from a TREC qrels file or questions.jsonl."""
    path = Path(path)
    qrels = {}
    if path.suffix == ".jsonl":
        for r in read_jsonl(path):
            gold = r.get("gold_ids") or []
            if gold:
                qrels[str(r["qid"])] = {str(d): 1 for d in gold}
        return qrels
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            if int(rel) > 0:
                qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def read_trec_run(path):
    """Return {qid: [docid, ...]} ordered by rank (then by score as a tiebreak)."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, score, _ = parts
            rows.setdefault(qid, []).append((int(rank), -float(score), docid))
    return {qid: [d for _, _, d in sorted(v)] for qid, v in rows.items()}
