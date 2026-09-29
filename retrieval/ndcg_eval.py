"""
NDCG@1/3/5 for one or more TREC runs, split by question type, plus coverage
at larger k for choosing how many passages to give the model.

Gain is the graded relevance label (linear, as in trec_eval and ranx's default
ndcg), so Walert's 2 = fully relevant and 1 = partially relevant both count.
Queries with qrels but no retrieved passages score 0, and are still counted
(trec_eval -c behaviour), so a run can't look better by returning nothing.

Usage:
    python ndcg_eval.py --qrels data/questions.jsonl --topics data/questions.jsonl \
        runs/fwo.minilm.txt

Options:
    --per-query out.csv   write per-query scores for significance testing
    --abstain-below X     out-of-scope: declined when the top score is below X;
                          in-scope: counts how many would be wrongly refused
    --coverage 3,5,8,10   for each k, the share of questions with at least one
                          correct passage in the top k (hit@k), the share of
                          correct passages found (recall@k), and the average
                          number of passages that aren't correct (noise@k)
    --corpus FILE         also report sources@k, the average number of distinct
                          source pages in the top k (for the dedup experiment)
"""

import argparse
import math
from pathlib import Path

from data_io import load_qrels, load_sources, load_topics, read_trec_run

CUTOFFS = (1, 3, 5)


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, judged, k):
    """ranked: docids best first. judged: {docid: rel}."""
    ideal = dcg(sorted(judged.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([judged.get(d, 0) for d in ranked[:k]]) / ideal


def coverage_at_k(ranked, judged, k, sources=None):
    top = ranked[:k]
    found = sum(1 for d in top if d in judged)
    row = {
        f"hit@{k}": 1.0 if found else 0.0,
        f"recall@{k}": found / len(judged) if judged else 0.0,
        f"noise@{k}": float(len(top) - found),
    }
    if sources is not None:
        row[f"sources@{k}"] = float(len({sources.get(d, d) for d in top}))
    return row


def top_scores(path):
    best = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 6 and p[3] == "1":
                best[p[0]] = float(p[4])
    return best


def evaluate(run, qrels, topics, cov_ks, sources):
    rows = []
    for t in topics:
        if t["qid"] not in qrels:
            continue
        ranked = run.get(t["qid"], [])
        row = {"qid": t["qid"], "type": t["type"]}
        for k in CUTOFFS:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, qrels[t["qid"]], k)
        for k in cov_ks:
            row.update(coverage_at_k(ranked, qrels[t["qid"]], k, sources))
        rows.append(row)
    return rows


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--per-query")
    ap.add_argument("--abstain-below", type=float)
    ap.add_argument("--coverage", default="",
                    help="comma-separated k values, e.g. 3,5,8,10")
    ap.add_argument("--corpus", help="passages.jsonl, to report sources@k")
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    topics = load_topics(args.topics)
    oos = [t for t in topics if t["type"] == "out_of_scope"]
    cov_ks = [int(x) for x in args.coverage.split(",") if x.strip()]
    sources = load_sources(args.corpus) if args.corpus else None

    bad = [t["qid"] for t in topics if (t["type"] == "out_of_scope") == (t["qid"] in qrels)]
    if bad:
        print(f"warning: type/qrels mismatch for {len(bad)} questions, e.g. {bad[:5]}")

    types = [q for q in ("known", "inferred", "unspecified")
             if any(t["type"] == q for t in topics)]
    metrics = [f"ndcg@{k}" for k in CUTOFFS]
    cov_cols = ["hit", "recall", "noise"] + (["sources"] if sources else [])
    all_rows = []

    for run_path in args.runs:
        name = Path(run_path).name
        run = read_trec_run(run_path)
        rows = evaluate(run, qrels, topics, cov_ks, sources)
        all_rows += [{"run": name, **r} for r in rows]

        print(f"\n{name}")
        print(f"  {'set':<10}{'n':>4}  " + "  ".join(f"{m.upper():>8}" for m in metrics))
        for qtype in types:
            sub = [r for r in rows if r["type"] == qtype]
            if sub:
                print(f"  {qtype:<10}{len(sub):>4}  "
                      + "  ".join(f"{mean(sub, m):>8.4f}" for m in metrics))

        if cov_ks and rows:
            short = [len(run.get(r["qid"], [])) for r in rows]
            for qtype in types:
                sub = [r for r in rows if r["type"] == qtype]
                if not sub:
                    continue
                print(f"\n  coverage, {qtype} questions (n={len(sub)})")
                print(f"  {'k':>4}  " + "  ".join(f"{c + '@k':>10}" for c in cov_cols))
                for k in cov_ks:
                    print(f"  {k:>4}  " + "  ".join(f"{mean(sub, f'{c}@{k}'):>10.3f}"
                                                 for c in cov_cols))
            if min(short) < max(cov_ks):
                print(f"  note: some queries have fewer than {max(cov_ks)} passages in this run")

        if args.abstain_below is not None:
            best = top_scores(run_path)
            below = lambda t: best.get(t["qid"], float("-inf")) < args.abstain_below
            in_scope = [t for t in topics if t["qid"] in qrels]
            if oos:
                ok = sum(1 for t in oos if below(t))
                print(f"  out-of-scope correctly declined (top < {args.abstain_below}): {ok}/{len(oos)}")
            if in_scope:
                wrong = sum(1 for t in in_scope if below(t))
                print(f"  in-scope wrongly declined   (top < {args.abstain_below}): {wrong}/{len(in_scope)}")

    if args.per_query and all_rows:
        cols = [c for c in all_rows[0] if c not in ("run", "qid", "type")]
        with open(args.per_query, "w", encoding="utf-8") as f:
            f.write("run,qid,type," + ",".join(cols) + "\n")
            for r in all_rows:
                f.write(f"{r['run']},{r['qid']},{r['type']},"
                        + ",".join(f"{r[c]:.4f}" for c in cols) + "\n")
        print(f"\nper-query scores -> {args.per_query}")


if __name__ == "__main__":
    main()
