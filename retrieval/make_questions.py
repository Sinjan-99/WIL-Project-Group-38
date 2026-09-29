"""
Convert Kripa's questions.csv into data/questions.jsonl in the format app.py
and ndcg_eval.py expect: {"qid", "question", "gold_ids", "type", ...}.

questions.csv has a gold_summary but no gold passage ids, and NDCG needs ids.
GOLD below maps each question group to the passages that state the answer in
its gold_summary. These are DRAFT labels picked by reading passages.jsonl:
the team should check them (see gold_review.csv) before results are reported.

Also maps question_type "out_of_kb" -> "out_of_scope", which is the value
app.py checks for.

Usage:
    python make_questions.py --csv data/questions.csv --passages data/processed/passages.jsonl --output data/questions.jsonl
"""

import argparse
import csv
import json

# group_id -> passages that directly state the gold_summary answer (DRAFT, verify)
GOLD = {
    "G001": ["P0072", "P0006"],                    # annual leave: all except casuals
    "G002": ["P0063", "P0190", "P0191", "P0055", "P0064"],  # no paid sick leave for casuals; P0055/P0064 added after review: paid sick leave is for full-time and part-time
    "G003": ["P0063", "P0191", "P0190", "P0058", "P0070", "P0006", "P0057"],  # no paid, but 2 days unpaid carer's; P0057 added after review
    "G004": ["P0076", "P0079", "P0006"],           # 10 days paid FDV incl. casuals
    "G005": ["P0008", "P0010", "P0011"],           # change to permanent / employee choice
    "G006": ["P0006", "P0001"],                    # casual loading or specific casual rate
    "G007": ["P0025", "P0027"],                    # right to refuse out-of-hours contact
    "G008": ["P0135", "P0206"],                    # pay slip within 1 working day
    "G009": ["P0263", "P0121"],                    # casuals needn't give notice when resigning
    "G010": ["P0085", "P0098"],                    # public holidays differ by state; P0098 added after review (holidays depend on where you are based)
    # --- inferred questions: each needs two facts, often from different sections (DRAFT, verify)
    "G015": ["P0011", "P0014", "P0010"],           # eligibility 6/12 months + employer consults, responds within 21 days
    "G016": ["P0063", "P0191", "P0190", "P0058", "P0070", "P0006", "P0057", "P0055", "P0064"],  # no paid sick/carer's + 2 days unpaid carer's
    "G017": ["P0135", "P0206", "P0123", "P0136", "P0194"],  # pay slip within 1 day, English + records kept 7 years
    "G018": ["P0098", "P0085"],                    # holidays where based for work + differ by state
    "G019": ["P0263", "P0265", "P0180", "P0120", "P0121"],  # casuals no notice + deduct up to 1 week's wages
    "G020": ["P0255", "P0256", "P0250"],           # adult rate when serving alcohol + junior % of adult rate
    "G021": ["P0279", "P0006", "P0281", "P0282"],  # ways penalties combine with loading + rate includes loading
    "G022": ["P0176", "P0166", "P0184"],           # what final pay includes + in lieu must equal full amount, paid by termination day
}

TYPE_MAP = {"out_of_kb": "out_of_scope"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--review", default="gold_review.csv",
                    help="sheet listing each gold passage with its text, for checking")
    args = ap.parse_args()

    with open(args.passages, encoding="utf-8") as f:
        passages = {p["passage_id"]: p for p in (json.loads(l) for l in f if l.strip())}

    missing = sorted({pid for ids in GOLD.values() for pid in ids} - passages.keys())
    if missing:
        raise SystemExit(f"Gold ids not in passages.jsonl: {missing}. "
                         "The corpus changed; update GOLD.")

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out, problems = [], []
    for r in rows:
        qtype = TYPE_MAP.get(r["question_type"].strip(), r["question_type"].strip())
        gold = [] if qtype == "out_of_scope" else GOLD.get(r["group_id"], [])
        if qtype != "out_of_scope" and not gold:
            problems.append(r["question_id"])
        out.append({
            "qid": r["question_id"],
            "question": r["question"].strip(),
            "gold_ids": gold,
            "type": qtype,
            "group_id": r["group_id"],
            "category": r["category"],
            "gold_summary": r["gold_summary"],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(args.review, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "gold_summary", "passage_id", "title", "passage", "correct? (Y/N)"])
        summaries = {r["group_id"]: r["gold_summary"] for r in rows}
        for gid, ids in GOLD.items():
            for pid in ids:
                p = passages[pid]
                w.writerow([gid, summaries.get(gid, ""), pid, p["title"], p["passage"], ""])

    counts = {}
    for rec in out:
        counts[rec["type"]] = counts.get(rec["type"], 0) + 1
    print(f"{len(out)} questions -> {args.output}  {counts}")
    print(f"review sheet -> {args.review}")
    if problems:
        print(f"warning: no gold ids for {problems}; add their group to GOLD")


if __name__ == "__main__":
    main()
