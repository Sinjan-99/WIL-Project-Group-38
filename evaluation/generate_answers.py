import csv
import json
import sys
from pathlib import Path

# Allow imports from project folders
ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "retrieval"))

from data_io import read_trec_run
from src.generation.generation import generate_answer


QUESTIONS_FILE = ROOT / "evaluation" / "questions.csv"
PASSAGES_FILE = ROOT / "data" / "processed" / "passages.jsonl"
RUN_FILE = ROOT / "retrieval" / "runs" / "fwo_final.minilm.txt"
OUTPUT_FILE = ROOT / "evaluation" / "generated_answers.csv"


def load_passages(path):
    passages = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            passages[str(record["passage_id"])] = {
                "title": record.get("title", ""),
                "text": record.get("passage", "")
            }

    return passages


def load_questions(path):
    questions = []

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Only Known and Inferred questions for answer evaluation
            if row["question_type"] in ("known", "inferred"):
                questions.append(row)

    return questions


def main():

    print("Loading passages...")
    passages = load_passages(PASSAGES_FILE)

    print("Loading questions...")
    questions = load_questions(QUESTIONS_FILE)

    print("Loading retrieval run...")
    run = read_trec_run(RUN_FILE)

    print(f"Passages: {len(passages)}")
    print(f"Questions: {len(questions)}")

    results = []

    for number, q in enumerate(questions, start=1):

        qid = q["question_id"]
        question = q["question"]

        ranked_ids = run.get(qid, [])

        print(f"\n[{number}/{len(questions)}] {qid}")
        print(question)

        row = {
            "question_id": qid,
            "question_type": q["question_type"],
            "question": question,
            "gold_summary": q["gold_summary"]
        }

        for k in (1, 3, 5):

            top_ids = ranked_ids[:k]

            context = [
                passages[pid]
                for pid in top_ids
                if pid in passages
            ]

            print(f"  Generating K={k}...")

            answer = generate_answer(question, context)

            row[f"retrieved_k{k}"] = "|".join(top_ids)
            row[f"answer_k{k}"] = answer

        results.append(row)

    fieldnames = [
        "question_id",
        "question_type",
        "question",
        "gold_summary",
        "retrieved_k1",
        "answer_k1",
        "retrieved_k3",
        "answer_k3",
        "retrieved_k5",
        "answer_k5",
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nGeneration complete.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()