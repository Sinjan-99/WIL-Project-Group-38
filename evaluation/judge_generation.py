"""LLM-as-a-judge scoring for the Casual Work Rights Assistant.

Adds a third generation metric alongside ROUGE-L and BERTScore. Those two
compare the generated answer to the reference by word overlap and embedding
similarity, so an answer that is correct but worded differently scores badly.
A judge model reads the question, our answer and the reference, and rates
correctness directly.

The judge is a different model from the one that wrote the answers. A model
judging its own output shares its own blind spots, so the score would be
weaker evidence than it looks.

Install the judge model first:

    ollama pull mistral

Then run from the project root:

    python evaluation/judge_generation.py

Input:  evaluation/generated_answers.csv   (produced by generate_answers.py)
Output: evaluation/judge_evaluation.csv

Roughly 170 judgements at a few seconds each, so expect 15 to 40 minutes.
Progress is saved after every row: stop it with Ctrl+C and run it again to
pick up where it left off.
"""

import os
import re
import sys

import ollama
import pandas as pd

JUDGE_MODEL = "mistral"
ANSWER_MODEL = "llama3.2"  # only used to warn if the judge is the same model

INPUT_FILE = "evaluation/generated_answers.csv"
OUTPUT_FILE = "evaluation/judge_evaluation.csv"

K_VALUES = [1, 3, 5]

# The rubric is spelled out level by level. Given a vague scale, the judge
# drifts between questions; given defined levels, the scores stay comparable.
JUDGE_PROMPT = """###Task Description:
You are given a question, a response to evaluate, a reference answer that would
score 5, and a rubric. The subject is Australian workplace rights.

1. Write brief feedback assessing the response strictly against the rubric.
2. Then write a score: an integer from 1 to 5.
3. Format your output exactly as: Feedback: {{your feedback}} [RESULT] {{score}}
4. Write nothing else. You must include [RESULT] in your output.

###Question:
{question}

###Response to evaluate:
{response}

###Reference answer (score 5):
{reference}

###Rubric:
[Is the response correct and factual compared to the reference answer?]
Score 1: Completely incorrect, or contradicts the reference.
Score 2: Mostly incorrect, with a small amount of correct information.
Score 3: Partly correct, but omits something important or adds something wrong.
Score 4: Correct, but less precise or less complete than the reference.
Score 5: Correct and factually equivalent to the reference. Different wording
is fine as long as the meaning matches.

###Feedback:"""


def judge_is_installed():
    try:
        installed = ollama.list()
    except Exception:
        return False
    names = [m.get("model", m.get("name", "")) for m in installed.get("models", [])]
    return any(n.split(":")[0] == JUDGE_MODEL.split(":")[0] for n in names)


def parse_judgement(text):
    """Pull the score and feedback out of the judge's reply.

    Returns (score, feedback). Score is None when the reply doesn't parse,
    which is kept in the output rather than silently dropped - a judge that
    ignores the format often has something else wrong with it too.
    """
    if "[RESULT]" not in text:
        return None, text.strip()

    feedback, _, tail = text.partition("[RESULT]")
    match = re.search(r"[1-5]", tail)
    score = int(match.group()) if match else None

    feedback = feedback.replace("Feedback:", "", 1).strip()
    return score, feedback


def judge_one(question, response, reference):
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        reference=reference,
    )
    reply = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 4096},
        keep_alive="10m",
    )
    return parse_judgement(reply["message"]["content"])


def load_done():
    """Rows already judged, so an interrupted run can resume."""
    if not os.path.exists(OUTPUT_FILE):
        return pd.DataFrame(), set()
    done = pd.read_csv(OUTPUT_FILE)
    keys = set(zip(done["question_id"], done["k"]))
    return done, keys


def main():
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"Not found: {INPUT_FILE}. Run generate_answers.py first.")

    if JUDGE_MODEL.split(":")[0] == ANSWER_MODEL.split(":")[0]:
        print(
            f"WARNING: the judge ({JUDGE_MODEL}) is the same model that wrote "
            "the answers. Scores will be less trustworthy. Say so if you "
            "report them."
        )

    if not judge_is_installed():
        sys.exit(
            f"Judge model '{JUDGE_MODEL}' is not installed.\n"
            f"Run: ollama pull {JUDGE_MODEL}"
        )

    df = pd.read_csv(INPUT_FILE)
    done_rows, done_keys = load_done()

    todo = [
        (row, k)
        for _, row in df.iterrows()
        for k in K_VALUES
        if (row["question_id"], k) not in done_keys
    ]

    if not todo:
        print(f"Nothing left to judge. {len(done_rows)} rows already in {OUTPUT_FILE}.")
        summarise()
        return

    print(f"{len(todo)} judgements to make with {JUDGE_MODEL}.")
    print("Saving after each one, so Ctrl+C is safe.\n")

    results = list(done_rows.to_dict("records"))

    for n, (row, k) in enumerate(todo, start=1):
        answer = str(row[f"answer_k{k}"])
        reference = str(row["gold_summary"])

        try:
            score, feedback = judge_one(row["question"], answer, reference)
        except Exception as error:
            print(f"  {row['question_id']} k={k}: judge failed ({error})")
            continue

        results.append({
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "k": k,
            "judge_score": score,
            "judge_feedback": feedback,
        })

        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        shown = score if score is not None else "unparsed"
        print(f"[{n}/{len(todo)}] {row['question_id']} k={k}: {shown}")

    print(f"\nWritten to {OUTPUT_FILE}")
    summarise()


def summarise():
    """Mean score by question type and retrieval depth."""
    if not os.path.exists(OUTPUT_FILE):
        return

    d = pd.read_csv(OUTPUT_FILE)
    scored = d.dropna(subset=["judge_score"])

    unparsed = len(d) - len(scored)
    if unparsed:
        print(f"\n{unparsed} judgements could not be parsed and are excluded.")

    if scored.empty:
        return

    print("\nMean judge score (1-5) by question type and k:\n")
    table = scored.groupby(["question_type", "k"])["judge_score"].agg(["mean", "count"])
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
