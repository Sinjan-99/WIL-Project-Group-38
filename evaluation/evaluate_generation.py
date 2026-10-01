import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score


INPUT_FILE = "evaluation/generated_answers.csv"
OUTPUT_FILE = "evaluation/generation_evaluation.csv"


def main():

    # -------------------------
    # Load data
    # -------------------------

    df = pd.read_csv(INPUT_FILE)

    # -------------------------
    # ROUGE-L
    # -------------------------

    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    # Store generated answers
    all_generated = []
    all_references = []
    metadata = []

    for _, row in df.iterrows():

        reference = str(row["gold_summary"])

        for k in [1, 3, 5]:

            generated = str(row[f"answer_k{k}"])

            all_generated.append(generated)
            all_references.append(reference)

            metadata.append({
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "k": k
            })

    # -------------------------
    # Calculate BERTScore
    # -------------------------

    print("Calculating BERTScore...")

    P, R, F1 = score(
        all_generated,
        all_references,
        lang="en",
        verbose=True
    )

    # -------------------------
    # Calculate ROUGE-L
    # -------------------------

    print("Calculating ROUGE-L...")

    results = []

    for i, item in enumerate(metadata):

        rouge_result = rouge.score(
            all_references[i],
            all_generated[i]
        )

        results.append({
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "k": item["k"],
            "rougeL": rouge_result["rougeL"].fmeasure,
            "bertscore": F1[i].item()
        })

    output = pd.DataFrame(results)

    # -------------------------
    # Save results
    # -------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -------------------------
    # Overall results
    # -------------------------

    print("\nOverall results:")

    overall = output.groupby("k")[
        ["rougeL", "bertscore"]
    ].mean()

    print(overall)

    # -------------------------
    # Results by question type
    # -------------------------

    print("\nResults by question type:")

    by_type = output.groupby(
        ["question_type", "k"]
    )[
        ["rougeL", "bertscore"]
    ].mean()

    print(by_type)

    print(
        f"\nSaved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()