"""
Casual Work Rights Assistant
Group 38, COSC2669 - WIL Project

Retrieval-augmented assistant answering questions about casual employment
rights in Australia from Fair Work Ombudsman source pages.

Retrieval is dense: passages and questions are encoded with a sentence
embedding model and compared by cosine similarity.

Generation runs locally through Ollama (see src/generation.py). If Ollama is
not running the app falls back to extractive mode and shows the retrieved
passage instead of a written answer, so retrieval can still be demonstrated
and evaluated on its own.

Run:
    python -m pip install -r requirements.txt
    ollama pull llama3.2
    streamlit run app.py

Data files:
    data/processed/passages.jsonl
        {
            "passage_id",
            "source_id",
            "title",
            "url",
            "passage",
            "passage_index",
            "word_count",
            "scraped_on"
        }

    data/questions.jsonl
        {
            "qid",
            "question",
            "gold_ids": [...],
            "type": "known|inferred|out_of_scope"
        }
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st


# --------------------------------------------------------------- configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
MIN_SIMILARITY = 0.35

# MIN_SIMILARITY decides whether retrieved passages are relevant enough
# to answer the question.
#
# Cosine similarity runs from -1 to 1.
# Tune this value against evaluation questions, especially out-of-scope
# questions, before final evaluation.

ROOT = Path(__file__).parent
DATA = ROOT / "data"

PASSAGES = DATA / "processed" / "passages.jsonl"
QUESTIONS = DATA / "questions.jsonl"

sys.path.insert(0, str(ROOT / "src"))


# ----------------------------------------------------------------------- data

def load_jsonl(path):
    """Load a JSONL file and skip malformed lines."""

    if not path.exists():
        return []

    records = []

    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError:
                st.warning(
                    f"Skipped malformed line {n} in {path.name}"
                )

    return records


@st.cache_data(show_spinner=False)
def load_passages():
    """
    Load the Fair Work passage corpus.

    Current passage schema:
        passage_id
        source_id
        title
        url
        passage
        passage_index
        word_count
        scraped_on
    """

    passages = load_jsonl(PASSAGES)

    # Basic schema check so database/app mismatches are easier to identify.
    required_fields = {
        "passage_id",
        "source_id",
        "title",
        "url",
        "passage",
        "passage_index",
        "word_count",
        "scraped_on",
    }

    valid_passages = []

    for p in passages:

        missing = required_fields - set(p.keys())

        if missing:
            st.warning(
                f"Skipping passage with missing fields: "
                f"{', '.join(sorted(missing))}"
            )
            continue

        valid_passages.append(p)

    return valid_passages


# ------------------------------------------------------------------- encoding

@st.cache_resource(
    show_spinner="Loading the embedding model, one moment..."
)
def load_model():
    """Load the sentence-transformer embedding model."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Encoding passages...")
def build_index():
    """
    Encode every passage once and keep the vectors in memory.

    Passage title and passage text are encoded together so that section
    context can help retrieval.

    Vectors are normalised to unit length, so cosine similarity between
    a question and a passage can be calculated using the dot product.
    """

    passages = load_passages()

    if not passages:
        return None, []

    model = load_model()

    # UPDATED DATABASE FIELD:
    # old: p["text"]
    # new: p["passage"]

    texts = [
        f"{p['title']}. {p['passage']}"
        for p in passages
    ]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors), passages


# ------------------------------------------------------------------ retrieval

def retrieve(
    query,
    k=TOP_K,
    min_similarity=MIN_SIMILARITY,
):
    """
    Retrieve the most relevant passages for a question.

    Returns:
        [(passage_record, similarity_score), ...]
    """

    vectors, passages = build_index()

    if vectors is None or not query.strip():
        return []

    model = load_model()

    question_vector = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    similarities = vectors @ question_vector

    order = np.argsort(similarities)[::-1][:k]

    hits = [
        (
            passages[i],
            float(similarities[i]),
        )
        for i in order
        if similarities[i] >= min_similarity
    ]

    return hits


def should_abstain(hits):
    """
    Return True when retrieval found no passage above the
    minimum similarity threshold.
    """

    return not hits


# ------------------------------------------------------------------ generation

REFUSED = object()  # model read the passages and said they don't answer the question


def generate(question, hits):
    """
    Generate an answer using retrieved Fair Work passages.

    Returns None when:
    - no passages were retrieved
    - generation.py cannot be imported
    - Ollama is unavailable
    - the generation model chooses the fallback response

    When None is returned, the interface displays the highest-ranked
    retrieved passage directly.
    """

    if not hits:
        return None

    try:
        from generation import (
            generate_answer,
            GenerationUnavailable,
            FALLBACK,
        )

    except ImportError:

        st.info(
            "Generation module not found - showing retrieved passages."
        )

        return None

    try:

        answer = generate_answer(
            question,
            hits,
        )

    except GenerationUnavailable as error:

        st.warning(
            "The local model isn't responding, so I'm showing the "
            "retrieved passage instead. Check that Ollama is running."
        )

        st.caption(f"{error}")

        return None

    if not answer:
        return None

    if answer.strip() == FALLBACK:
        return REFUSED

    return answer


# ------------------------------------------------------------------ evaluation

def ndcg_at_k(
    ranked_ids,
    gold_ids,
    k,
):
    """
    Calculate binary-relevance NDCG@k.

    ranked_ids:
        passage IDs returned by retrieval

    gold_ids:
        passage IDs considered relevant for the question
    """

    gains = [
        1.0 if pid in gold_ids else 0.0
        for pid in ranked_ids[:k]
    ]

    dcg = sum(
        gain / np.log2(i + 2)
        for i, gain in enumerate(gains)
    )

    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(
            min(len(gold_ids), k)
        )
    )

    return (
        dcg / idcg
        if idcg > 0
        else 0.0
    )


def run_evaluation(questions):
    """
    Evaluate retrieval performance.

    Answerable questions:
        NDCG@1
        NDCG@3
        NDCG@5

    Out-of-scope questions:
        checks whether the system correctly abstains.

    Generation is not used during retrieval evaluation.
    """

    retrieval_rows = []
    abstention_rows = []

    for q in questions:

        question = q.get("question", "").strip()

        if not question:
            continue

        hits = retrieve(question)

        # ------------------------------------------------ out-of-scope

        if q.get("type") == "out_of_scope":

            abstention_rows.append(
                {
                    "qid": q.get("qid", ""),
                    "question": question,
                    "declined": should_abstain(hits),
                }
            )

            continue

        # ------------------------------------------------ answerable

        # UPDATED DATABASE FIELD:
        # old: p["id"]
        # new: p["passage_id"]

        ranked = [
            p["passage_id"]
            for p, _ in hits
        ]

        retrieval_rows.append(
            {
                "qid": q.get("qid", ""),
                "type": q.get(
                    "type",
                    "unspecified",
                ),
                "ndcg@1": ndcg_at_k(
                    ranked,
                    q.get("gold_ids", []),
                    1,
                ),
                "ndcg@3": ndcg_at_k(
                    ranked,
                    q.get("gold_ids", []),
                    3,
                ),
                "ndcg@5": ndcg_at_k(
                    ranked,
                    q.get("gold_ids", []),
                    5,
                ),
            }
        )

    return retrieval_rows, abstention_rows


# -------------------------------------------------------------------------- UI

st.set_page_config(
    page_title="Casual Work Rights Assistant",
    page_icon="⚖️",
)

passages = load_passages()


# --------------------------------------------------------------------- sidebar

with st.sidebar:

    st.subheader("Knowledge base")

    if passages:

        st.success(
            f"{len(passages)} passages indexed"
        )

        # Useful database information
        source_count = len(
            {
                p["source_id"]
                for p in passages
            }
        )

        st.caption(
            f"{source_count} Fair Work source pages"
        )

    else:

        st.error(
            "No corpus loaded"
        )

        st.caption(
            f"Expected {PASSAGES.relative_to(ROOT)}"
        )

    # -------------------------------------------------------------- reload

    if st.button("Reload corpus"):

        load_passages.clear()
        build_index.clear()

        st.rerun()

    # -------------------------------------------------------------- warm up

    if st.button("Warm up model"):

        try:

            build_index()

            st.success(
                "Retrieval ready"
            )

        except ImportError:

            st.error(
                "sentence-transformers is not installed."
            )

    # ----------------------------------------------------------- generation

    st.subheader("Generation")

    try:

        from generation import (
            is_available,
            MODEL_NAME,
        )

        if is_available():

            st.success(
                f"{MODEL_NAME} ready"
            )

        else:

            st.warning(
                "Ollama not reachable"
            )

            st.caption(
                "Extractive mode: answers show the "
                "retrieved passage."
            )

    except ImportError:

        st.warning(
            "Generation module not found"
        )

    st.caption(
        "Retrieval: dense - "
        f"{EMBEDDING_MODEL.split('/')[-1]}"
    )


# ------------------------------------------------------------------------ tabs

ask_tab, eval_tab = st.tabs(
    [
        "Ask",
        "Evaluation",
    ]
)


# --------------------------------------------------------------------- ask tab

with ask_tab:

    st.title(
        "Casual Work Rights Assistant"
    )

    st.caption(
        "Answers come only from Fair Work source pages held in the "
        "knowledge base. Pay rates change on 1 July each year, so "
        "confirm current figures on the Fair Work website."
    )

    question = st.text_input(
        "Your question",
        placeholder="Do casual employees get sick leave?",
    )

    if question:

        if not passages:

            st.error(
                "No corpus loaded, so nothing can be retrieved."
            )

        else:

            hits = retrieve(question)

            # ------------------------------------------------------- abstain

            if should_abstain(hits):

                st.warning(
                    "I don't have enough information in my sources "
                    "to answer that. Try the Fair Work Ombudsman "
                    "website, or call the Fair Work Infoline on "
                    "13 13 94."
                )

            # ------------------------------------------------------- answer

            else:

                with st.spinner(
                    "Writing an answer..."
                ):

                    answer = generate(
                        question,
                        hits,
                    )

                # ------------------------------------------ generated answer

                if answer is REFUSED:

                    st.warning(
                        "I found related pages but none of them answer "
                        "that question. Try the Fair Work Ombudsman "
                        "website, or call the Fair Work Infoline on "
                        "13 13 94."
                    )

                elif answer:

                    st.markdown(answer)

                    st.subheader(
                        "Sources"
                    )

                    for p, score in hits:

                        label = (
                            f"{p['title']} · "
                            f"{p['passage_id']} · "
                            f"similarity {score:.2f}"
                        )

                        with st.expander(label):

                            # UPDATED:
                            # old p["text"]
                            # new p["passage"]

                            st.write(
                                p["passage"]
                            )

                            st.caption(
                                f"Source: {p['url']}"
                            )

                # ------------------------------------------ extractive answer

                else:

                    top, score = hits[0]

                    st.markdown(
                        f"**{top['title']}**"
                    )

                    st.caption(
                        f"Passage {top['passage_id']} "
                        f"· similarity {score:.2f}"
                    )

                    st.write(
                        top["passage"]
                    )

                    st.caption(
                        f"Source: {top['url']}"
                    )

                    # ------------------------------------ additional passages

                    if len(hits) > 1:

                        st.subheader(
                            "Other passages retrieved"
                        )

                        for p, score in hits[1:]:

                            label = (
                                f"{p['title']} · "
                                f"{p['passage_id']} · "
                                f"similarity {score:.2f}"
                            )

                            with st.expander(label):

                                st.write(
                                    p["passage"]
                                )

                                st.caption(
                                    f"Source: {p['url']}"
                                )


# ---------------------------------------------------------------- evaluation tab

with eval_tab:

    st.title(
        "Evaluation"
    )

    questions = load_jsonl(
        QUESTIONS
    )

    if not questions:

        st.info(
            f"Add {QUESTIONS.relative_to(ROOT)} "
            "to run the evaluation."
        )

        st.caption(
            "One JSON object per line:"
        )

        # IMPORTANT:
        # Gold IDs must now use Pxxxx passage IDs from the
        # final Collection.csv/passages.jsonl database.

        st.code(
            '{"qid": "q1", '
            '"question": "Do casual employees get paid sick leave?", '
            '"gold_ids": ["P0007"], '
            '"type": "known"}\n'
            '{"qid": "q2", '
            '"question": "Can a casual employee become permanent?", '
            '"gold_ids": ["P0012"], '
            '"type": "inferred"}\n'
            '{"qid": "q3", '
            '"question": "How much is stamp duty in Victoria?", '
            '"gold_ids": [], '
            '"type": "out_of_scope"}',
            language="json",
        )

    else:

        st.caption(
            f"{len(questions)} questions loaded. "
            "Retrieval only - generation is not called here."
        )

        if st.button(
            "Run evaluation"
        ):

            with st.spinner(
                "Scoring..."
            ):

                (
                    retrieval_rows,
                    abstention_rows,
                ) = run_evaluation(
                    questions
                )

            # -------------------------------------------- retrieval results

            for qtype in (
                "known",
                "inferred",
                "unspecified",
            ):

                subset = [
                    r
                    for r in retrieval_rows
                    if r["type"] == qtype
                ]

                if not subset:
                    continue

                st.subheader(
                    f"{qtype.title()} questions "
                    f"({len(subset)})"
                )

                cols = st.columns(3)

                for col, metric in zip(
                    cols,
                    (
                        "ndcg@1",
                        "ndcg@3",
                        "ndcg@5",
                    ),
                ):

                    mean = float(
                        np.mean(
                            [
                                r[metric]
                                for r in subset
                            ]
                        )
                    )

                    col.metric(
                        metric.upper(),
                        f"{mean:.4f}",
                    )

            # ------------------------------------------- abstention results

            if abstention_rows:

                declined = sum(
                    r["declined"]
                    for r in abstention_rows
                )

                st.subheader(
                    "Out-of-scope questions "
                    f"({len(abstention_rows)})"
                )

                st.metric(
                    "Correctly declined",
                    f"{declined}/"
                    f"{len(abstention_rows)}",
                )

                missed = [
                    r["question"]
                    for r in abstention_rows
                    if not r["declined"]
                ]

                if missed:

                    st.caption(
                        "Answered when it should "
                        "have declined:"
                    )

                    for q in missed:

                        st.caption(
                            f"· {q}"
                        )

            # ---------------------------------------------------- download

            if retrieval_rows:

                header = (
                    "qid,type,"
                    "ndcg@1,ndcg@3,ndcg@5\n"
                )

                body = "\n".join(
                    f"{r['qid']},"
                    f"{r['type']},"
                    f"{r['ndcg@1']:.4f},"
                    f"{r['ndcg@3']:.4f},"
                    f"{r['ndcg@5']:.4f}"
                    for r in retrieval_rows
                )

                st.download_button(
                    "Download results (CSV)",
                    header + body,
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                )