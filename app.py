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

TOP_K = 8
MIN_SIMILARITY = 0.50

# Retrieval settings, tuned by Krishi (Retrieval / ML) on the 282-passage
# corpus with 58 questions: 30 known, 24 inferred (each needing two facts)
# and 4 out-of-scope. Evidence is in the retrieval/ run files.
#
# MIN_SIMILARITY = 0.50 (was 0.35, set when the corpus had 3 passages)
#   The top passage's cosine similarity must reach this or the app declines.
#   Sweep 0.35-0.65: 0.50 was the only value that declined all 4 out-of-scope
#   questions while refusing none of the 54 answerable ones. 0.45 lets one
#   out-of-scope question through; 0.55 refuses 2/54 answerable. The margin
#   is narrow, so recheck as the question set grows.
#
# TOP_K = 8 (was 5)
#   On the inferred questions, both required facts were in the top 5 for
#   16/24 questions and in the top 8 for 21/24; k = 10 gave no further gain.
#   The missing fact was usually another section of an already-retrieved
#   page. Cost: ~2.3 more non-relevant passages per question and a longer
#   prompt, so generation needs a large enough context window (num_ctx).
#   hit@k is 1.000 at both 5 and 8, and abstention only uses the top score.
#
# No deduplication by source_id
#   Tested at k = 8. Capping each page at 2 or 1 passages cut the inferred
#   questions with both facts retrieved from 21/24 to 16/24 and 8/24, and
#   lowered known hit@8 from 1.000 to 0.967 / 0.900. Two facts usually come
#   from different sections of the same page, so capping removes the second.

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

def rank(query, k=None):
    """
    Rank passages for a question by cosine similarity, with no threshold.

    Used by retrieve() for the app, and directly by the evaluation so that
    ranking quality (NDCG) is measured separately from abstention.

    Returns:
        [(passage_record, similarity_score), ...], best first
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

    order = np.argsort(similarities)[::-1]

    if k is not None:
        order = order[:k]

    return [
        (
            passages[i],
            float(similarities[i]),
        )
        for i in order
    ]


def retrieve(
    query,
    k=TOP_K,
    min_similarity=MIN_SIMILARITY,
):
    """
    Retrieve the passages the model will see: the top k, keeping only
    those at or above the similarity threshold.

    Returns:
        [(passage_record, similarity_score), ...]
    """

    return [
        (p, score)
        for p, score in rank(query, k)
        if score >= min_similarity
    ]


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


# Question files may label unanswerable questions "out_of_kb" (questions.csv).
TYPE_ALIASES = {"out_of_kb": "out_of_scope"}

SWEEP_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def question_type(q):
    t = str(q.get("type", "unspecified")).strip()
    return TYPE_ALIASES.get(t, t)


def run_evaluation(questions, k=TOP_K):
    """
    Evaluate retrieval, matching retrieval/ndcg_eval.py.

    Answerable questions:
        NDCG@1/3/5 on the full ranking (no threshold applied), plus
        hit@k and recall@k at the k the app uses.

    Every question:
        the top similarity score, so abstention can be scored at any
        threshold without re-running retrieval.

    Generation is not used during retrieval evaluation.
    """

    retrieval_rows = []
    abstention_rows = []

    for q in questions:

        question = q.get("question", "").strip()

        if not question:
            continue

        ranked_hits = rank(question, k=max(5, k))
        top_score = ranked_hits[0][1] if ranked_hits else float("-inf")
        qtype = question_type(q)

        abstention_rows.append(
            {
                "qid": q.get("qid", ""),
                "question": question,
                "type": qtype,
                "top_score": top_score,
                "top_title": ranked_hits[0][0]["title"] if ranked_hits else "",
            }
        )

        if qtype == "out_of_scope":
            continue

        gold = set(q.get("gold_ids", []))
        ranked = [p["passage_id"] for p, _ in ranked_hits]
        found = sum(1 for pid in ranked[:k] if pid in gold)

        retrieval_rows.append(
            {
                "qid": q.get("qid", ""),
                "type": qtype,
                "ndcg@1": ndcg_at_k(ranked, gold, 1),
                "ndcg@3": ndcg_at_k(ranked, gold, 3),
                "ndcg@5": ndcg_at_k(ranked, gold, 5),
                f"hit@{k}": 1.0 if found else 0.0,
                f"recall@{k}": found / len(gold) if gold else 0.0,
                "top_score": top_score,
            }
        )

    return retrieval_rows, abstention_rows


def abstention_counts(abstention_rows, threshold):
    """(out-of-scope declined, out-of-scope total,
        answerable refused, answerable total) at a threshold."""

    oos = [r for r in abstention_rows if r["type"] == "out_of_scope"]
    ans = [r for r in abstention_rows if r["type"] != "out_of_scope"]

    return (
        sum(r["top_score"] < threshold for r in oos),
        len(oos),
        sum(r["top_score"] < threshold for r in ans),
        len(ans),
    )


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
            "Retrieval only - generation is not called here. "
            f"Settings: k = {TOP_K}, MIN_SIMILARITY = {MIN_SIMILARITY}."
        )

        # Results are kept in session state so they survive reruns
        # (for example when the download button is clicked).
        if st.button(
            "Run evaluation"
        ):

            with st.spinner(
                "Scoring..."
            ):

                st.session_state["eval_results"] = run_evaluation(
                    questions
                )

        if "eval_results" in st.session_state:

            (
                retrieval_rows,
                abstention_rows,
            ) = st.session_state["eval_results"]

            # -------------------------------------------- retrieval results

            st.caption(
                "NDCG is scored on the full ranking, before the similarity "
                "threshold is applied, so it measures ranking quality alone. "
                "Abstention is scored separately below."
            )

            hit_key = f"hit@{TOP_K}"
            recall_key = f"recall@{TOP_K}"

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

                cols = st.columns(5)

                for col, metric in zip(
                    cols,
                    (
                        "ndcg@1",
                        "ndcg@3",
                        "ndcg@5",
                        hit_key,
                        recall_key,
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
                        metric.upper() if metric.startswith("ndcg") else metric,
                        f"{mean:.4f}" if metric.startswith("ndcg") else f"{mean:.3f}",
                    )

            if retrieval_rows:
                st.caption(
                    f"{hit_key}: share of questions with at least one correct "
                    f"passage in the top {TOP_K}. {recall_key}: share of all "
                    f"correct passages found in the top {TOP_K}."
                )

                misses = [
                    r["qid"]
                    for r in retrieval_rows
                    if r["ndcg@1"] == 0
                ]

                if misses:
                    st.caption(
                        "Top passage not a gold passage: "
                        + ", ".join(misses)
                    )

            # ------------------------------------------- abstention results

            if abstention_rows:

                ok, n_oos, wrong, n_ans = abstention_counts(
                    abstention_rows,
                    MIN_SIMILARITY,
                )

                st.subheader(
                    f"Abstention at MIN_SIMILARITY = {MIN_SIMILARITY}"
                )

                cols = st.columns(2)

                if n_oos:
                    cols[0].metric(
                        "Out-of-scope correctly declined",
                        f"{ok}/{n_oos}",
                    )

                if n_ans:
                    cols[1].metric(
                        "Answerable wrongly refused",
                        f"{wrong}/{n_ans}",
                    )

                missed = [
                    r
                    for r in abstention_rows
                    if r["type"] == "out_of_scope"
                    and r["top_score"] >= MIN_SIMILARITY
                ]

                if missed:

                    st.caption(
                        "Answered when it should "
                        "have declined:"
                    )

                    for r in missed:

                        st.caption(
                            f"· {r['question']} "
                            f"(top match: {r['top_title']}, "
                            f"{r['top_score']:.3f})"
                        )

                refused = [
                    r
                    for r in abstention_rows
                    if r["type"] != "out_of_scope"
                    and r["top_score"] < MIN_SIMILARITY
                ]

                if refused:

                    st.caption(
                        "Refused although answerable:"
                    )

                    for r in refused:

                        st.caption(
                            f"· {r['question']} "
                            f"({r['top_score']:.3f})"
                        )

                with st.expander(
                    "Threshold sweep"
                ):

                    sweep = []

                    for t in SWEEP_THRESHOLDS:

                        ok_t, n_oos, wrong_t, n_ans = abstention_counts(
                            abstention_rows,
                            t,
                        )

                        sweep.append(
                            {
                                "Threshold": f"{t:.2f}"
                                + (" (current)" if abs(t - MIN_SIMILARITY) < 1e-9 else ""),
                                "Out-of-scope declined": f"{ok_t}/{n_oos}",
                                "Answerable refused": f"{wrong_t}/{n_ans}",
                            }
                        )

                    st.table(sweep)

                    st.caption(
                        "Choose the highest threshold that refuses no "
                        "answerable question."
                    )

            # ---------------------------------------------------- download

            if retrieval_rows:

                cols_out = [
                    "ndcg@1",
                    "ndcg@3",
                    "ndcg@5",
                    hit_key,
                    recall_key,
                    "top_score",
                ]

                header = (
                    "qid,type,"
                    + ",".join(cols_out)
                    + "\n"
                )

                body = "\n".join(
                    f"{r['qid']},{r['type']},"
                    + ",".join(
                        f"{r[c]:.4f}"
                        for c in cols_out
                    )
                    for r in retrieval_rows
                )

                st.download_button(
                    "Download results (CSV)",
                    header + body,
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                )
