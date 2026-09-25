"""
Casual Work Rights Assistant
Group 38, COSC2669 - WIL Project

Retrieval-augmented assistant answering questions about casual employment
rights in Australia from Fair Work Ombudsman source pages.

Retrieval is dense: passages and questions are encoded with a sentence
embedding model and compared by cosine similarity. This replaces the BM25
baseline used in the Milestone 1 reproduction, where dense retrieval reached
NDCG@1 of 0.6905 on known questions against 0.5119 for BM25, with the gap
widening on paraphrased questions.

Generation runs locally through Ollama (see src/generation.py). If Ollama is
not running the app falls back to extractive mode and shows the retrieved
passage instead of a written answer, so retrieval can still be demonstrated
and evaluated on its own.

Run:
    python -m pip install -r requirements.txt
    ollama pull llama3.2
    streamlit run app.py

Data files:
    data/processed/passages.jsonl   {"id", "url", "title", "text"}
    data/questions.jsonl            {"qid", "question", "gold_ids": [...], "type": "known|inferred|out_of_scope"}
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# --------------------------------------------------------------- configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5              # passages retrieved per question
MIN_SIMILARITY = 0.35  # below this, nothing is relevant enough to answer from

# MIN_SIMILARITY is the one retrieval number that decides behaviour. Cosine
# similarity runs from -1 to 1; in practice a clearly relevant passage sits
# above 0.5, a loosely related one around 0.3, and an unrelated one near 0.
# Tune it against the out-of-scope questions and record what you chose.

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PASSAGES = DATA / "processed" / "passages.jsonl"
QUESTIONS = DATA / "questions.jsonl"

sys.path.insert(0, str(ROOT / "src"))


# ----------------------------------------------------------------------- data

def load_jsonl(path):
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
                st.warning(f"Skipped malformed line {n} in {path.name}")
    return records


@st.cache_data(show_spinner=False)
def load_passages():
    """Read the corpus only. No model, no encoding, so the page renders at once."""
    return load_jsonl(PASSAGES)


# ------------------------------------------------------------------- encoding

@st.cache_resource(show_spinner="Loading the embedding model, one moment...")
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Encoding passages...")
def build_index():
    """Encode every passage once and keep the vectors in memory.

    Vectors are normalised to unit length, so cosine similarity between a
    question and a passage is just their dot product. Called lazily, so nothing
    here runs until a question is asked or the model is warmed up.
    """
    passages = load_passages()
    if not passages:
        return None, []
    model = load_model()
    texts = [f"{p['title']}. {p['text']}" for p in passages]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors), passages


# ------------------------------------------------------------------ retrieval

def retrieve(query, k=TOP_K, min_similarity=MIN_SIMILARITY):
    """Return (passage, similarity) pairs above the relevance floor, best first."""
    vectors, passages = build_index()
    if vectors is None or not query.strip():
        return []
    model = load_model()
    q = model.encode([query], normalize_embeddings=True)[0]
    similarities = vectors @ q
    order = np.argsort(similarities)[::-1][:k]
    return [
        (passages[i], float(similarities[i]))
        for i in order
        if similarities[i] >= min_similarity
    ]


def should_abstain(hits):
    """Decline when nothing retrieved is close enough to the question."""
    return not hits


# ------------------------------------------------------------------ generation

def generate(question, hits):
    """Write an answer from the retrieved passages.

    Returns None when no answer should be shown, which sends the interface to
    extractive mode. That happens when Ollama is unreachable, or when the model
    itself reports that the passages do not answer the question.
    """
    if not hits:
        return None

    try:
        from generation import generate_answer, GenerationUnavailable, FALLBACK
    except ImportError:
        st.info("Generation module not found - showing retrieved passages.")
        return None

    try:
        answer = generate_answer(question, hits)
    except GenerationUnavailable as error:
        st.warning(
            "The local model isn't responding, so I'm showing the retrieved "
            "passage instead. Check that Ollama is running."
        )
        st.caption(f"{error}")
        return None

    if answer.strip() == FALLBACK:
        return None
    return answer


# ------------------------------------------------------------------ evaluation

def ndcg_at_k(ranked_ids, gold_ids, k):
    gains = [1.0 if pid in gold_ids else 0.0 for pid in ranked_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(questions):
    """Score retrieval on answerable questions and abstention on the rest.

    Retrieval only - generation is not called here, so this runs quickly and
    measures the stage that sets the ceiling for everything downstream.
    """
    retrieval_rows, abstention_rows = [], []
    for q in questions:
        hits = retrieve(q["question"])
        if q.get("type") == "out_of_scope":
            abstention_rows.append({
                "qid": q["qid"],
                "question": q["question"],
                "declined": should_abstain(hits),
            })
            continue
        ranked = [p["id"] for p, _ in hits]
        retrieval_rows.append({
            "qid": q["qid"],
            "type": q.get("type", "unspecified"),
            "ndcg@1": ndcg_at_k(ranked, q.get("gold_ids", []), 1),
            "ndcg@3": ndcg_at_k(ranked, q.get("gold_ids", []), 3),
            "ndcg@5": ndcg_at_k(ranked, q.get("gold_ids", []), 5),
        })
    return retrieval_rows, abstention_rows


# -------------------------------------------------------------------------- ui

st.set_page_config(page_title="Casual Work Rights Assistant", page_icon="⚖️")

passages = load_passages()

with st.sidebar:
    st.subheader("Knowledge base")
    if passages:
        st.success(f"{len(passages)} passages indexed")
    else:
        st.error("No corpus loaded")
        st.caption(f"Expected {PASSAGES.relative_to(ROOT)}")

    if st.button("Reload corpus"):
        load_passages.clear()
        build_index.clear()
        st.rerun()

    if st.button("Warm up model"):
        try:
            build_index()
            st.success("Retrieval ready")
        except ImportError:
            st.error("sentence-transformers is not installed.")

    st.subheader("Generation")
    try:
        from generation import is_available, MODEL_NAME
        if is_available():
            st.success(f"{MODEL_NAME} ready")
        else:
            st.warning("Ollama not reachable")
            st.caption("Extractive mode: answers show the retrieved passage.")
    except ImportError:
        st.warning("Generation module not found")

    st.caption(f"Retrieval: dense - {EMBEDDING_MODEL.split('/')[-1]}")

ask_tab, eval_tab = st.tabs(["Ask", "Evaluation"])

with ask_tab:
    st.title("Casual Work Rights Assistant")
    st.caption(
        "Answers come only from Fair Work source pages held in the knowledge "
        "base. Pay rates change on 1 July each year, so confirm current figures "
        "on the Fair Work website."
    )

    question = st.text_input("Your question", placeholder="Do casual employees get sick leave?")

    if question:
        if not passages:
            st.error("No corpus loaded, so nothing can be retrieved.")
        else:
            hits = retrieve(question)

            if should_abstain(hits):
                st.warning(
                    "I don't have enough information in my sources to answer "
                    "that. Try the Fair Work Ombudsman website, or call the "
                    "Fair Work Infoline on 13 13 94."
                )
            else:
                with st.spinner("Writing an answer..."):
                    answer = generate(question, hits)

                if answer:
                    st.markdown(answer)
                    st.subheader("Sources")
                    for p, score in hits:
                        with st.expander(f"{p['title']} · similarity {score:.2f}"):
                            st.write(p["text"])
                            st.caption(p["url"])
                else:
                    top, score = hits[0]
                    st.markdown(f"**{top['title']}**")
                    st.write(top["text"])
                    st.caption(f"Source: {top['url']}")

                    if len(hits) > 1:
                        st.subheader("Other passages retrieved")
                        for p, score in hits[1:]:
                            with st.expander(f"{p['title']} · similarity {score:.2f}"):
                                st.write(p["text"])
                                st.caption(p["url"])

with eval_tab:
    st.title("Evaluation")
    questions = load_jsonl(QUESTIONS)

    if not questions:
        st.info(f"Add {QUESTIONS.relative_to(ROOT)} to run the evaluation.")
        st.caption("One JSON object per line:")
        st.code(
            '{"qid": "q1", "question": "What is casual loading?", '
            '"gold_ids": ["fwo_001_0000"], "type": "known"}\n'
            '{"qid": "q2", "question": "Why is my hourly rate higher than my '
            'friend\'s?", "gold_ids": ["fwo_001_0000"], "type": "inferred"}\n'
            '{"qid": "q3", "question": "How much is stamp duty in Victoria?", '
            '"gold_ids": [], "type": "out_of_scope"}',
            language="json",
        )
    else:
        st.caption(f"{len(questions)} questions loaded. Retrieval only - generation is not called here.")
        if st.button("Run evaluation"):
            with st.spinner("Scoring..."):
                retrieval_rows, abstention_rows = run_evaluation(questions)

            for qtype in ("known", "inferred", "unspecified"):
                subset = [r for r in retrieval_rows if r["type"] == qtype]
                if not subset:
                    continue
                st.subheader(f"{qtype.title()} questions ({len(subset)})")
                cols = st.columns(3)
                for col, metric in zip(cols, ("ndcg@1", "ndcg@3", "ndcg@5")):
                    mean = float(np.mean([r[metric] for r in subset]))
                    col.metric(metric.upper(), f"{mean:.4f}")

            if abstention_rows:
                declined = sum(r["declined"] for r in abstention_rows)
                st.subheader(f"Out-of-scope questions ({len(abstention_rows)})")
                st.metric("Correctly declined", f"{declined}/{len(abstention_rows)}")
                missed = [r["question"] for r in abstention_rows if not r["declined"]]
                if missed:
                    st.caption("Answered when it should have declined:")
                    for q in missed:
                        st.caption(f"· {q}")

            if retrieval_rows:
                header = "qid,type,ndcg@1,ndcg@3,ndcg@5\n"
                body = "\n".join(
                    f"{r['qid']},{r['type']},{r['ndcg@1']:.4f},"
                    f"{r['ndcg@3']:.4f},{r['ndcg@5']:.4f}"
                    for r in retrieval_rows
                )
                st.download_button(
                    "Download results (CSV)",
                    header + body,
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                )
