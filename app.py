"""
Casual Work Rights Assistant
Group 38, COSC2669 - WIL Project

Retrieval-augmented assistant answering questions about casual employment
rights in Australia from Fair Work Ombudsman source pages.

Retrieval is dense: passages and questions are encoded with a sentence
embedding model and compared by cosine similarity. This replaces the BM25
baseline used in the Milestone 1 reproduction. The reason is in our own
results - BM25 reached NDCG@1 of 0.5119 on known questions against 0.6905 for
dense retrieval, and the gap widens on paraphrased questions, which is how
casual workers actually ask.

The interface loads immediately. The embedding model is only loaded when it is
first needed, either by asking a question or by pressing Warm up model in the
sidebar. Use that button before a demo so the wait happens off screen.

Run:
    python -m pip install streamlit sentence-transformers numpy
    streamlit run app.py

First use downloads the embedding model (about 90 MB) and needs internet.
Afterwards it is cached locally and runs offline.

Data files:
    data/passages.jsonl   {"id", "url", "title", "text"}
    data/questions.jsonl  {"qid", "question", "gold_ids": [...], "type": "known|inferred|out_of_scope"}
"""

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st

# --------------------------------------------------------------- configuration

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5              # passages retrieved per question
MIN_SIMILARITY = 0.35  # below this, nothing is relevant enough to answer from

# MIN_SIMILARITY is the one number that decides behaviour. Cosine similarity
# runs from -1 to 1; in practice a clearly relevant passage sits above 0.5, a
# loosely related one around 0.3, and an unrelated one near 0. Set this against
# the out-of-scope questions in data/questions.jsonl and record what you chose.

DATA = Path(__file__).parent / "data"
PASSAGES = DATA / "processed" / "passages.jsonl"
QUESTIONS = DATA / "questions.jsonl"


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
    return SentenceTransformer(MODEL_NAME)


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
    """Return passages ranked by cosine similarity, above the relevance floor."""
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

SYSTEM_PROMPT = """You are helping casual employees in Australia understand their
workplace rights.

Rules:
- Answer only from the passages provided. Do not use outside knowledge.
- Cite the source URL for every claim you make.
- If the passages do not answer the question, say so plainly. Do not guess.
- Pay rates change on 1 July each year, so tell the user to confirm current
  figures on the Fair Work website.
"""


def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{i + 1}] {p['title']} ({p['url']})\n{p['text']}"
        for i, (p, _) in enumerate(hits)
    )
    return f"{SYSTEM_PROMPT}\n\nPassages:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def generate(question, hits):
    """Return a generated answer, or None to fall back to extractive mode.

    To enable generation, set the key in your environment and fill this in.
    Keep the key out of the repository.

        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=700,
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
        )
        return msg.content[0].text
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return None


# ------------------------------------------------------------------ evaluation

def ndcg_at_k(ranked_ids, gold_ids, k):
    gains = [1.0 if pid in gold_ids else 0.0 for pid in ranked_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(questions):
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
        st.caption("Add data/passages.jsonl, then reload.")

    if st.button("Reload corpus"):
        load_passages.clear()
        build_index.clear()
        st.rerun()

    if st.button("Warm up model"):
        try:
            build_index()
            st.success("Ready")
        except ImportError:
            st.error("sentence-transformers is not installed.")

    st.caption(f"Retrieval: dense - {MODEL_NAME.split('/')[-1]}")

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
            try:
                hits = retrieve(question)
            except ImportError:
                st.error("The sentence-transformers package is not installed.")
                st.code("python -m pip install sentence-transformers", language="bash")
                st.stop()

            if should_abstain(hits):
                st.warning(
                    "I don't have enough information in my sources to answer "
                    "that. Try the Fair Work Ombudsman website, or call the "
                    "Fair Work Infoline on 13 13 94."
                )
            else:
                answer = generate(question, hits)
                if answer:
                    st.markdown(answer)
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
        st.info("Add data/questions.jsonl to run the evaluation.")
        st.caption("One JSON object per line:")
        st.code(
            '{"qid": "q1", "question": "What is casual loading?", '
            '"gold_ids": ["fwo_001_001"], "type": "known"}\n'
            '{"qid": "q2", "question": "Why is my hourly rate higher than my '
            'friend\'s?", "gold_ids": ["fwo_001_001"], "type": "inferred"}\n'
            '{"qid": "q3", "question": "How much is stamp duty in Victoria?", '
            '"gold_ids": [], "type": "out_of_scope"}',
            language="json",
        )
    else:
        st.caption(f"{len(questions)} questions loaded.")
        if st.button("Run evaluation"):
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
