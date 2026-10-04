import html

import streamlit as st

from src.retrieval.retrieval import (
    load_knowledge_base,
    load_model,
    build_passage_embeddings,
    build_tfidf_index,
    retrieve
)

from src.generation.generation import generate_answer

# generation.py may or may not define GenerationUnavailable depending on which
# version is on the branch. Fall back to Exception so the app degrades either
# way instead of failing at import time.
try:
    from src.generation.generation import GenerationUnavailable
except ImportError:
    GenerationUnavailable = Exception


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Casual Employee Work-Rights Assistant",
    page_icon="💼"
)


# --------------------------------------------------
# Load knowledge base and model
# --------------------------------------------------

@st.cache_data
def load_data():
    return load_knowledge_base()


@st.cache_resource
def load_retrieval_model():
    return load_model()


# A leading underscore tells Streamlit not to hash the argument. DataFrames
# are unhashable, so without it the cache raises UnhashableParamError.
@st.cache_resource
def load_embeddings(_df, _model):
    return build_passage_embeddings(_df, _model)


@st.cache_resource
def load_tfidf_index(_df):
    return build_tfidf_index(_df)


df = load_data()
model = load_retrieval_model()
passage_embeddings = load_embeddings(df, model)
tfidf_vectorizer, tfidf_embeddings = load_tfidf_index(df)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "turns" not in st.session_state:
    st.session_state.turns = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# How confident retrieval must be before we let the model answer.
# Lower this if too many real questions get flagged as unanswered;
# raise it if wrong answers are getting through.
CONFIDENCE_THRESHOLD = 0.15

PRESETS = {
    "Pay & wages": "What is the pay rate for casual employees?",
    "Leave entitlements": "What leave are casual employees entitled to?",
    "Casual employment": "What is casual employment?",
}

st.markdown(
    """
    <style>
    /* Centered header */
    .st-key-header h1,
    .st-key-header p,
    .st-key-header div[data-testid="stMarkdownContainer"] {
        text-align: center;
        justify-content: center;
    }
    .st-key-header [data-testid="stIconMaterial"] {
        font-size: 8rem;
    }

    /* Topic pills */
    .st-key-pills {
        flex-direction: row !important;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.6rem !important;
    }
    .st-key-pills > div { width: auto !important; }
    .st-key-pills button {
        border-radius: 999px;
        padding: 0.4rem 1.2rem;
        border: 1px solid rgba(128,128,128,0.35);
        background: transparent;
    }
    .st-key-pills button:hover {
        background: rgba(128,128,128,0.15);
    }

    /* User question bubble */
    .user-row {
        display: flex;
        justify-content: flex-end;
        margin: 1.2rem 0 0.8rem 0;
    }
    .user-bubble {
        background-color: rgba(128,128,128,0.18);
        padding: 0.7rem 1.2rem;
        border-radius: 18px;
        max-width: 80%;
    }

    /* Answer card */
    [class*="st-key-answer_card"] {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 16px;
        padding: 0.5rem 0.8rem;
    }
    [class*="st-key-answer_card"] [data-testid="stExpander"] {
        border-radius: 10px;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

with st.container(key="header"):
    st.markdown("# :material/balance:")
    st.title("Work Rights Assistant")
    st.write("Ask questions about casual employment and workplace rights")

with st.container(key="pills"):
    for label, preset_q in PRESETS.items():
        if st.button(label, key=f"pill_{label}"):
            st.session_state.pending_question = preset_q


# --------------------------------------------------
# Conversation
# --------------------------------------------------

for turn_idx, turn in enumerate(st.session_state.turns):

    st.markdown(
        f"<div class='user-row'><div class='user-bubble'>"
        f"{html.escape(turn['question'])}</div></div>",
        unsafe_allow_html=True
    )

    show_sources = True

    with st.container(key=f"answer_card_{turn_idx}"):

        if turn.get("model_down"):
            # Generation could not run at all. Show the top passage so the
            # retrieval half is still usable, rather than a blank card.
            st.warning(
                "The local model isn't responding, so I'm showing the "
                "retrieved passage instead. Check that Ollama is running."
            )
            if turn["results"]:
                top = turn["results"][0]
                st.write(top["text"])
                st.caption(f"Source: {top['url']}")
            show_sources = False

        elif turn.get("unanswered"):
            st.warning(
                "I found related pages but none of them answer that "
                "question. Try the Fair Work Ombudsman website, or call "
                "the Fair Work Infoline on 13 13 94."
            )
            show_sources = False

        else:
            st.write(turn["answer"])

        if turn["results"] and show_sources:
            st.caption("Learn more")
            for rank, result in enumerate(turn["results"], start=1):
                with st.expander(
                    f"Source {rank}: {result['title']} "
                    f"(similarity: {result['score']:.4f})"
                ):
                    st.write(result["text"])
                    st.caption(f"Passage ID: {result['passage_id']}")
                    st.caption(f"Source: {result['url']}")


# --------------------------------------------------
# Input
# --------------------------------------------------

typed_question = st.chat_input("Ask about your workplace rights...")
question = st.session_state.pending_question or typed_question

if question and question.strip():

    st.session_state.pending_question = None
    clean_question = question.strip()

    results = retrieve(
        clean_question, df, model, passage_embeddings,
        tfidf_vectorizer, tfidf_embeddings, k=10
    )
    top_results = results[:3]

    FALLBACK_PHRASES = [
        "the information is not available in the knowledge base",
    ]

    low_confidence = (
        len(top_results) == 0
        or top_results[0]["score"] < CONFIDENCE_THRESHOLD
    )

    answer = None
    unanswered = False
    model_down = False

    if low_confidence:
        unanswered = True
    else:
        try:
            answer = generate_answer(clean_question, top_results)
        except GenerationUnavailable:
            # Ollama is not running, or the model failed. Degrade to showing
            # the retrieved passages rather than crashing with a traceback.
            model_down = True

        if not model_down:
            unanswered = any(
                phrase in str(answer).lower() for phrase in FALLBACK_PHRASES
            )
            if unanswered:
                answer = None

    st.session_state.turns.append({
        "question": clean_question,
        "answer": answer,
        "results": top_results,
        "unanswered": unanswered,
        "model_down": model_down
    })

    st.rerun()
