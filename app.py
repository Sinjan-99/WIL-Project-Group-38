import streamlit as st

from src.retrieval.retrieval import (
    load_knowledge_base,
    load_model,
    build_passage_embeddings,
    build_tfidf_index,
    retrieve
)

from src.generation.generation import generate_answer


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


@st.cache_resource
def load_embeddings(df, _model):
    return build_passage_embeddings(df, _model)

@st.cache_resource
def load_tfidf_index(df):
    return build_tfidf_index(df)

df = load_data()
model = load_retrieval_model()
passage_embeddings = load_embeddings(df, model)
tfidf_vectorizer, tfidf_embeddings = load_tfidf_index(df)

# # CHANGE 8 
import html

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

    with st.container(key=f"answer_card_{turn_idx}"):
        if turn.get("unanswered"):
            st.warning(
                "I found related pages but none of them answer that "
                "question. Try the Fair Work Ombudsman website, or call "
                "the Fair Work Infoline on 13 13 94."
            )
        else:
            st.write(turn["answer"])

        if turn["results"] and not turn.get("unanswered"):
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

    if low_confidence:
        answer = None
        unanswered = True
    else:
        answer = generate_answer(clean_question, top_results)
        unanswered = any(
            phrase in answer.lower() for phrase in FALLBACK_PHRASES
        )
        if unanswered:
            answer = None

    st.session_state.turns.append({
        "question": clean_question,
        "answer": answer,
        "results": top_results,
        "unanswered": unanswered
    })

    st.rerun()


# # CHANGE 7 
# import html

# # --------------------------------------------------
# # Session state
# # --------------------------------------------------

# if "turns" not in st.session_state:
#     st.session_state.turns = []
# if "pending_question" not in st.session_state:
#     st.session_state.pending_question = None

# PRESETS = {
#     "Pay & wages": "What is the pay rate for casual employees?",
#     "Leave entitlements": "What leave are casual employees entitled to?",
#     "Casual employment": "What is casual employment?",
# }

# st.markdown(
#     """
#     <style>
#     /* Centered header */
#     .st-key-header h1,
#     .st-key-header p,
#     .st-key-header div[data-testid="stMarkdownContainer"] {
#         text-align: center;
#         justify-content: center;
#     }
#     .st-key-header [data-testid="stIconMaterial"] {
#         font-size: 8rem;
#     }

#     /* Topic pills */
#     .st-key-pills {
#         flex-direction: row !important;
#         flex-wrap: wrap;
#         justify-content: center;
#         gap: 0.6rem !important;
#     }
#     .st-key-pills > div { width: auto !important; }
#     .st-key-pills button {
#         border-radius: 999px;
#         padding: 0.4rem 1.2rem;
#         border: 1px solid rgba(128,128,128,0.35);
#         background: transparent;
#     }
#     .st-key-pills button:hover {
#         background: rgba(128,128,128,0.15);
#     }

#     /* User question bubble */
#     .user-row {
#         display: flex;
#         justify-content: flex-end;
#         margin: 1.2rem 0 0.8rem 0;
#     }
#     .user-bubble {
#         background-color: rgba(128,128,128,0.18);
#         padding: 0.7rem 1.2rem;
#         border-radius: 18px;
#         max-width: 80%;
#     }

#     /* Answer card */
#     [class*="st-key-answer_card"] {
#         border: 1px solid rgba(128,128,128,0.3);
#         border-radius: 16px;
#         padding: 0.5rem 0.8rem;
#     }
#     [class*="st-key-answer_card"] [data-testid="stExpander"] {
#         border-radius: 10px;
#     }

#     /* Chat input */
#     [data-testid="stChatInput"] {
#         border-radius: 16px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # --------------------------------------------------
# # Header
# # --------------------------------------------------

# with st.container(key="header"):
#     st.markdown("# :material/balance:")
#     st.title("Work Rights Assistant")
#     st.write("Ask questions about casual employment and workplace rights")

# with st.container(key="pills"):
#     for label, preset_q in PRESETS.items():
#         if st.button(label, key=f"pill_{label}"):
#             st.session_state.pending_question = preset_q


# # --------------------------------------------------
# # Conversation
# # --------------------------------------------------

# for turn_idx, turn in enumerate(st.session_state.turns):

#     st.markdown(
#         f"<div class='user-row'><div class='user-bubble'>"
#         f"{html.escape(turn['question'])}</div></div>",
#         unsafe_allow_html=True
#     )

#     with st.container(key=f"answer_card_{turn_idx}"):
#         st.write(turn["answer"])

#         st.caption("Learn more")
#         for rank, result in enumerate(turn["results"], start=1):
#             with st.expander(
#                 f"Source {rank}: {result['title']} "
#                 f"(similarity: {result['score']:.4f})"
#             ):
#                 st.write(result["text"])
#                 st.caption(f"Passage ID: {result['passage_id']}")
#                 st.caption(f"Source: {result['url']}")


# # --------------------------------------------------
# # Input
# # --------------------------------------------------

# typed_question = st.chat_input("Ask about your workplace rights...")
# question = st.session_state.pending_question or typed_question

# if question and question.strip():

#     st.session_state.pending_question = None
#     clean_question = question.strip()

#     results = retrieve(
#         clean_question, df, model, passage_embeddings,
#         tfidf_vectorizer, tfidf_embeddings, k=10
#     )

#     answer = generate_answer(clean_question, results[:3])

#     st.session_state.turns.append({
#         "question": clean_question,
#         "answer": answer,
#         "results": results[:3]
#     })

#     st.rerun()



# # CHANGE 6 
# import html

# # --------------------------------------------------
# # Session state
# # --------------------------------------------------

# if "turns" not in st.session_state:
#     st.session_state.turns = []
# if "pending_question" not in st.session_state:
#     st.session_state.pending_question = None

# PRESETS = {
#     "Pay & wages": "What is the pay rate for casual employees?",
#     "Leave entitlements": "What leave are casual employees entitled to?",
#     "Casual employment": "What is casual employment?",
# }

# st.markdown(
#     """
#     <style>
#     /* Page */
#     .stApp { background-color: #0b1120; color: #e6edf7; }
#     [data-testid="stHeader"] { background: transparent; }
#     [data-testid="stBottom"] > div { background-color: #0b1120; }

#     /* Centered header */
#     .st-key-header h1,
#     .st-key-header p,
#     .st-key-header div[data-testid="stMarkdownContainer"] {
#         text-align: center;
#         justify-content: center;
#     }
#     .st-key-header [data-testid="stIconMaterial"] {
#         color: #6ea8fe;
#         font-size: 3.5rem;
#     }

#     /* Topic pills */
#     .st-key-pills {
#         flex-direction: row !important;
#         flex-wrap: wrap;
#         justify-content: center;
#         gap: 0.6rem !important;
#     }
#     .st-key-pills > div { width: auto !important; }
#     .st-key-pills button {
#         border-radius: 999px;
#         padding: 0.4rem 1.2rem;
#         border: 1px solid rgba(120,150,220,0.35);
#         background: rgba(255,255,255,0.03);
#     }
#     .st-key-pills button:hover {
#         background: rgba(110,168,254,0.15);
#         border-color: rgba(110,168,254,0.6);
#     }

#     /* User question bubble */
#     .user-row {
#         display: flex;
#         justify-content: flex-end;
#         margin: 1.2rem 0 0.8rem 0;
#     }
#     .user-bubble {
#         background-color: rgba(59,110,200,0.28);
#         border: 1px solid rgba(110,168,254,0.25);
#         padding: 0.7rem 1.2rem;
#         border-radius: 18px;
#         max-width: 80%;
#     }

#     /* Answer card */
#     [class*="st-key-answer_card"] {
#         background: rgba(255,255,255,0.03);
#         border: 1px solid rgba(120,150,220,0.25);
#         border-radius: 16px;
#         padding: 0.5rem 0.8rem;
#     }
#     [class*="st-key-answer_card"] [data-testid="stIconMaterial"] {
#         color: #6ea8fe;
#     }
#     [class*="st-key-answer_card"] [data-testid="stExpander"] {
#         border-radius: 10px;
#         border: 1px solid rgba(120,150,220,0.25);
#     }

#     /* Feedback icons: one tight row */
#     [class*="st-key-feedback_icons"] {
#         flex-direction: row !important;
#         gap: 0rem !important;
#     }
#     [class*="st-key-feedback_icons"] > div { width: auto !important; }
#     [class*="st-key-feedback_icons"] button {
#         border: none;
#         background: transparent;
#         padding: 0.1rem 0.3rem;
#         min-height: 2rem;
#     }
#     [class*="st-key-feedback_icons"] button:hover {
#         background-color: rgba(128,128,128,0.15);
#     }

#     /* Chat input */
#     [data-testid="stChatInput"] {
#         border-radius: 16px;
#         border: 1px solid rgba(120,150,220,0.35);
#     }
#     [data-testid="stChatInputSubmitButton"] {
#         background-color: #4a6fb5;
#         border-radius: 50%;
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # --------------------------------------------------
# # Header
# # --------------------------------------------------

# with st.container(key="header"):
#     st.markdown("# :material/smart_toy:")
#     st.title("Work Rights Assistant")
#     st.write("Ask questions about casual employment and workplace rights.")

# with st.container(key="pills"):
#     for label, preset_q in PRESETS.items():
#         if st.button(label, key=f"pill_{label}"):
#             st.session_state.pending_question = preset_q


# # --------------------------------------------------
# # Conversation
# # --------------------------------------------------

# for turn_idx, turn in enumerate(st.session_state.turns):

#     st.markdown(
#         f"<div class='user-row'><div class='user-bubble'>"
#         f"{html.escape(turn['question'])}</div></div>",
#         unsafe_allow_html=True
#     )

#     with st.container(key=f"answer_card_{turn_idx}"):
#         icon_col, text_col = st.columns([1, 15])
#         with icon_col:
#             st.markdown(":material/smart_toy:")
#         with text_col:
#             st.write(turn["answer"])

#         st.caption("Learn more")
#         for rank, result in enumerate(turn["results"], start=1):
#             with st.expander(
#                 f"Source {rank}: {result['title']} "
#                 f"(similarity: {result['score']:.4f})"
#             ):
#                 st.write(result["text"])
#                 st.caption(f"Passage ID: {result['passage_id']}")
#                 st.caption(f"Source: {result['url']}")

#         with st.container(key=f"feedback_icons_{turn_idx}"):
#             st.button(" ", icon=":material/thumb_up:", key=f"thumbup_btn_{turn_idx}")
#             st.button(" ", icon=":material/thumb_down:", key=f"thumbdown_btn_{turn_idx}")


# # --------------------------------------------------
# # Input
# # --------------------------------------------------

# typed_question = st.chat_input("Ask about your workplace rights...")
# question = st.session_state.pending_question or typed_question

# if question and question.strip():

#     st.session_state.pending_question = None
#     clean_question = question.strip()

#     results = retrieve(
#         clean_question, df, model, passage_embeddings,
#         tfidf_vectorizer, tfidf_embeddings, k=10
#     )

#     answer = generate_answer(clean_question, results[:3])

#     st.session_state.turns.append({
#         "question": clean_question,
#         "answer": answer,
#         "results": results[:3]
#     })

#     st.rerun()


# # CHANGE 5
# import html

# # --------------------------------------------------
# # Session state
# # --------------------------------------------------

# if "turns" not in st.session_state:
#     st.session_state.turns = []
# if "input_key_counter" not in st.session_state:
#     st.session_state.input_key_counter = 0

# st.markdown(
#     """
#     <style>
#     /* Question bubble (right aligned, like Claude) */
#     .user-row {
#         display: flex;
#         justify-content: flex-end;
#         margin: 1rem 0 0.5rem 0;
#     }
#     .user-bubble {
#         background-color: rgba(128,128,128,0.18);
#         padding: 0.7rem 1.1rem;
#         border-radius: 18px;
#         max-width: 80%;
#     }

#     /* Feedback icons: one tight row */
#     [class*="st-key-feedback_icons"] {
#         flex-direction: row !important;
#         gap: 0rem !important;
#     }
#     [class*="st-key-feedback_icons"] > div {
#         width: auto !important;
#     }
#     [class*="st-key-feedback_icons"] button {
#         border: none;
#         background: transparent;
#         padding: 0.1rem 0.3rem;
#         min-height: 2rem;
#     }
#     [class*="st-key-feedback_icons"] button:hover {
#         background-color: rgba(128,128,128,0.15);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # --------------------------------------------------
# # User interface
# # --------------------------------------------------

# st.title("Work Rights Assistant")

# st.write(
#     "Ask a question about casual employment and workplace rights"
# )

# # ---- Show every turn so far ----
# for turn_idx, turn in enumerate(st.session_state.turns):

#     st.markdown(
#         f"<div class='user-row'><div class='user-bubble'>"
#         f"{html.escape(turn['question'])}</div></div>",
#         unsafe_allow_html=True
#     )

#     st.write(turn["answer"])

#     st.markdown("##### Sources")
#     for rank, result in enumerate(turn["results"], start=1):
#         with st.expander(
#             f"Source {rank}: {result['title']} "
#             f"(similarity: {result['score']:.4f})"
#         ):
#             st.write(result["text"])
#             st.caption(f"Passage ID: {result['passage_id']}")
#             st.caption(f"Source: {result['url']}")

#     with st.container(key=f"feedback_icons_{turn_idx}"):
#         st.button(" ", icon=":material/thumb_up:", key=f"thumbup_btn_{turn_idx}")
#         st.button(" ", icon=":material/thumb_down:", key=f"thumbdown_btn_{turn_idx}")

#     st.divider()

# # ---- Input for a new question ----
# question = st.text_input(
#     "Enter your question:",
#     placeholder="e.g. Do casual employees get annual leave?",
#     label_visibility="collapsed",
#     key=f"question_input_{st.session_state.input_key_counter}"
# )

# if st.button("Ask") and question.strip():

#     clean_question = question.strip()

#     results = retrieve(
#         clean_question, df, model, passage_embeddings,
#         tfidf_vectorizer, tfidf_embeddings, k=10
#     )

#     answer = generate_answer(clean_question, results[:3])

#     st.session_state.turns.append({
#         "question": clean_question,
#         "answer": answer,
#         "results": results[:3]
#     })

#     st.session_state.input_key_counter += 1
#     st.rerun()





# # CHANGE 4
# import html

# # --------------------------------------------------
# # Sidebar: new chat, search, recent
# # --------------------------------------------------

# if "conversations" not in st.session_state:
#     st.session_state.conversations = [[]]
# if "current_conv" not in st.session_state:
#     st.session_state.current_conv = 0
# if "input_key_counter" not in st.session_state:
#     st.session_state.input_key_counter = 0
# if "show_search" not in st.session_state:
#     st.session_state.show_search = False

# st.markdown(
#     """
#     <style>
#     /* Sidebar */
#     .st-key-sidebar_nav {
#         gap: 0rem !important;
#     }
#     .st-key-sidebar_nav button {
#         border: none;
#         background: transparent;
#         text-align: left;
#         padding: 0.2rem 0.4rem;
#         min-height: 2rem;
#         width: 100%;
#         justify-content: flex-start;
#     }
#     .st-key-sidebar_nav button:hover {
#         background-color: rgba(128,128,128,0.15);
#     }

#     /* Question bubble (right aligned, like Claude) */
#     .user-row {
#         display: flex;
#         justify-content: flex-end;
#         margin: 1rem 0 0.5rem 0;
#     }
#     .user-bubble {
#         background-color: rgba(128,128,128,0.18);
#         padding: 0.7rem 1.1rem;
#         border-radius: 18px;
#         max-width: 80%;
#     }

#     /* Feedback icons: one tight row */
#     [class*="st-key-feedback_icons"] {
#         flex-direction: row !important;
#         gap: 0rem !important;
#     }
#     [class*="st-key-feedback_icons"] > div {
#         width: auto !important;
#     }
#     [class*="st-key-feedback_icons"] button {
#         border: none;
#         background: transparent;
#         padding: 0.1rem 0.3rem;
#         min-height: 2rem;
#     }
#     [class*="st-key-feedback_icons"] button:hover {
#         background-color: rgba(128,128,128,0.15);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# with st.sidebar:
#     with st.container(key="sidebar_nav"):
#         if st.button(" New chat", icon=":material/add:", key="new_chat_btn"):
#             st.session_state.conversations.append([])
#             st.session_state.current_conv = len(st.session_state.conversations) - 1
#             st.session_state.input_key_counter += 1
#             st.rerun()

#         if st.button(" Search chats", icon=":material/search:", key="search_chat_btn"):
#             st.session_state.show_search = not st.session_state.show_search

#         if st.session_state.show_search:
#             search_term = st.text_input(
#                 "Search", placeholder="Search chats", label_visibility="collapsed"
#             )
#         else:
#             search_term = ""

#         st.caption("Recent")

#         for i, conv in enumerate(st.session_state.conversations):
#             if len(conv) == 0:
#                 continue
#             title = conv[0]["question"][:35]
#             if search_term.lower() in title.lower():
#                 if st.button(title, key=f"hist_{i}"):
#                     st.session_state.current_conv = i
#                     st.session_state.input_key_counter += 1
#                     st.rerun()


# # --------------------------------------------------
# # User interface
# # --------------------------------------------------

# st.title("Work Rights Assistant")

# st.write(
#     "Ask a question about casual employment and workplace rights."
# )

# current = st.session_state.conversations[st.session_state.current_conv]

# # ---- Show every turn so far in this conversation ----
# for turn_idx, turn in enumerate(current):

#     st.markdown(
#         f"<div class='user-row'><div class='user-bubble'>"
#         f"{html.escape(turn['question'])}</div></div>",
#         unsafe_allow_html=True
#     )

#     st.markdown("#### Answer")
#     st.write(turn["answer"])

#     st.subheader("Retrieved sources")
#     for rank, result in enumerate(turn["results"], start=1):
#         with st.expander(
#             f"Source {rank}: {result['title']} "
#             f"(similarity: {result['score']:.4f})"
#         ):
#             st.write(result["text"])
#             st.caption(f"Passage ID: {result['passage_id']}")
#             st.caption(f"Source: {result['url']}")

#     with st.container(key=f"feedback_icons_{turn_idx}"):
#         st.button(" ", icon=":material/edit:", key=f"edit_btn_{turn_idx}")
#         st.button(" ", icon=":material/thumb_up:", key=f"thumbup_btn_{turn_idx}")
#         st.button(" ", icon=":material/thumb_down:", key=f"thumbdown_btn_{turn_idx}")

#     st.divider()

# # ---- Input for a new question ----
# question = st.text_input(
#     "Enter your question:",
#     placeholder="e.g. Do casual employees get annual leave?",
#     label_visibility="collapsed",
#     key=f"question_input_{st.session_state.input_key_counter}"
# )

# if st.button("Ask") and question.strip():

#     clean_question = question.strip()

#     results = retrieve(
#         clean_question, df, model, passage_embeddings,
#         tfidf_vectorizer, tfidf_embeddings, k=10
#     )

#     answer = generate_answer(clean_question, results[:3])

#     current.append({
#         "question": clean_question,
#         "answer": answer,
#         "results": results[:3]
#     })

#     st.session_state.input_key_counter += 1
#     st.rerun()




# CHANGE 3
# # --------------------------------------------------
# # Sidebar: new chat, search, recent
# # --------------------------------------------------

# if "conversations" not in st.session_state:
#     st.session_state.conversations = [[]]  # list of conversations; each is a list of turns
# if "current_conv" not in st.session_state:
#     st.session_state.current_conv = 0
# if "input_key_counter" not in st.session_state:
#     st.session_state.input_key_counter = 0
# if "show_search" not in st.session_state:
#     st.session_state.show_search = False

# st.markdown(
#     """
#     <style>
#     .st-key-sidebar_nav button {
#         border: none;
#         background: transparent;
#         text-align: left;
#         padding: 0.3rem 0.4rem;
#         width: 100%;
#         justify-content: flex-start;
#     }
#     .st-key-sidebar_nav button:hover {
#         background-color: rgba(128,128,128,0.15);
#     }
#     .st-key-sidebar_nav div[data-testid="stVerticalBlock"] {
#         gap: 0.1rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# with st.sidebar:
#     with st.container(key="sidebar_nav"):
#         if st.button(" New chat", icon=":material/add:", key="new_chat_btn"):
#             st.session_state.conversations.append([])
#             st.session_state.current_conv = len(st.session_state.conversations) - 1
#             st.session_state.input_key_counter += 1
#             st.rerun()

#         if st.button(" Search chats", icon=":material/search:", key="search_chat_btn"):
#             st.session_state.show_search = not st.session_state.show_search

#         if st.session_state.show_search:
#             search_term = st.text_input(
#                 "Search", placeholder="Search chats", label_visibility="collapsed"
#             )
#         else:
#             search_term = ""

#         st.caption("Recent")

#         for i, conv in enumerate(st.session_state.conversations):
#             if len(conv) == 0:
#                 continue
#             title = conv[0]["question"][:35]
#             if search_term.lower() in title.lower():
#                 if st.button(title, key=f"hist_{i}"):
#                     st.session_state.current_conv = i
#                     st.session_state.input_key_counter += 1
#                     st.rerun()


# # --------------------------------------------------
# # User interface
# # --------------------------------------------------

# st.title("Work Rights Assistant")

# st.write(
#     "Ask a question about casual employment and workplace rights."
# )

# current = st.session_state.conversations[st.session_state.current_conv]

# # ---- Show every turn so far in this conversation ----
# for turn_idx, turn in enumerate(current):

#     st.markdown(f"**{turn['question']}**")

#     st.markdown("#### Answer")
#     st.write(turn["answer"])

#     st.subheader("Retrieved sources")
#     for rank, result in enumerate(turn["results"], start=1):
#         with st.expander(
#             f"Source {rank}: {result['title']} "
#             f"(similarity: {result['score']:.4f})"
#         ):
#             st.write(result["text"])
#             st.caption(f"Passage ID: {result['passage_id']}")
#             st.caption(f"Source: {result['url']}")

#     st.markdown(
#         """
#         <style>
#         .st-key-feedback_icons button {
#             border: none;
#             background: transparent;
#             padding: 0.1rem 0.2rem;
#         }
#         .st-key-feedback_icons button:hover {
#             background-color: rgba(128,128,128,0.15);
#         }
#         .st-key-feedback_icons div[data-testid="stHorizontalBlock"] {
#             gap: 0.02rem;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     with st.container(key="feedback_icons"):
#         col1, col2, col3, col4 = st.columns([1, 1, 1, 9])
#         with col1:
#             st.button(" ", icon=":material/edit:", key=f"edit_btn_{turn_idx}")
#         with col2:
#             st.button(" ", icon=":material/thumb_up:", key=f"thumbup_btn_{turn_idx}")
#         with col3:
#             st.button(" ", icon=":material/thumb_down:", key=f"thumbdown_btn_{turn_idx}")

#     st.divider()

# # ---- Input for a new question ----
# question = st.text_input(
#     "Enter your question:",
#     placeholder="e.g. Do casual employees get annual leave?",
#     label_visibility="collapsed",
#     key=f"question_input_{st.session_state.input_key_counter}"
# )

# if st.button("Ask") and question:

#     results = retrieve(
#         question, df, model, passage_embeddings,
#         tfidf_vectorizer, tfidf_embeddings, k=10
#     )

#     answer = generate_answer(question, results[:3])

#     current.append({
#         "question": question,
#         "answer": answer,
#         "results": results[:3]
#     })

#     st.session_state.input_key_counter += 1
#     st.rerun()




# CHANGE 2
# # --------------------------------------------------
# # Sidebar: new chat, search, recent
# # --------------------------------------------------

# if "history" not in st.session_state:
#     st.session_state.history = []
# if "last_question" not in st.session_state:
#     st.session_state.last_question = ""
# if "last_answer" not in st.session_state:
#     st.session_state.last_answer = None
# if "last_results" not in st.session_state:
#     st.session_state.last_results = None
# if "input_key_counter" not in st.session_state:
#     st.session_state.input_key_counter = 0
# if "show_search" not in st.session_state:
#     st.session_state.show_search = False

# st.markdown(
#     """
#     <style>
#     .st-key-sidebar_nav button {
#         border: none;
#         background: transparent;
#         text-align: left;
#         padding: 0.3rem 0.4rem;
#         width: 100%;
#         justify-content: flex-start;
#     }
#     .st-key-sidebar_nav button:hover {
#         background-color: rgba(128,128,128,0.15);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# with st.sidebar:
#     with st.container(key="sidebar_nav"):
#         if st.button(" New chat", icon=":material/add:", key="new_chat_btn"):
#             st.session_state.last_question = ""
#             st.session_state.last_answer = None
#             st.session_state.last_results = None
#             st.session_state.input_key_counter += 1
#             st.rerun()

#         if st.button(" Search chats", icon=":material/search:", key="search_chat_btn"):
#             st.session_state.show_search = not st.session_state.show_search

#         if st.session_state.show_search:
#             search_term = st.text_input(
#                 "Search", placeholder="Search chats", label_visibility="collapsed"
#             )
#         else:
#             search_term = ""

#         st.caption("Recent")

#         for i, item in enumerate(reversed(st.session_state.history)):
#             if search_term.lower() in item["question"].lower():
#                 if st.button(item["question"][:35], key=f"hist_{i}"):
#                     st.session_state.last_question = item["question"]
#                     st.session_state.last_answer = item["answer"]
#                     st.session_state.last_results = item["results"]
#                     st.session_state.input_key_counter += 1
#                     st.rerun()


# # --------------------------------------------------
# # User interface
# # --------------------------------------------------

# st.title("Work Rights Assistant")

# st.write(
#     "Ask a question about casual employment and workplace rights."
# )

# question = st.text_input(
#     "Enter your question:",
#     value=st.session_state.last_question,
#     placeholder="e.g. Do casual employees get annual leave?",
#     label_visibility="collapsed",
#     key=f"question_input_{st.session_state.input_key_counter}"
# )

# if st.button("Ask") and question:

#     results = retrieve(
#         question, df, model, passage_embeddings,
#         tfidf_vectorizer, tfidf_embeddings, k=10
#     )

#     answer = generate_answer(question, results[:3])

#     st.session_state.last_question = question
#     st.session_state.last_answer = answer
#     st.session_state.last_results = results[:3]
#     st.session_state.history.append({
#         "question": question,
#         "answer": answer,
#         "results": results[:3]
#     })

# if st.session_state.last_answer:

#     # --------------------------------------------------
#     # Display answer
#     # --------------------------------------------------

#     st.markdown("#### Answer")
#     st.write(st.session_state.last_answer)

#     # --------------------------------------------------
#     # Display retrieved sources
#     # --------------------------------------------------

#     st.subheader("Retrieved sources")

#     for rank, result in enumerate(st.session_state.last_results, start=1):
#         with st.expander(
#             f"Source {rank}: {result['title']} "
#             f"(similarity: {result['score']:.4f})"
#         ):
#             st.write(result["text"])
#             st.caption(f"Passage ID: {result['passage_id']}")
#             st.caption(f"Source: {result['url']}")

#     # --------------------------------------------------
#     # Feedback icons
#     # --------------------------------------------------

#     st.markdown(
#         """
#         <style>
#         .st-key-feedback_icons button {
#             border: none;
#             background: transparent;
#             padding: 0.2rem 0.4rem;
#         }
#         .st-key-feedback_icons button:hover {
#             background-color: rgba(128,128,128,0.15);
#         }
#         .st-key-feedback_icons div[data-testid="stHorizontalBlock"] {
#             gap: 0.1rem;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     with st.container(key="feedback_icons"):
#         col1, col2, col3, col4 = st.columns([1, 1, 1, 9])
#         with col1:
#             st.button(" ", icon=":material/edit:", key="edit_btn")
#         with col2:
#             st.button(" ", icon=":material/thumb_up:", key="thumbup_btn")
#         with col3:
#             st.button(" ", icon=":material/thumb_down:", key="thumbdown_btn")





# CHANGE 1 
# # --------------------------------------------------
# # User interface
# # --------------------------------------------------

# st.title("Work Rights Assistant")

# st.write(
#     "Ask a question about casual employment and workplace rights."
# )

# question = st.text_input(
#     "Enter your question:",
#     placeholder="e.g. Do casual employees get annual leave?"
# )


# if st.button("Ask") and question:

#     # Retrieve relevant documents
#     results = retrieve(
#         question,
#         df,
#         model,
#         passage_embeddings,
#         tfidf_vectorizer,
#         tfidf_embeddings,
#         k=10
#     )

#     # Generate answer
#     answer = generate_answer(
#         question,
#         results[:3]
#     )

#     # --------------------------------------------------
#     # Display answer
#     # --------------------------------------------------

#     st.markdown("#### Answer")

#     st.write(answer)


#     # --------------------------------------------------
#     # Display retrieved sources
#     # --------------------------------------------------

#     st.subheader("Retrieved sources")

#     for rank, result in enumerate(results[:3], start=1):

#         with st.expander(
#             f"Source {rank}: {result['title']} "
#             f"(similarity: {result['score']:.4f})"
#         ):

#             st.write(result["text"])

#             st.caption(
#                 f"Passage ID: {result['passage_id']}"
#             )

#             st.caption(
#                 f"Source: {result['url']}"
#             )

#     # --------------------------------------------------
#     # Feedback icons
#     # --------------------------------------------------

#     st.markdown(
#         """
#         <style>
#         .st-key-feedback_icons button {
#             border: none;
#             background: transparent;
#             padding: 0.2rem 0.4rem;
#         }
#         .st-key-feedback_icons button:hover {
#             background-color: rgba(128,128,128,0.15);
#         }
#         .st-key-feedback_icons div[data-testid="stHorizontalBlock"] {
#             gap: 0.2rem;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     with st.container(key="feedback_icons"):
#         col1, col2, col3, col4 = st.columns([1, 1, 1, 9])
#         with col1:
#             st.button(" ", icon=":material/edit:", key="edit_btn")
#         with col2:
#             st.button(" ", icon=":material/thumb_up:", key="thumbup_btn")
#         with col3:
#             st.button(" ", icon=":material/thumb_down:", key="thumbdown_btn")






## INITIAL CODE 
# # --------------------------------------------------
# # User interface
# # --------------------------------------------------

# st.title("Work Rights Assistant")

# st.write(
#     "Ask a question about casual employment and workplace rights."
# )

# question = st.text_input(
#     "Enter your question:",
#     placeholder="e.g. Do casual employees get annual leave?"
# )


# if st.button("Ask") and question:

#     # Retrieve relevant documents
#     results = retrieve(
#         question,
#         df,
#         model,
#         passage_embeddings,
#         tfidf_vectorizer,
#         tfidf_embeddings,
#         k=10
#     )

#     # Generate answer
#     answer = generate_answer(
#         question,
#         results[:3]
#     )

#     # --------------------------------------------------
#     # Display answer
#     # --------------------------------------------------

#     st.subheader("Answer")

#     st.write(answer)


#     # --------------------------------------------------
#     # Display retrieved sources
#     # --------------------------------------------------

#     st.subheader("Retrieved sources")

#     for rank, result in enumerate(results[:3], start=1):

#         with st.expander(
#             f"Source {rank}: {result['title']} "
#             f"(similarity: {result['score']:.4f})"
#         ):

#             st.write(result["text"])

#             st.caption(
#                 f"Passage ID: {result['passage_id']}"
#             )

#             st.caption(
#                 f"Source: {result['url']}"
#             )