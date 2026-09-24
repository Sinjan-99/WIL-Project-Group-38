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

# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("💼 Australian Casual Employee Work-Rights Assistant")

st.write(
    "Ask a question about casual employment and workplace rights."
)

question = st.text_input(
    "Enter your question:",
    placeholder="e.g. Do casual employees get annual leave?"
)


if st.button("Ask") and question:

    # Retrieve relevant documents
    results = retrieve(
        question,
        df,
        model,
        passage_embeddings,
        tfidf_vectorizer,
        tfidf_embeddings,
        k=10
    )

    # Generate answer
    answer = generate_answer(
        question,
        results[:3]
    )

    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    st.subheader("Answer")

    st.write(answer)


    # --------------------------------------------------
    # Display retrieved sources
    # --------------------------------------------------

    st.subheader("Retrieved sources")

    for rank, result in enumerate(results[:3], start=1):

        with st.expander(
            f"Source {rank}: {result['title']} "
            f"(similarity: {result['score']:.4f})"
        ):

            st.write(result["text"])

            st.caption(
                f"Passage ID: {result['passage_id']}"
            )

            st.caption(
                f"Source: {result['url']}"
            )