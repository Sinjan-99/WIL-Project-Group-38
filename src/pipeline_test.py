from retrieval.retrieval import (
    load_knowledge_base,
    load_model,
    build_passage_embeddings,
    build_tfidf_index,
    retrieve,
    extract_relevant_sentences
)

from generation.generation import generate_answer


# Load knowledge base
df = load_knowledge_base()

print(f"Loaded {len(df)} passages.")

# Load embedding model
model = load_model()

# Build embeddings
passage_embeddings = build_passage_embeddings(
    df,
    model
)
tfidf_vectorizer, tfidf_embeddings = build_tfidf_index(
    df
)

# Question
question = "Do casual employees get sick leave?"

results = retrieve(
    question,
    df,
    model,
    passage_embeddings,
    tfidf_vectorizer,
    tfidf_embeddings,
    k=10
)

top_results = results[:3]

answer = generate_answer(
    question,
    results[:3]
)

print("\n" + "=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print(answer)