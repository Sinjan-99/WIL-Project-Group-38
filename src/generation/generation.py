import ollama

MODEL_NAME = "llama3.2"

FALLBACK = "The information is not available in the knowledge base."

SYSTEM_PROMPT = f"""You are a workplace rights assistant for Australian employees and employers.
You answer questions using ONLY the Fair Work documents provided in the user message.

How to answer:
- Answer the exact question asked. If the question is about one entitlement (e.g. annual leave), do not answer about a different one (e.g. personal/carer's leave).
- State the answer directly in the first sentence. Include specific figures (weeks, hours, dollar amounts, percentages) exactly as they appear in the documents.
- If the answer depends on a condition stated in the documents (e.g. full-time vs part-time vs casual, award coverage, length of service), include that condition.
- If the documents answer only part of the question, answer that part and say the rest is not covered.
- If the documents do not contain the answer, reply with exactly: "{FALLBACK}"
- Never use outside knowledge, even if you believe it is correct.
- Do not mention "the documents", "Document 1", or your reasoning.
- Keep the answer to 1-3 sentences."""


def format_context(context):
    blocks = []
    for i, passage in enumerate(context, start=1):
        blocks.append(
            f"<document id=\"{i}\">\n"
            f"Title: {passage['title']}\n"
            f"{passage['text'].strip()}\n"
            f"</document>"
        )
    return "\n\n".join(blocks)


def build_messages(question, context):
    # Documents first, question last: small models attend most to the end of the prompt.
    user_content = (
        "<documents>\n"
        f"{format_context(context)}\n"
        "</documents>\n\n"
        f"Question: {question.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(question, context):
    if not context:
        return FALLBACK

    response = ollama.chat(
        model=MODEL_NAME,
        messages=build_messages(question, context),
        options={
            "temperature": 0,   # deterministic, less paraphrasing drift
            "num_ctx": 8192,    # Ollama's default context can silently truncate retrieved docs
        },
    )
    return response["message"]["content"].strip()