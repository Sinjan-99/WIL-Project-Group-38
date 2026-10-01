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


class GenerationUnavailable(RuntimeError):
    """Raised when the model cannot be reached or fails to respond."""


def passage_text(passage):

    for field in ("passage", "text"):
        if field in passage:
            return passage[field]
    raise KeyError(
        "Passage record has neither a 'passage' nor a 'text' field: "
        f"{sorted(passage.keys())}"
    )


def normalise_context(context):
    passages = []
    for item in context:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            passages.append(item[0])
        else:
            passages.append(item)
    return passages


def format_context(context):
    blocks = []
    for i, passage in enumerate(normalise_context(context), start=1):
        blocks.append(
            f"<document id=\"{i}\">\n"
            f"Title: {passage['title']}\n"
            f"{passage_text(passage).strip()}\n"
            f"</document>"
        )
    return "\n\n".join(blocks)


def build_messages(question, context):
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


    # -------------------------
    # Generation
    # -------------------------

def is_available():
    try:
        installed = ollama.list()
    except Exception:
        return False

    names = [m.get("model", m.get("name", "")) for m in installed.get("models", [])]
    return any(n.split(":")[0] == MODEL_NAME.split(":")[0] for n in names)


def generate_answer(question, context):
    if not context:
        return FALLBACK

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=build_messages(question, context),
            options={
                "temperature": 0,   # deterministic, less paraphrasing drift
                "num_ctx": 8192,    # Ollama's default context can silently truncate retrieved docs
            },
            keep_alive="10m",       # hold the model in memory between questions
        )
    except Exception as error:
        raise GenerationUnavailable(str(error)) from error

    try:
        return response["message"]["content"].strip()
    except (KeyError, TypeError) as error:
        raise GenerationUnavailable("Unexpected response shape from Ollama") from error


    # -------------------------
    # Testing
    # -------------------------

if __name__ == "__main__":
    sample = [
        {
            "passage_id": "P0007",
            "title": "Sick and carers leave",
            "passage": (
                "Casual employees do not get paid sick or carers leave. However, "
                "casual employees are entitled to 2 days of unpaid carers leave "
                "per occasion."
            ),
        }
    ]

    print("Ollama available:", is_available())
    print()

    for q in [
        "do casual employees get sick leave",
        "how much is stamp duty in Victoria",
    ]:
        print(f"Q: {q}")
        try:
            print(f"A: {generate_answer(q, sample)}\n")
        except GenerationUnavailable as e:
            print(f"unavailable: {e}\n")
