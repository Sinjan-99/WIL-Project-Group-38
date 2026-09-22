
import json
from collections import defaultdict

from build_database import (
    PASSAGES_JSONL,
    CHUNK_SIZE,
    OVERLAP,
    FIELDS,
    load_pages,
    clean_text,
    chunk_text,
)


# ==================================================
# 1. LOAD THE GENERATED DATABASE
# ==================================================

def load_passages():

    if not PASSAGES_JSONL.exists():
        raise FileNotFoundError(
            f"Database not found: {PASSAGES_JSONL}"
        )

    records = []

    with PASSAGES_JSONL.open(
        "r", encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, 1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} is not "
                    f"a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(
            "The passage database is empty."
        )

    return records


# ==================================================
# 2. VALIDATE THE DATABASE
# ==================================================

def validate_database():

    pages = load_pages()
    records = load_passages()

    source_by_id = {
        page["id"].strip(): page
        for page in pages
    }

    if len(source_by_id) != len(pages):
        raise ValueError(
            "Duplicate source document IDs."
        )

    seen_ids = set()
    grouped = defaultdict(list)

    for record in records:

        # ------------------------------------------
        # Check required schema fields.
        # ------------------------------------------

        assert set(record.keys()) == set(FIELDS), (
            f"Invalid schema: {record}"
        )

        passage_id = record["id"]
        document_id = record["document_id"]

        for field in [
            "id",
            "document_id",
            "title",
            "url",
            "text",
            "scraped_on",
        ]:

            assert isinstance(record[field], str), (
                f"Invalid {field}: {passage_id}"
            )

            assert record[field].strip(), (
                f"Empty {field}: {passage_id}"
            )

        # ------------------------------------------
        # Check unique passage IDs.
        # ------------------------------------------

        assert passage_id not in seen_ids, (
            f"Duplicate passage ID: {passage_id}"
        )

        seen_ids.add(passage_id)

        # ------------------------------------------
        # Check chunk index and word count.
        # ------------------------------------------

        assert type(record["chunk_index"]) is int
        assert record["chunk_index"] >= 0

        assert type(record["word_count"]) is int

        actual_word_count = len(
            record["text"].split()
        )

        assert (
            record["word_count"] == actual_word_count
        ), (
            f"Incorrect word count: {passage_id}"
        )

        assert 0 < actual_word_count <= CHUNK_SIZE, (
            f"Invalid passage length: {passage_id}"
        )

        # ------------------------------------------
        # Check stable passage ID format.
        # ------------------------------------------

        expected_id = (
            f"{document_id}_"
            f"{record['chunk_index']:04d}"
        )

        assert passage_id == expected_id, (
            f"Incorrect passage ID: {passage_id}"
        )

        # ------------------------------------------
        # Check source metadata.
        # ------------------------------------------

        assert document_id in source_by_id, (
            f"Unknown source document: {document_id}"
        )

        source = source_by_id[document_id]

        assert (
            record["url"] == source["url"].strip()
        ), (
            f"Incorrect source URL: {passage_id}"
        )

        assert (
            record["title"] == source["title"].strip()
        ), (
            f"Incorrect source title: {passage_id}"
        )

        assert (
            record["scraped_on"]
            == source["scraped_on"].strip()
        ), (
            f"Incorrect scraping date: {passage_id}"
        )

        # ------------------------------------------
        # Reconstruct the expected source chunk.
        # ------------------------------------------

        expected_chunks = chunk_text(
            clean_text(
                source["text"],
                document_id=document_id
            )
        )

        index = record["chunk_index"]

        assert index < len(expected_chunks), (
            f"Invalid chunk index: {passage_id}"
        )

        assert (
            record["text"] == expected_chunks[index]
        ), (
            f"Passage text does not match "
            f"the source: {passage_id}"
        )

        grouped[document_id].append(record)

    # ==================================================
    # 3. CHECK THAT ALL WEBPAGES ARE REPRESENTED
    # ==================================================

    missing_documents = (
        set(source_by_id) - set(grouped)
    )

    if missing_documents:
        raise ValueError(
            "No passages generated for these "
            "source documents: "
            + ", ".join(sorted(missing_documents))
        )

    # ==================================================
    # 4. CHECK CHUNK ORDER AND 50-WORD OVERLAP
    # ==================================================

    checked_pairs = 0

    for document_id, passages in grouped.items():

        passages.sort(
            key=lambda record: record["chunk_index"]
        )

        # Every source chunk should be present because
        # the builder does not automatically remove
        # duplicate passages.
        actual_indices = [
            record["chunk_index"]
            for record in passages
        ]

        expected_indices = list(
            range(len(passages))
        )

        assert actual_indices == expected_indices, (
            f"Missing or incorrectly ordered chunks "
            f"for {document_id}"
        )

        for first, second in zip(
            passages,
            passages[1:]
        ):

            first_words = first["text"].split()
            second_words = second["text"].split()

            overlap_length = min(
                OVERLAP,
                len(second_words)
            )

            assert (
                first_words[-overlap_length:]
                == second_words[:overlap_length]
            ), (
                "Incorrect overlap between "
                f"{first['id']} and {second['id']}"
            )

            checked_pairs += 1

    # ==================================================
    # 5. PRINT VALIDATION RESULTS
    # ==================================================

    print("\nDATABASE VALIDATION RESULTS")
    print("---------------------------")

    print("Schema: PASSED")
    print("Unique passage IDs: PASSED")
    print("Word counts: PASSED")
    print("Source metadata: PASSED")
    print("Source text consistency: PASSED")
    print("All source webpages represented: PASSED")
    print("Chunk order and overlap: PASSED")

    print("\nSource webpages:", len(pages))
    print("Passages:", len(records))
    print(
        "Source documents represented:",
        len(grouped)
    )
    print(
        "Consecutive chunk pairs checked:",
        checked_pairs
    )

    print(
        "\nAll automated validation "
        "checks passed."
    )


if __name__ == "__main__":
    validate_database()