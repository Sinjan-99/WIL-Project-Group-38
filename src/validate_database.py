import csv
import json
import re
from collections import defaultdict

from build_database import (
    PASSAGES_JSONL,
    COLLECTION_CSV,
    FIELDS,
    MAX_PASSAGE_WORDS,
    FAST_FOOD_MAX_WORDS,
    load_pages,
    clean_text,
    create_passages,
    create_fast_food_passages,
    normalise,
)


# ==================================================
# 1. LOAD GENERATED PASSAGES
# ==================================================

def load_passages():

    if not PASSAGES_JSONL.exists():
        raise FileNotFoundError(
            f"Database not found: {PASSAGES_JSONL}"
        )

    records = []

    with PASSAGES_JSONL.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            1,
        ):

            if not line.strip():
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):

                raise ValueError(
                    f"Line {line_number} is not "
                    f"a JSON object."
                )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            "The passage database is empty."
        )

    return records


# ==================================================
# 2. LOAD COLLECTION.CSV
# ==================================================

def load_collection():

    if not COLLECTION_CSV.exists():

        raise FileNotFoundError(
            f"Collection not found: "
            f"{COLLECTION_CSV}"
        )

    with COLLECTION_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        expected_fields = [
            "passage_id",
            "passage",
        ]

        if (
            reader.fieldnames
            != expected_fields
        ):

            raise ValueError(
                "Collection.csv must contain "
                "exactly: passage_id, passage"
            )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "Collection.csv is empty."
        )

    return rows


# ==================================================
# 3. VALIDATE DATABASE
# ==================================================

def validate_database():

    pages = load_pages()
    records = load_passages()
    collection = load_collection()

    source_by_id = {
        page["id"].strip():
            page
        for page in pages
    }

    if len(source_by_id) != len(pages):

        raise ValueError(
            "Duplicate source document IDs "
            "exist in raw_pages.jsonl."
        )

    seen_passage_ids = set()

    grouped = defaultdict(
        list
    )

    warnings = []

    # ==================================================
    # 4. VALIDATE EACH PASSAGE
    # ==================================================

    for record in records:

        # ------------------------------------------
        # Schema
        # ------------------------------------------

        assert (
            set(record.keys())
            == set(FIELDS)
        ), (
            "Invalid passage schema for "
            f"record: {record}"
        )

        passage_id = (
            record["passage_id"]
        )

        source_id = (
            record["source_id"]
        )

        passage = (
            record["passage"]
        )

        # ------------------------------------------
        # String fields
        # ------------------------------------------

        for field in [
            "passage_id",
            "source_id",
            "title",
            "url",
            "passage",
            "scraped_on",
        ]:

            assert isinstance(
                record[field],
                str,
            ), (
                f"Invalid {field}: "
                f"{passage_id}"
            )

            assert (
                record[field].strip()
            ), (
                f"Empty {field}: "
                f"{passage_id}"
            )

        # ------------------------------------------
        # Passage ID
        # ------------------------------------------

        assert re.fullmatch(
            r"P\d{4,}",
            passage_id,
        ), (
            f"Invalid passage ID format: "
            f"{passage_id}"
        )

        assert (
            passage_id
            not in seen_passage_ids
        ), (
            f"Duplicate passage ID: "
            f"{passage_id}"
        )

        seen_passage_ids.add(
            passage_id
        )

        # ------------------------------------------
        # passage_index
        # ------------------------------------------

        assert (
            type(
                record["passage_index"]
            )
            is int
        ), (
            f"Invalid passage_index: "
            f"{passage_id}"
        )

        assert (
            record["passage_index"]
            >= 0
        ), (
            f"Negative passage_index: "
            f"{passage_id}"
        )

        # ------------------------------------------
        # word_count
        # ------------------------------------------

        assert (
            type(
                record["word_count"]
            )
            is int
        ), (
            f"Invalid word_count: "
            f"{passage_id}"
        )

        actual_word_count = len(
            passage.split()
        )

        assert (
            record["word_count"]
            == actual_word_count
        ), (
            f"Incorrect word count: "
            f"{passage_id}"
        )

        assert (
            actual_word_count > 0
        ), (
            f"Empty passage: "
            f"{passage_id}"
        )

        # fwo_015 intentionally uses the larger
        # table-aware passage limit.
        if source_id == "fwo_015":

            allowed_max = (
                FAST_FOOD_MAX_WORDS
            )

        else:

            allowed_max = (
                MAX_PASSAGE_WORDS
            )

        if (
            actual_word_count
            > allowed_max
        ):

            warnings.append(
                f"{passage_id}: "
                f"{actual_word_count} words "
                f"(above {allowed_max})"
            )

        # ------------------------------------------
        # Source exists
        # ------------------------------------------

        assert (
            source_id
            in source_by_id
        ), (
            f"Unknown source: "
            f"{source_id}"
        )

        source = (
            source_by_id[source_id]
        )

        # ------------------------------------------
        # Source metadata
        # ------------------------------------------

        assert (
            record["title"]
            == source[
                "title"
            ].strip()
        ), (
            f"Incorrect title: "
            f"{passage_id}"
        )

        assert (
            record["url"]
            == source[
                "url"
            ].strip()
        ), (
            f"Incorrect URL: "
            f"{passage_id}"
        )

        assert (
            record["scraped_on"]
            == source[
                "scraped_on"
            ].strip()
        ), (
            f"Incorrect scraped_on: "
            f"{passage_id}"
        )

        # ------------------------------------------
        # Obvious webpage artefacts
        # ------------------------------------------

        forbidden_artifacts = [
            "On this page:",
            "Industry Embedded Filter Placeholder",
            "Tools and resources",
        ]

        for artifact in (
            forbidden_artifacts
        ):

            assert (
                artifact
                not in passage
            ), (
                f"Webpage artefact "
                f"'{artifact}' found in "
                f"{passage_id}"
            )

        # ------------------------------------------
        # Encoding artefacts
        # ------------------------------------------

        encoding_artifacts = [
            "â€™",
            "â€˜",
            "â€œ",
            "â€",
            "Â ",
        ]

        for artifact in (
            encoding_artifacts
        ):

            assert (
                artifact
                not in passage
            ), (
                f"Encoding artefact "
                f"'{artifact}' found in "
                f"{passage_id}"
            )

        # ------------------------------------------
        # Example heading must not be stranded
        # ------------------------------------------

        lines = [
            line.strip()
            for line
            in passage.splitlines()
            if line.strip()
        ]

        if lines:

            assert not (
                lines[-1].startswith(
                    "Example:"
                )
            ), (
                "Example heading without "
                f"example text: {passage_id}"
            )

        grouped[
            source_id
        ].append(
            record
        )

    # ==================================================
    # 5. ALL SOURCES REPRESENTED
    # ==================================================

    missing_sources = (
        set(source_by_id)
        - set(grouped)
    )

    if missing_sources:

        raise ValueError(
            "No passages generated for: "
            + ", ".join(
                sorted(
                    missing_sources
                )
            )
        )

    # ==================================================
    # 6. PASSAGE INDEX ORDER PER SOURCE
    # ==================================================

    for (
        source_id,
        source_records,
    ) in grouped.items():

        source_records.sort(
            key=lambda record:
                record[
                    "passage_index"
                ]
        )

        actual_indices = [
            record[
                "passage_index"
            ]
            for record
            in source_records
        ]

        expected_indices = list(
            range(
                len(
                    source_records
                )
            )
        )

        assert (
            actual_indices
            == expected_indices
        ), (
            "Incorrect passage_index "
            f"sequence for {source_id}. "
            f"Got {actual_indices}"
        )

    # ==================================================
    # 7. GLOBAL PASSAGE ID SEQUENCE
    # ==================================================

    expected_ids = [
        f"P{number:04d}"
        for number in range(
            1,
            len(records) + 1,
        )
    ]

    actual_ids = [
        record["passage_id"]
        for record in records
    ]

    assert (
        actual_ids
        == expected_ids
    ), (
        "Passage IDs are not sequential "
        "from P0001 onwards."
    )

    # ==================================================
    # 8. RECONSTRUCT PASSAGES FROM RAW SOURCES
    # ==================================================

    for (
        source_id,
        source_records,
    ) in grouped.items():

        source = (
            source_by_id[source_id]
        )

        cleaned = clean_text(
            source["text"],
            document_id=source_id,
        )

        # Reproduce exactly the same construction
        # decision as build_database.py.
        if source_id == "fwo_015":

            expected_passages = (
                create_fast_food_passages(
                    cleaned
                )
            )

        else:

            expected_passages = (
                create_passages(
                    cleaned
                )
            )

        # Same-source exact duplicate removal,
        # matching build_database.py.
        unique_expected = []
        seen_texts = set()

        for passage in (
            expected_passages
        ):

            key = normalise(
                passage
            )

            if key in seen_texts:
                continue

            seen_texts.add(
                key
            )

            unique_expected.append(
                passage
            )

        source_records.sort(
            key=lambda record:
                record[
                    "passage_index"
                ]
        )

        actual_passages = [
            record["passage"]
            for record
            in source_records
        ]

        assert (
            actual_passages
            == unique_expected
        ), (
            "Generated passages do not "
            "match reconstructed source "
            f"passages for {source_id}"
        )

    # ==================================================
    # 9. EXACT SAME-SOURCE DUPLICATE CHECK
    # ==================================================

    seen_texts = {}

    for record in records:

        key = (
            record["source_id"],
            normalise(
                record["passage"]
            ),
        )

        assert (
            key not in seen_texts
        ), (
            "Exact same-source duplicate: "
            f"{seen_texts[key]} and "
            f"{record['passage_id']}"
        )

        seen_texts[key] = (
            record["passage_id"]
        )

    # ==================================================
    # 10. COLLECTION.CSV MUST MATCH JSONL
    # ==================================================

    assert (
        len(collection)
        == len(records)
    ), (
        "Collection.csv and "
        "passages.jsonl have different "
        "numbers of passages."
    )

    for record, row in zip(
        records,
        collection,
    ):

        assert (
            row["passage_id"]
            == record["passage_id"]
        ), (
            "Collection passage ID "
            f"mismatch: "
            f"{record['passage_id']}"
        )

        assert (
            row["passage"]
            == record["passage"]
        ), (
            "Collection passage text "
            f"mismatch: "
            f"{record['passage_id']}"
        )

    # ==================================================
    # 11. CHECK COLLECTION IDS UNIQUE
    # ==================================================

    collection_ids = [
        row["passage_id"]
        for row in collection
    ]

    assert (
        len(collection_ids)
        == len(
            set(collection_ids)
        )
    ), (
        "Duplicate passage IDs found "
        "in Collection.csv."
    )

    # ==================================================
    # 12. PRINT RESULTS
    # ==================================================

    print()
    print(
        "DATABASE VALIDATION RESULTS"
    )

    print(
        "---------------------------"
    )

    print("Schema: PASSED")
    print("Passage ID format: PASSED")
    print("Unique passage IDs: PASSED")
    print("Sequential passage IDs: PASSED")
    print("Word counts: PASSED")
    print("Source metadata: PASSED")
    print(
        "All source webpages represented: "
        "PASSED"
    )
    print("Passage indices: PASSED")
    print("Source reconstruction: PASSED")
    print(
        "Exact same-source duplicates: "
        "PASSED"
    )
    print(
        "Webpage artefact checks: PASSED"
    )
    print(
        "Example boundary checks: PASSED"
    )
    print(
        "Collection.csv consistency: PASSED"
    )

    print()

    print(
        "Source webpages:",
        len(pages),
    )

    print(
        "Passages:",
        len(records),
    )

    print(
        "Sources represented:",
        len(grouped),
    )

    if warnings:

        print()
        print("WARNINGS")
        print("--------")

        for warning in warnings:
            print(
                "-",
                warning,
            )

    else:

        print()

        print(
            "No passage-length warnings."
        )

    print()

    print(
        "All automated database "
        "validation checks passed."
    )

    print()

    print(
        "NOTE: Automated validation does "
        "not guarantee semantic passage "
        "quality."
    )

    print(
        "Manually review passage "
        "boundaries and near-duplicate "
        "passages before freezing "
        "passage IDs."
    )


# ==================================================
# 13. RUN
# ==================================================

if __name__ == "__main__":

    validate_database()