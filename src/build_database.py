import csv
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "processed" / "raw_pages.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed"

PASSAGES_JSONL = OUTPUT_DIR / "passages.jsonl"
COLLECTION_CSV = OUTPUT_DIR / "Collection.csv"
REPORT_FILE = OUTPUT_DIR / "database_report.json"

# Normal Fair Work article passage limits.
MIN_PASSAGE_WORDS = 15
TARGET_PASSAGE_WORDS = 120
MAX_PASSAGE_WORDS = 220

# Short standalone passages below this size are merged
# with a neighbouring passage from the same source when safe.
CONTEXT_MERGE_WORDS = 30

# The Fast Food Award page is an interactive/table-like
# page. A slightly larger safety limit helps keep related
# rows and their context together.
FAST_FOOD_TARGET_WORDS = 170
FAST_FOOD_MAX_WORDS = 260

FIELDS = [
    "passage_id",
    "source_id",
    "title",
    "url",
    "passage",
    "passage_index",
    "word_count",
    "scraped_on",
]


# ==================================================
# 2. LOAD SCRAPED WEBPAGES
# ==================================================

def load_pages():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    pages = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:

        for line_number, line in enumerate(file, 1):

            if not line.strip():
                continue

            try:
                page = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            if not isinstance(page, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            for field in [
                "id",
                "url",
                "title",
                "text",
                "scraped_on",
            ]:

                value = page.get(field)

                if (
                    not isinstance(value, str)
                    or not value.strip()
                ):
                    raise ValueError(
                        f"Missing or invalid '{field}' "
                        f"on line {line_number}"
                    )

            pages.append(page)

    if not pages:
        raise ValueError(
            "No source webpages were loaded."
        )

    return pages


# ==================================================
# 3. BASIC HELPERS
# ==================================================

def normalise(text):

    return re.sub(
        r"\s+",
        " ",
        text.casefold(),
    ).strip()


def word_count(text):

    return len(text.split())


def is_list_item(text):

    text = text.strip()

    return bool(
        re.match(
            r"^(?:[-•*]|\d+[.)])\s+",
            text,
        )
    )


def is_complete_sentence(text):

    return text.rstrip().endswith(
        (".", "?", "!", ".”", ".’")
    )


# Introductory webpage text that is useful for website navigation
# but does not add substantive information to retrieval passages.
UNHELPFUL_INTRO_LINES = {
    "learn more about casual employees and what entitlements they receive.",
}


def is_unhelpful_intro_line(text):

    return (
        normalise(text)
        in UNHELPFUL_INTRO_LINES
    )


# ==================================================
# 4. CLEAN SOURCE TEXT
# ==================================================

def clean_text(text, document_id="unknown"):

    replacements = {
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€\x9d": "”",
        "â€“": "–",
        "â€”": "—",
        "Â ": " ",
        "\u00a0": " ",
    }

    for incorrect, correct in replacements.items():
        text = text.replace(
            incorrect,
            correct,
        )

    # Preserve scraper block boundaries.
    lines = [
        re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Remove known scraper/webpage text that does not
    # contribute useful employment-rights information.
    lines = [
        line
        for line in lines
        if (
            line
            != "Industry Embedded Filter Placeholder"
            and not is_unhelpful_intro_line(line)
        )
    ]

    # Remove "On this page:" table of contents.
    if "On this page:" in lines:

        toc_start = lines.index(
            "On this page:"
        )

        toc_end = toc_start + 1

        while (
            toc_end < len(lines)
            and is_list_item(lines[toc_end])
        ):
            toc_end += 1

        lines = (
            lines[:toc_start]
            + lines[toc_end:]
        )

    # Remove footer/navigation beginning at
    # "Tools and resources".
    footer_start = None

    for index, line in enumerate(lines):

        if line == "Tools and resources":
            footer_start = index
            break

    if footer_start is not None:
        lines = lines[:footer_start]

    # Remove only consecutive duplicate blocks.
    deduplicated = []

    for line in lines:

        if (
            deduplicated
            and normalise(line)
            == normalise(deduplicated[-1])
        ):
            continue

        deduplicated.append(line)

    lines = deduplicated

    cleaned_text = "\n".join(lines)

    cleaned_text = re.sub(
        r"\.{2,}",
        ".",
        cleaned_text,
    )

    return cleaned_text.strip()


# ==================================================
# 5. HEADING DETECTION
# ==================================================

def looks_like_heading(block):
    """
    Conservative heading detection for ordinary
    Fair Work article pages.
    """

    block = block.strip()

    if not block:
        return False

    if is_list_item(block):
        return False

    if block.startswith(
        "Source reference for page:"
    ):
        return False

    if block.startswith(
        "Tip:"
    ):
        return False

    if block.startswith(
        ("http://", "https://")
    ):
        return False

    words = block.split()

    if len(words) > 12:
        return False

    if block.endswith(
        (
            ".",
            "?",
            "!",
            ",",
            ";",
            ":",
        )
    ):
        return False

    return True


def is_example_heading(block):

    return block.strip().startswith(
        "Example:"
    )


def is_dependent_leadin(block):
    """Return True for a source line that introduces following content."""
    return block.strip().endswith(":")


# ==================================================
# 6. SPLIT EXTREMELY LONG PROSE BLOCK
# ==================================================

def split_long_block(block):

    if word_count(block) <= MAX_PASSAGE_WORDS:
        return [block]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        block.strip(),
    )

    pieces = []
    current = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = word_count(
            sentence
        )

        # Rare fallback for one extremely long
        # source sentence.
        if sentence_words > MAX_PASSAGE_WORDS:

            if current:
                pieces.append(
                    " ".join(current)
                )
                current = []

            words = sentence.split()

            for start in range(
                0,
                len(words),
                MAX_PASSAGE_WORDS,
            ):

                pieces.append(
                    " ".join(
                        words[
                            start:
                            start
                            + MAX_PASSAGE_WORDS
                        ]
                    )
                )

            continue

        candidate = " ".join(
            current + [sentence]
        )

        if (
            current
            and word_count(candidate)
            > MAX_PASSAGE_WORDS
        ):
            pieces.append(
                " ".join(current)
            )

            current = [sentence]

        else:
            current.append(sentence)

    if current:
        pieces.append(
            " ".join(current)
        )

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


# ==================================================
# 7. GROUP NORMAL ARTICLES INTO SECTIONS
# ==================================================

def create_sections(text):
    """
    Turn cleaned scraper blocks into logical sections.

    Example headings remain inside their surrounding
    topic section.
    """

    blocks = [
        block.strip()
        for block in text.splitlines()
        if block.strip()
    ]

    if not blocks:
        return []

    sections = []
    current = []

    for block in blocks:

        heading = looks_like_heading(
            block
        )

        example_heading = (
            is_example_heading(block)
        )

        if (
            heading
            and not example_heading
            and current
        ):

            sections.append(current)
            current = []

        current.append(block)

    if current:
        sections.append(current)

    return sections


# ==================================================
# 8. TURN NORMAL SECTION INTO PASSAGES
# ==================================================

def section_to_passages(section):
    """
    Keep a logical section together where possible.

    If it becomes too large, split at source block
    boundaries rather than arbitrary overlapping
    word windows.
    """

    if not section:
        return []

    expanded = []

    for block in section:
        expanded.extend(
            split_long_block(block)
        )

    section = expanded

    whole_section = "\n".join(
        section
    ).strip()

    if (
        word_count(whole_section)
        <= MAX_PASSAGE_WORDS
    ):
        return [whole_section]

    first_block_is_heading = (
        looks_like_heading(section[0])
        and not is_example_heading(
            section[0]
        )
    )

    section_heading = (
        section[0]
        if first_block_is_heading
        else None
    )

    content_blocks = (
        section[1:]
        if section_heading
        else section
    )

    passages = []
    current = []

    if section_heading:
        current.append(
            section_heading
        )

    i = 0

    while i < len(content_blocks):

        block = content_blocks[i]

        # Keep Example heading with at least the first
        # block of its example.
        if (
            is_example_heading(block)
            and i + 1 < len(content_blocks)
        ):

            next_block = (
                content_blocks[i + 1]
            )

            example_unit = (
                block
                + "\n"
                + next_block
            )

            candidate = "\n".join(
                current
                + [example_unit]
            )

            if (
                current
                and word_count(candidate)
                > MAX_PASSAGE_WORDS
            ):

                if not (
                    section_heading
                    and current
                    == [section_heading]
                ):

                    passages.append(
                        "\n".join(
                            current
                        ).strip()
                    )

                    current = []

                    if section_heading:
                        current.append(
                            section_heading
                        )

            current.append(
                example_unit
            )

            i += 2
            continue

        candidate = "\n".join(
            current + [block]
        )

        if (
            current
            and word_count(candidate)
            > MAX_PASSAGE_WORDS
        ):

            if (
                section_heading
                and current
                == [section_heading]
            ):

                current.append(
                    block
                )

                i += 1
                continue

            passages.append(
                "\n".join(
                    current
                ).strip()
            )

            current = []

            if section_heading:
                current.append(
                    section_heading
                )

            current.append(
                block
            )

        else:

            current.append(
                block
            )

        i += 1

    if current:

        final_text = "\n".join(
            current
        ).strip()

        if not (
            section_heading
            and current
            == [section_heading]
        ):

            passages.append(
                final_text
            )

    return passages


# ==================================================
# 9. CREATE NORMAL FAIR WORK PASSAGES
# ==================================================

def create_passages(text):

    sections = create_sections(text)

    passages = []

    for section in sections:

        passages.extend(
            section_to_passages(
                section
            )
        )

    # --------------------------------------------------
    # Merge short contextual passages.
    #
    # A short passage should not be left alone when the
    # following passage provides the explanation needed
    # to make it useful for retrieval.
    #
    # If the complete next passage fits, merge it.
    #
    # If it does not fit, take complete source blocks
    # from the beginning of the next passage until the
    # current passage has useful supporting context.
    # The unused blocks remain as a separate passage.
    # --------------------------------------------------

    merged = []

    i = 0

    while i < len(passages):

        passage = passages[i].strip()
        wc = word_count(passage)

        if (
            wc < CONTEXT_MERGE_WORDS
            and i + 1 < len(passages)
        ):

            next_passage = passages[i + 1].strip()

            # ------------------------------------------
            # Case 1: the complete next passage fits.
            # ------------------------------------------

            if (
                wc + word_count(next_passage)
                <= MAX_PASSAGE_WORDS
            ):

                merged.append(
                    passage
                    + "\n"
                    + next_passage
                )

                i += 2
                continue

            # ------------------------------------------
            # Case 2: the complete next passage is too
            # large.
            #
            # Borrow complete source blocks from its
            # beginning so the short passage receives
            # useful explanatory context.
            # ------------------------------------------

            next_blocks = [
                block.strip()
                for block in next_passage.splitlines()
                if block.strip()
            ]

            borrowed = []
            remaining = []

            current_words = wc

            block_index = 0

            while block_index < len(next_blocks):

                block = next_blocks[block_index]

                # Keep an Example heading together with the first
                # block of its example.
                if (
                    is_example_heading(block)
                    and block_index + 1 < len(next_blocks)
                ):

                    unit = (
                        block
                        + "\n"
                        + next_blocks[block_index + 1]
                    )

                    unit_words = word_count(unit)
                    step = 2

                # A line ending in ':' introduces the content that
                # follows it. Never borrow the lead-in by itself.
                # If it introduces a list, keep the complete
                # consecutive list with the lead-in.
                elif (
                    is_dependent_leadin(block)
                    and block_index + 1 < len(next_blocks)
                ):

                    unit_blocks = [block]
                    lookahead = block_index + 1

                    if is_list_item(next_blocks[lookahead]):
                        while (
                            lookahead < len(next_blocks)
                            and is_list_item(next_blocks[lookahead])
                        ):
                            unit_blocks.append(next_blocks[lookahead])
                            lookahead += 1
                    else:
                        unit_blocks.append(next_blocks[lookahead])
                        lookahead += 1

                    unit = "\n".join(unit_blocks)
                    unit_words = word_count(unit)
                    step = lookahead - block_index

                else:

                    unit = block
                    unit_words = word_count(unit)
                    step = 1

                if (
                    current_words + unit_words
                    <= TARGET_PASSAGE_WORDS
                ):

                    borrowed.append(unit)
                    current_words += unit_words
                    block_index += step

                else:

                    remaining = next_blocks[
                        block_index:
                    ]

                    break

            else:

                remaining = []


            # Only perform the split when useful context
            # was actually borrowed.
            if borrowed:

                combined = (
                    passage
                    + "\n"
                    + "\n".join(borrowed)
                ).strip()

                merged.append(combined)

                # Preserve the unused part of the next
                # passage instead of losing information.
                if remaining:

                    passages[i + 1] = (
                        "\n".join(remaining)
                    )

                    i += 1

                else:

                    i += 2

                continue

        # ----------------------------------------------
        # If forward merging was not possible, merge a
        # very short passage backwards when safe.
        # ----------------------------------------------

        if (
            wc < CONTEXT_MERGE_WORDS
            and merged
            and (
                word_count(merged[-1]) + wc
                <= MAX_PASSAGE_WORDS
            )
        ):

            merged[-1] = (
                merged[-1].rstrip()
                + "\n"
                + passage
            )

        else:

            merged.append(passage)

        i += 1

    return [
        passage.strip()
        for passage in merged
        if passage.strip()
    ]
    
# ==================================================
# 10. FAST FOOD AWARD TABLE-AWARE PASSAGES
# ==================================================

def create_fast_food_passages(text):
    """
    Special passage construction for fwo_015.

    The Fast Food Industry Award page contains an
    interactive/table-like structure. Short lines such
    as "Laundry", "Full-time", "$ Saturday", "N/A",
    "Rest break (paid)" and similar values must not
    automatically be treated as article headings.

    Therefore this function:
    - preserves the Fair Work wording
    - preserves source order
    - does not invent labels
    - uses only explicit major source headings as
      strong boundaries
    - keeps table rows together where possible
    - avoids fixed overlapping chunks
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    # Explicit high-level headings visible in the
    # Fair Work interactive content.
    major_headings = {
        "Employment Status Overview",
        "Pay Overview",
        "Employment Level Overview",
        "Hours of work",
        "Allowances",
        "Leave",
        "NOTICE OF TERMINATION",
        "WITHHOLDING MONEY FROM FINAL PAY",
        "I’m not sure if this is my award",
        "I'm not sure if this is my award",
    }

    passages = []
    current = []

    def current_words():

        return word_count(
            "\n".join(current)
        )

    def flush():

        nonlocal current

        if not current:
            return

        passage = "\n".join(
            current
        ).strip()

        if passage:
            passages.append(
                passage
            )

        current = []

    for line in lines:

        # Major topic boundary.
        if (
            line in major_headings
            and current
            and current_words()
            >= MIN_PASSAGE_WORDS
        ):

            flush()

        candidate = (
            current + [line]
        )

        candidate_words = word_count(
            "\n".join(candidate)
        )

        # If a table/topic block becomes very large,
        # split only before adding the next complete
        # source line.
        if (
            current
            and candidate_words
            > FAST_FOOD_MAX_WORDS
            and current_words()
            >= FAST_FOOD_TARGET_WORDS
        ):

            flush()

        current.append(
            line
        )

    flush()

    # Merge very small fragments so headings and
    # short table tails are not left alone.
    merged = []

    for passage in passages:

        wc = word_count(
            passage
        )

        if (
            merged
            and wc < MIN_PASSAGE_WORDS
            and (
                word_count(merged[-1])
                + wc
                <= FAST_FOOD_MAX_WORDS
            )
        ):

            merged[-1] = (
                merged[-1].rstrip()
                + "\n"
                + passage.lstrip()
            )

        else:

            merged.append(
                passage
            )

    if (
        len(merged) > 1
        and word_count(merged[0])
        < MIN_PASSAGE_WORDS
        and (
            word_count(merged[0])
            + word_count(merged[1])
            <= FAST_FOOD_MAX_WORDS
        )
    ):

        merged[1] = (
            merged[0].rstrip()
            + "\n"
            + merged[1].lstrip()
        )

        merged = merged[1:]

    return [
        passage.strip()
        for passage in merged
        if passage.strip()
    ]


# Compatibility alias for existing code that imports
# chunk_text.
def chunk_text(text):

    return create_passages(text)


# ==================================================
# 11. BUILD DATABASE
# ==================================================

def build_database(pages):

    records = []

    seen_document_ids = set()
    seen_urls = set()

    passage_number = 1

    for page in sorted(
        pages,
        key=lambda p: p["id"],
    ):

        source_id = (
            page["id"].strip()
        )

        url = (
            page["url"].strip()
        )

        title = (
            page["title"].strip()
        )

        if source_id in seen_document_ids:
            raise ValueError(
                f"Duplicate document ID: "
                f"{source_id}"
            )

        if url in seen_urls:
            raise ValueError(
                f"Duplicate source URL: "
                f"{url}"
            )

        parsed = urlparse(url)

        if parsed.scheme not in (
            "http",
            "https",
        ):
            raise ValueError(
                f"Invalid source URL: {url}"
            )

        if not parsed.netloc:
            raise ValueError(
                f"Invalid source URL: {url}"
            )

        seen_document_ids.add(
            source_id
        )

        seen_urls.add(
            url
        )

        cleaned_text = clean_text(
            page["text"],
            document_id=source_id,
        )

        if not cleaned_text:
            raise ValueError(
                f"Cleaning removed all text "
                f"from {source_id}: {url}"
            )

        # fwo_015 is the table-heavy Fast Food
        # Industry Award page.
        if source_id == "fwo_015":

            passages = (
                create_fast_food_passages(
                    cleaned_text
                )
            )

        else:

            passages = create_passages(
                cleaned_text
            )

        if not passages:
            raise ValueError(
                f"No passages generated for "
                f"{source_id}: {url}"
            )

        for passage_index, passage in enumerate(
            passages
        ):

            passage_id = (
                f"P{passage_number:04d}"
            )

            record = {
                "passage_id":
                    passage_id,

                "source_id":
                    source_id,

                "title":
                    title,

                "url":
                    url,

                "passage":
                    passage,

                "passage_index":
                    passage_index,

                "word_count":
                    word_count(passage),

                "scraped_on":
                    page[
                        "scraped_on"
                    ].strip(),
            }

            records.append(
                record
            )

            passage_number += 1

    return records


# ==================================================
# 12. EXACT DUPLICATES
# ==================================================

def find_exact_duplicates(records):

    seen = {}
    duplicates = []

    for record in records:

        text = normalise(
            record["passage"]
        )

        if text in seen:

            duplicates.append({
                "passage_id_1":
                    seen[text],

                "passage_id_2":
                    record["passage_id"],
            })

        else:

            seen[text] = (
                record["passage_id"]
            )

    return duplicates


# ==================================================
# 13. NEAR DUPLICATES
# ==================================================

def find_near_duplicates(records):

    flagged = []

    for i, first in enumerate(records):

        text_a = normalise(
            first["passage"]
        )

        for second in records[
            i + 1:
        ]:

            text_b = normalise(
                second["passage"]
            )

            if text_a == text_b:
                continue

            if not text_a or not text_b:
                continue

            length_ratio = (
                min(
                    len(text_a),
                    len(text_b),
                )
                /
                max(
                    len(text_a),
                    len(text_b),
                )
            )

            if length_ratio < 0.92:
                continue

            words_a = set(
                text_a.split()
            )

            words_b = set(
                text_b.split()
            )

            union = (
                words_a | words_b
            )

            if not union:
                continue

            jaccard = (
                len(
                    words_a & words_b
                )
                /
                len(union)
            )

            if jaccard < 0.85:
                continue

            similarity = (
                SequenceMatcher(
                    None,
                    text_a,
                    text_b,
                    autojunk=False,
                ).ratio()
            )

            if similarity >= 0.96:

                flagged.append({
                    "passage_id_1":
                        first[
                            "passage_id"
                        ],

                    "passage_id_2":
                        second[
                            "passage_id"
                        ],

                    "similarity":
                        round(
                            similarity,
                            4,
                        ),
                })

    return flagged


# ==================================================
# 14. REMOVE SAME-SOURCE EXACT DUPLICATES
# ==================================================

def remove_exact_duplicate_passages(records):

    seen = set()
    unique_records = []

    for record in records:

        key = (
            record["source_id"],
            normalise(
                record["passage"]
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_records.append(
            record
        )

    # Reassign stable sequential IDs after removal.
    for number, record in enumerate(
        unique_records,
        start=1,
    ):

        record["passage_id"] = (
            f"P{number:04d}"
        )

    # Reassign per-source passage indexes.
    source_indexes = {}

    for record in unique_records:

        source_id = (
            record["source_id"]
        )

        index = source_indexes.get(
            source_id,
            0,
        )

        record["passage_index"] = (
            index
        )

        source_indexes[source_id] = (
            index + 1
        )

    return unique_records


# ==================================================
# 15. SAVE DATABASE
# ==================================================

def save_database(records, report):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Rich database for retrieval/application.
    with PASSAGES_JSONL.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Walert-style simple collection used for
    # question/relevance mapping and evaluation.
    with COLLECTION_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "passage_id",
                "passage",
            ],
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                "passage_id":
                    record[
                        "passage_id"
                    ],

                "passage":
                    record[
                        "passage"
                    ],
            })

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ==================================================
# 16. MAIN
# ==================================================

def main():

    pages = load_pages()

    print(
        "Source webpages:",
        len(pages),
    )

    records = build_database(
        pages
    )

    records = (
        remove_exact_duplicate_passages(
            records
        )
    )

    if not records:
        raise ValueError(
            "No passages were generated."
        )

    print(
        "Passages generated:",
        len(records),
    )

    print(
        "Checking exact duplicates..."
    )

    exact_duplicates = (
        find_exact_duplicates(
            records
        )
    )

    print(
        "Exact duplicate pairs:",
        len(exact_duplicates),
    )

    print(
        "Checking near-duplicates..."
    )

    near_duplicates = (
        find_near_duplicates(
            records
        )
    )

    print(
        "Near-duplicate pairs:",
        len(near_duplicates),
    )

    lengths = [
        record["word_count"]
        for record in records
    ]

    report = {
        "source_webpages":
            len(pages),

        "passages_generated":
            len(records),

        "passage_method":
            (
                "Fair Work section-aware passage "
                "construction with table-aware "
                "handling for fwo_015"
            ),

        "minimum_passage_words":
            min(lengths),

        "maximum_passage_words":
            max(lengths),

        "average_passage_words":
            round(
                sum(lengths)
                / len(lengths),
                2,
            ),

        "exact_duplicate_pairs_for_review":
            exact_duplicates,

        "near_duplicate_pairs_for_review":
            near_duplicates,

        "passages_per_document":
            dict(
                Counter(
                    record["source_id"]
                    for record in records
                )
            ),
    }

    save_database(
        records,
        report,
    )

    print()
    print(
        "Database files created:"
    )

    print(
        PASSAGES_JSONL
    )

    print(
        COLLECTION_CSV
    )

    print(
        REPORT_FILE
    )

    print()

    print(
        "Review passage boundaries "
        "before freezing passage IDs."
    )


if __name__ == "__main__":
    main()