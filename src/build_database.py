
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
PASSAGES_CSV = OUTPUT_DIR / "passages.csv"
REPORT_FILE = OUTPUT_DIR / "database_report.json"

CHUNK_SIZE = 250
OVERLAP = 50
STEP = CHUNK_SIZE - OVERLAP

FIELDS = [
    "id",
    "document_id",
    "title",
    "url",
    "text",
    "chunk_index",
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
        "r", encoding="utf-8-sig"
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
                "id", "url", "title", "text", "scraped_on"
            ]:

                value = page.get(field)

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Missing or invalid '{field}' "
                        f"on line {line_number}"
                    )

            pages.append(page)

    if not pages:
        raise ValueError("No source webpages were loaded.")

    return pages


# ==================================================
# 3. CLEAN SCRAPED TEXT
# ==================================================

def clean_text(text, document_id="unknown"):

    # Repair common character-encoding problems.
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
        text = text.replace(incorrect, correct)

    # Preserve line boundaries while normalising whitespace.
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]

    # ----------------------------------------------
    # Remove the "On this page" table of contents.
    # ----------------------------------------------

    toc_present = "On this page:" in lines
    toc_removed = False

    if toc_present:

        toc_start = lines.index("On this page:")

        first_heading = next(
            (
                line
                for line in lines[toc_start + 1:]
                if line
            ),
            None
        )

        if first_heading is not None:

            try:
                # Locate the first TOC heading when it
                # appears again in the article body.
                article_start = lines.index(
                    first_heading,
                    toc_start + 2
                )

                lines = (
                    lines[:toc_start]
                    + lines[article_start:]
                )

                toc_removed = True

            except ValueError:
                print(
                    f"WARNING: TOC removal failed for "
                    f"{document_id}. Footer removal "
                    f"skipped. Review this page manually."
                )

        else:
            print(
                f"WARNING: No TOC headings found for "
                f"{document_id}. Footer removal skipped. "
                f"Review this page manually."
            )

    # ----------------------------------------------
    # Remove identifiable footer navigation.
    # ----------------------------------------------

    # If TOC removal failed, footer headings may still
    # appear inside the TOC. Do not risk cutting the
    # article at one of those earlier occurrences.
    if toc_removed or not toc_present:

        footer_headings = {
            "Tools and resources",
            "Related information",
            "Tags",
        }

        footer_start = None

        for index, line in enumerate(lines):

            if line in footer_headings:
                footer_start = index
                break

        if footer_start is not None:
            lines = lines[:footer_start]

    # Join the retained text into a single string.
    cleaned_text = "\n".join(line for line in lines if line)


    # Remove the unwanted placeholder.
    cleaned_text = cleaned_text.replace(
        "Industry Embedded Filter Placeholder", ""
    )

    # Fix repeated full stops.
    cleaned_text = re.sub(r"\.{2,}", ".", cleaned_text)

    # Normalise whitespace after cleaning.
    cleaned_text = "\n".join(
    re.sub(r"\s+", " ", line).strip()
    for line in cleaned_text.splitlines()
    if line.strip()
)

    return cleaned_text


# ==================================================
# 4. PARAGRAPH-AWARE CHUNKING
# ==================================================

def chunk_text(text):
    paragraphs = [
        paragraph.strip()
        for paragraph in text.splitlines()
        if paragraph.strip()
    ]

    chunks = []
    current_paragraphs = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_word_count = len(paragraph_words)

        # If adding this paragraph would exceed the chunk size,
        # save the current chunk first.
        if (
            current_paragraphs
            and current_word_count + paragraph_word_count > CHUNK_SIZE
        ):
            chunks.append("\n".join(current_paragraphs))

            # Keep the last paragraph as overlap when possible.
            overlap_paragraph = current_paragraphs[-1]

            current_paragraphs = [overlap_paragraph]
            current_word_count = len(overlap_paragraph.split())

        # If a single paragraph is larger than the chunk size,
        # split it into smaller pieces.
        if paragraph_word_count > CHUNK_SIZE:
            words = paragraph_words

            for start in range(0, len(words), STEP):
                end = min(start + CHUNK_SIZE, len(words))

                chunk = " ".join(words[start:end])

                if chunk:
                    chunks.append(chunk)

                if end == len(words):
                    break

            current_paragraphs = []
            current_word_count = 0

        else:
            current_paragraphs.append(paragraph)
            current_word_count += paragraph_word_count

    # Add remaining paragraphs.
    if current_paragraphs:
        chunks.append("\n".join(current_paragraphs))

    return chunks

    # The scraper stores paragraphs, headings and list items
    # on separate lines. Keep those boundaries when chunking.
    blocks = [
        block.strip()
        for block in text.splitlines()
        if block.strip()
    ]

    chunks = []
    current_blocks = []
    current_word_count = 0

    for block in blocks:

        block_word_count = len(block.split())

        # If adding this block would make the passage too large,
        # save the current passage first.
        if (
            current_blocks
            and current_word_count + block_word_count > CHUNK_SIZE
        ):
            chunks.append("\n\n".join(current_blocks))

            # Keep the last block as contextual overlap.
            overlap_blocks = []
            overlap_words = 0

            for previous_block in reversed(current_blocks):

                previous_words = len(previous_block.split())

                if overlap_words + previous_words > OVERLAP:
                    break

                overlap_blocks.insert(0, previous_block)
                overlap_words += previous_words

            current_blocks = overlap_blocks
            current_word_count = overlap_words

        current_blocks.append(block)
        current_word_count += block_word_count

    # Add the final passage.
    if current_blocks:
        chunks.append("\n\n".join(current_blocks))

    return chunks


# ==================================================
# 5. NORMALISE TEXT FOR DUPLICATE CHECKING
# ==================================================

def normalise(text):

    return re.sub(
        r"\s+", " ", text.casefold()
    ).strip()


# ==================================================
# 6. BUILD PASSAGE RECORDS
# ==================================================

def build_database(pages):

    records = []

    seen_document_ids = set()
    seen_urls = set()

    for page in sorted(pages, key=lambda p: p["id"]):

        document_id = page["id"].strip()
        url = page["url"].strip()

        if document_id in seen_document_ids:
            raise ValueError(
                f"Duplicate document ID: {document_id}"
            )

        if url in seen_urls:
            raise ValueError(
                f"Duplicate source URL: {url}"
            )

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid source URL: {url}"
            )

        if not parsed.netloc:
            raise ValueError(
                f"Invalid source URL: {url}"
            )

        seen_document_ids.add(document_id)
        seen_urls.add(url)

        cleaned_text = clean_text(
            page["text"],
            document_id=document_id
        )

        # Prevent an individual webpage from
        # disappearing silently during cleaning.
        if not cleaned_text:
            raise ValueError(
                f"Cleaning removed all text from "
                f"{document_id}: {url}"
            )

        chunks = chunk_text(cleaned_text)

        if not chunks:
            raise ValueError(
                f"No passages generated for "
                f"{document_id}: {url}"
            )

        for index, chunk in enumerate(chunks):

            passage_id = (
                f"{document_id}_{index:04d}"
            )

            record = {
                "id": passage_id,
                "document_id": document_id,
                "title": page["title"].strip(),
                "url": url,
                "text": chunk,
                "chunk_index": index,
                "word_count": len(chunk.split()),
                "scraped_on": page["scraped_on"].strip(),
            }

            records.append(record)

    return records


# ==================================================
# 7. FIND EXACT DUPLICATES
# ==================================================

def find_exact_duplicates(records):

    seen_texts = {}

    duplicates = []

    for record in records:

        normalised = normalise(record["text"])

        if normalised in seen_texts:

            duplicates.append({
                "passage_id_1": seen_texts[normalised],
                "passage_id_2": record["id"],
            })

        else:
            seen_texts[normalised] = record["id"]

    return duplicates


# ==================================================
# 8. FIND NEAR-DUPLICATES FOR MANUAL REVIEW
# ==================================================

def find_near_duplicates(records):

    flagged = []

    for i, first in enumerate(records):

        text_a = normalise(first["text"])

        for second in records[i + 1:]:

            # Adjacent passages intentionally overlap.
            if (
                first["document_id"] == second["document_id"]
                and abs(
                    first["chunk_index"]
                    - second["chunk_index"]
                ) <= 1
            ):
                continue

            text_b = normalise(second["text"])

            # Exact duplicates are reported separately.
            if text_a == text_b:
                continue

            length_ratio = (
                min(len(text_a), len(text_b))
                / max(len(text_a), len(text_b))
            )

            if length_ratio < 0.95:
                continue

            similarity = SequenceMatcher(
                None,
                text_a,
                text_b,
                autojunk=False,
            ).ratio()

            if similarity >= 0.98:

                flagged.append({
                    "passage_id_1": first["id"],
                    "passage_id_2": second["id"],
                    "similarity": round(similarity, 4),
                })

    return flagged


# ==================================================
# 9. SAVE DATABASE FILES
# ==================================================

def save_database(records, report):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with PASSAGES_JSONL.open(
        "w", encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    with PASSAGES_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(records)

    with REPORT_FILE.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )


# ==================================================
# 10. MAIN PROGRAM
# ==================================================

def main():

    pages = load_pages()

    print("Source webpages:", len(pages))

    records = build_database(pages)

    if not records:
        raise ValueError(
            "No passages were generated."
        )

    print("Passages generated:", len(records))

    print("Checking exact duplicates...")
    exact_duplicates = find_exact_duplicates(records)

    print(
        "Exact duplicate pairs:",
        len(exact_duplicates)
    )

    print("Checking near-duplicates...")
    near_duplicates = find_near_duplicates(records)

    print(
        "Near-duplicate pairs:",
        len(near_duplicates)
    )

    report = {
        "source_webpages": len(pages),
        "passages_generated": len(records),
        "exact_duplicate_pairs_for_review": (
            exact_duplicates
        ),
        "near_duplicate_pairs_for_review": (
            near_duplicates
        ),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "passages_per_document": dict(
            Counter(
                record["document_id"]
                for record in records
            )
        ),
    }

    save_database(records, report)

    print("\nDatabase files created:")
    print(PASSAGES_JSONL)
    print(PASSAGES_CSV)
    print(REPORT_FILE)

    print(
        "\nReview duplicate pairs and inspect "
        "cleaned article boundaries before "
        "finalising the database."
    )


if __name__ == "__main__":
    main()