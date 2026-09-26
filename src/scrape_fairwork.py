"""
Scrape Fair Work Ombudsman pages for the Casual Work Rights Assistant.

Automatic pages are downloaded with requests. Pages that block scripts
(403 errors, e.g. anything under /tools-and-resources/library/) are saved
by hand from a browser into data/raw/manual/ and cleaned the same way.

Output:
    data/raw/<id>.html
        Original HTML of scraped pages

    data/processed/raw_pages.jsonl
        One cleaned and structured page per line
"""

import json
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# ==============================================================
# URLS
# ==============================================================

# Pages the script can download itself
URLS = [
    "https://www.fairwork.gov.au/casual",
    "https://www.fairwork.gov.au/employeechoice",
    "https://www.fairwork.gov.au/ceis",
    "https://www.fairwork.gov.au/right-to-disconnect",
    "https://www.fairwork.gov.au/minimum-wages",
    "https://www.fairwork.gov.au/NES",

    # Breaks
    "https://www.fairwork.gov.au/employee-entitlements/hours-of-work-breaks-and-rosters/breaks",

    # Leave
    "https://www.fairwork.gov.au/sickandcarersleave",
    "https://www.fairwork.gov.au/leave/sick-and-carers-leave/paid-sick-and-carers-leave",
    "https://www.fairwork.gov.au/language-help/english/holidays-and-taking-time-off-work/sick-and-carers-leave",
    "https://www.fairwork.gov.au/annualleave",
    "https://www.fairwork.gov.au/fdvleave",
    "https://www.fairwork.gov.au/publicholidays",

    # Pay
    "https://www.fairwork.gov.au/pay-and-wages/penalty-rates-allowances-and-other-payments/penalty-rates",
    "https://www.fairwork.gov.au/find-help-for/fast-food-restaurants-cafes/fast-food-industry",
    "https://www.fairwork.gov.au/recordkeeping",
    "https://www.fairwork.gov.au/language-help/english/pay-slips-and-record-keeping",
    "https://www.fairwork.gov.au/pay/unpaid-work",
    "https://www.fairwork.gov.au/flexibility",

    # Ending employment
    "https://www.fairwork.gov.au/ending-employment/notice-and-final-pay",
    "https://www.fairwork.gov.au/ending-employment/notice-and-final-pay/final-pay",

    # Fact sheets
    "https://www.fairwork.gov.au/tools-and-resources/fact-sheets/minimum-workplace-entitlements/personal-leave-and-compassionate-leave",
    "https://www.fairwork.gov.au/tools-and-resources/fact-sheets/rights-and-obligations/record-keeping-pay-slips",
    "https://www.fairwork.gov.au/tools-and-resources/fact-sheets/unpaid-work/unpaid-work-unpaid-work",

    # Award, junior pay, resignation, notice
    "https://www.fairwork.gov.au/employment-conditions/awards/awards-summary/ma000009-summary",
    "https://www.fairwork.gov.au/pay-and-wages/minimum-wages/junior-pay-rates",
    "https://www.fairwork.gov.au/ending-employment/resignation",
    "https://www.fairwork.gov.au/language-help/english/ending-employment",
]


# ==============================================================
# MANUAL PAGES
# ==============================================================

# Pages saved manually from Chrome.
# File name in data/raw/manual/ -> real page URL
MANUAL_URLS = {
    "K600467.html":
        "https://www.fairwork.gov.au/tools-and-resources/library/K600467_How-do-penalties-loadings-interact-",

    "K600162.html":
        "https://www.fairwork.gov.au/tools-and-resources/library/K600162_Casual-penalty-rates-in-the-Retail-Award",
}


# ==============================================================
# PATHS
# ==============================================================

BASE = Path(__file__).resolve().parent.parent

RAW = BASE / "data" / "raw"

MANUAL = RAW / "manual"

OUT = (
    BASE
    / "data"
    / "processed"
    / "raw_pages.jsonl"
)


# ==============================================================
# REQUEST HEADERS
# ==============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-AU,en;q=0.9",
}


# ==============================================================
# CLEANING CONFIGURATION
# ==============================================================

# Fair Work pages often end with one of these sections.
# Stop collecting article content when encountered.
END_MARKERS = (
    "View references",
    "What to do next",
    "Give us feedback on this article",
    "Page reference No",
    "© Copyright Fair Work Ombudsman",
)


# Individual lines that are never useful article content.
SKIP_LINES = {
    "Skip to main content",
    "You’re offline. This is a read only version of the page.",
    "You're offline. This is a read only version of the page.",
}


# Elements that normally contain interface/navigation content.
DROP_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "button",
    "svg",
]


# Elements representing meaningful article blocks.
BLOCK_TAGS = [
    "p",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "td",
    "th",
    "dt",
    "dd",
    "blockquote",
    "pre",
]


GENERIC_TITLES = {
    "",
    "Fair Work Ombudsman",
    "FWO Search",
}


# Exact UI text that can safely be ignored.
UI_LINES = {
    "Print this page",
    "Share this page",
    "Listen",
    "Download",
    "Back to top",
}


# ==============================================================
# BASIC TEXT HELPERS
# ==============================================================

def normalise_text(text):
    """
    Normalise whitespace and punctuation without destroying
    paragraph/list structure.
    """

    if not text:
        return ""

    text = (
        text
        .replace("\xa0", " ")
        .replace("\u200b", "")
    )

    text = " ".join(text.split())

    # Remove spaces before punctuation.
    text = re.sub(
        r"\s+([,.;:!?)])",
        r"\1",
        text,
    )

    # Remove spaces directly after opening brackets.
    text = re.sub(
        r"([(])\s+",
        r"\1",
        text,
    )

    return text.strip()


def comparison_text(text):
    """
    Produce a normalised form used only for duplicate comparison.
    """

    text = text.removeprefix("- ").strip()

    return normalise_text(text).casefold()


def is_generic(title):
    """Return True if a page title is not useful."""

    if not title:
        return True

    return (
        title in GENERIC_TITLES
        or "Article Preview" in title
    )


# ==============================================================
# TITLE EXTRACTION
# ==============================================================

def find_title(soup):
    """
    Find the most useful page title.

    Priority:
        1. h1
        2. og:title
        3. HTML title
    """

    candidates = []

    h1 = soup.find("h1")

    if h1:
        candidates.append(
            h1.get_text(" ", strip=True)
        )

    og = soup.find(
        "meta",
        attrs={"property": "og:title"},
    )

    if og and og.get("content"):
        candidates.append(
            og["content"]
        )

    if soup.title:
        candidates.append(
            soup.title.get_text(
                " ",
                strip=True,
            )
        )

    for candidate in candidates:

        candidate = normalise_text(
            candidate
        )

        candidate = (
            candidate
            .replace(
                " | Fair Work Ombudsman",
                "",
            )
            .replace(
                " - Fair Work Ombudsman",
                "",
            )
            .strip()
        )

        if not is_generic(candidate):
            return candidate

    return "Fair Work Ombudsman"


# ==============================================================
# MAIN ARTICLE LOCATION
# ==============================================================

def find_body(soup):
    """
    Locate the main article area.

    The skip-to-content target is checked first,
    followed by common main-content selectors.
    """

    options = []

    skip = soup.find(
        "a",
        string=lambda s:
            s
            and "skip to main content"
            in s.lower(),
    )

    if (
        skip
        and skip.get(
            "href",
            "",
        ).startswith("#")
    ):

        target_id = (
            skip["href"][1:]
        )

        options.append(
            soup.find(
                id=target_id
            )
        )

    selectors = (
        "main",
        "[role=main]",
        "#main-content",
        "#main",
        "article",
    )

    for selector in selectors:

        options.append(
            soup.select_one(
                selector
            )
        )

    for element in options:

        if element is None:
            continue

        words = (
            element
            .get_text(
                " ",
                strip=True,
            )
            .split()
        )

        if len(words) > 50:
            return element

    return soup.body


# ==============================================================
# REMOVE OBVIOUS NON-ARTICLE COMPONENTS
# ==============================================================

def remove_non_content_elements(body):
    """
    Remove elements that are clearly interface components rather
    than article content.

    This intentionally stays conservative so useful Fair Work
    information is not accidentally deleted.
    """

    if body is None:
        return

    # Standard HTML interface elements.
    for tag in body.find_all(DROP_TAGS):
        tag.decompose()

    # Remove hidden elements.
    for element in body.select(
        '[hidden], '
        '[aria-hidden="true"]'
    ):
        element.decompose()


# ==============================================================
# EXTRACT STRUCTURED BLOCKS
# ==============================================================

def block_lines(body):
    """
    Extract meaningful article blocks.

    Returns:
        (tag_name, text)

    Headings, paragraphs and list items remain separate so
    structure is not lost during scraping.
    """

    if body is None:
        return

    for element in body.find_all(
        BLOCK_TAGS
    ):

        # Ignore blocks nested inside another selected block.
        # This prevents duplicate extraction.
        if element.find(BLOCK_TAGS):
            continue

        text = normalise_text(
            element.get_text(
                " ",
                strip=True,
            )
        )

        if not text:
            continue

        # Preserve list structure.
        if element.name == "li":
            text = "- " + text

        yield element.name, text


# ==============================================================
# IDENTIFY OBVIOUS UI BLOCKS
# ==============================================================

def should_skip_line(
    tag_name,
    line,
):
    """
    Decide whether an extracted block is clearly interface
    content rather than article content.

    Keep this conservative.
    """

    plain = (
        line
        .removeprefix("- ")
        .strip()
    )

    if plain in SKIP_LINES:
        return True

    if plain in UI_LINES:
        return True

    if not any(
        character.isalnum()
        for character in plain
    ):
        return True

    # Video headings themselves are interface content.
    # Do not broadly remove surrounding paragraphs because they
    # may contain useful employment information.
    if (
        tag_name
        in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }
        and plain.casefold().startswith(
            "video:"
        )
    ):
        return True

    return False


# ==============================================================
# CLEAN HTML
# ==============================================================

def clean(html):
    """
    Convert Fair Work HTML into a title and structured article
    text.

    The function:
        - finds the real article
        - removes navigation/interface HTML
        - preserves headings
        - preserves paragraphs
        - preserves list items
        - removes duplicate title/H1
        - removes consecutive duplicate blocks
        - cuts known footer boilerplate
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Find these before deleting anything.
    title = find_title(soup)

    body = find_body(soup)

    if body is None:
        return title, ""

    remove_non_content_elements(
        body
    )

    kept = []

    for tag_name, line in block_lines(
        body
    ):

        plain = (
            line
            .removeprefix("- ")
            .strip()
        )

        # ------------------------------------------
        # Stop at known end-of-article sections
        # ------------------------------------------

        if any(
            plain.startswith(marker)
            for marker in END_MARKERS
        ):
            break

        # ------------------------------------------
        # Skip obvious UI/noise
        # ------------------------------------------

        if should_skip_line(
            tag_name,
            line,
        ):
            continue

        # ------------------------------------------
        # Remove duplicate page title from body
        # ------------------------------------------

        if (
            comparison_text(plain)
            == comparison_text(title)
        ):
            continue

        # ------------------------------------------
        # Remove consecutive duplicate blocks
        # ------------------------------------------

        if kept:

            if (
                comparison_text(line)
                == comparison_text(
                    kept[-1]
                )
            ):
                continue

        kept.append(line)

    # ----------------------------------------------
    # Fallback title
    # ----------------------------------------------

    if (
        is_generic(title)
        and kept
    ):

        title = (
            kept[0]
            .removeprefix("- ")
            .strip()
        )

    # Blank lines preserve logical article structure.
    text = "\n\n".join(kept)

    return title, text


# ==============================================================
# RECORD CREATION
# ==============================================================

def make_record(
    pid,
    url,
    requested_url,
    title,
    text,
    method,
):

    word_count = len(
        text.split()
    )

    if word_count < 80:

        print(
            f"check {pid}: "
            f"only {word_count} words"
        )

    return {
        "id": pid,
        "url": url,
        "requested_url": requested_url,
        "title": title,
        "text": text,
        "scraped_on":
            date.today().isoformat(),
        "method": method,
    }


# ==============================================================
# AUTOMATIC SCRAPING
# ==============================================================

def scrape_pages(
    records,
    seen,
):

    failed = []

    for n, url in enumerate(
        URLS,
        1,
    ):

        pid = f"fwo_{n:03d}"

        print(
            f"Scraping {pid}: {url}"
        )

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=20,
            )

            response.raise_for_status()

        except requests.RequestException as error:

            failed.append(
                (
                    url,
                    str(error),
                )
            )

            print(
                f"FAILED {pid}"
            )

            continue

        # ------------------------------------------
        # Avoid duplicate redirected pages
        # ------------------------------------------

        final_url = response.url

        if final_url in seen:

            print(
                f"duplicate, skipped: "
                f"{url}"
            )

            continue

        seen.add(final_url)

        # ------------------------------------------
        # Save original HTML
        # ------------------------------------------

        html_path = (
            RAW
            / f"{pid}.html"
        )

        html_path.write_text(
            response.text,
            encoding="utf-8",
        )

        # ------------------------------------------
        # Clean page
        # ------------------------------------------

        title, text = clean(
            response.text
        )

        records.append(
            make_record(
                pid,
                final_url,
                url,
                title,
                text,
                "scraped",
            )
        )

        print(
            f"  title: {title}"
        )

        print(
            f"  words: "
            f"{len(text.split())}"
        )

        time.sleep(1.5)

    return failed


# ==============================================================
# MANUAL PAGES
# ==============================================================

def add_manual_pages(
    records,
    seen,
):

    # IDs continue after automatic URL list.
    start = len(URLS) + 1

    for n, (
        filename,
        url,
    ) in enumerate(
        MANUAL_URLS.items(),
        start,
    ):

        pid = f"fwo_{n:03d}"

        path = (
            MANUAL
            / filename
        )

        if not path.exists():

            print(
                f"missing manual file: "
                f"{path}"
            )

            continue

        if url in seen:

            print(
                f"duplicate, skipped: "
                f"{url}"
            )

            continue

        seen.add(url)

        print(
            f"Processing manual page "
            f"{pid}: {filename}"
        )

        html = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        title, text = clean(
            html
        )

        records.append(
            make_record(
                pid,
                url,
                url,
                title,
                text,
                "manual",
            )
        )

        print(
            f"  title: {title}"
        )

        print(
            f"  words: "
            f"{len(text.split())}"
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    RAW.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANUAL.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    seen = set()

    # ----------------------------------------------
    # Scrape automatic pages
    # ----------------------------------------------

    failed = scrape_pages(
        records,
        seen,
    )

    # ----------------------------------------------
    # Add manually saved pages
    # ----------------------------------------------

    add_manual_pages(
        records,
        seen,
    )

    # ----------------------------------------------
    # Write JSONL database source
    # ----------------------------------------------

    with OUT.open(
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

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    scraped = sum(
        record["method"]
        == "scraped"
        for record in records
    )

    manual = (
        len(records)
        - scraped
    )

    print()
    print(
        "SCRAPING COMPLETE"
    )
    print(
        "-----------------"
    )

    print(
        f"Pages saved: "
        f"{len(records)}"
    )

    print(
        f"Automatically scraped: "
        f"{scraped}"
    )

    print(
        f"Manual pages: "
        f"{manual}"
    )

    print(
        f"Output: {OUT}"
    )

    if failed:

        print()
        print(
            "FAILED PAGES"
        )
        print(
            "------------"
        )

        for url, error in failed:

            print(
                f"{url}: {error}"
            )


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":
    main()