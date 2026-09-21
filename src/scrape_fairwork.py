"""Scrape Fair Work Ombudsman pages for the Casual Work Rights Assistant.

Automatic pages are downloaded with requests. Pages that block scripts
(403 errors, e.g. anything under /tools-and-resources/library/) are saved
by hand from a browser into data/raw/manual/ and cleaned the same way.

Output:
    data/raw/<id>.html                   original HTML of scraped pages
    data/processed/raw_pages.jsonl       one cleaned page per line
"""
import json
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Pages the script can download itself
URLS = [
    "https://www.fairwork.gov.au/casual",
    "https://www.fairwork.gov.au/employeechoice",
    "https://www.fairwork.gov.au/ceis",
    "https://www.fairwork.gov.au/right-to-disconnect",
    "https://www.fairwork.gov.au/minimum-wages",
    "https://www.fairwork.gov.au/NES",
    # breaks
    "https://www.fairwork.gov.au/employee-entitlements/hours-of-work-breaks-and-rosters/breaks",
    # leave
    "https://www.fairwork.gov.au/sickandcarersleave",
    "https://www.fairwork.gov.au/leave/sick-and-carers-leave/paid-sick-and-carers-leave",
    "https://www.fairwork.gov.au/language-help/english/holidays-and-taking-time-off-work/sick-and-carers-leave",
    "https://www.fairwork.gov.au/annualleave",
    "https://www.fairwork.gov.au/fdvleave",
    "https://www.fairwork.gov.au/publicholidays",
    # pay
    "https://www.fairwork.gov.au/pay-and-wages/penalty-rates-allowances-and-other-payments/penalty-rates",
    "https://www.fairwork.gov.au/find-help-for/fast-food-restaurants-cafes/fast-food-industry",
    "https://www.fairwork.gov.au/recordkeeping",
    "https://www.fairwork.gov.au/language-help/english/pay-slips-and-record-keeping",
    "https://www.fairwork.gov.au/pay/unpaid-work",
    "https://www.fairwork.gov.au/flexibility",
    # ending employment
    "https://www.fairwork.gov.au/ending-employment/notice-and-final-pay",
    "https://www.fairwork.gov.au/ending-employment/notice-and-final-pay/final-pay",
    # fact sheets (may return 403 like the library pages)
    "https://www.fairwork.gov.au/tools-and-resources/fact-sheets/minimum-workplace-entitlements/personal-leave-and-compassionate-leave",
    "https://www.fairwork.gov.au/tools-and-resources/fact-sheets/rights-and-obligations/record-keeping-pay-slips",
    "https://www.fairwork.gov.au/tools-and-resources/fact-sheets/unpaid-work/unpaid-work-unpaid-work",

    # award, junior pay, resignation, notice
    "https://www.fairwork.gov.au/employment-conditions/awards/awards-summary/ma000009-summary",
    "https://www.fairwork.gov.au/pay-and-wages/minimum-wages/junior-pay-rates",
    "https://www.fairwork.gov.au/ending-employment/resignation",
    "https://www.fairwork.gov.au/language-help/english/ending-employment",
]

# Pages saved by hand from Chrome (Cmd+S, "Webpage, HTML Only")
# file name in data/raw/manual/  ->  the page's real URL
MANUAL_URLS = {
    "K600467.html": "https://www.fairwork.gov.au/tools-and-resources/library/K600467_How-do-penalties-loadings-interact-",
    "K600162.html": "https://www.fairwork.gov.au/tools-and-resources/library/K600162_Casual-penalty-rates-in-the-Retail-Award",
}

BASE = Path(__file__).resolve().parent.parent  # the WIL_38 folder
RAW = BASE / "data" / "raw"
MANUAL = RAW / "manual"
OUT = BASE / "data" / "processed" / "raw_pages.jsonl"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-AU,en;q=0.9",
}

# Fair Work pages end with this boilerplate. Cut from the first match.
END_MARKERS = ("View references", "What to do next",
               "Give us feedback on this article", "Page reference No",
               "© Copyright Fair Work Ombudsman")
# Single lines to drop wherever they appear
SKIP_LINES = {"Skip to main content",
              "You’re offline. This is a read only version of the page.",
              "You're offline. This is a read only version of the page."}
DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer",
             "aside", "form", "iframe", "button"]
BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th",
              "dt", "dd", "blockquote", "pre", "div", "section", "article"]
GENERIC_TITLES = {"", "Fair Work Ombudsman", "FWO Search"}


# ------------------------------------------------------------------ cleaning

def is_generic(t):
    return t in GENERIC_TITLES or "Article Preview" in t


def find_title(soup):
    """Take the first real title from the h1, og:title or <title> tag."""
    candidates = []
    h1 = soup.find("h1")
    if h1:
        candidates.append(h1.get_text(" ", strip=True))
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        candidates.append(og["content"])
    if soup.title:
        candidates.append(soup.title.get_text(" ", strip=True))
    for t in candidates:
        t = " ".join(t.split())  # collapse stray newlines and spaces
        t = t.replace(" | Fair Work Ombudsman", "").replace(" - Fair Work Ombudsman", "").strip()
        if not is_generic(t):
            return t
    return "Fair Work Ombudsman"


def find_body(soup):
    """Locate the article element, trying the skip link's target first."""
    options = []
    skip = soup.find("a", string=lambda s: s and "skip to main content" in s.lower())
    if skip and skip.get("href", "").startswith("#"):
        options.append(soup.find(id=skip["href"][1:]))
    for sel in ("main", "[role=main]", "#main-content", "#main", "article"):
        options.append(soup.select_one(sel))
    for el in options:
        # an empty anchor just marks the spot, so require real text
        if el is not None and len(el.get_text(" ", strip=True).split()) > 50:
            return el
    return soup.body


def block_lines(body):
    """One line per paragraph, heading or list item, with links kept inline."""
    for el in body.find_all(BLOCK_TAGS):
        if el.find(BLOCK_TAGS):  # only the innermost blocks, so nothing repeats
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        text = re.sub(r"\s+([,.;:!?)])", r"\1", text)  # "employees , for" -> "employees, for"
        if text:
            yield text


def clean(html):
    """Return (title, text) with menus, footers and page boilerplate removed."""
    soup = BeautifulSoup(html, "html.parser")
    title = find_title(soup)   # before deleting anything
    body = find_body(soup)     # before the header holding the skip link goes
    for tag in body(DROP_TAGS):
        tag.decompose()
    kept = []
    for line in block_lines(body):
        if line in SKIP_LINES:
            continue
        if not any(ch.isalnum() for ch in line):
            continue
        if line.startswith(END_MARKERS):
            break
        if kept and line == kept[-1]:  # same line twice in a row
            continue
        kept.append(line)
    if is_generic(title) and kept:
        title = kept[0]
    return title, "\n".join(kept)


# ------------------------------------------------------------------ collection

def make_record(pid, url, requested_url, title, text, method):
    if len(text.split()) < 80:
        print(f"check {pid}: only {len(text.split())} words")
    return {"id": pid, "url": url, "requested_url": requested_url,
            "title": title, "text": text,
            "scraped_on": date.today().isoformat(), "method": method}


def scrape_pages(records, seen):
    failed = []
    for n, url in enumerate(URLS, 1):
        pid = f"fwo_{n:03d}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            failed.append((url, str(e)))
            continue
        if r.url in seen:
            print(f"duplicate, skipped: {url}")
            continue
        seen.add(r.url)
        (RAW / f"{pid}.html").write_text(r.text, encoding="utf-8")
        title, text = clean(r.text)
        records.append(make_record(pid, r.url, url, title, text, "scraped"))
        time.sleep(1.5)
    return failed


def add_manual_pages(records, seen):
    start = len(URLS) + 1  # IDs continue after the scraped pages
    for n, (fname, url) in enumerate(MANUAL_URLS.items(), start):
        pid = f"fwo_{n:03d}"
        path = MANUAL / fname
        if not path.exists():
            print(f"missing manual file: {path}")
            continue
        if url in seen:
            print(f"duplicate, skipped: {url}")
            continue
        seen.add(url)
        title, text = clean(path.read_text(encoding="utf-8", errors="ignore"))
        records.append(make_record(pid, url, url, title, text, "manual"))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    MANUAL.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    records, seen = [], set()
    failed = scrape_pages(records, seen)
    add_manual_pages(records, seen)

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    scraped = sum(r["method"] == "scraped" for r in records)
    manual = len(records) - scraped
    print(f"{len(records)} pages saved to {OUT} ({scraped} scraped, {manual} manual)")
    for url, err in failed:
        print(f"FAILED {url}: {err}")


if __name__ == "__main__":
    main()