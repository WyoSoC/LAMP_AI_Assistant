"""Re-crawl the LAMP website subtree into data/corpus/*.md.

Run this on your own machine (it needs direct internet access) whenever you
want to refresh the corpus, then re-run `python -m app.ingest`.

- BFS restricted to https://www.uwyo.edu/science-initiative/lamp/
- HTML pages: main content extracted (boilerplate nav/footer stripped),
  converted to markdown with hyperlinks preserved.
- Linked PDFs under the LAMP subtree: text extracted with pypdf.
- Each output file carries YAML frontmatter: url, title, section, fetched.

Usage: python scripts/crawl.py [--delay 1.0]
Requires: requests, beautifulsoup4, markdownify, pypdf
"""
from __future__ import annotations

import argparse
import datetime
import io
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup

try:
    from markdownify import markdownify as md
except ImportError:
    md = None

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus"
BASE = "https://www.uwyo.edu/science-initiative/lamp/"
HEADERS = {"User-Agent": "LAMP-RAG-crawler/1.0 (research use; contact site owner via SI@uwyo.edu)"}
TODAY = datetime.date.today().isoformat()


def slugify(url: str) -> str:
    path = url.replace(BASE, "").strip("/")
    path = re.sub(r"\.(html?|pdf)$", "", path)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", path.replace("/", "__")) or "index"
    return slug


def extract_main(soup: BeautifulSoup) -> BeautifulSoup | None:
    main = soup.find("main") or soup.find(id="mainContent") or soup.find("body")
    if not main:
        return None
    for sel in ["nav", "header", "footer", "script", "style", "form",
                ".breadcrumb", "#mainFooter", ".uw-header", ".uw-footer"]:
        for el in main.select(sel):
            el.decompose()
    return main


def html_to_markdown(html: str) -> str:
    if md:
        text = md(html, heading_style="ATX", strip=["img"])
    else:
        text = BeautifulSoup(html, "html.parser").get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop the "Contact Us" footer block if present
    text = re.split(r"\n#{1,3}\s*Contact Us\s*\n", text)[0]
    return text.strip()


def save(url: str, title: str, body: str, doctype: str = "html"):
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"{slugify(url)}.md"
    fm = (f"---\nurl: {url}\ntitle: {title}\nsection: LAMP\n"
          f"fetched: {TODAY}\ndoctype: {doctype}\n---\n\n")
    fp.write_text(fm + body + "\n", encoding="utf-8")
    print(f"  saved {fp.name} ({len(body)} chars)")


def crawl(delay: float):
    seen, queue = set(), [BASE + "index.html"]
    pdfs = set()
    session = requests.Session()
    session.headers.update(HEADERS)

    while queue:
        url = queue.pop(0)
        url = urldefrag(url)[0]
        if url in seen or not url.startswith(BASE):
            continue
        seen.add(url)
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  FAIL {url}: {e}")
            continue
        print(f"fetch {url}")

        ctype = r.headers.get("Content-Type", "")
        if "pdf" in ctype or url.endswith(".pdf"):
            pdfs.add(url)
            continue
        if "html" not in ctype:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else slugify(url)
        main = extract_main(soup)
        if main:
            save(url, title, html_to_markdown(str(main)))

        for a in soup.find_all("a", href=True):
            link = urldefrag(urljoin(url, a["href"]))[0]
            if link.startswith(BASE):
                if link.endswith(".pdf"):
                    pdfs.add(link)
                elif re.search(r"\.html?$|/$", link) and link not in seen:
                    queue.append(link)
        time.sleep(delay)

    # PDFs
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf not installed — skipping PDF extraction")
        return
    for url in sorted(pdfs):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            reader = PdfReader(io.BytesIO(r.content))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
            if text:
                save(url, Path(url).name, text, doctype="pdf")
            else:
                print(f"  SKIP (no text layer): {url}")
        except Exception as e:
            print(f"  FAIL {url}: {e}")
        time.sleep(delay)

    print(f"\nDone: {len(seen)} pages visited, corpus in {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    crawl(ap.parse_args().delay)
