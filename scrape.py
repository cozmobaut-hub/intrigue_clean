#!/usr/bin/env python3

import os
import re
import time
import sys
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_ARCHIVE = "https://archives.internationalintrigue.io"
POSTS_ENDPOINT = BASE_ARCHIVE + "/posts"
OUTPUT_DIR = "intrigue_clean"
REQUEST_DELAY = 0.5
TIMEOUT = 15

# Browser-like headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://archives.internationalintrigue.io"
}


def slugify(text, max_length=80):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    return text or "untitled"


def fetch_json(session, page, per_page=6):
    params = {
        "page": page,
        "per_page": per_page,
        "_data": "routes/__loaders/posts",
    }
    resp = session.get(POSTS_ENDPOINT, headers=HEADERS, params=params, timeout=TIMEOUT)
    if resp.status_code == 403:
        raise RuntimeError(f"403 on posts endpoint at page {page}")
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError(f"Failed to parse JSON on page {page}: {e}")


def discover_all_posts(session):
    """
    Walk page=1,2,3,... via the posts endpoint until it returns empty.
    Returns a list of dicts with keys: slug, url, title, published_at.
    """
    page = 1
    all_posts = []
    seen_slugs = set()

    while True:
        try:
            data = fetch_json(session, page)
        except Exception as e:
            print(f"[WARN] Stopping pagination at page {page}: {e}")
            break

        # Heuristic: Beehiiv / loaders usually return a dict with "posts" or a list itself.
        # We'll support a couple patterns:
        posts = None
        if isinstance(data, dict):
            for key in ["posts", "data", "items"]:
                if key in data and isinstance(data[key], list):
                    posts = data[key]
                    break
            if posts is None and isinstance(data.get("posts"), list):
                posts = data["posts"]
        elif isinstance(data, list):
            posts = data

        if not posts:
            print(f"[INFO] No posts found on page {page}, stopping.")
            break

        new_count = 0
        for p in posts:
            # Try common Beehiiv keys
            slug = (
                p.get("slug")
                or p.get("path")
                or p.get("id")
            )
            if not slug:
                continue

            # If path already includes /p/, keep; otherwise prepend
            if not slug.startswith("/"):
                slug_path = f"/p/{slug}"
            else:
                slug_path = slug

            url = urljoin(BASE_ARCHIVE, slug_path)

            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            title = p.get("title") or p.get("headline") or ""
            published_at = p.get("published_at") or p.get("created_at") or ""

            all_posts.append(
                {
                    "slug": slug,
                    "url": url,
                    "title": title,
                    "published_at": published_at,
                }
            )
            new_count += 1

        print(f"[INFO] Page {page}: {new_count} new posts, {len(all_posts)} total.")

        if new_count == 0:
            print(f"[INFO] No new posts on page {page}, stopping.")
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    return all_posts


def fetch_soup(url, session):
    resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
    if resp.status_code == 403:
        raise RuntimeError(f"403 fetching {url}")
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_main_text(soup):
    candidates = []

    article = soup.find("article")
    if article:
        candidates.append(article)

    for cls in ["post-content", "content", "article-body", "newsletter-body", "prose"]:
        div = soup.find("div", class_=lambda c: c and cls in c)
        if div:
            candidates.append(div)

    if not candidates:
        candidates.append(soup.body or soup)

    best = max(candidates, key=lambda el: len(el.get_text(separator="\n", strip=True)))

    text = best.get_text(separator="\n", strip=True)
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned_lines = []
    for ln in lines:
        if ln.strip() == "" and (not cleaned_lines or cleaned_lines[-1] == ""):
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines).strip()


def save_newsletter(meta, session):
    url = meta["url"]

    try:
        soup = fetch_soup(url, session)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return False

    title = meta.get("title") or (soup.find("title").get_text(strip=True) if soup.find("title") else url)
    slug = slugify(title)

    date_str = None
    # try meta first
    if meta.get("published_at"):
        date_str = str(meta["published_at"])[:10].replace("/", "-").replace(".", "-")
    if not date_str:
        for meta_name in ["article:published_time", "date", "og:updated_time"]:
            m = soup.find("meta", attrs={"property": meta_name}) or soup.find(
                "meta", attrs={"name": meta_name}
            )
            if m and m.get("content"):
                date_str = m["content"][:10].replace("/", "-").replace(".", "-")
                break

    if date_str:
        fname = f"{date_str}-{slug}.txt"
    else:
        fname = f"{slug}.txt"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)

    text = extract_main_text(soup)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Title: {title}\n")
        if date_str:
            f.write(f"Date: {date_str}\n")
        f.write("\n")
        f.write(text)

    return True


def main():
    session = requests.Session()

    print(f"[INFO] Discovering all posts via {POSTS_ENDPOINT}")
    posts = discover_all_posts(session)

    print(f"[INFO] Total unique newsletters discovered: {len(posts)}")
    if not posts:
        print("[ERROR] No posts discovered; check the JSON structure or endpoint.")
        sys.exit(1)

    success = 0
    fail = 0

    with tqdm(total=len(posts), desc="Downloading newsletters", unit="post") as pbar:
        for meta in posts:
            if save_newsletter(meta, session):
                success += 1
            else:
                fail += 1
            pbar.update(1)
            time.sleep(REQUEST_DELAY)

    print(f"[INFO] Done. Success: {success}, Failed: {fail}")
    print(f"[INFO] Clean text saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
