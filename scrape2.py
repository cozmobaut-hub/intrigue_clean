#!/usr/bin/env python3

import os
import re
import time
import sys
import random
from urllib.parse import urljoin, parse_qs, urlparse
import json

try:
    import cloudscraper
    USE_CLOUDSCRAPER = True
    print("[INFO] Using cloudscraper for Cloudflare bypass")
except ImportError:
    import requests
    USE_CLOUDSCRAPER = False
    print("[WARN] cloudscraper not installed, using requests")

from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_ARCHIVE = "https://archives.internationalintrigue.io"
OUTPUT_DIR = "intrigue_clean"
REQUEST_DELAY = 1.5
TIMEOUT = 30

USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def get_session():
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "X-Requested-With": "XMLHttpRequest",
    }
    
    if USE_CLOUDSCRAPER:
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'linux', 'desktop': True},
            delay=10  # Higher delay for Cloudflare
        )
        scraper.headers.update(headers)
        return scraper
    else:
        session = requests.Session()
        session.headers.update(headers)
        return session


def slugify(text, max_length=80):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    return text or "untitled"


def fetch_with_retry(session, url, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 403 and attempt < retries - 1:
                time.sleep((attempt + 1) * 5)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 3)
                continue
            raise


def discover_all_posts_api(session):
    """
    Try to get all posts via API with high per_page first.
    If that fails, use cursor-based pagination.
    """
    all_posts = []
    seen_slugs = set()
    
    # Try getting everything at once first (Beehiiv usually supports up to 100)
    print("[INFO] Trying API with high per_page...")
    for per_page in [100, 50, 30]:
        try:
            params = {
                "page": 1,
                "per_page": per_page,
                "_data": "routes/__loaders/posts",
            }
            resp = fetch_with_retry(session, f"{BASE_ARCHIVE}/posts", retries=2)
            data = resp.json()
            
            posts = None
            if isinstance(data, dict):
                for key in ["posts", "data", "items"]:
                    if key in data:
                        posts = data[key]
                        break
            elif isinstance(data, list):
                posts = data
            
            if posts and len(posts) > 30:
                print(f"[INFO] Got {len(posts)} posts with per_page={per_page}")
                for p in posts:
                    slug = p.get("slug") or p.get("path") or p.get("id")
                    if slug and slug not in seen_slugs:
                        seen_slugs.add(slug)
                        all_posts.append({
                            "slug": slug.lstrip("/").replace("p/", ""),
                            "url": urljoin(BASE_ARCHIVE, f"/p/{slug.lstrip('/').replace('p/', '')}"),
                            "title": p.get("title") or p.get("headline") or "",
                            "published_at": p.get("published_at") or p.get("created_at") or "",
                        })
                if len(all_posts) >= 100:
                    return all_posts
        except Exception as e:
            print(f"[WARN] per_page={per_page} failed: {e}")
            continue
    
    # If bulk fetch didn't work, try cursor-based pagination
    print("[INFO] Trying cursor-based pagination...")
    cursor = None
    page = 1
    
    while True:
        try:
            params = {
                "page": page,
                "per_page": 30,
                "_data": "routes/__loaders/posts",
            }
            if cursor:
                params["cursor"] = cursor
                params["after"] = cursor
            
            resp = fetch_with_retry(session, f"{BASE_ARCHIVE}/posts", retries=2)
            data = resp.json()
            
            # Try to find cursor for next page
            next_cursor = None
            if isinstance(data, dict):
                if "next" in data and data["next"]:
                    next_cursor = data["next"]
                elif "cursor" in data:
                    next_cursor = data["cursor"]
                elif "pagination" in data and isinstance(data["pagination"], dict):
                    next_cursor = data["pagination"].get("next_cursor") or data["pagination"].get("cursor")
                
                posts = data.get("posts") or data.get("data") or data.get("items")
            elif isinstance(data, list):
                posts = data
            else:
                posts = []
            
            if not posts:
                break
            
            new_count = 0
            for p in posts:
                slug = p.get("slug") or p.get("path") or p.get("id")
                if not slug:
                    continue
                slug = slug.lstrip("/").replace("p/", "")
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                
                all_posts.append({
                    "slug": slug,
                    "url": urljoin(BASE_ARCHIVE, f"/p/{slug}"),
                    "title": p.get("title") or p.get("headline") or "",
                    "published_at": p.get("published_at") or p.get("created_at") or "",
                })
                new_count += 1
            
            print(f"[INFO] API Page {page}: {new_count} new posts (total: {len(all_posts)})")
            
            if new_count == 0:
                break
            
            # Update cursor for next iteration
            if next_cursor:
                cursor = next_cursor
            page += 1
            time.sleep(REQUEST_DELAY + random.uniform(0.5, 1))
            
        except Exception as e:
            print(f"[WARN] API pagination stopped at page {page}: {e}")
            break
    
    return all_posts


def discover_via_archive_html(session):
    """
    Scrape the /archive page which usually has working pagination or "Load More" simulation.
    """
    print("[INFO] Scraping HTML archive pages...")
    all_posts = []
    seen_slugs = set()
    
    # Try different archive URL patterns
    archive_patterns = [
        f"{BASE_ARCHIVE}/archive",
        f"{BASE_ARCHIVE}/archive/",
        BASE_ARCHIVE,
    ]
    
    for base_url in archive_patterns:
        page = 1
        while True:
            try:
                # Try query param pagination
                url = f"{base_url}?page={page}" if page > 1 else base_url
                resp = fetch_with_retry(session, url)
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Look for post links
                links = soup.find_all("a", href=re.compile(r'/p/[^/]+'))
                new_count = 0
                
                for link in links:
                    href = link.get("href", "")
                    match = re.search(r'/p/([^/?#]+)', href)
                    if not match:
                        continue
                    
                    slug = match.group(1)
                    if slug in seen_slugs:
                        continue
                    
                    seen_slugs.add(slug)
                    title = link.get_text(strip=True) or ""
                    
                    # Try to find date nearby
                    date_str = ""
                    parent = link.find_parent(["article", "div", "li"])
                    if parent:
                        date_elem = parent.find(text=re.compile(r'\d{4}-\d{2}-\d{2}'))
                        if date_elem:
                            date_str = str(date_elem).strip()
                    
                    all_posts.append({
                        "slug": slug,
                        "url": urljoin(BASE_ARCHIVE, f"/p/{slug}"),
                        "title": title,
                        "published_at": date_str,
                    })
                    new_count += 1
                
                print(f"[INFO] Archive page {page}: {new_count} new posts (total: {len(all_posts)})")
                
                if new_count == 0:
                    break
                
                page += 1
                time.sleep(REQUEST_DELAY)
                
                # Safety limit
                if page > 100:
                    break
                    
            except Exception as e:
                print(f"[WARN] Archive scraping stopped: {e}")
                break
        
        if len(all_posts) > 100:
            break
    
    return all_posts


def discover_via_sitemap(session):
    """Parse sitemap.xml for all post URLs"""
    print("[INFO] Checking sitemap...")
    posts = []
    seen_slugs = set()
    
    sitemaps = [
        f"{BASE_ARCHIVE}/sitemap.xml",
        f"{BASE_ARCHIVE}/sitemap_index.xml",
        f"{BASE_ARCHIVE}/rss.xml",
        f"{BASE_ARCHIVE}/feed",
    ]
    
    for sitemap_url in sitemaps:
        try:
            resp = fetch_with_retry(session, sitemap_url, retries=1)
            content = resp.text
            
            # Extract all /p/ URLs
            urls = re.findall(r'(https?://[^<\s"]+/p/[^<\s"]+)', content)
            
            for url in urls:
                match = re.search(r'/p/([^/?#]+)', url)
                if match:
                    slug = match.group(1)
                    if slug not in seen_slugs:
                        seen_slugs.add(slug)
                        posts.append({
                            "slug": slug,
                            "url": url,
                            "title": "",
                            "published_at": "",
                        })
            
            if len(posts) > 100:
                print(f"[INFO] Found {len(posts)} posts via sitemap")
                return posts
                
        except Exception:
            continue
    
    return posts


def fetch_soup(session, url):
    resp = fetch_with_retry(session, url)
    return BeautifulSoup(resp.text, "html.parser")


def extract_main_text(soup):
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    
    candidates = []
    
    article = soup.find("article")
    if article:
        candidates.append(article)
    
    for cls in ["post-content", "content", "article-body", "newsletter-body", "prose", "entry-content"]:
        elems = soup.find_all("div", class_=lambda c: c and cls in c.lower())
        candidates.extend(elems)
    
    if not candidates:
        main = soup.find("main") or soup.find("div", role="main")
        if main:
            candidates.append(main)
    
    if not candidates:
        candidates.append(soup.body or soup)
    
    best = max(candidates, key=lambda el: len(el.get_text(separator="\n", strip=True)))
    text = best.get_text(separator="\n", strip=True)
    
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = []
    prev_empty = False
    for line in lines:
        if line == "":
            if not prev_empty:
                cleaned.append(line)
                prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False
    
    return "\n".join(cleaned).strip()


def save_newsletter(meta, session):
    url = meta["url"]
    
    try:
        soup = fetch_soup(session, url)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return False
    
    title = meta.get("title", "")
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
    
    slug = slugify(title) if title else meta["slug"]
    
    date_str = None
    if meta.get("published_at"):
        date_str = str(meta["published_at"])[:10].replace("/", "-")
    
    if not date_str:
        for meta_name in ["article:published_time", "date", "og:updated_time"]:
            m = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
            if m and m.get("content"):
                date_str = m["content"][:10]
                break
    
    if date_str:
        fname = f"{date_str}-{slug}.txt"
    else:
        fname = f"{slug}.txt"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)
    
    if os.path.exists(path):
        return True
    
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
    session = get_session()
    
    print(f"[INFO] Discovering posts...")
    
    # Try API first
    posts = discover_all_posts_api(session)
    
    # Fallback to HTML scraping if API returned few results
    if len(posts) < 100:
        html_posts = discover_via_archive_html(session)
        existing_slugs = {p["slug"] for p in posts}
        for p in html_posts:
            if p["slug"] not in existing_slugs:
                posts.append(p)
    
    # Last resort: sitemap
    if len(posts) < 100:
        sitemap_posts = discover_via_sitemap(session)
        existing_slugs = {p["slug"] for p in posts}
        for p in sitemap_posts:
            if p["slug"] not in existing_slugs:
                posts.append(p)
    
    print(f"[INFO] Total unique newsletters discovered: {len(posts)}")
    
    if len(posts) < 100:
        print("[WARN] Expected 900+ articles but only found {len(posts)}")
        print("      The site may require browser automation (selenium/playwright)")
    
    if not posts:
        sys.exit(1)
    
    success = 0
    fail = 0
    
    with tqdm(total=len(posts), desc="Downloading", unit="post") as pbar:
        for meta in posts:
            if save_newsletter(meta, session):
                success += 1
            else:
                fail += 1
            pbar.update(1)
            time.sleep(REQUEST_DELAY + random.uniform(0.3, 0.8))
    
    print(f"[INFO] Done. Success: {success}, Failed: {fail}")
    print(f"[INFO] Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
