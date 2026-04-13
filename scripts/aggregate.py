# scripts/aggregate.py
import feedparser
import json
import os
import re
import requests
import unicodedata
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from feeds import FEEDS

# ── 定数 ──────────────────────────────────────────────
STORE_TTL_DAYS = 7
POPULAR_INTERVAL_SECONDS = 3600  # 1時間

POPULAR_FEEDS = {
    "news":   "https://feeds.mtmx.jp/news/all/popular/feed.xml",
    "entame": "https://feeds.mtmx.jp/entame/all/popular/feed.xml",
    "neta":   "https://feeds.mtmx.jp/neta/all/popular/feed.xml",
    "life":   "https://feeds.mtmx.jp/life/all/popular/feed.xml",
    "sports": "https://feeds.mtmx.jp/sports/feed.xml",
    "anige":  "https://feeds.mtmx.jp/anige/feed.xml",
}

HATENA_API = "https://bookmark.hatenaapis.com/count/entries"
UA = {"User-Agent": "Mozilla/5.0 (compatible; MatomeAggregator/1.0)"}


# ── ユーティリティ ────────────────────────────────────
def parse_date(s: str) -> datetime | None:
    """RFC2822 / ISO8601 対応の日付パーサー（rank_sites.pyから流用）"""
    if not s:
        return None
    try:
        return parsedate_to_datetime(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def sort_key(article: dict) -> datetime:
    dt = parse_date(article.get("published", ""))
    return dt if dt else datetime.min.replace(tzinfo=timezone.utc)


# ── サムネイル抽出 ────────────────────────────────────
def extract_thumbnail(entry: dict) -> str | None:
    media_thumbnail = entry.get("media_thumbnail")
    if media_thumbnail:
        return media_thumbnail[0].get("url")

    media_content = entry.get("media_content")
    if media_content:
        url = media_content[0].get("url", "")
        if url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            return url

    enclosures = entry.get("enclosures", [])
    for enc in enclosures:
        if enc.get("type", "").startswith("image/"):
            return enc.get("href") or enc.get("url")

    for field in ("summary", "content"):
        text = ""
        val = entry.get(field)
        if isinstance(val, list) and val:
            text = val[0].get("value", "")
        elif isinstance(val, str):
            text = val
        match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', text)
        if match:
            url = match.group(1)
            if url.startswith("http") and not re.search(r'[1-9]x[1-9]\.', url):
                return url

    return None


def proxy_thumbnail(url: str | None) -> str | None:
    """画像URLをwsrv.nlプロキシ経由のWebP縮小URLに変換"""
    if not url:
        return None
    return f"https://wsrv.nl/?url={quote(url, safe='')}&w=400&h=400&output=webp&q=80"


# ── はてなブックマーク ────────────────────────────────
def fetch_hatena_batch(urls: list[str]) -> dict[str, int]:
    try:
        resp = requests.get(HATENA_API, params=[("url", u) for u in urls], timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[WARN] Hatena API error: {e}")
        return {}


def fetch_hatena_counts(urls: list[str]) -> dict[str, int]:
    chunks = [urls[i:i + 50] for i in range(0, len(urls), 50)]
    counts = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_hatena_batch, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            counts.update(future.result())
    return counts


# ── Store（7日間記事蓄積）────────────────────────────
def _empty_store() -> dict:
    return {"schema_version": 1, "articles": {}, "popular_updated_at": None}


def load_from_pages(filename: str) -> dict | None:
    """gh-pagesからJSONを取得。失敗時はNone"""
    base = os.environ.get("PAGES_BASE_URL")
    if not base:
        print(f"[WARN] PAGES_BASE_URL not set, cannot load {filename}")
        return None
    url = f"{base.rstrip('/')}/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            print(f"[INFO] {filename} not found on gh-pages (first run?)")
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[WARN] Failed to load {filename}: {e}")
        return None


def load_store() -> dict:
    data = load_from_pages("store.json")
    if data and data.get("schema_version") == 1:
        return data
    return _empty_store()


def merge_into_store(store: dict, new_articles: list[dict]) -> dict:
    articles = store.get("articles", {})
    cutoff = datetime.now(timezone.utc) - timedelta(days=STORE_TTL_DAYS)
    valid_site_ids = {f["id"] for f in FEEDS}

    for a in new_articles:
        key = a.get("url")
        if key:
            articles[key] = a

    pruned = {}
    for url, a in articles.items():
        dt = parse_date(a.get("published", ""))
        if dt and dt < cutoff:
            continue
        if a.get("site_id") not in valid_site_ids:
            continue
        pruned[url] = a

    store["articles"] = pruned
    return store


# ── RSSフィード取得 ───────────────────────────────────
def fetch_feed(feed_info: dict) -> dict:
    try:
        resp = requests.get(feed_info["url"], headers=UA, timeout=10)
        resp.raise_for_status()
        d = feedparser.parse(resp.content)

        if d.bozo and not d.entries:
            raise ValueError(f"Feed parse error: {d.bozo_exception}")

        articles = []
        for entry in d.entries[:30]:
            published = entry.get("published", "") or entry.get("updated", "")
            articles.append({
                "id":        entry.get("id") or entry.get("link", ""),
                "title":     entry.get("title", "").strip(),
                "url":       entry.get("link", ""),
                "published": published,
                "thumbnail": proxy_thumbnail(extract_thumbnail(entry)),
                "site_id":   feed_info["id"],
                "site_name": feed_info["name"],
            })

        return {"site_id": feed_info["id"], "articles": articles, "ok": True}

    except Exception as e:
        print(f"[ERROR] {feed_info['id']}: {e}")
        return {"site_id": feed_info["id"], "articles": [], "ok": False, "error": str(e)}


# ── 人気記事 ──────────────────────────────────────────
def should_update_popular(store: dict) -> bool:
    if os.environ.get("FORCE_POPULAR"):
        return True
    ts = store.get("popular_updated_at")
    if not ts:
        return True
    try:
        last = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - last).total_seconds() >= POPULAR_INTERVAL_SECONDS
    except Exception:
        return True


def _fetch_one_popular(category: str, url: str) -> list[dict]:
    """1つの人気フィードを取得し、カテゴリ付きエントリを返す"""
    try:
        resp = requests.get(url, headers=UA, timeout=10)
        resp.raise_for_status()
        d = feedparser.parse(resp.content)
        entries = []
        for rank, entry in enumerate(d.entries, start=1):
            entries.append({
                "title": entry.get("title", "").strip(),
                "category": category,
                "rank": rank,
            })
        return entries
    except Exception as e:
        print(f"[WARN] Popular feed ({category}): {e}")
        return []


def fetch_popular_feeds() -> list[dict]:
    """6カテゴリの人気フィードを並列取得"""
    all_entries = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_fetch_one_popular, cat, url): cat
            for cat, url in POPULAR_FEEDS.items()
        }
        for future in as_completed(futures):
            all_entries.extend(future.result())
    return all_entries


def _normalize_title(title: str) -> str:
    """タイトル比較用に正規化（空白・全角半角の差異を吸収）"""
    t = unicodedata.normalize("NFKC", title)
    # 全種類の空白を除去
    t = re.sub(r'\s+', '', t)
    return t.lower()


def build_popular(store: dict, popular_entries: list[dict]) -> list[dict]:
    """人気エントリのタイトルでstoreの記事とマッチング（正規化済み）"""
    store_articles = store.get("articles", {})

    # 正規化タイトル → 記事の逆引きインデックス
    title_index: dict[str, dict] = {}
    for article in store_articles.values():
        title = article.get("title", "").strip()
        if title:
            title_index[_normalize_title(title)] = article

    matched = []
    seen_urls: set[str] = set()
    for entry in popular_entries:
        key = _normalize_title(entry["title"])
        category = entry["category"]
        article = title_index.get(key)
        if article and article["url"] not in seen_urls:
            item = dict(article)
            item["category"] = category
            item["rank"] = entry.get("rank", 999)
            matched.append(item)
            seen_urls.add(article["url"])

    return matched


# ── メイン ────────────────────────────────────────────
def main():
    os.makedirs("output", exist_ok=True)
    now = datetime.now(timezone.utc)

    # 1. Store読み込み
    store = load_store()
    print(f"[STORE] Store loaded: {len(store.get('articles', {}))} articles")

    # 2. RSSフィード取得
    all_articles = []
    site_status = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_feed, f): f for f in FEEDS}
        for future in as_completed(futures):
            result = future.result()
            site_status[result["site_id"]] = result["ok"]
            all_articles.extend(result["articles"])

    print(f"[RSS] Fetched {len(all_articles)} articles from {len(FEEDS)} feeds")

    # 3. Storeにマージ（7日超の記事を削除）
    store = merge_into_store(store, all_articles)
    store_articles = store.get("articles", {})
    print(f"[STORE] Store after merge: {len(store_articles)} articles")

    # 4. feed.json: storeから新しい順に3000件（未来の記事を除外）
    past_articles = [a for a in store_articles.values() if sort_key(a) <= now]
    past_articles.sort(key=sort_key, reverse=True)
    feed_articles = past_articles[:3000]

    article_urls = [a["url"] for a in feed_articles if a["url"]]
    hatena_counts = fetch_hatena_counts(article_urls)
    for article in feed_articles:
        article["hatena_bookmarks"] = hatena_counts.get(article["url"], 0)

    print(f"[HATENA] Hatena bookmarks fetched for {len(article_urls)} articles")

    feed_output = {
        "schema_version": 1,
        "updated_at":     now.isoformat(),
        "total":          len(feed_articles),
        "articles":       feed_articles,
        "site_status":    site_status,
    }
    with open("output/feed.json", "w", encoding="utf-8") as f:
        json.dump(feed_output, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[OK] {len(feed_articles)} articles → output/feed.json")

    # 5. popular.json: 1時間に1回更新
    if should_update_popular(store):
        print("[POP] Updating popular articles...")
        popular_entries = fetch_popular_feeds()
        print(f"[POP] Fetched {len(popular_entries)} popular entries from mtmx.jp")

        popular_articles = build_popular(store, popular_entries)

        # 人気記事のはてブ数取得（feed.jsonで取得済みの分は再利用）
        popular_urls = [a["url"] for a in popular_articles if a["url"] and a["url"] not in hatena_counts]
        if popular_urls:
            extra_counts = fetch_hatena_counts(popular_urls)
            hatena_counts.update(extra_counts)
        for article in popular_articles:
            article["hatena_bookmarks"] = hatena_counts.get(article["url"], 0)

        popular_output = {
            "schema_version": 1,
            "updated_at":     now.isoformat(),
            "total":          len(popular_articles),
            "articles":       popular_articles,
        }
        with open("output/popular.json", "w", encoding="utf-8") as f:
            json.dump(popular_output, f, ensure_ascii=False, separators=(",", ":"))

        store["popular_updated_at"] = now.isoformat()
        print(f"[POP] {len(popular_articles)} popular articles → output/popular.json")
    else:
        print("[SKIP] Popular update skipped (less than 1 hour since last update)")

    # 6. store.json保存
    store["updated_at"] = now.isoformat()
    store["total"] = len(store_articles)
    with open("output/store.json", "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[SAVE] Store saved: {len(store_articles)} articles → output/store.json")


if __name__ == "__main__":
    main()
