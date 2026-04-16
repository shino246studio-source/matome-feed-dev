# scripts/trend.py
"""
記事タイトルからトレンドワードを抽出するモジュール
SudachiPy (full辞書) による形態素解析
"""

import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone


# ── ストップワード読み込み ──────────────────────────────
def _load_stopwords() -> set[str]:
    path = os.path.join(os.path.dirname(__file__), "stopwords.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip() and not line.startswith("#")}
    return set()


STOP_WORDS = _load_stopwords()

# 品詞フィルタ
ALLOW_POS = {
    ("名詞", "固有名詞"),
    ("名詞", "普通名詞"),
}


# ── トークナイザー（シングルトン）──────────────────────
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from sudachipy import Dictionary
    except ImportError:
        print("[ERROR] sudachipy is required: pip install sudachipy sudachidict_full")
        sys.exit(1)
    for dict_type in ("full", "core", None):
        try:
            _tokenizer = Dictionary(dict=dict_type).create() if dict_type else Dictionary().create()
            return _tokenizer
        except Exception:
            continue
    print("[ERROR] Failed to create Sudachi tokenizer")
    sys.exit(1)


# ── タイトル前処理 ────────────────────────────────────
def clean_title(title: str) -> str:
    title = re.sub(r"【[^】]*】", " ", title)
    title = re.sub(r"https?://\S+", " ", title)
    title = re.sub(r"[wWｗＷ]+", " ", title)
    title = re.sub(r"[！？!?…→←↑↓★☆♪♡◆■□●○▲△▼▽※＊\.\-]", " ", title)
    return title.strip()


# ── ワード抽出 ────────────────────────────────────────
def extract_words(titles: list[str]) -> tuple[Counter, dict[str, list[str]]]:
    import sudachipy

    tokenizer = _get_tokenizer()
    mode = sudachipy.SplitMode.C

    word_counter = Counter()
    word_articles: dict[str, list[str]] = {}

    for title in titles:
        cleaned = clean_title(title)
        seen_in_title: set[str] = set()

        for morpheme in tokenizer.tokenize(cleaned, mode):
            pos = morpheme.part_of_speech()
            surface = morpheme.surface()
            base = morpheme.normalized_form()

            pos_major = (pos[0], pos[1]) if len(pos) >= 2 else (pos[0], "")
            if pos_major not in ALLOW_POS:
                continue
            if len(surface) <= 1:
                continue
            if re.match(r"^[\d０-９]+$", surface):
                continue
            if re.match(r"^[wWｗＷ]+$", surface):
                continue
            if base in STOP_WORDS or surface in STOP_WORDS:
                continue

            word = surface
            if word not in seen_in_title:
                seen_in_title.add(word)
                word_counter[word] += 1
                if word not in word_articles:
                    word_articles[word] = []
                if len(word_articles[word]) < 3:
                    word_articles[word].append(title)

    return word_counter, word_articles


# ── メインAPI ─────────────────────────────────────────
def extract_trends(
    articles: list[dict],
    top_n: int = 10,
    min_count: int = 3,
    hours_window: int = 24,
) -> list[dict]:
    """
    記事リストからトレンドワードを抽出する。

    Args:
        articles: {"title": str, "published": str, ...} のリスト
        top_n: 返すワード数の上限
        min_count: 最低出現回数
        hours_window: 直近N時間に絞る（0で全件）

    Returns:
        [{"rank": 1, "word": "京都", "count": 85, "sample_titles": [...]}]
    """
    # 時間フィルタ
    if hours_window > 0:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours_window)
        filtered = []
        for a in articles:
            try:
                pub = datetime.fromisoformat(a.get("published", ""))
                if pub >= cutoff:
                    filtered.append(a)
            except (ValueError, KeyError, TypeError):
                pass
        articles = filtered

    titles = [a["title"] for a in articles if a.get("title")]
    if not titles:
        return []

    word_counter, word_articles = extract_words(titles)

    # min_count でフィルタ、降順でtop_n件
    filtered = {w: c for w, c in word_counter.items() if c >= min_count}
    ranking = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        {
            "rank": i,
            "word": word,
            "count": count,
            "sample_titles": word_articles.get(word, []),
        }
        for i, (word, count) in enumerate(ranking, 1)
    ]
