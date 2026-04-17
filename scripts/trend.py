# scripts/trend.py
"""
記事タイトルからトレンドワードを抽出するモジュール
SudachiPy (full辞書) による形態素解析 + N-gram結合ユニット
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

# 結合ユニット関連の設定
PHRASE_MIN_COUNT = 5     # 結合ユニットとして採用する最小出現回数
PHRASE_MAX_N = 3         # 結合の最大長（trigram）


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
    # wが2文字以上連続するものだけ除去（単独Wは保持: WBC/W杯等）
    title = re.sub(r"[wｗ]{2,}", " ", title)
    title = re.sub(r"[！？!?…→←↑↓★☆♪♡◆■□●○▲△▼▽※＊\.\-]", " ", title)
    return title.strip()


# ── ノイズ判定 ────────────────────────────────────────
def _is_noise_token(surface: str) -> bool:
    if len(surface) <= 1:
        return True
    if re.match(r"^[\d０-９]+$", surface):
        return True
    # w/ｗが2文字以上連続する場合のみノイズ（単独WはOK）
    if re.match(r"^[wｗ]{2,}$", surface):
        return True
    return False


# ── 記事ごとの名詞列を抽出 ────────────────────────────
def _extract_noun_sequences(titles: list[str]) -> list[list]:
    """
    各タイトルを (surface, is_stopword) or None の列に変換。
    None は名詞以外で連続が途切れた箇所を示す。
    """
    import sudachipy
    tokenizer = _get_tokenizer()
    mode = sudachipy.SplitMode.C

    all_seqs = []
    for title in titles:
        cleaned = clean_title(title)
        seq = []
        for morpheme in tokenizer.tokenize(cleaned, mode):
            pos = morpheme.part_of_speech()
            surface = morpheme.surface()
            base = morpheme.normalized_form()

            pos_major = (pos[0], pos[1]) if len(pos) >= 2 else (pos[0], "")
            if pos_major not in ALLOW_POS:
                seq.append(None)
                continue
            if _is_noise_token(surface):
                seq.append(None)
                continue

            is_stop = (base in STOP_WORDS) or (surface in STOP_WORDS)
            seq.append((surface, is_stop))
        all_seqs.append(seq)
    return all_seqs


def _split_by_none(seq: list) -> list[list]:
    """Noneで分割して連続名詞のラン（run）リストを返す"""
    runs = []
    current = []
    for item in seq:
        if item is None:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(item)
    if current:
        runs.append(current)
    return runs


def _collect_ngrams(all_seqs: list[list]) -> Counter:
    """
    各タイトルの連続名詞ランから N-gram (2..PHRASE_MAX_N) を収集。
    記事単位でユニークカウント。
    """
    phrase_counter = Counter()
    for seq in all_seqs:
        seen_in_title = set()
        for run in _split_by_none(seq):
            for n in range(2, PHRASE_MAX_N + 1):
                for i in range(len(run) - n + 1):
                    phrase = "".join(t[0] for t in run[i:i + n])
                    if phrase not in seen_in_title:
                        seen_in_title.add(phrase)
                        phrase_counter[phrase] += 1
    return phrase_counter


# ── メインAPI ─────────────────────────────────────────
def extract_trends(
    articles: list[dict],
    top_n: int = 10,
    min_count: int = 3,
    hours_window: int = 24,
) -> list[dict]:
    """
    記事リストからトレンドワードを抽出する。
    連続名詞の頻出パターンを結合ユニットとして1ワード化する。

    Args:
        articles: {"title": str, "published": str, ...} のリスト
        top_n: 返すワード数の上限
        min_count: ランキング掲載の最低出現回数
        hours_window: 直近N時間に絞る（0で全件）

    Returns:
        [{"rank": 1, "word": "安達優季容疑者", "count": 26, "sample_titles": [...]}]
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

    # Step 1-2: 名詞列抽出 + N-gram候補収集
    all_seqs = _extract_noun_sequences(titles)
    phrase_counter = _collect_ngrams(all_seqs)

    # Step 3: 結合ユニット確定
    merge_units = {p for p, c in phrase_counter.items() if c >= PHRASE_MIN_COUNT}

    # Step 4-5: 最長一致で結合してカウント
    word_counter = Counter()
    word_articles: dict[str, list[str]] = {}

    for title, seq in zip(titles, all_seqs):
        seen_in_title: set[str] = set()
        for run in _split_by_none(seq):
            i = 0
            while i < len(run):
                matched_word = None
                match_n = 1

                # PHRASE_MAX_N → ... → 2 の順で最長マッチ探索
                for n in range(min(PHRASE_MAX_N, len(run) - i), 1, -1):
                    candidate = "".join(t[0] for t in run[i:i + n])
                    if candidate in merge_units:
                        # 結合ユニット自体がSTOP_WORDSに明示登録されていれば除外
                        if candidate not in STOP_WORDS:
                            matched_word = candidate
                        else:
                            matched_word = None
                        match_n = n
                        break

                if matched_word is not None:
                    word = matched_word
                elif match_n == 1:
                    # 単独名詞: ストップワード除外
                    surface, is_stop = run[i]
                    if is_stop:
                        i += 1
                        continue
                    word = surface
                else:
                    # 結合マッチだがSTOP_WORDS登録済み → スキップ
                    i += match_n
                    continue

                if word not in seen_in_title:
                    seen_in_title.add(word)
                    word_counter[word] += 1
                    if word not in word_articles:
                        word_articles[word] = []
                    if len(word_articles[word]) < 3:
                        word_articles[word].append(title)

                i += match_n if matched_word is not None else 1

    # Step 6: ランキング出力
    filtered_counts = {w: c for w, c in word_counter.items() if c >= min_count}
    ranking = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        {
            "rank": i,
            "word": word,
            "count": count,
            "sample_titles": word_articles.get(word, []),
        }
        for i, (word, count) in enumerate(ranking, 1)
    ]
