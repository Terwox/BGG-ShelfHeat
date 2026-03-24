"""
Stage 5: BGG collection loading + embedding-based name matching.

Supports two import paths:
  - BGG XML API v2 (fetches live from boardgamegeek.com)
  - CSV export (offline, user-provided)

Matching uses sentence-transformers cosine similarity, NOT fuzzy string matching.
"""

import csv
import io
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

BGG_API_BASE = "https://boardgamegeek.com/xmlapi2"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"

# Thresholds from architecture spec
AUTO_MATCH_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.70


class BGGCollection:
    """A user's BGG collection with embedding-based name matching."""

    def __init__(self, games: list[dict]):
        self.games = games  # [{name, bgg_id, play_count, last_played, user_rating, ...}]
        self._embed_model = None
        self._name_embeddings = None

    @classmethod
    def from_api(cls, username: str, include_plays: bool = True) -> "BGGCollection":
        """Fetch collection from BGG XML API v2."""
        print(f"[match] Fetching BGG collection for '{username}'...")
        games = _fetch_collection(username)
        print(f"[match] {len(games)} games in collection")

        if include_plays and games:
            print("[match] Fetching play history (for last-played dates)...")
            plays = _fetch_plays(username)
            _merge_play_dates(games, plays)
            played = sum(1 for g in games if g.get("last_played"))
            print(f"[match] Play dates found for {played}/{len(games)} games")

        return cls(games)

    @classmethod
    def from_csv(cls, csv_path: str) -> "BGGCollection":
        """Load from a BGG collection CSV export."""
        print(f"[match] Loading collection from {csv_path}...")
        path = Path(csv_path)
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            games = []
            for row in reader:
                game = _parse_csv_row(row)
                if game:
                    games.append(game)

        print(f"[match] {len(games)} games loaded from CSV")
        return cls(games)

    def game_names(self) -> list[str]:
        return [g["name"] for g in self.games]

    def image_urls(self) -> dict[str, str]:
        """Return {game_name: thumbnail_url} for games with image URLs."""
        return {
            g["name"]: g["image"]
            for g in self.games
            if g.get("image")
        }

    def match(self, query: str, inherit_plays: bool = True) -> dict | None:
        """
        Match a query string against collection game names.

        Returns the best match dict with added "match_score" field,
        or None if below REVIEW_THRESHOLD.

        If inherit_plays is True and the matched game has no play data,
        checks BGG family/version links to find play data from related
        games (e.g., base game plays applied to a Saga Collection edition).
        """
        if not self.games:
            return None

        self._ensure_embeddings()
        query_emb = self._embed_model.encode([query], normalize_embeddings=True)
        scores = (query_emb @ self._name_embeddings.T)[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < REVIEW_THRESHOLD:
            return None

        result = {**self.games[best_idx]}
        result["match_score"] = round(best_score, 4)
        result["match_quality"] = "auto" if best_score >= AUTO_MATCH_THRESHOLD else "review"

        # Inherit play data from related games if this one has none
        if inherit_plays and not result.get("last_played") and result.get("bgg_id"):
            inherited = self._inherit_plays_from_family(result["bgg_id"])
            if inherited:
                result["last_played"] = inherited["last_played"]
                result["play_count"] = inherited["play_count"]
                result["play_inherited_from"] = inherited["from_name"]
                result["play_inherited_from_id"] = inherited["from_id"]

        return result

    def match_top_k(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k matches with scores."""
        if not self.games:
            return []

        self._ensure_embeddings()
        query_emb = self._embed_model.encode([query], normalize_embeddings=True)
        scores = (query_emb @ self._name_embeddings.T)[0]
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0.3:  # don't return garbage
                break
            result = {**self.games[idx]}
            result["match_score"] = round(score, 4)
            results.append(result)
        return results

    def _inherit_plays_from_family(self, bgg_id: int) -> dict | None:
        """
        Check BGG implementation/compilation links for related games with play data.

        NOTE: BGG API now requires app registration (XMLcalypse). This method
        will silently return None until auth is configured. Left as a hook for
        when BGG app IDs become available.

        Only checks boardgameimplementation and boardgamecompilation links —
        NOT boardgamefamily (too broad, e.g. Agricola ≠ Agricola: ACBS).
        """
        if not hasattr(self, "_family_cache"):
            self._family_cache = {}

        if bgg_id in self._family_cache:
            return self._family_cache[bgg_id]

        related_ids = _fetch_related_game_ids(bgg_id)
        if not related_ids:
            self._family_cache[bgg_id] = None
            return None

        # Build a lookup of our collection by bgg_id
        if not hasattr(self, "_id_lookup"):
            self._id_lookup = {g["bgg_id"]: g for g in self.games if g.get("bgg_id")}

        # Check if any related game is in our collection WITH play data
        for rid in related_ids:
            if rid in self._id_lookup:
                related_game = self._id_lookup[rid]
                if related_game.get("last_played") or related_game.get("play_count", 0) > 0:
                    result = {
                        "last_played": related_game.get("last_played"),
                        "play_count": related_game.get("play_count", 0),
                        "from_name": related_game["name"],
                        "from_id": rid,
                    }
                    self._family_cache[bgg_id] = result
                    print(f"    ↳ Inherited plays from: {related_game['name']} (id={rid})")
                    return result

        self._family_cache[bgg_id] = None
        return None

    def _ensure_embeddings(self):
        if self._name_embeddings is not None:
            return

        from sentence_transformers import SentenceTransformer

        print(f"[match] Loading embedding model ({EMBED_MODEL_ID})...")
        self._embed_model = SentenceTransformer(EMBED_MODEL_ID)
        names = self.game_names()
        print(f"[match] Encoding {len(names)} game names...")
        self._name_embeddings = self._embed_model.encode(names, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# BGG XML API helpers
# ---------------------------------------------------------------------------

def _fetch_collection(username: str) -> list[dict]:
    """Fetch collection via XML API with retry on 202 (queued)."""
    url = f"{BGG_API_BASE}/collection"
    params = {"username": username, "stats": 1, "own": 1}

    for attempt in range(6):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            break
        if resp.status_code == 202:
            wait = 3 * (attempt + 1)
            print(f"  BGG is generating collection... retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    else:
        raise RuntimeError("BGG API did not return collection after 6 attempts")

    return _parse_collection_xml(resp.text)


def _parse_collection_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    games = []
    for item in root.findall("item"):
        name_el = item.find("name")
        numplays_el = item.find("numplays")
        stats_el = item.find("stats")

        user_rating = None
        if stats_el is not None:
            rating_el = stats_el.find("rating")
            if rating_el is not None:
                val = rating_el.get("value", "N/A")
                if val not in ("N/A", ""):
                    user_rating = float(val)

        games.append(
            {
                "name": name_el.text if name_el is not None else "Unknown",
                "bgg_id": int(item.get("objectid", 0)),
                "play_count": int(numplays_el.text) if numplays_el is not None else 0,
                "last_played": None,  # filled by _merge_play_dates
                "user_rating": user_rating,
                "year_published": _text_or_none(item, "yearpublished"),
                "image": _text_or_none(item, "thumbnail"),
            }
        )
    return games


def _text_or_none(parent, tag):
    el = parent.find(tag)
    return el.text if el is not None and el.text else None


def _fetch_plays(username: str, max_pages: int = 10) -> dict[int, str]:
    """Fetch play history. Returns {bgg_id: most_recent_date_str}."""
    latest: dict[int, str] = {}
    url = f"{BGG_API_BASE}/plays"

    for page in range(1, max_pages + 1):
        resp = requests.get(url, params={"username": username, "page": page}, timeout=30)
        if resp.status_code != 200:
            break

        root = ET.fromstring(resp.text)
        plays = root.findall("play")
        if not plays:
            break

        for play in plays:
            date = play.get("date", "")
            item_el = play.find("item")
            if item_el is None:
                continue
            bgg_id = int(item_el.get("objectid", 0))
            if bgg_id and date:
                if bgg_id not in latest or date > latest[bgg_id]:
                    latest[bgg_id] = date

        # BGG returns 100 per page; fewer means last page
        if len(plays) < 100:
            break

        time.sleep(1)  # be nice to BGG

    return latest


def _fetch_related_game_ids(bgg_id: int) -> list[int]:
    """
    Fetch related game IDs from BGG's thing API.

    Looks for: implementations, compilations, reimplementations, and
    games in the same family. Returns a list of related BGG IDs.
    """
    url = f"{BGG_API_BASE}/thing"
    try:
        resp = requests.get(url, params={"id": bgg_id, "type": "boardgame"}, timeout=15)
        if resp.status_code != 200:
            return []
    except Exception:
        return []

    try:
        root = ET.fromstring(resp.text)
    except Exception:
        return []

    related = set()
    item = root.find("item")
    if item is None:
        return []

    # Link types that indicate related games
    link_types = [
        "boardgameimplementation",   # reimplements / is implemented by
        "boardgamecompilation",      # compiles / is compiled in
        "boardgameintegration",      # integrates with
    ]

    for link in item.findall("link"):
        if link.get("type") in link_types:
            try:
                rid = int(link.get("id", 0))
                if rid and rid != bgg_id:
                    related.add(rid)
            except ValueError:
                pass

    return list(related)


def _merge_play_dates(games: list[dict], plays: dict[int, str]):
    for game in games:
        bgg_id = game.get("bgg_id")
        if bgg_id and bgg_id in plays:
            game["last_played"] = plays[bgg_id]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

# Common BGG CSV column name variants
_CSV_NAME_COLS = ("objectname", "name", "game", "title")
_CSV_ID_COLS = ("objectid", "bggid", "gameid", "id")
_CSV_PLAYS_COLS = ("numplays", "plays", "num plays")
_CSV_RATING_COLS = ("rating", "userrating", "your rating")
_CSV_LASTPLAYED_COLS = ("lastplayed", "last_played", "last played", "lastplay")


def _find_col(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_headers = {h.lower().strip(): h for h in headers}
    for c in candidates:
        if c in lower_headers:
            return lower_headers[c]
    return None


def _parse_csv_row(row: dict) -> dict | None:
    headers = list(row.keys())
    name_col = _find_col(headers, _CSV_NAME_COLS)
    if not name_col or not row.get(name_col):
        return None

    id_col = _find_col(headers, _CSV_ID_COLS)
    plays_col = _find_col(headers, _CSV_PLAYS_COLS)
    rating_col = _find_col(headers, _CSV_RATING_COLS)

    bgg_id = 0
    if id_col and row.get(id_col):
        try:
            bgg_id = int(row[id_col])
        except ValueError:
            pass

    play_count = 0
    if plays_col and row.get(plays_col):
        try:
            play_count = int(row[plays_col])
        except ValueError:
            pass

    user_rating = None
    if rating_col and row.get(rating_col):
        val = row[rating_col].strip()
        if val and val not in ("N/A", ""):
            try:
                user_rating = float(val)
            except ValueError:
                pass

    # Last played date
    lp_col = _find_col(headers, _CSV_LASTPLAYED_COLS)
    last_played = None
    if lp_col and row.get(lp_col):
        val = row[lp_col].strip()
        if val:
            # Handle both "2024-01-15" and "2024-01-15 12:30:00" formats
            last_played = val[:10] if len(val) >= 10 else val

    return {
        "name": row[name_col].strip(),
        "bgg_id": bgg_id,
        "play_count": play_count,
        "last_played": last_played,
        "user_rating": user_rating,
    }
