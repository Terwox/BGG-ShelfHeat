"""
Local BGG database for zero-API game matching.

Downloads the bgg-ranking-historicals daily CSV (~25K ranked games)
and builds a local SQLite database for instant name→ID lookups.
No BGG API auth needed. Updated daily by the community.

Source: https://github.com/beefsack/bgg-ranking-historicals
"""

import csv
import io
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Where to cache the database
_DEFAULT_DB_DIR = Path.home() / ".cache" / "shelfheat"
_DB_FILENAME = "bgg_games.db"

# Source: daily CSV snapshots of BGG rankings
_RANKINGS_BASE = "https://raw.githubusercontent.com/beefsack/bgg-ranking-historicals/master"


def get_db_path(db_dir: str | Path | None = None) -> Path:
    """Return the SQLite database path, creating the directory if needed."""
    d = Path(db_dir) if db_dir else _DEFAULT_DB_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DB_FILENAME


def ensure_db(db_dir: str | Path | None = None, max_age_days: int = 7) -> Path:
    """
    Ensure a local BGG database exists and is fresh enough.

    Downloads from bgg-ranking-historicals if missing or stale.
    Returns the path to the SQLite file.
    """
    db_path = get_db_path(db_dir)

    if db_path.exists():
        age = datetime.now().timestamp() - db_path.stat().st_mtime
        if age < max_age_days * 86400:
            count = _count_games(db_path)
            if count > 10000:
                print(f"[bggdb] Using cached database ({count:,} games, {age/86400:.1f}d old)")
                return db_path

    print("[bggdb] Downloading fresh BGG rankings...")
    csv_text = _download_latest_csv()
    if not csv_text:
        if db_path.exists():
            print("[bggdb] Download failed, using stale cache")
            return db_path
        raise RuntimeError("Could not download BGG data and no cache exists")

    _build_db(db_path, csv_text)
    return db_path


def _download_latest_csv() -> str | None:
    """Try downloading today's CSV, then yesterday's, etc. (up to 5 days back)."""
    for days_back in range(5):
        date = datetime.now() - timedelta(days=days_back)
        date_str = date.strftime("%Y-%m-%d")
        url = f"{_RANKINGS_BASE}/{date_str}.csv"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 1000:
                print(f"[bggdb] Downloaded {date_str}.csv ({len(resp.text) // 1024:,} KB)")
                return resp.text
        except Exception as e:
            print(f"[bggdb] Failed to fetch {date_str}: {e}")
    return None


def _build_db(db_path: Path, csv_text: str) -> None:
    """Parse the CSV and create/replace the SQLite database."""
    # Parse CSV
    reader = csv.DictReader(io.StringIO(csv_text))
    games = []
    for row in reader:
        try:
            games.append({
                "bgg_id": int(row["ID"]),
                "name": row["Name"],
                "year": int(row.get("Year", 0) or 0),
                "rank": int(row.get("Rank", 0) or 0),
                "average": float(row.get("Average", 0) or 0),
                "bayes_average": float(row.get("Bayes average", 0) or 0),
                "users_rated": int(row.get("Users rated", 0) or 0),
                "thumbnail": row.get("Thumbnail", ""),
            })
        except (ValueError, KeyError):
            continue

    # Build SQLite
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE games (
            bgg_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            name_lower TEXT NOT NULL,
            year INTEGER,
            rank INTEGER,
            average REAL,
            bayes_average REAL,
            users_rated INTEGER,
            thumbnail TEXT
        )
    """)
    conn.execute("CREATE INDEX idx_name_lower ON games(name_lower)")

    conn.executemany(
        "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?)",
        [(g["bgg_id"], g["name"], g["name"].lower(), g["year"], g["rank"],
          g["average"], g["bayes_average"], g["users_rated"], g["thumbnail"])
         for g in games],
    )

    # Also create an FTS5 virtual table for fuzzy name search
    conn.execute("""
        CREATE VIRTUAL TABLE games_fts USING fts5(
            name, bgg_id UNINDEXED, content=games, content_rowid=bgg_id
        )
    """)
    conn.execute("""
        INSERT INTO games_fts(rowid, name, bgg_id)
        SELECT bgg_id, name, bgg_id FROM games
    """)

    conn.commit()
    conn.close()

    print(f"[bggdb] Built database: {len(games):,} games -> {db_path}")


def _count_games(db_path: Path) -> int:
    """Quick count of games in the database."""
    try:
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

class BGGLocalDB:
    """Query interface for the local BGG database."""

    def __init__(self, db_dir: str | Path | None = None, max_age_days: int = 7):
        self.db_path = ensure_db(db_dir, max_age_days)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def lookup_id(self, bgg_id: int) -> dict | None:
        """Look up a game by BGG ID."""
        row = self.conn.execute(
            "SELECT * FROM games WHERE bgg_id = ?", (bgg_id,)
        ).fetchone()
        return dict(row) if row else None

    def search_name(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search by game name. Returns best matches."""
        # FTS5 search with prefix matching
        fts_query = " ".join(f'"{w}"*' for w in query.split() if w)
        try:
            rows = self.conn.execute(
                "SELECT g.* FROM games_fts f JOIN games g ON f.rowid = g.bgg_id "
                "WHERE games_fts MATCH ? ORDER BY g.users_rated DESC LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass

        # Fallback: LIKE search
        rows = self.conn.execute(
            "SELECT * FROM games WHERE name_lower LIKE ? ORDER BY users_rated DESC LIMIT ?",
            (f"%{query.lower()}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def exact_name(self, name: str) -> dict | None:
        """Exact case-insensitive name lookup."""
        row = self.conn.execute(
            "SELECT * FROM games WHERE name_lower = ?", (name.lower(),)
        ).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict:
        """Database stats."""
        count = self.conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        newest = self.conn.execute(
            "SELECT MAX(year) FROM games WHERE year > 0"
        ).fetchone()[0]
        return {"total_games": count, "newest_year": newest, "db_path": str(self.db_path)}
