"""Read-only access to a local Danbooru wiki SQLite snapshot."""

from __future__ import annotations

import sqlite3
import json
import re
from contextlib import closing
from pathlib import Path
from urllib.parse import quote


DEFAULT_DB = Path(__file__).resolve().parent / "data" / "danbooru_wiki.sqlite3"
PAGE_SIZE = 30
WIKI_LINK = re.compile(r"\[\[([^\]]+)\]\]")


def active_wiki_db() -> Path:
    if not DEFAULT_DB.is_file():
        raise OSError("Bundled wiki database is missing from nodes/data.")
    return DEFAULT_DB


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or active_wiki_db()
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    con = None
    try:
        con = sqlite3.connect(uri, uri=True, timeout=2)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        return con
    except sqlite3.Error as exc:
        if con is not None:
            con.close()
        raise OSError("Wiki database could not be opened.") from exc


def _escaped_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _other_names(value: str | None) -> list[str]:
    try:
        names = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [name for name in names if isinstance(name, str)] if isinstance(names, list) else []


def _snippet(body: str | None, query: str) -> str:
    plain = WIKI_LINK.sub(lambda match: match.group(1).split("|", 1)[-1], body or "")
    plain = re.sub(r"\[[^\]]+\]|\bh[4-6]\.\s*", " ", plain)
    plain = " ".join(plain.split())
    position = plain.casefold().find(query.casefold()) if query else -1
    start = max(0, position - 45) if position >= 0 else 0
    return ("…" if start else "") + plain[start:start + 180] + (
        "…" if start + 180 < len(plain) else "")


def wiki_categories(*, path: Path | None = None) -> dict:
    try:
        with closing(_connect(path)) as con:
            rows = con.execute(
                "SELECT category_name, COUNT(*) AS count FROM wiki_pages "
                "WHERE is_deleted = 0 GROUP BY category_name ORDER BY count DESC"
            ).fetchall()
    except sqlite3.Error as exc:
        raise OSError("Wiki categories could not be read.") from exc
    return {"total": sum(row["count"] for row in rows),
            "categories": [{"name": row["category_name"], "count": row["count"]}
                           for row in rows]}


def search_wiki(query: str, category: str = "", page: int = 1,
                *, path: Path | None = None) -> dict:
    if not isinstance(query, str):
        raise ValueError("Search text must be a string.")
    query = query.strip()
    if len(query) > 150:
        raise ValueError("Search text is too long.")
    if not isinstance(category, str) or len(category) > 50:
        raise ValueError("Invalid wiki category.")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1 or page > 10000:
        raise ValueError("Invalid wiki page number.")
    normalized = query.replace(" ", "_")
    title_pattern = f"%{_escaped_like(normalized)}%"
    alias_pattern = f"%{_escaped_like(query)}%"
    clauses = ["w.is_deleted = 0"]
    args: list[object] = []
    if category:
        clauses.append("w.category_name = ?")
        args.append(category)
    if query:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if tokens:
            expression = " AND ".join('"' + token + '"' for token in tokens)
            clauses.append("(w.title LIKE ? ESCAPE '\\' OR w.other_names_json LIKE ? ESCAPE '\\' "
                           "OR w.id IN (SELECT rowid FROM wiki_fts WHERE wiki_fts MATCH ?))")
            args.extend((title_pattern, alias_pattern, expression))
        else:
            clauses.append("(w.title LIKE ? ESCAPE '\\' OR w.other_names_json LIKE ? ESCAPE '\\')")
            args.extend((title_pattern, alias_pattern))
        order = ("CASE WHEN lower(w.title) = lower(?) THEN 0 "
                 "WHEN lower(w.title) LIKE lower(?) ESCAPE '\\' THEN 1 "
                 "WHEN w.other_names_json LIKE ? ESCAPE '\\' THEN 2 ELSE 3 END, "
                 "COALESCE(w.post_count, 0) DESC, w.title")
        args.extend((normalized, _escaped_like(normalized) + "%", alias_pattern))
    else:
        order = "COALESCE(w.post_count, 0) DESC, w.title"
    sql = ("SELECT w.id, w.title, w.body, w.category_name, w.post_count, "
           "w.other_names_json FROM wiki_pages w WHERE " + " AND ".join(clauses) +
           " ORDER BY " + order + " LIMIT ? OFFSET ?")
    args.extend((PAGE_SIZE + 1, (page - 1) * PAGE_SIZE))
    try:
        with closing(_connect(path)) as con:
            rows = con.execute(sql, args).fetchall()
    except sqlite3.Error as exc:
        raise OSError("Wiki search failed. Check the database format.") from exc
    return {"items": [{"id": row["id"], "title": row["title"],
                       "category": row["category_name"], "post_count": row["post_count"],
                       "other_names": _other_names(row["other_names_json"])[:4],
                       "snippet": _snippet(row["body"], query)} for row in rows[:PAGE_SIZE]],
            "page": page, "has_more": len(rows) > PAGE_SIZE}


def wiki_page(page_id: int | None = None, title: str = "", *, path: Path | None = None) -> dict | None:
    if page_id is not None and (not isinstance(page_id, int) or isinstance(page_id, bool) or page_id < 0):
        raise ValueError("Invalid wiki page ID.")
    if page_id is None and (not isinstance(title, str) or not title or len(title) > 200):
        raise ValueError("Invalid wiki title.")
    try:
        with closing(_connect(path)) as con:
            where = "id = ?" if page_id is not None else "title = ? COLLATE NOCASE"
            value = page_id if page_id is not None else re.sub(r"\s+", "_", title.strip())
            row = con.execute("SELECT id, title, body, category_name, post_count, source_url, "
                              "other_names_json, updated_at FROM wiki_pages WHERE " + where +
                              " AND is_deleted = 0", (value,)).fetchone()
    except sqlite3.Error as exc:
        raise OSError("Wiki page could not be read.") from exc
    if row is None:
        return None
    result = dict(row)
    result["category"] = result.pop("category_name")
    result["other_names"] = _other_names(result.pop("other_names_json"))
    return result
