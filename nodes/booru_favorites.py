"""JSON-backed saved prompts for the Booru tag prompter."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "favorites"
CONFIG_PATH = Path(__file__).resolve().parents[1] / ".favorite_settings.json"
FAVORITES_PATH = DEFAULT_ROOT / "prompts.json"
MAX_PROMPT_LENGTH = 100_000
_LOCK = threading.Lock()


class FavoriteConflictError(ValueError):
    """The selected favorite no longer matches the current file contents."""


def _validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a prompt before saving it as a favorite.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("The prompt is too long to save as a favorite.")


def _favorite_from_line(line: str) -> str | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) and value.strip() else None


def _empty_store() -> dict:
    return {"version": 1, "favorites": [], "unparsed_legacy_lines": []}


def _write_store_unlocked(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=".prompts-", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(store, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_store_unlocked(path: Path) -> dict:
    if path.exists():
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise OSError("The favorites JSON file could not be read. It was not changed.") from exc
        if (not isinstance(store, dict) or store.get("version") != 1
                or not isinstance(store.get("favorites"), list)
                or not isinstance(store.get("unparsed_legacy_lines", []), list)
                or any(not isinstance(value, str) or not value.strip()
                       for value in store["favorites"])
                or any(not isinstance(value, str)
                       for value in store.get("unparsed_legacy_lines", []))):
            raise OSError("The favorites JSON file has an invalid structure. It was not changed.")
        return store

    legacy = path.with_suffix(".txt")
    if not legacy.exists():
        return _empty_store()
    try:
        lines = legacy.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OSError("The legacy favorites file could not be read. It was not changed.") from exc
    store = _empty_store()
    for line in lines:
        if not line.strip():
            continue
        value = _favorite_from_line(line)
        if value is None:
            store["unparsed_legacy_lines"].append(line)
        else:
            store["favorites"].append(value)
    _write_store_unlocked(path, store)
    return store


def active_favorite_root() -> Path:
    try:
        settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DEFAULT_ROOT.resolve()
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise OSError("Favorite folder settings could not be read.") from exc
    if not isinstance(settings, dict):
        raise OSError("Favorite folder settings are invalid.")
    configured = settings.get("favorite_path", "")
    if configured:
        if not isinstance(configured, str):
            raise OSError("Favorite folder settings are invalid.")
        path = Path(configured)
        if not path.is_absolute() or not path.is_dir():
            raise OSError("Configured favorite folder is unavailable.")
        return path.resolve()
    return DEFAULT_ROOT.resolve()


def set_favorite_path(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Favorite path must be text.")
    value = value.strip()
    if value:
        path = Path(value)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("Enter an existing absolute folder path, or leave it blank for the built-in folder.")
        value = str(path.resolve())
    temporary = CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps({"favorite_path": value}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(CONFIG_PATH)
    return value


def browse_favorite_folder() -> str:
    from .booru_wildcards import browse_wildcard_folder
    return browse_wildcard_folder("Select favorite folder")


def list_favorites(path: Path | None = None) -> list[str]:
    """Return saved prompts in file order, migrating a legacy text file once."""
    path = path if path is not None else active_favorite_root() / "prompts.json"
    with _LOCK:
        return list(_read_store_unlocked(path)["favorites"])


def save_favorite(prompt: str, path: Path | None = None) -> list[str]:
    _validate_prompt(prompt)
    path = path if path is not None else active_favorite_root() / "prompts.json"
    with _LOCK:
        store = _read_store_unlocked(path)
        store["favorites"].append(prompt)
        _write_store_unlocked(path, store)
        return list(store["favorites"])


def change_favorite(index: int, original: str, *, replacement: str | None = None,
                    delete: bool = False, path: Path | None = None) -> list[str]:
    """Replace or remove one verified entry, writing the JSON file atomically."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or not isinstance(original, str):
        raise ValueError("Invalid favorite selection.")
    if not delete:
        _validate_prompt(replacement)
    path = path if path is not None else active_favorite_root() / "prompts.json"
    with _LOCK:
        store = _read_store_unlocked(path)
        favorites = store["favorites"]
        if index >= len(favorites) or favorites[index] != original:
            raise FavoriteConflictError("This favorite changed. Refresh the list and try again.")
        if delete:
            del favorites[index]
        else:
            favorites[index] = replacement
        _write_store_unlocked(path, store)
        return list(favorites)
