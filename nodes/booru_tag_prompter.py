"""Danbooru tag entry with source-preserving text output and local suggestions."""

from __future__ import annotations

import logging
from functools import lru_cache

from .anima_tags import get_db, normalize
from .booru_tag_presets import append_preset_tags
from .booru_wildcards import TOKEN, expand_wildcards, wildcard_files, wildcard_preview, open_wildcard_folder, browse_wildcard_folder, set_wildcard_path, active_wildcard_root
from .booru_favorites import (FavoriteConflictError, active_favorite_root,
                              browse_favorite_folder, change_favorite,
                              list_favorites, save_favorite, set_favorite_path)
from .booru_wiki import download_wiki_db, search_wiki, wiki_categories, wiki_download_status, wiki_page


LOG = logging.getLogger(__name__)
CATEGORIES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
PAGE_SIZE = 50


class BooruTagPrompter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tags": ("STRING", {
                "default": "", "multiline": True, "dynamicPrompts": False,
                "pysssss.autocomplete": False,
                "tooltip": "Type Danbooru tags or __wildcard__ calls from this node's wildcards folder. Each call samples one non-empty text line on every run. Choose tag suggestions with click, Tab, or Enter.",
            }),
        }, "optional": {
            "camera": ("TOYXYZ_BOORU_PRESET", {
                "tooltip": "Connect booru tag camera. Camera tags and pose-neutral view prose are appended without reordering your text. Framing uses tags only. Pose, subject count, clothing and background remain user-controlled.",
            }),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "emit"
    CATEGORY = "ToyxyzTestNodes/Prompt"
    DESCRIPTION = "Compose Danbooru tags with local suggestions while preserving authored text."

    @classmethod
    def IS_CHANGED(cls, tags="", **kwargs):
        return float("nan") if TOKEN.search(tags) else False

    def emit(self, tags, camera=None, preset=None, **_legacy_inputs):
        # Accept stale saved API inputs without exposing or applying model settings.
        tags = append_preset_tags(expand_wildcards(tags), camera if camera is not None else preset)
        return (tags,)


def _match_rank(name: str, query: str, alias: bool) -> int | None:
    position = name.find(query)
    if position < 0:
        return None
    if name == query:
        return 0 if not alias else 1
    if position == 0:
        return 2 if not alias else 3
    if name[position - 1] == "_":
        return 4 if not alias else 5
    return 6 if not alias else 7


def search_tags(query: str, limit: int = PAGE_SIZE, db=None, offset: int = 0) -> list[dict]:
    """Rank exact, prefix, word-boundary, and substring hits by popularity."""
    if not isinstance(query, str):
        return []
    needle = normalize(query)
    if not needle or limit <= 0 or offset < 0:
        return []
    db = db or get_db()
    candidates: dict[str, tuple[int, str | None]] = {}

    for name in db.canonical:
        rank = _match_rank(name, needle, False)
        if rank is not None:
            candidates[name] = rank, None
    for alias, canonical in db.aliases.items():
        rank = _match_rank(alias, needle, True)
        if rank is not None and (canonical not in candidates or rank < candidates[canonical][0]):
            candidates[canonical] = rank, alias
    for alias, canonical in db.punctuation_aliases.items():
        rank = _match_rank(alias, needle, True)
        if rank is not None and (canonical not in candidates or rank < candidates[canonical][0]):
            candidates[canonical] = rank, alias

    best = sorted(candidates.items(),
                  key=lambda item: (item[1][0], -db.counts[item[0]], len(item[0]), item[0]))[
                      offset:offset + limit]
    return [{"tag": tag, "count": db.counts[tag],
             "category": CATEGORIES.get(db.canonical[tag], "other"),
             **({"alias": match} if match else {})}
            for tag, (_, match) in best]


@lru_cache(maxsize=128)
def cached_suggestions(query: str, offset: int = 0, limit: int = PAGE_SIZE) -> list[dict]:
    return search_tags(query, limit, offset=offset)


def register_routes():
    try:
        import asyncio
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    if not getattr(PromptServer, "instance", None):
        return

    def local_request(request):
        from urllib.parse import urlsplit
        origin = request.headers.get("Origin")
        return request.remote in ("127.0.0.1", "::1") and (not origin or urlsplit(origin).netloc == request.host)

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/wiki/status")
    async def wiki_status_route(request):
        if not local_request(request):
            return web.json_response({"error": "Wiki status requires local access."}, status=403)
        return web.json_response(await asyncio.to_thread(wiki_download_status))

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/wiki/download")
    async def wiki_download_route(request):
        if not local_request(request):
            return web.json_response({"error": "Wiki download requires local access."}, status=403)
        try:
            await asyncio.to_thread(download_wiki_db)
        except OSError as exc:
            LOG.warning("Wiki download failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=503)
        return web.json_response({"ready": True})

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/wiki/categories")
    async def wiki_categories_route(request):
        if not local_request(request):
            return web.json_response({"error": "Wiki categories require local access."}, status=403)
        try:
            data = await asyncio.to_thread(wiki_categories)
        except OSError as exc:
            return web.json_response({"error": str(exc)}, status=503)
        return web.json_response(data)

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/wiki/search")
    async def search_wiki_route(request):
        if not local_request(request):
            return web.json_response({"error": "Wiki search requires local access."}, status=403)
        try:
            results = await asyncio.to_thread(search_wiki, request.query.get("q", ""),
                                              request.query.get("category", ""),
                                              int(request.query.get("page", "1")))
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError as exc:
            return web.json_response({"error": str(exc)}, status=503)
        return web.json_response(results)

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/wiki/page")
    async def wiki_page_route(request):
        if not local_request(request):
            return web.json_response({"error": "Wiki pages require local access."}, status=403)
        try:
            page_id = request.query.get("id")
            page = await asyncio.to_thread(wiki_page, int(page_id) if page_id else None,
                                           request.query.get("title", ""))
        except ValueError:
            return web.json_response({"error": "Invalid wiki page ID."}, status=400)
        except OSError as exc:
            return web.json_response({"error": str(exc)}, status=503)
        return web.json_response(page if page is not None else {"error": "Wiki page not found."},
                                 status=200 if page is not None else 404)

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/wildcards/settings")
    async def configure_wildcards(request):
        if not local_request(request):
            return web.json_response({"error": "Wildcard settings require local access."}, status=403)
        try:
            data = await request.json()
            path = await asyncio.to_thread(set_wildcard_path, data.get("path"))
        except (ValueError, TypeError, AttributeError):
            return web.json_response({"error": "Enter an existing absolute folder path, or leave it blank for the built-in folder."}, status=400)
        return web.json_response({"path": path})

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/wildcards/settings")
    async def current_wildcard_settings(request):
        return web.json_response({"path": str(active_wildcard_root())})

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/wildcards/browse")
    async def browse_wildcards(request):
        if not local_request(request):
            return web.json_response({"error": "Folder browsing requires local access."}, status=403)
        try:
            selected = await asyncio.to_thread(browse_wildcard_folder)
        except (OSError, AttributeError, TimeoutError):
            LOG.exception("Wildcard folder picker failed")
            return web.json_response({"error": "Unable to open the folder picker on this computer."}, status=500)
        return web.json_response({"path": selected})

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/wildcards")
    async def list_wildcards(request):
        names = await asyncio.to_thread(lambda: sorted(wildcard_files()))
        return web.json_response({"wildcards": names})

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/wildcards/preview")
    async def preview_wildcard(request):
        if not local_request(request):
            return web.json_response({"error": "Wildcard previews require local access."}, status=403)
        try:
            preview = await asyncio.to_thread(wildcard_preview, request.query.get("name"))
        except KeyError:
            return web.json_response({"error": "Wildcard file not found."}, status=404)
        except OSError:
            LOG.exception("Could not read wildcard preview")
            return web.json_response({"error": "Unable to read wildcard file."}, status=500)
        return web.json_response(preview)

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/favorites/list")
    async def get_favorites(request):
        if not local_request(request):
            return web.json_response({"error": "Favorites require local access."}, status=403)
        try:
            prompts = await asyncio.to_thread(list_favorites)
        except OSError as exc:
            LOG.exception("Could not read favorite prompts")
            return web.json_response({"error": str(exc) or "Unable to load favorites."}, status=500)
        return web.json_response({"favorites": prompts})

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/favorites/settings/current")
    async def current_favorite_settings(request):
        if not local_request(request):
            return web.json_response({"error": "Favorite settings require local access."}, status=403)
        try:
            path = await asyncio.to_thread(active_favorite_root)
        except OSError as exc:
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response({"path": str(path)})

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/favorites/settings")
    async def configure_favorites(request):
        if not local_request(request):
            return web.json_response({"error": "Favorite settings require local access."}, status=403)
        try:
            data = await request.json()
            path = await asyncio.to_thread(set_favorite_path, data.get("path"))
        except (ValueError, TypeError, AttributeError):
            return web.json_response({"error": "Enter an existing absolute folder path, or leave it blank for the built-in folder."}, status=400)
        except OSError:
            LOG.exception("Could not save favorite folder setting")
            return web.json_response({"error": "Unable to save favorite folder setting."}, status=500)
        return web.json_response({"path": path})

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/favorites/browse")
    async def browse_favorites(request):
        if not local_request(request):
            return web.json_response({"error": "Folder browsing requires local access."}, status=403)
        try:
            selected = await asyncio.to_thread(browse_favorite_folder)
        except (OSError, AttributeError, TimeoutError):
            LOG.exception("Favorite folder picker failed")
            return web.json_response({"error": "Unable to open the folder picker on this computer."}, status=500)
        return web.json_response({"path": selected})

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/favorites")
    async def add_favorite(request):
        if not local_request(request):
            return web.json_response({"error": "Favorites require local access."}, status=403)
        try:
            data = await request.json()
            prompts = await asyncio.to_thread(save_favorite, data.get("prompt"))
        except (ValueError, TypeError, AttributeError) as exc:
            return web.json_response({"error": str(exc) or "Invalid prompt."}, status=400)
        except OSError:
            LOG.exception("Could not save favorite prompt")
            return web.json_response({"error": "Unable to save the favorite."}, status=500)
        return web.json_response({"favorites": prompts})

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/favorites/change")
    async def update_favorite(request):
        if not local_request(request):
            return web.json_response({"error": "Favorites require local access."}, status=403)
        try:
            data = await request.json()
            action = data.get("action")
            if action not in ("edit", "delete"):
                raise ValueError("Invalid favorite action.")
            prompts = await asyncio.to_thread(
                change_favorite, data.get("index"), data.get("original"),
                replacement=data.get("prompt"), delete=action == "delete")
        except FavoriteConflictError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (ValueError, TypeError, AttributeError) as exc:
            return web.json_response({"error": str(exc) or "Invalid favorite change."}, status=400)
        except OSError:
            LOG.exception("Could not change favorite prompt")
            return web.json_response({"error": "Unable to change the favorite."}, status=500)
        return web.json_response({"favorites": prompts})

    @PromptServer.instance.routes.post("/toyxyz/booru-tags/wildcards/open-folder")
    async def open_folder(request):
        # Folder launch is a local desktop action. Reject remote clients and
        # cross-origin requests; no path or shell command is accepted.
        if not local_request(request):
            return web.json_response({"error": "Open Folder is available only on the ComfyUI computer."}, status=403)
        try:
            await asyncio.to_thread(open_wildcard_folder)
        except (OSError, AttributeError):
            LOG.exception("Could not open wildcard folder")
            return web.json_response({"error": "Unable to open the wildcard folder on this computer."}, status=500)
        return web.json_response({"ok": True})

    @PromptServer.instance.routes.get("/toyxyz/booru-tags/suggest")
    async def suggest(request):
        query = request.query.get("q", "")
        try:
            offset = int(request.query.get("offset", "0"))
        except ValueError:
            return web.json_response({"error": "Invalid offset."}, status=400)
        if offset < 0:
            return web.json_response({"error": "Invalid offset."}, status=400)
        try:
            results = await asyncio.to_thread(cached_suggestions, query, offset, PAGE_SIZE + 1)
        except Exception:
            LOG.exception("Booru tag suggestion lookup failed")
            return web.json_response({"error": "Tag suggestions unavailable."}, status=500)
        return web.json_response({"suggestions": results[:PAGE_SIZE],
                                  "has_more": len(results) > PAGE_SIZE})


register_routes()
