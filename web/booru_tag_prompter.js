import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { currentTagRange, formatSelectedTag, insertTag, matchRanges } from "./util/booru_tag_input.js";
import { textareaCaretRect } from "./util/textarea_caret.js";
import { currentWildcardRange, insertWildcard, installWildcards, matchingWildcards } from "./util/booru_wildcards.js";
import { installWiki } from "./util/booru_wiki.js";

if (!document.querySelector('link[data-toyxyz-booru-style]')) {
    const style = document.createElement("link");
    style.rel = "stylesheet";
    style.href = new URL("./booru_tag_prompter.css", import.meta.url).href;
    style.dataset.toyxyzBooruStyle = "true";
    document.head.append(style);
}

const TYPE = "ToyxyzBooruTagPrompter";
const WILDCARD_PATH_SETTING='toyxyz_test_nodes.WildcardPath';
const FAVORITE_PATH_SETTING='toyxyz_test_nodes.FavoritePath';
let wildcardPathTimer,favoritePathTimer;
async function syncWildcardPath(value) {
    const response=await api.fetchApi('/toyxyz/booru-tags/wildcards/settings',{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:value}),
    });
    if(!response.ok){
        const data=await response.json().catch(()=>({}));
        throw new Error(data.error||`HTTP ${response.status}`);
    }
    window.dispatchEvent(new Event('toyxyz-wildcards-changed'));
}
async function browseWildcardPath(apiClient=api) {
    const response=await apiClient.fetchApi('/toyxyz/booru-tags/wildcards/browse',{method:'POST'});
    const data=await response.json().catch(()=>({}));
    if(!response.ok)throw new Error(data.error||`Folder picker failed (HTTP ${response.status}).`);
    return typeof data.path==='string'?data.path:'';
}
async function syncFavoritePath(value) {
    const response=await api.fetchApi('/toyxyz/booru-tags/favorites/settings',{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:value}),
    });
    if(!response.ok){
        const data=await response.json().catch(()=>({}));
        throw new Error(data.error||`HTTP ${response.status}`);
    }
    window.dispatchEvent(new Event('toyxyz-favorites-changed'));
}
async function browseFavoritePath(apiClient=api) {
    const response=await apiClient.fetchApi('/toyxyz/booru-tags/favorites/browse',{method:'POST'});
    const data=await response.json().catch(()=>({}));
    if(!response.ok)throw new Error(data.error||`Folder picker failed (HTTP ${response.status}).`);
    return typeof data.path==='string'?data.path:'';
}
function folderPathControl(name,setter,value,placeholder,browse) {
    const row=document.createElement('div');row.style.cssText='display:flex;align-items:center;gap:8px;width:100%;min-width:0;';
    const input=document.createElement('input');input.type='text';input.value=typeof value==='string'?value:'';
    input.placeholder=placeholder;input.setAttribute('aria-label',name);
    input.style.cssText='flex:1;min-width:0;padding:6px 8px;border:1px solid #666;border-radius:5px;background:#222;color:#eee;';
    const button=document.createElement('button');button.type='button';button.textContent='Browse';
    button.title='Choose a folder on the ComfyUI computer';
    button.style.cssText='flex-shrink:0;padding:6px 10px;border:1px solid #666;border-radius:5px;background:#353535;color:#eee;cursor:pointer;';
    const status=document.createElement('span');status.setAttribute('role','status');status.style.cssText='font-size:11px;color:#e7b7a2;';
    input.addEventListener('change',()=>setter(input.value));
    button.addEventListener('click',async()=>{
        button.disabled=true;status.textContent='';
        try {
            const path=await browse();
            if(path){input.value=path;setter(path);}
        }catch(error){status.textContent=error.message;}
        finally{button.disabled=false;}
    });
    row.append(input,button,status);
    return row;
}
function wildcardPathControl(name,setter,value) {
    return folderPathControl(name,setter,value,'Default: this node’s wildcards folder',browseWildcardPath);
}
function favoritePathControl(name,setter,value) {
    return folderPathControl(name,setter,value,'Default: this node’s favorites folder',browseFavoritePath);
}

function attachSuggestions(node) {
    const widget = node.widgets?.find(item => item.name === "tags");
    const input = widget?.inputEl;
    if (!(input instanceof HTMLTextAreaElement) || input.dataset.toyxyzBooruAttached) return;
    input.dataset.toyxyzBooruAttached = "true";
    input.setAttribute("autocomplete", "off");
    installWildcards(node,widget,input,api);
    installWiki(node,widget,input,api);

    const list = document.createElement("div");
    list.className = "toyxyz-booru-suggestions";
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", "Tag and wildcard suggestions");
    let suggestions = [];
    let selected = -1;
    let activeRange = null;
    let activeKind = "tag";
    let wildcardNames = null;
    let nextOffset = 0;
    let hasMore = false;
    let loading = false;
    let pointerInsideList = false;
    let requestId = 0;
    let timer;
    let composing = false;
    let removed = false;
    let positionFrame = 0;
    let lastInputRect = "";

    const hide = () => {
        clearTimeout(timer);
        cancelAnimationFrame(positionFrame);
        positionFrame = 0;
        lastInputRect = "";
        requestId++;
        list.remove();
        suggestions = [];
        activeRange = null;
        nextOffset = 0;
        hasMore = false;
        loading = false;
        pointerInsideList = false;
    };

    const position = () => {
        if (!list.isConnected) return;
        const rect = input.getBoundingClientRect();
        const caret = textareaCaretRect(input);
        if (caret.bottom < rect.top || caret.top > rect.bottom) { hide(); return; }
        const style = getComputedStyle(input);
        const wordStart = activeRange ? textareaCaretRect(input, activeRange.start) : caret;
        const anchorLeft = Math.abs(wordStart.top - caret.top) < 1
            ? Math.max(rect.left, Math.min(wordStart.left, rect.right)) : caret.left;
        const scale = caret.scale;
        const fontSize = Number.parseFloat(style.fontSize) || 13;
        list.style.fontSize = `${fontSize * scale}px`;
        list.style.width = "max-content";
        list.style.maxWidth = `${Math.min(420, rect.width * 0.5, window.innerWidth - 16)}px`;
        const width = list.getBoundingClientRect().width;
        list.style.left = `${Math.max(8, Math.min(anchorLeft, window.innerWidth - width - 8))}px`;
        const top = caret.bottom + 2;
        list.style.top = `${top}px`;
        list.style.maxHeight = `${Math.max(32, Math.min(300 * scale, window.innerHeight - top - 8))}px`;
    };

    const watchPosition = () => {
        if (!list.isConnected || removed) { positionFrame = 0; return; }
        const rect = input.getBoundingClientRect();
        const key = `${rect.left},${rect.top},${rect.width},${rect.height}`;
        if (key !== lastInputRect) {
            lastInputRect = key;
            position();
        }
        positionFrame = list.isConnected ? requestAnimationFrame(watchPosition) : 0;
    };

    const select = index => {
        selected = (index + suggestions.length) % suggestions.length;
        [...list.children].forEach((item, i) =>
            item.setAttribute("aria-selected", String(i === selected)));
        list.children[selected]?.scrollIntoView({ block: "nearest" });
    };

    const insert = index => {
        const choice = suggestions[index];
        const range = activeKind === "wildcard"
            ? currentWildcardRange(input.value, input.selectionStart, input.selectionEnd)
            : currentTagRange(input.value, input.selectionStart, input.selectionEnd);
        if (!choice || !range || range.query !== activeRange?.query) return;
        const next = activeKind === "wildcard"
            ? insertWildcard(input.value, range.start, range.end, choice.tag)
            : insertTag(input.value, range, formatSelectedTag(choice.tag));
        input.value = next.value;
        input.setSelectionRange(next.caret, next.caret);
        input.dispatchEvent(new Event("input", { bubbles: true }));
        if (widget.value !== input.value) {
            widget.value = input.value;
            widget.callback?.(widget.value);
        }
        node.setDirtyCanvas?.(true, true);
        input.focus();
        hide();
    };

    const appendHighlighted = (element, label, query) => {
        let cursor = 0;
        for (const [start, end] of matchRanges(label, query)) {
            element.append(document.createTextNode(label.slice(cursor, start)));
            const match = document.createElement("span");
            match.className = "toyxyz-booru-match";
            match.textContent = label.slice(start, end);
            element.append(match);
            cursor = end;
        }
        element.append(document.createTextNode(label.slice(cursor)));
    };

    const render = (results, append = false) => {
        if (!append) {
            suggestions = [];
            selected = -1;
            list.replaceChildren();
            list.scrollTop = 0;
        }
        if (!results.length && !suggestions.length) { hide(); return; }
        for (const result of results) {
            const index = suggestions.length;
            suggestions.push(result);
            const button = document.createElement("button");
            button.type = "button";
            button.className = "toyxyz-booru-suggestion";
            button.setAttribute("role", "option");
            button.setAttribute("aria-selected", "false");
            const name = document.createElement("span");
            name.className = "toyxyz-booru-suggestion-name";
            appendHighlighted(name, activeKind === "wildcard" ? `__${result.tag}__` : result.tag, activeRange.query);
            const detail = document.createElement("span");
            detail.className = "toyxyz-booru-suggestion-detail";
            detail.append(document.createTextNode(activeKind === "wildcard"
                ? "wildcard" : `${result.category} · ${result.count.toLocaleString()}`));
            if (result.alias) {
                detail.append(document.createTextNode(" · "));
                appendHighlighted(detail, result.alias, activeRange.query);
            }
            button.append(name, detail);
            button.addEventListener("pointerdown", event => event.preventDefault());
            button.addEventListener("click", () => insert(index));
            list.append(button);
        }
        if (!list.isConnected) document.body.append(list);
        position();
        if (list.isConnected && !positionFrame) positionFrame = requestAnimationFrame(watchPosition);
    };

    const loadPage = async (range, currentRequest, offset, append) => {
        if (loading) return;
        loading = true;
        try {
            let data;
            if (activeKind === "wildcard") {
                if (!wildcardNames) {
                    const response = await api.fetchApi('/toyxyz/booru-tags/wildcards');
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const result = await response.json();
                    wildcardNames = Array.isArray(result.wildcards) ? result.wildcards : [];
                }
                data = matchingWildcards(wildcardNames, range.query, offset);
            } else {
                const response = await api.fetchApi(
                    `/toyxyz/booru-tags/suggest?q=${encodeURIComponent(range.query)}&offset=${offset}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                data = await response.json();
            }
            if (currentRequest !== requestId ||
                    (document.activeElement !== input && !pointerInsideList) || removed) return;
            const page = Array.isArray(data.suggestions) ? data.suggestions : [];
            nextOffset = offset + page.length;
            hasMore = Boolean(data.has_more);
            render(page, append);
        } catch (error) {
            if (currentRequest === requestId) {
                hasMore = false;
                if (!append) hide();
            }
            console.warn("booru tag prompter suggestions unavailable", error);
        } finally {
            if (currentRequest === requestId) loading = false;
        }
    };

    const update = () => {
        clearTimeout(timer);
        hide();
        const wildcardRange = currentWildcardRange(input.value, input.selectionStart, input.selectionEnd);
        activeKind = wildcardRange ? "wildcard" : "tag";
        const range = wildcardRange || currentTagRange(input.value, input.selectionStart, input.selectionEnd);
        activeRange = range;
        const currentRequest = ++requestId;
        if (!range || (activeKind === "tag" && range.query.includes('__')) || composing || removed) return;
        timer = setTimeout(() => loadPage(range, currentRequest, 0, false), 100);
    };

    const onWildcardListChanged = () => {
        wildcardNames = null;
        if (activeKind === "wildcard" && document.activeElement === input) update();
    };

    list.addEventListener("scroll", () => {
        if (hasMore && !loading && activeRange &&
                list.scrollTop + list.clientHeight >= list.scrollHeight - 60) {
            loadPage(activeRange, requestId, nextOffset, true);
        }
    });
    list.addEventListener("pointerenter", () => { pointerInsideList = true; });
    list.addEventListener("pointerleave", () => {
        pointerInsideList = false;
        if (document.activeElement !== input) hide();
    });

    const onKeyDown = event => {
        // Leave ComfyUI's Ctrl+arrow weights and Ctrl+Enter action untouched.
        if (event.ctrlKey && (event.key.startsWith("Arrow") || event.key === "Enter")) return;
        if (!list.isConnected || !suggestions.length) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            event.stopPropagation();
            if (event.key === "ArrowUp" && selected === 0) return;
            if (event.key === "ArrowDown" && selected === suggestions.length - 1 && hasMore) {
                const nextIndex = suggestions.length;
                const currentRequest = requestId;
                if (!loading) {
                    loadPage(activeRange, currentRequest, nextOffset, true).then(() => {
                        if (currentRequest === requestId && list.isConnected &&
                                suggestions.length > nextIndex) select(nextIndex);
                    });
                }
                return;
            }
            select(selected < 0 ? 0 : selected + (event.key === "ArrowDown" ? 1 : -1));
        } else if (event.key === "Enter" || event.key === "Tab") {
            event.preventDefault();
            event.stopPropagation();
            insert(selected < 0 ? 0 : selected);
        } else if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            hide();
        }
    };
    const onOutsidePointer = event => {
        if (event.target !== input && !list.contains(event.target)) hide();
    };
    const onBlur = () => setTimeout(() => {
        if (!pointerInsideList && !list.contains(document.activeElement)) hide();
    }, 120);

    input.addEventListener("input", update);
    input.addEventListener("click", update);
    input.addEventListener("keyup", event => {
        if (event.ctrlKey && event.key.startsWith("Arrow")) return;
        if (["ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
            update();
        }
    });
    input.addEventListener("keydown", onKeyDown);
    input.addEventListener("compositionstart", () => { composing = true; hide(); });
    input.addEventListener("compositionend", () => { composing = false; update(); });
    input.addEventListener("blur", onBlur);
    input.addEventListener("scroll", position);
    document.addEventListener("pointerdown", onOutsidePointer);
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    window.addEventListener("toyxyz-wildcard-list-updated", onWildcardListChanged);
    window.addEventListener("toyxyz-wildcards-changed", onWildcardListChanged);

    const originalRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        removed = true;
        clearTimeout(timer);
        hide();
        document.removeEventListener("pointerdown", onOutsidePointer);
        window.removeEventListener("resize", position);
        window.removeEventListener("scroll", position, true);
        window.removeEventListener("toyxyz-wildcard-list-updated", onWildcardListChanged);
        window.removeEventListener("toyxyz-wildcards-changed", onWildcardListChanged);
        input.removeEventListener("scroll", position);
        return originalRemoved?.apply(this, args);
    };
}

app.registerExtension({
    name: "toyxyz.booru_tag_prompter",
    settings: [{
        id: WILDCARD_PATH_SETTING,
        name: 'Wildcard folder path',
        category: ['toyxyz_test_nodes', 'Wildcards', 'Wildcard folder path'],
        type: wildcardPathControl,
        defaultValue: '',
        tooltip: 'Enter an existing absolute folder path or choose Browse on the ComfyUI computer. Leave blank to use this custom node’s wildcards folder.',
        onChange(value) {
            clearTimeout(wildcardPathTimer);
            wildcardPathTimer=setTimeout(async()=>{
                try {
                    await syncWildcardPath(value);
                }catch(error){console.warn('Wildcard folder path was not applied:',error);}
            },400);
        },
    },{
        id: FAVORITE_PATH_SETTING,
        name: 'Favorite folder path',
        category: ['toyxyz_test_nodes', 'Favorites', 'Favorite folder path'],
        type: favoritePathControl,
        defaultValue: '',
        tooltip: 'Enter an existing absolute folder path or choose Browse on the ComfyUI computer. Leave blank to use this custom node’s favorites folder.',
        onChange(value) {
            clearTimeout(favoritePathTimer);
            favoritePathTimer=setTimeout(async()=>{
                try {await syncFavoritePath(value);}
                catch(error){console.warn('Favorite folder path was not applied:',error);}
            },400);
        },
    }],
    async setup(app) {
        const saved=app.ui.settings.getSettingValue(WILDCARD_PATH_SETTING);
        if(typeof saved==='string'&&saved.trim()) {
            try {await syncWildcardPath(saved);}catch(error){console.warn('Saved wildcard folder path was not applied:',error);}
        }
        const favoritePath=app.ui.settings.getSettingValue(FAVORITE_PATH_SETTING);
        if(typeof favoritePath==='string'&&favoritePath.trim()) {
            try {await syncFavoritePath(favoritePath);}catch(error){console.warn('Saved favorite folder path was not applied:',error);}
        }
    },
    beforeRegisterNodeDef(nodeType, data) {
        if (data.name !== TYPE) return;
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            const legacy = this.inputs?.find(input => input.name === "preset");
            if (legacy) legacy.name = "camera";
            const obsoleteWidget = this.widgets?.findIndex(widget => widget.name === "model_type");
            if (obsoleteWidget >= 0) this.widgets.splice(obsoleteWidget, 1);
            return result;
        };
    },
    nodeCreated(node) {
        if (node.comfyClass === TYPE || node.type === TYPE) attachSuggestions(node);
    },
});
