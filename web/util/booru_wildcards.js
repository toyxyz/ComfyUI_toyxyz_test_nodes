export function wildcardSegments(text) {
    const result=[];let cursor=0;
    for(const match of text.matchAll(/__([^\r\n]+?)__/g)) {
        result.push({text:text.slice(cursor,match.index),wildcard:false},{text:match[0],wildcard:true});
        cursor=match.index+match[0].length;
    }
    result.push({text:text.slice(cursor),wildcard:false});return result;
}
export function insertWildcard(text,start,end,name) {
    const token=`__${name}__`;
    const prefix=start>0&&!/[\s,(]$/.test(text.slice(0,start))?', ':'';
    const suffix=end<text.length&&!/^[\s,):]/.test(text.slice(end))?', ':'';
    const inserted=prefix+token+suffix;
    return {value:text.slice(0,start)+inserted+text.slice(end),caret:start+prefix.length+token.length};
}
export function insertFavorite(text,caret,prompt) {
    const position=Math.max(0,Math.min(caret,text.length));
    return {value:text.slice(0,position)+prompt+text.slice(position),caret:position+prompt.length};
}
export function currentWildcardRange(text,selectionStart,selectionEnd=selectionStart) {
    if(selectionStart!==selectionEnd)return null;
    const cursor=Math.max(0,Math.min(selectionStart,text.length));
    const segmentStart=Math.max(text.lastIndexOf(',',cursor-1),text.lastIndexOf(';',cursor-1),text.lastIndexOf('\n',cursor-1))+1;
    const markers=[...text.slice(segmentStart,cursor).matchAll(/__/g)];
    if(markers.length%2===0)return null;
    const start=segmentStart+markers.at(-1).index;
    const query=text.slice(start+2,cursor);
    if(/[():]/.test(query))return null;
    const close=text.indexOf('__',cursor);
    const nextDelimiter=text.slice(cursor).search(/[,;\n]/);
    const end=close>=0&&(nextDelimiter<0||close-cursor<nextDelimiter)?close+2:cursor;
    return {start,end,query};
}
export function matchingWildcards(names,query,offset=0,limit=50) {
    const needle=query.toLocaleLowerCase();
    const matches=names.filter(name=>name.toLocaleLowerCase().includes(needle));
    matches.sort((a,b)=>Number(b.toLocaleLowerCase().startsWith(needle))-Number(a.toLocaleLowerCase().startsWith(needle))||a.localeCompare(b));
    return {suggestions:matches.slice(offset,offset+limit).map(tag=>({tag,category:'wildcard'})),has_more:offset+limit<matches.length};
}
export async function openWildcardFolder(api) {
    const response=await api.fetchApi('/toyxyz/booru-tags/wildcards/open-folder',{method:'POST'});
    if(response.ok)return;
    if(response.status===404||response.status===405)throw new Error('Restart ComfyUI to enable Open Folder, then refresh the browser.');
    let message=`Unable to open the wildcard folder (HTTP ${response.status}).`;
    try {
        const result=await response.json();
        if(typeof result.error==='string')message=result.error;
    }catch { /* Plain-text server errors are not JSON. Keep the useful HTTP error. */ }
    throw new Error(message);
}
export function wildcardTree(names) {
    const root={folders:new Map(),files:[]};
    for(const path of [...new Set(names)].sort()) {
        const parts=path.split('/');let branch=root;
        for(const name of parts.slice(0,-1)) {
            if(!branch.folders.has(name))branch.folders.set(name,{folders:new Map(),files:[]});
            branch=branch.folders.get(name);
        }
        branch.files.push({name:parts.at(-1),path});
    }
    return root;
}
export function installWildcards(node,widget,input,api) {
    if(!node.addDOMWidget)return;
    const root=document.createElement('div');root.className='toyxyz-wildcard-editor';
    const editor=document.createElement('div');editor.className='toyxyz-wildcard-text';
    const backdrop=document.createElement('div');backdrop.className='toyxyz-wildcard-backdrop';
    const highlight=document.createElement('pre');backdrop.append(highlight);
    input.classList.add('toyxyz-wildcard-input');
    input.style.cssText='position:absolute;inset:0;width:100%;height:100%;box-sizing:border-box;resize:none;margin:0;';
    editor.append(backdrop,input);
    const sidebar=document.createElement('aside');sidebar.className='toyxyz-wildcard-sidebar';
    const tabs=document.createElement('div');tabs.className='toyxyz-wildcard-tabs';tabs.setAttribute('role','tablist');
    const wildcardTab=document.createElement('button');wildcardTab.type='button';wildcardTab.textContent='W';wildcardTab.title='Wildcards';wildcardTab.setAttribute('aria-label','Wildcards');
    const favoriteTab=document.createElement('button');favoriteTab.type='button';favoriteTab.textContent='F';favoriteTab.title='Favorites';favoriteTab.setAttribute('aria-label','Favorites');
    tabs.append(wildcardTab,favoriteTab);
    const header=document.createElement('div');header.className='toyxyz-wildcard-header';
    const title=document.createElement('span');title.textContent='Wildcards';
    const folder=document.createElement('button');folder.type='button';folder.textContent='Folder';folder.title='Open wildcard folder';folder.setAttribute('aria-label','Open wildcard folder');
    const save=document.createElement('button');save.type='button';save.textContent='Save';save.title='Save the entire current prompt as a favorite';save.hidden=true;
    const refresh=document.createElement('button');refresh.type='button';refresh.textContent='↻';refresh.title='Refresh wildcard files';
    const list=document.createElement('div');list.className='toyxyz-wildcard-files';
    const favoritesList=document.createElement('div');favoritesList.className='toyxyz-favorite-files';favoritesList.hidden=true;
    const status=document.createElement('div');status.className='toyxyz-wildcard-status';status.setAttribute('role','status');status.hidden=true;
    header.append(title,folder,save,refresh);sidebar.append(tabs,header,status,list,favoritesList);root.append(editor,sidebar);
    root.style.setProperty('--toyxyz-node-bg',node.bgcolor||'#353535');
    let removed=false,request=0,favoriteRequest=0,activeTab='wildcards';
    let favoriteWritePending=false,reloadFavoritesAfterWrite=false;
    const openFolders=new Set();
    const previewCache=new Map();
    const preview=document.createElement('div');preview.className='toyxyz-wildcard-preview';preview.hidden=true;
    document.body?.append(preview);
    const favoriteMenu=document.createElement('div');favoriteMenu.className='toyxyz-favorite-menu';favoriteMenu.hidden=true;
    const editAction=document.createElement('button');editAction.type='button';editAction.textContent='Edit';
    const deleteAction=document.createElement('button');deleteAction.type='button';deleteAction.textContent='Delete';
    favoriteMenu.append(editAction,deleteAction);document.body?.append(favoriteMenu);
    const editOverlay=document.createElement('div');editOverlay.className='toyxyz-favorite-overlay';editOverlay.hidden=true;
    editOverlay.setAttribute('role','dialog');editOverlay.setAttribute('aria-modal','true');
    const editPanel=document.createElement('div');editPanel.className='toyxyz-favorite-dialog';
    const editTitle=document.createElement('h3');editTitle.textContent='Edit favorite';
    const editText=document.createElement('textarea');editText.setAttribute('aria-label','Favorite prompt');editText.maxLength=100000;
    const editStatus=document.createElement('div');editStatus.className='toyxyz-wildcard-status';editStatus.setAttribute('role','status');
    const editButtons=document.createElement('div');editButtons.className='toyxyz-favorite-dialog-actions';
    const cancelEdit=document.createElement('button');cancelEdit.type='button';cancelEdit.textContent='Cancel';
    const saveEdit=document.createElement('button');saveEdit.type='button';saveEdit.textContent='Save changes';
    editButtons.append(cancelEdit,saveEdit);editPanel.append(editTitle,editText,editStatus,editButtons);
    editOverlay.append(editPanel);document.body?.append(editOverlay);
    let selectedFavorite=null;
    const hideFavoriteMenu=()=>{favoriteMenu.hidden=true;};
    const closeFavoriteEditor=()=>{editOverlay.hidden=true;editStatus.textContent='';selectedFavorite=null;};
    favoriteMenu.addEventListener('pointerdown',event=>event.stopPropagation());
    editOverlay.addEventListener('pointerdown',event=>event.stopPropagation());
    const outsidePointer=()=>hideFavoriteMenu();
    const escapeFavorite=event=>{if(event.key==='Escape'){hideFavoriteMenu();closeFavoriteEditor();}};
    window.addEventListener('pointerdown',outsidePointer);
    window.addEventListener('keydown',escapeFavorite);
    let hovered=null,previewHideTimer;
    const hidePreview=()=>{clearTimeout(previewHideTimer);hovered=null;preview.hidden=true;};
    const schedulePreviewHide=()=>{
        clearTimeout(previewHideTimer);
        previewHideTimer=setTimeout(hidePreview,180);
    };
    preview.addEventListener('mouseenter',()=>clearTimeout(previewHideTimer));
    preview.addEventListener('mouseleave',hidePreview);
    const showPreview=async(button,path)=>{
        hovered=button;
        const rect=button.getBoundingClientRect();
        const width=Math.min(480,window.innerWidth-16);
        preview.style.left=`${Math.max(8,Math.min(rect.right+8,window.innerWidth-width-8))}px`;
        preview.style.top=`${Math.max(8,Math.min(rect.top,window.innerHeight-428))}px`;
        preview.textContent=`${path}.txt\n\nLoading…`;preview.hidden=false;
        try {
            if(!previewCache.has(path)){
                const response=await api.fetchApi(`/toyxyz/booru-tags/wildcards/preview?name=${encodeURIComponent(path)}`);
                if(!response.ok)throw new Error('Unable to load wildcard contents.');
                previewCache.set(path,await response.json());
            }
            if(removed||hovered!==button||activeTab!=='wildcards')return;
            const data=previewCache.get(path);
            preview.textContent=`${path}.txt\n\n${data.content||'(empty file)'}${data.truncated?'\n\n[Preview truncated]':''}`;
        }catch(error){if(!removed&&hovered===button)preview.textContent=`${path}.txt\n\n${error.message}`;}
    };
    const paint=()=>{
        const style=getComputedStyle(input);
        for(const key of ['fontFamily','fontSize','fontWeight','lineHeight','letterSpacing','padding','borderWidth'])highlight.style[key]=style[key];
        highlight.style.width=`${input.clientWidth}px`;
        highlight.replaceChildren();
        for(const segment of wildcardSegments(input.value)) {
            const span=document.createElement(segment.wildcard?'mark':'span');span.textContent=segment.text;highlight.append(span);
        }
        highlight.append(document.createTextNode('\n'));
        highlight.style.transform=`translate(${-input.scrollLeft}px,${-input.scrollTop}px)`;
    };
    const load=async()=>{
        const id=++request;refresh.disabled=true;hidePreview();previewCache.clear();
        try {
            const response=await api.fetchApi('/toyxyz/booru-tags/wildcards');
            if(!response.ok)throw new Error('Wildcard files unavailable');
            const data=await response.json();if(removed||id!==request)return;
            window.dispatchEvent(new Event('toyxyz-wildcard-list-updated'));
            list.replaceChildren();
            const renderTree=(branch,parent,prefix='')=>{
                for(const [name,child] of branch.folders) {
                    const path=prefix+name,folder=document.createElement('details');folder.className='toyxyz-wildcard-directory';folder.open=openFolders.has(path);
                    const summary=document.createElement('summary'),icon=document.createElement('span'),label=document.createElement('span');
                    icon.className='toyxyz-wildcard-folder-icon';icon.setAttribute('aria-hidden','true');label.textContent=name;
                    summary.title='Expand or collapse '+name;summary.append(icon,label);
                    const contents=document.createElement('div');contents.className='toyxyz-wildcard-directory-contents';
                    folder.append(summary,contents);parent.append(folder);
                    folder.addEventListener('toggle',()=>{if(folder.open)openFolders.add(path);else openFolders.delete(path);});
                    renderTree(child,contents,path+'/');
                }
                for(const {name,path} of branch.files) {
                const button=document.createElement('button');button.type='button';button.textContent=name;button.title=`Insert __${path}__`;
                button.addEventListener('mouseenter',()=>{button.title='';showPreview(button,path);});
                button.addEventListener('mouseleave',()=>{button.title=`Insert __${path}__`;schedulePreviewHide();});
                button.addEventListener('pointerdown',event=>event.preventDefault());
                button.addEventListener('click',()=>{
                    const next=insertWildcard(input.value,input.selectionStart,input.selectionEnd,path);
                    input.value=next.value;input.focus();input.setSelectionRange(next.caret,next.caret);
                    input.dispatchEvent(new Event('input',{bubbles:true}));
                    widget.value=input.value;widget.callback?.(widget.value);node.setDirtyCanvas?.(true,true);
                });parent.append(button);
                }
            };
            renderTree(wildcardTree(data.wildcards||[]),list);
            if(!list.children.length)list.textContent='Add .txt files to wildcards/';
        }catch(error){if(!removed&&id===request)list.textContent='Unable to load wildcard files. Use Refresh to retry.';}
        finally{if(!removed&&id===request)refresh.disabled=false;}
    };
    const renderFavorites=prompts=>{
        hideFavoriteMenu();
        favoritesList.replaceChildren();
        for(const [index,prompt] of prompts.entries()) {
            const button=document.createElement('button');button.type='button';
            button.textContent=`${index+1}. ${prompt.replace(/\s+/g,' ').slice(0,70)}`;
            button.title=prompt;button.setAttribute('aria-label',`Insert favorite ${index+1}: ${prompt}`);
            button.addEventListener('contextmenu',event=>{
                event.preventDefault();event.stopPropagation();
                selectedFavorite={index,prompt};
                favoriteMenu.style.left=`${Math.max(8,Math.min(event.clientX,window.innerWidth-150))}px`;
                favoriteMenu.style.top=`${Math.max(8,Math.min(event.clientY,window.innerHeight-90))}px`;
                favoriteMenu.hidden=false;
            });
            button.addEventListener('pointerdown',event=>event.preventDefault());
            button.addEventListener('click',()=>{
                const next=insertFavorite(input.value,input.selectionStart,prompt);
                input.value=next.value;input.focus();input.setSelectionRange(next.caret,next.caret);
                input.dispatchEvent(new Event('input',{bubbles:true}));
                widget.value=input.value;widget.callback?.(widget.value);node.setDirtyCanvas?.(true,true);
            });
            favoritesList.append(button);
        }
        if(!prompts.length)favoritesList.textContent='No favorites saved yet.';
    };
    const beginFavoriteWrite=()=>{
        if(favoriteWritePending)throw new Error('Wait for the current favorite change to finish.');
        favoriteWritePending=true;
        ++favoriteRequest; // Invalidate any GET started before this write.
        refresh.disabled=true;
    };
    const finishFavoriteWrite=()=>{
        favoriteWritePending=false;
        if(reloadFavoritesAfterWrite&&!removed){
            reloadFavoritesAfterWrite=false;
            loadFavorites();
        }else if(!removed)refresh.disabled=false;
    };
    const changeFavorite=async(action,selection,replacement)=>{
        beginFavoriteWrite();
        try {
            const response=await api.fetchApi('/toyxyz/booru-tags/favorites/change',{
                method:'POST',headers:{'Content-Type':'application/json'},
                body:JSON.stringify({action,index:selection.index,original:selection.prompt,prompt:replacement})});
            const data=await response.json().catch(()=>({}));
            if(!response.ok)throw new Error(data.error||`Unable to ${action} favorite.`);
            if(!Array.isArray(data.favorites))throw new Error('Invalid favorites response. Refresh the list.');
            if(!removed&&!reloadFavoritesAfterWrite)renderFavorites(data.favorites);
        }finally{finishFavoriteWrite();}
    };
    editAction.addEventListener('click',()=>{
        if(!selectedFavorite)return;
        hideFavoriteMenu();editText.value=selectedFavorite.prompt;
        editStatus.textContent='';editOverlay.hidden=false;editText.focus();
    });
    deleteAction.addEventListener('click',async()=>{
        const selection=selectedFavorite;hideFavoriteMenu();
        if(!selection||!window.confirm(`Delete this favorite?\n\n${selection.prompt.slice(0,200)}`))return;
        deleteAction.disabled=true;status.hidden=true;
        try {await changeFavorite('delete',selection);selectedFavorite=null;}
        catch(error){if(!removed){status.textContent=error.message;status.hidden=false;}}
        finally{if(!removed)deleteAction.disabled=false;}
    });
    cancelEdit.addEventListener('click',closeFavoriteEditor);
    saveEdit.addEventListener('click',async()=>{
        if(!selectedFavorite)return;
        saveEdit.disabled=true;editStatus.textContent='';
        try {await changeFavorite('edit',selectedFavorite,editText.value);closeFavoriteEditor();}
        catch(error){if(!removed)editStatus.textContent=error.message;}
        finally{if(!removed)saveEdit.disabled=false;}
    });
    const loadFavorites=async()=>{
        if(favoriteWritePending){reloadFavoritesAfterWrite=true;return;}
        const id=++favoriteRequest;refresh.disabled=true;
        try {
            const response=await api.fetchApi('/toyxyz/booru-tags/favorites/list');
            const data=await response.json().catch(()=>({}));
            if(response.status===404)throw new Error('Restart ComfyUI to activate the updated Favorites route.');
            if(!response.ok)throw new Error(data.error||`Unable to load favorites (HTTP ${response.status}).`);
            if(!Array.isArray(data.favorites))throw new Error('Invalid favorites response. Try Refresh.');
            if(removed||id!==favoriteRequest)return;
            renderFavorites(data.favorites);status.textContent='';status.hidden=true;
        }catch(error){if(!removed&&id===favoriteRequest){status.textContent=error.message;status.hidden=false;}}
        finally{if(!removed&&id===favoriteRequest)refresh.disabled=false;}
    };
    const selectTab=kind=>{
        hidePreview();hideFavoriteMenu();closeFavoriteEditor();activeTab=kind;const favorites=kind==='favorites';
        wildcardTab.setAttribute('aria-selected',String(!favorites));favoriteTab.setAttribute('aria-selected',String(favorites));
        wildcardTab.classList.toggle('toyxyz-tab-active',!favorites);favoriteTab.classList.toggle('toyxyz-tab-active',favorites);
        title.textContent=favorites?'Favorites':'Wildcards';folder.hidden=favorites;save.hidden=!favorites;
        list.hidden=favorites;favoritesList.hidden=!favorites;refresh.title=favorites?'Refresh favorites':'Refresh wildcard files';
        status.hidden=true;status.textContent='';
        if(favorites)loadFavorites();
    };
    wildcardTab.addEventListener('click',()=>selectTab('wildcards'));
    favoriteTab.addEventListener('click',()=>selectTab('favorites'));
    refresh.addEventListener('click',()=>activeTab==='favorites'?loadFavorites():load());
    const favoritePathChanged=()=>{if(activeTab==='favorites')loadFavorites();};
    window.addEventListener('toyxyz-favorites-changed',favoritePathChanged);
    save.addEventListener('click',async()=>{
        save.disabled=true;status.hidden=true;status.textContent='';
        try {
            beginFavoriteWrite();
            try {
                const response=await api.fetchApi('/toyxyz/booru-tags/favorites',{
                    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:input.value})});
                const data=await response.json().catch(()=>({}));
                if(!response.ok)throw new Error(data.error||'Unable to save favorite.');
                if(!Array.isArray(data.favorites))throw new Error('Invalid favorites response. Refresh the list.');
                if(!removed&&!reloadFavoritesAfterWrite)renderFavorites(data.favorites);
            }finally{finishFavoriteWrite();}
        }catch(error){if(!removed){status.textContent=error.message;status.hidden=false;}}
        finally{if(!removed)save.disabled=false;}
    });
    window.addEventListener('toyxyz-wildcards-changed',load);
    folder.addEventListener('click',async()=>{
        folder.disabled=true;status.hidden=true;status.textContent='';
        try {
            await openWildcardFolder(api);
        }catch(error){if(!removed){status.textContent=error.message;status.hidden=false;}}
        finally{if(!removed)folder.disabled=false;}
    });
    input.addEventListener('input',paint);input.addEventListener('scroll',paint);
    // Keep the original DOM widget as the sole owner. Hiding it while moving
    // its textarea into another DOM widget lets ComfyUI hide/reparent the input.
    // Its existing getValue/setValue and textarea bindings remain untouched.
    widget.element=root;
    const dom=widget;
    const height=()=>Math.max(180,(node.size?.[1]||340)-85);
    dom.computeSize=width=>[width||node.size[0],height()];
    dom.computeLayoutSize=()=>({minHeight:height(),maxHeight:height()});
    const observer=typeof ResizeObserver==='undefined'?null:new ResizeObserver(paint);observer?.observe(input);
    const configure=node.onConfigure;node.onConfigure=function(){const result=configure?.apply(this,arguments);paint();return result;};
    const remove=node.onRemoved;node.onRemoved=function(){removed=true;request++;favoriteRequest++;hidePreview();preview.remove?.();favoriteMenu.remove?.();editOverlay.remove?.();observer?.disconnect();input.removeEventListener('input',paint);input.removeEventListener('scroll',paint);window.removeEventListener('pointerdown',outsidePointer);window.removeEventListener('keydown',escapeFavorite);window.removeEventListener('toyxyz-wildcards-changed',load);window.removeEventListener('toyxyz-favorites-changed',favoritePathChanged);return remove?.apply(this,arguments);};
    node.setSize?.([Math.max(node.size[0],520),Math.max(node.size[1],340)]);
    selectTab('wildcards');paint();load();
}
