import {formatSelectedTag} from './booru_tag_input.js';

export function insertWikiTag(text,start,end,tag) {
    start=Math.max(0,Math.min(start,text.length));
    end=Math.max(start,Math.min(end,text.length));
    const before=text.slice(0,start),after=text.slice(end),value=formatSelectedTag(tag);
    const prefix=before&&/[,.]$/.test(before)?' '
        :before&&!/[\s,(.;\n]$/.test(before)?', ':'';
    const suffix=after&&!/^[\s,.;)\n]/.test(after)?', '
        :after?'':', ';
    return {value:before+prefix+value+suffix+after,caret:start+prefix.length+value.length};
}

const label=title=>String(title||'').replaceAll('_',' ');
const number=value=>Number(value||0).toLocaleString();
const node=(tag,className='',value='')=>{
    const element=document.createElement(tag);
    if(className)element.className=className;
    if(value)element.textContent=value;
    return element;
};

// Render the small DText subset used for wiki navigation without injecting HTML.
export function renderWikiDText(body,openTitle) {
    const article=node('article','toyxyz-wiki-article');
    let parent=article;
    const inline=(container,text)=>{
        const tokens=/\[\[([^\]]+)\]\]|\[(\/?)((?:b|i|u|s))\]/gi;
        const tags={b:'strong',i:'em',u:'u',s:'s'};
        const stack=[{name:'',element:container}];
        let last=0;
        for(const match of text.matchAll(tokens)){
            if(match.index>last)stack.at(-1).element.append(document.createTextNode(text.slice(last,match.index)));
            if(match[1]!==undefined){
                const [target,display]=match[1].split('|',2);
                const title=target.trim().split('#',1)[0].replace(/\s+/g,'_').toLowerCase();
                const link=node('button','toyxyz-wiki-inline-link',(display||target).trim());
                link.type='button';link.title=`Open ${label(title)}`;
                link.addEventListener('click',()=>openTitle(title));
                stack.at(-1).element.append(link);
            }else if(match[2]){
                if(stack.at(-1).name===match[3].toLowerCase())stack.pop();
                else stack.at(-1).element.append(document.createTextNode(match[0]));
            }else{
                const name=match[3].toLowerCase(),element=node(tags[name]);
                stack.at(-1).element.append(element);stack.push({name,element});
            }
            last=match.index+match[0].length;
        }
        if(last<text.length)stack.at(-1).element.append(document.createTextNode(text.slice(last)));
    };
    for(const raw of String(body||'').split(/\r?\n/)){
        const line=raw.trim();
        if(!line){parent.append(node('div','toyxyz-wiki-gap'));continue;}
        if(line==='[quote]'){const quote=node('blockquote');parent.append(quote);parent=quote;continue;}
        if(line==='[/quote]'){parent=article;continue;}
        if(/^\[\/?expand(?:=.*)?\]$/.test(line))continue;
        const heading=/^h([4-6])(?:#[\w-]+)?\.\s*(.*)$/.exec(line);
        if(heading){const element=node(`h${Number(heading[1])-2}`);inline(element,heading[2]);parent.append(element);continue;}
        const list=/^(\*+)\s*(.*)$/.exec(line);
        if(list){
            const element=node('div','toyxyz-wiki-list-line');
            element.style.marginLeft=`${Math.min(list[1].length-1,5)*14}px`;
            element.append(node('span','toyxyz-wiki-bullet','•'));
            inline(element,list[2]);parent.append(element);continue;
        }
        const paragraph=node('p');inline(paragraph,line);parent.append(paragraph);
    }
    return article;
}

export function installWiki(nodeRef,widget,input,api) {
    const root=widget.element;
    const tabs=root?.querySelector?.('.toyxyz-wildcard-tabs');
    if(!tabs)return;
    const toggle=node('button','','Wiki');toggle.type='button';
    toggle.title='Browse the offline tag wiki';toggle.setAttribute('aria-expanded','false');tabs.append(toggle);

    const panel=node('section','toyxyz-wiki-panel');panel.hidden=true;
    panel.setAttribute('aria-label','Tag Wiki browser');
    const header=node('header','toyxyz-wiki-header');
    header.title='Drag to move the wiki window';
    header.append(node('strong','','Tag Wiki Browser'),node('span','toyxyz-wiki-header-note','Offline · Read only'));
    const close=node('button','','×');close.type='button';close.title='Close wiki';
    close.setAttribute('aria-label','Close wiki');header.append(close);
    const layout=node('div','toyxyz-wiki-layout');
    const sidebar=node('aside','toyxyz-wiki-sidebar');sidebar.setAttribute('aria-label','Wiki categories');
    sidebar.append(node('div','toyxyz-wiki-eyebrow','LIBRARY'),node('h2','','Explore the Wiki'));
    const total=node('p','toyxyz-wiki-total','Loading categories…');sidebar.append(total);
    sidebar.append(node('h3','toyxyz-wiki-section-title','Categories'));
    const categories=node('nav','toyxyz-wiki-categories');sidebar.append(categories);
    sidebar.append(node('h3','toyxyz-wiki-section-title','Quick links'));
    const quickLinks=node('div','toyxyz-wiki-quicklinks');sidebar.append(quickLinks);
    sidebar.append(node('p','toyxyz-wiki-sidebar-help',
        'Categories are tag types. Browse topic lists in Tag Groups. Wiki links open here; Insert adds the current title to your prompt.'));

    const browse=node('main','toyxyz-wiki-browse');
    browse.append(node('div','toyxyz-wiki-eyebrow','EXPLORE THE WIKI'));
    browse.append(node('h2','','Find a tag or topic'));
    browse.append(node('p','toyxyz-wiki-help','Search titles, aliases, and wiki text.'));
    const search=node('input','toyxyz-wiki-search');search.type='search';
    search.placeholder='Search wiki entries';search.autocomplete='off';
    search.setAttribute('aria-label','Search wiki entries');browse.append(search);
    const status=node('div','toyxyz-wiki-status');status.setAttribute('role','status');browse.append(status);
    const results=node('div','toyxyz-wiki-results');browse.append(results);
    const pager=node('div','toyxyz-wiki-pager');
    const previous=node('button','','← Previous'),pageLabel=node('span','','Page 1'),next=node('button','','Next →');
    previous.type=next.type='button';pager.append(previous,pageLabel,next);browse.append(pager);

    const detail=node('section','toyxyz-wiki-detail');detail.setAttribute('aria-label','Wiki entry');
    detail.append(node('p','toyxyz-wiki-placeholder','Select an entry or follow a wiki link.'));
    const resizeGrip=node('div','toyxyz-wiki-resize-grip');
    resizeGrip.title='Drag to resize the wiki window';
    resizeGrip.setAttribute('role','separator');
    resizeGrip.setAttribute('aria-label','Resize wiki window');
    resizeGrip.tabIndex=0;
    layout.append(sidebar,browse,detail);panel.append(header,layout,resizeGrip);document.body.append(panel);

    const downloadDialog=node('div','toyxyz-wiki-download-backdrop');downloadDialog.hidden=true;
    const downloadBox=node('section','toyxyz-wiki-download-box');
    downloadBox.setAttribute('role','dialog');downloadBox.setAttribute('aria-modal','true');
    downloadBox.setAttribute('aria-label','Download Tag Wiki');
    downloadBox.append(node('h2','','Download Tag Wiki'));
    const downloadMessage=node('p','','The offline Tag Wiki database is not installed. Download approximately 272 MiB from Hugging Face? The file will be saved in this custom node folder.');
    const progressRow=node('div','toyxyz-wiki-download-progress');progressRow.hidden=true;
    const progressBar=node('progress');progressBar.max=100;progressBar.value=0;
    progressBar.setAttribute('aria-label','Wiki download progress');
    const progressLabel=node('output','','0%');
    progressRow.append(progressBar,progressLabel);
    const downloadError=node('p','toyxyz-wiki-download-error');downloadError.setAttribute('role','alert');
    const downloadActions=node('div','toyxyz-wiki-download-actions');
    const dismissDownload=node('button','','Cancel'),startDownload=node('button','','Download');
    dismissDownload.type=startDownload.type='button';
    downloadActions.append(dismissDownload,startDownload);
    downloadBox.append(downloadMessage,progressRow,downloadError,downloadActions);
    downloadDialog.append(downloadBox);document.body.append(downloadDialog);

    const state={open:false,disposed:false,category:'',query:'',page:1,
        searchToken:0,detailToken:0,history:[],frame:0,lastRect:'',timer:0,
        manualPosition:false,dragging:null,resizing:null,opening:false,downloading:false,
        progressTimer:0,progressPending:false};
    const readJSON=async(url,options)=>{
        const response=await api.fetchApi(url,options);
        const data=await response.json().catch(()=>({}));
        if(!response.ok)throw new Error(response.status===404&&url.includes('/wiki/')
            ?'Restart ComfyUI to activate the Tag Wiki Browser.'
            :data.error||`Wiki request failed (HTTP ${response.status}).`);
        return data;
    };
    const place=()=>{
        if(!state.open||state.disposed)return;
        const rect=root.getBoundingClientRect(),right=window.innerWidth-rect.right-8,left=rect.left-8;
        const available=Math.max(right,left),width=Math.min(1050,Math.max(600,available-8));
        panel.style.width=`${Math.min(width,window.innerWidth-16)}px`;
        panel.style.left=`${right>=left&&right>=width?rect.right+8:
            left>=width?rect.left-width-8:Math.max(8,(window.innerWidth-width)/2)}px`;
        panel.style.top=`${Math.max(8,Math.min(rect.top,window.innerHeight-panel.offsetHeight-8))}px`;
    };
    const clampPosition=(left,top)=>{
        const width=panel.offsetWidth||Number.parseFloat(panel.style.width)||0;
        const height=panel.offsetHeight||0;
        panel.style.left=`${Math.max(8,Math.min(left,Math.max(8,window.innerWidth-width-8)))}px`;
        panel.style.top=`${Math.max(8,Math.min(top,Math.max(8,window.innerHeight-height-8)))}px`;
    };
    const fitManualPanel=()=>{
        const maxWidth=Math.max(100,window.innerWidth-16);
        const maxHeight=Math.max(100,window.innerHeight-16);
        if(panel.offsetWidth>maxWidth)panel.style.width=`${maxWidth}px`;
        if(panel.offsetHeight>maxHeight)panel.style.height=`${maxHeight}px`;
        clampPosition(Number.parseFloat(panel.style.left)||8,Number.parseFloat(panel.style.top)||8);
    };
    const resizeTo=(width,height)=>{
        const left=Number.parseFloat(panel.style.left)||8,top=Number.parseFloat(panel.style.top)||8;
        const maxWidth=Math.max(100,window.innerWidth-left-8);
        const maxHeight=Math.max(100,window.innerHeight-top-8);
        const minWidth=Math.min(560,maxWidth),minHeight=Math.min(320,maxHeight);
        panel.style.width=`${Math.max(minWidth,Math.min(width,maxWidth))}px`;
        panel.style.height=`${Math.max(minHeight,Math.min(height,maxHeight))}px`;
        state.manualPosition=true;
    };
    const watch=()=>{
        if(!state.open||state.disposed){state.frame=0;return;}
        const rect=root.getBoundingClientRect();
        const key=`${rect.left},${rect.top},${rect.width},${rect.height},${window.innerWidth},${window.innerHeight}`;
        if(key!==state.lastRect){
            state.lastRect=key;
            if(state.manualPosition)fitManualPanel();
            else place();
        }
        state.frame=requestAnimationFrame(watch);
    };
    header.addEventListener('pointerdown',event=>{
        if(event.button!==0||event.target===close||close.contains?.(event.target))return;
        const rect=panel.getBoundingClientRect();
        state.dragging={id:event.pointerId,offsetX:event.clientX-rect.left,offsetY:event.clientY-rect.top};
        header.setPointerCapture?.(event.pointerId);
        event.preventDefault();event.stopPropagation();
    });
    header.addEventListener('pointermove',event=>{
        if(!state.dragging||event.pointerId!==state.dragging.id)return;
        state.manualPosition=true;
        clampPosition(event.clientX-state.dragging.offsetX,event.clientY-state.dragging.offsetY);
        event.preventDefault();event.stopPropagation();
    });
    const endDrag=event=>{
        if(!state.dragging||event.pointerId!==state.dragging.id)return;
        state.dragging=null;
        if(header.hasPointerCapture?.(event.pointerId))header.releasePointerCapture(event.pointerId);
        event.stopPropagation();
    };
    header.addEventListener('pointerup',endDrag);
    header.addEventListener('pointercancel',endDrag);
    resizeGrip.addEventListener('pointerdown',event=>{
        if(event.button!==0)return;
        state.resizing={id:event.pointerId,x:event.clientX,y:event.clientY,
            width:panel.offsetWidth,height:panel.offsetHeight};
        resizeGrip.setPointerCapture?.(event.pointerId);
        event.preventDefault();event.stopPropagation();
    });
    resizeGrip.addEventListener('pointermove',event=>{
        if(!state.resizing||event.pointerId!==state.resizing.id)return;
        resizeTo(state.resizing.width+event.clientX-state.resizing.x,
            state.resizing.height+event.clientY-state.resizing.y);
        event.preventDefault();event.stopPropagation();
    });
    const endResize=event=>{
        if(!state.resizing||event.pointerId!==state.resizing.id)return;
        state.resizing=null;
        if(resizeGrip.hasPointerCapture?.(event.pointerId))resizeGrip.releasePointerCapture(event.pointerId);
        event.stopPropagation();
    };
    resizeGrip.addEventListener('pointerup',endResize);
    resizeGrip.addEventListener('pointercancel',endResize);
    resizeGrip.addEventListener('keydown',event=>{
        const step=event.shiftKey?5:20;
        if(!['ArrowRight','ArrowLeft','ArrowDown','ArrowUp'].includes(event.key))return;
        resizeTo(panel.offsetWidth+(event.key==='ArrowRight'?step:event.key==='ArrowLeft'?-step:0),
            panel.offsetHeight+(event.key==='ArrowDown'?step:event.key==='ArrowUp'?-step:0));
        event.preventDefault();event.stopPropagation();
    });
    const insert=title=>{
        const result=insertWikiTag(input.value,input.selectionStart,input.selectionEnd,title);
        input.value=result.value;input.focus();input.setSelectionRange(result.caret,result.caret);
        input.dispatchEvent(new Event('input',{bubbles:true}));
        widget.value=input.value;widget.callback?.(widget.value);nodeRef.setDirtyCanvas?.(true,true);
    };
    let openPage;
    const addCategory=(name,count)=>{
        const button=node('button','toyxyz-wiki-category');button.type='button';
        button.append(node('span','toyxyz-wiki-category-icon',name?name[0]:'∞'),
            node('span','toyxyz-wiki-category-name',name||'All entries'),
            node('span','toyxyz-wiki-category-count',number(count)));
        button.title=name||'All entries';
        button.addEventListener('click',()=>{
            state.category=name;state.page=1;
            for(const item of categories.children){
                const active=item===button;item.classList.toggle('toyxyz-wiki-active',active);
                item.setAttribute('aria-pressed',String(active));
            }
            loadResults();
        });
        button.classList.toggle('toyxyz-wiki-active',name===state.category);
        button.setAttribute('aria-pressed',String(name===state.category));categories.append(button);
    };
    const loadCategories=async()=>{
        try{
            const data=await readJSON('/toyxyz/booru-tags/wiki/categories');
            if(state.disposed)return;
            total.textContent=`${number(data.total)} entries`;
            categories.replaceChildren();addCategory('',data.total);
            for(const item of data.categories||[])addCategory(item.name,item.count);
        }catch(error){if(!state.disposed)total.textContent=error.message;}
    };
    const loadResults=async()=>{
        const token=++state.searchToken;
        status.textContent=state.query?`Results for “${state.query}”`:
            state.category?`${state.category} entries`:'Popular entries';
        results.replaceChildren(node('p','toyxyz-wiki-placeholder','Loading entries…'));
        const params=new URLSearchParams({q:state.query,category:state.category,page:String(state.page)});
        try{
            const data=await readJSON(`/toyxyz/booru-tags/wiki/search?${params}`);
            if(token!==state.searchToken||state.disposed)return;
            results.replaceChildren();
            if(!data.items?.length)results.append(node('p','toyxyz-wiki-placeholder','No matching entries. Try another search or category.'));
            for(const item of data.items||[]){
                const card=node('button','toyxyz-wiki-result');card.type='button';
                card.wikiPageId=item.id;
                const row=node('span','toyxyz-wiki-result-top');
                row.append(node('strong','',label(item.title)),node('small','',item.category||'Wiki-only'));
                card.append(row,node('span','toyxyz-wiki-result-snippet',item.snippet||'No description available.'),
                    node('span','toyxyz-wiki-result-meta',item.other_names?.length
                        ?`Aliases: ${item.other_names.join(', ')}`:`${number(item.post_count)} posts`));
                card.title=`Read ${label(item.title)}`;
                card.addEventListener('click',()=>openPage({id:item.id}));results.append(card);
            }
            pageLabel.textContent=`Page ${data.page}`;previous.disabled=data.page<=1;next.disabled=!data.has_more;
        }catch(error){if(token===state.searchToken&&!state.disposed)results.replaceChildren(
            node('p','toyxyz-wiki-placeholder',error.message));}
    };
    const renderDetail=data=>{
        for(const card of results.children){
            card.classList?.toggle('toyxyz-wiki-selected',card.wikiPageId===data.id);
        }
        detail.replaceChildren();
        const top=node('div','toyxyz-wiki-detail-top');
        const back=node('button','','← Back');back.type='button';back.disabled=!state.history.length;
        back.addEventListener('click',()=>{
            const previousPage=state.history.pop();if(previousPage)openPage({...previousPage,remember:false});
        });
        top.append(back,node('span','',`WIKI ENTRY #${data.id}`));detail.append(top);
        detail.append(node('h2','toyxyz-wiki-detail-title',label(data.title)));
        detail.append(node('p','toyxyz-wiki-aliases',data.other_names?.length
            ?`Aliases: ${data.other_names.join(' · ')}`:'No aliases listed.'));
        const facts=node('div','toyxyz-wiki-facts');
        facts.append(node('span','',`Type: ${data.category||'Wiki-only'}`),
            node('span','',`Posts: ${number(data.post_count)}`));
        if(data.updated_at)facts.append(node('span','',`Updated: ${data.updated_at.slice(0,10)}`));
        detail.append(facts);
        const actions=node('div','toyxyz-wiki-actions');
        if(data.category&&data.category.toLowerCase()!=='wiki-only'){
            const add=node('button','toyxyz-wiki-insert','Insert');add.type='button';
            add.title='Insert this tag at the prompt cursor';add.addEventListener('click',()=>insert(data.title));
            actions.append(add);
        }
        if(typeof data.source_url==='string'&&data.source_url.startsWith('https://danbooru.donmai.us/wiki_pages/')){
            const source=node('a','','Source ↗');source.href=data.source_url;source.target='_blank';
            source.rel='noopener noreferrer';actions.append(source);
        }
        if(actions.children.length)detail.append(actions);
        detail.append(renderWikiDText(data.body,title=>openPage({title})));
        detail.scrollTop=0;
    };
    openPage=async({id,title,remember=true})=>{
        const token=++state.detailToken;
        const params=new URLSearchParams(id?{id:String(id)}:{title});
        detail.replaceChildren(node('p','toyxyz-wiki-placeholder','Loading wiki entry…'));
        try{
            const data=await readJSON(`/toyxyz/booru-tags/wiki/page?${params}`);
            if(token!==state.detailToken||state.disposed)return;
            if(remember&&state.currentPage)state.history.push(state.currentPage);
            state.currentPage={id:data.id};renderDetail(data);
        }catch(error){if(token===state.detailToken&&!state.disposed)detail.replaceChildren(
            node('p','toyxyz-wiki-placeholder',error.message));}
    };
    for(const [title,name] of [['help:home','Wiki Help'],['tag_groups','Tag Group Index'],
        ['help:glossary','Glossary']]){
        const button=node('button','',name+' ↗');button.type='button';
        button.addEventListener('click',()=>openPage({title}));quickLinks.append(button);
    }
    const setOpen=value=>{
        state.open=value;panel.hidden=!value;
        toggle.classList.toggle('toyxyz-tab-active',value);
        toggle.setAttribute('aria-expanded',String(value));
        if(value){
            state.lastRect='';
            if(state.manualPosition)fitManualPanel();
            else place();
            if(!state.frame)state.frame=requestAnimationFrame(watch);
            loadCategories();loadResults();
            openPage(state.currentPage?{id:state.currentPage.id,remember:false}:
                {title:'help:home',remember:false});
            search.focus();
        }else{
            if(state.dragging&&header.hasPointerCapture?.(state.dragging.id)){
                header.releasePointerCapture(state.dragging.id);
            }
            state.dragging=null;
            if(state.resizing&&resizeGrip.hasPointerCapture?.(state.resizing.id)){
                resizeGrip.releasePointerCapture(state.resizing.id);
            }
            state.resizing=null;
            clearTimeout(state.timer);cancelAnimationFrame(state.frame);state.frame=0;
            ++state.searchToken;++state.detailToken;
        }
    };
    const showDownload=()=>{
        downloadDialog.hidden=false;
        downloadMessage.textContent='The offline Tag Wiki database is not installed. Download approximately 272 MiB from Hugging Face? The file will be saved in this custom node folder.';
        downloadError.textContent='';
        dismissDownload.disabled=false;startDownload.disabled=false;
        startDownload.textContent='Download';
        progressBar.value=0;progressLabel.textContent='0%';progressRow.hidden=true;
    };
    dismissDownload.addEventListener('click',()=>{if(!state.downloading)downloadDialog.hidden=true;});
    startDownload.addEventListener('click',async()=>{
        if(state.downloading||state.disposed)return;
        state.downloading=true;dismissDownload.disabled=true;startDownload.disabled=true;
        downloadError.textContent='';
        downloadMessage.textContent='Downloading and verifying the Tag Wiki database. Please wait; the wiki will open when ready.';
        progressBar.value=0;progressLabel.textContent='0%';progressRow.hidden=false;
        const refreshProgress=async()=>{
            if(state.progressPending||!state.downloading||state.disposed)return;
            state.progressPending=true;
            try{
                const result=await readJSON('/toyxyz/booru-tags/wiki/status');
                if(!state.downloading||state.disposed)return;
                const percent=Math.max(0,Math.min(99,Number(result.percent)||0));
                progressBar.value=percent;progressLabel.textContent=`${percent}%`;
            }catch{ /* The download request reports a final error if status is unavailable. */ }
            finally{state.progressPending=false;}
        };
        state.progressTimer=setInterval(refreshProgress,250);
        try{
            await readJSON('/toyxyz/booru-tags/wiki/download', {method:'POST'});
            if(state.disposed)return;
            progressBar.value=100;progressLabel.textContent='100%';
            downloadDialog.hidden=true;setOpen(true);
        }catch(error){
            if(!state.disposed){
                downloadMessage.textContent='The download did not complete. Your prompt remains available.';
                downloadError.textContent=error.message;
                dismissDownload.disabled=false;startDownload.disabled=false;
                startDownload.textContent='Retry';
            }
        }finally{
            state.downloading=false;
            clearInterval(state.progressTimer);state.progressTimer=0;
        }
    });
    const openWiki=async()=>{
        if(state.open){setOpen(false);return;}
        if(state.opening||state.downloading)return;
        state.opening=true;toggle.disabled=true;
        try{
            const result=await readJSON('/toyxyz/booru-tags/wiki/status');
            if(state.disposed)return;
            if(result.ready)setOpen(true);
            else showDownload();
        }catch(error){
            if(!state.disposed){showDownload();downloadError.textContent=error.message;}
        }finally{state.opening=false;toggle.disabled=false;}
    };
    search.addEventListener('input',()=>{
        clearTimeout(state.timer);state.timer=setTimeout(()=>{
            state.query=search.value.trim();state.page=1;loadResults();
        },220);
    });
    search.addEventListener('keydown',event=>{
        if(event.key==='Enter'){
            event.preventDefault();clearTimeout(state.timer);
            state.query=search.value.trim();state.page=1;loadResults();
        }
        if(event.key==='Escape'){event.preventDefault();setOpen(false);}
    });
    previous.addEventListener('click',()=>{if(state.page>1){--state.page;loadResults();}});
    next.addEventListener('click',()=>{++state.page;loadResults();});
    toggle.addEventListener('click',openWiki);close.addEventListener('click',()=>setOpen(false));
    const removed=nodeRef.onRemoved;
    nodeRef.onRemoved=function(){state.disposed=true;clearInterval(state.progressTimer);setOpen(false);panel.remove();downloadDialog.remove();return removed?.apply(this,arguments);};
}
