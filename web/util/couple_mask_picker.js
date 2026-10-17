// Native options cannot reliably contain swatches. Keep the select as the
// value/change control and show an accessible, zoom-aware colored listbox.
export class MaskLayerPicker {
    constructor(select,parent){
        this.select=select;select.style.display='none';
        this.button=document.createElement('button');this.button.type='button';this.button.setAttribute('role','combobox');this.button.setAttribute('aria-label','Layer Mask');this.button.setAttribute('aria-haspopup','listbox');this.button.setAttribute('aria-expanded','false');
        this.button.style.cssText='flex:1;min-width:0;display:flex;align-items:center;gap:7px;text-align:left;height:32px';
        this.swatch=this.makeSwatch();this.text=document.createElement('span');this.text.style.cssText='flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
        const arrow=document.createElement('span');arrow.textContent='⌄';arrow.setAttribute('aria-hidden','true');this.button.append(this.swatch,this.text,arrow);parent.append(this.button);
        this.button.onclick=()=>this.open?this.close():this.show();
        this.button.addEventListener('keydown',e=>this.keydown(e));
        this.outside=e=>{if(this.open&&!this.button.contains(e.target)&&!this.popup.contains(e.target))this.close(false);};
        this.reposition=()=>{if(this.open)this.position();};
    }
    makeSwatch(){const swatch=document.createElement('span');swatch.setAttribute('aria-hidden','true');swatch.style.cssText='display:block;width:13px;height:13px;flex:0 0 13px;border:1px solid #aaa;box-sizing:border-box';return swatch;}
    sync(colors){
        this.colors=colors;this.items=[...this.select.options].map(option=>({id:option.value,label:option.textContent,title:option.title||option.textContent,disabled:option.disabled,color:colors.get(option.value)}));
        const item=this.items.find(item=>item.id===this.select.value)??this.items[0];this.button.disabled=this.select.disabled;this.text.textContent=item?.label??'None';this.swatch.style.backgroundColor=item?.color??'transparent';this.swatch.style.visibility=item?.color?'visible':'hidden';this.button.title=item?.color?`${item.title} · ${item.color}`:this.select.title;
        if(this.button.disabled)this.close(false);else if(this.open){const active=this.items[this.active]?.id;this.build();this.active=this.items.findIndex(item=>item.id===active&&!item.disabled);if(this.active<0)this.active=this.items.findIndex(item=>item.id===this.select.value&&!item.disabled);this.position();this.highlight();}
    }
    show(){
        if(this.open||this.button.disabled)return;this.open=true;this.button.setAttribute('aria-expanded','true');
        this.popup=document.createElement('div');this.popup.id=`painter-mask-${crypto.randomUUID()}`;this.popup.setAttribute('role','listbox');this.popup.setAttribute('aria-label','Mask layers');this.popup.tabIndex=-1;this.button.setAttribute('aria-controls',this.popup.id);
        this.popup.style.cssText='position:fixed;z-index:2147483000;box-sizing:border-box;background:#303030;color:#ddd;border:1px solid #777;border-radius:5px;padding:4px;overflow:auto;overscroll-behavior:contain;box-shadow:0 6px 20px #0009;font-family:sans-serif';
        for(const type of ['pointerdown','pointerup','pointermove','click','wheel','dblclick'])this.popup.addEventListener(type,e=>e.stopPropagation());
        this.popup.addEventListener('keydown',e=>this.keydown(e));this.popup.addEventListener('blur',()=>{queueMicrotask(()=>{if(this.open&&document.activeElement!==this.button&&!this.popup.contains(document.activeElement))this.close(false);});});
        document.body.append(this.popup);this.build();this.active=this.items.findIndex(item=>item.id===this.select.value&&!item.disabled);this.position();this.highlight();this.popup.focus({preventScroll:true});
        document.addEventListener('pointerdown',this.outside,true);window.addEventListener('resize',this.reposition);window.addEventListener('scroll',this.reposition,true);
    }
    build(){
        this.popup.replaceChildren();this.rows=this.items.map((item,index)=>{
            const row=document.createElement('div');row.id=`${this.popup.id}-${index}`;row.setAttribute('role','option');row.setAttribute('aria-selected',String(item.id===this.select.value));row.setAttribute('aria-disabled',String(item.disabled));row.dataset.value=item.id;row.title=item.color?`${item.title} · ${item.color}`:item.title;
            row.style.cssText=`display:flex;align-items:center;gap:8px;padding:7px 8px;border-radius:3px;cursor:${item.disabled?'not-allowed':'pointer'};opacity:${item.disabled?'.4':'1'}`;
            const swatch=this.makeSwatch();swatch.style.backgroundColor=item.color??'transparent';swatch.style.visibility=item.color?'visible':'hidden';const text=document.createElement('span');text.textContent=item.label;text.style.cssText='flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';row.append(swatch,text);
            row.onpointermove=()=>{if(!item.disabled&&this.active!==index){this.active=index;this.highlight();}};
            row.onclick=()=>this.choose(index);this.popup.append(row);return row;
        });
    }
    position(){
        const rect=this.button.getBoundingClientRect(),scale=Math.max(.6,rect.height/(this.button.offsetHeight||32)),margin=8;
        const width=Math.min(Math.max(rect.width,Math.min(480*scale,640)),Math.max(0,window.innerWidth-margin*2));
        const below=window.innerHeight-rect.bottom-margin,above=rect.top-margin,desired=Math.min(320*scale,window.innerHeight-margin*2),flip=below<Math.min(desired,150*scale)&&above>below,height=Math.max(0,Math.min(desired,flip?above:below));
        this.popup.style.width=`${width}px`;this.popup.style.maxHeight=`${height}px`;this.popup.style.left=`${Math.max(margin,Math.min(rect.left,window.innerWidth-width-margin))}px`;this.popup.style.fontSize=`${Math.max(10,13*scale)}px`;
        for(const row of this.rows){row.style.padding=`${7*scale}px ${8*scale}px`;row.style.gap=`${8*scale}px`;Object.assign(row.firstChild.style,{width:`${13*scale}px`,height:`${13*scale}px`,flexBasis:`${13*scale}px`});}
        this.popup.style.top=`${flip?Math.max(margin,rect.top-this.popup.getBoundingClientRect().height):rect.bottom}px`;
    }
    highlight(){
        this.rows?.forEach((row,index)=>{row.style.backgroundColor=index===this.active?'#34566e':'';});
        const row=this.rows?.[this.active];if(row){this.popup.setAttribute('aria-activedescendant',row.id);this.button.setAttribute('aria-activedescendant',row.id);const top=row.offsetTop,bottom=top+row.offsetHeight;if(top<this.popup.scrollTop)this.popup.scrollTop=top;else if(bottom>this.popup.scrollTop+this.popup.clientHeight)this.popup.scrollTop=bottom-this.popup.clientHeight;}
    }
    choose(index){
        const item=this.items[index];if(!item||item.disabled)return;this.close();this.select.value=item.id;this.select.dispatchEvent(new Event('change',{bubbles:true}));
    }
    keydown(event){
        if(event.ctrlKey||event.metaKey||event.altKey||event.isComposing||event.keyCode===229)return;
        if(!['ArrowDown','ArrowUp','Home','End','Enter',' ','Escape','Tab'].includes(event.key))return;
        if(event.key==='Tab'){this.close();return;}
        event.preventDefault();event.stopPropagation();
        if(event.key==='Escape'){this.close();return;}
        if(!this.open){this.show();return;}
        if(['Enter',' '].includes(event.key)){this.choose(this.active);return;}
        const enabled=this.items.map((item,index)=>item.disabled?-1:index).filter(index=>index>=0),at=enabled.indexOf(this.active);
        this.active=event.key==='Home'?enabled[0]:event.key==='End'?enabled.at(-1):enabled[Math.max(0,Math.min(enabled.length-1,at+(event.key==='ArrowDown'?1:-1)))];this.highlight();
    }
    close(focus=true){
        if(!this.open)return;this.open=false;this.popup.remove();this.popup=null;this.rows=null;this.button.setAttribute('aria-expanded','false');this.button.removeAttribute('aria-controls');this.button.removeAttribute('aria-activedescendant');
        document.removeEventListener('pointerdown',this.outside,true);window.removeEventListener('resize',this.reposition);window.removeEventListener('scroll',this.reposition,true);if(focus&&!this.button.disabled)this.button.focus({preventScroll:true});
    }
    dispose(){this.close(false);}
}
