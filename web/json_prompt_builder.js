import {app} from "../../scripts/app.js";
import {pointInRect,dragRegion,hitRegionHandle,squareGridLines,validateRegions} from "./util/json_builder_geometry.js";

const liveDocks=new Set();
let installedCanvas=null, dockFrame=0, framesLeft=0;
function syncDocks() {
    for(const node of liveDocks)node._jsonBuilderUpdateDock?.();
}
function wakeDocks() {
    framesLeft=8;
    if(!dockFrame)dockFrame=requestAnimationFrame(tickDocks);
}
function tickDocks() {
    dockFrame=0;
    syncDocks();
    if(--framesLeft>0&&liveDocks.size)dockFrame=requestAnimationFrame(tickDocks);
}
function installDockWake() {
    const canvas=app.canvas;
    if(!canvas||installedCanvas===canvas)return;
    installedCanvas=canvas;
    const previous=canvas.onDrawForeground;
    canvas.onDrawForeground=function(...args) {
        const result=previous?.apply(this,args);
        syncDocks();
        wakeDocks();
        return result;
    };
    wakeDocks();
}
window.addEventListener("resize",wakeDocks);

app.registerExtension({
    name:"toyxyz.json_prompter_builder",
    setup(){installDockWake();},
    async beforeRegisterNodeDef(NodeType,data) {
        if(data.name!=="ToyxyzJsonPrompterBuilder") return;
        const created=NodeType.prototype.onNodeCreated;
        NodeType.prototype.onNodeCreated=function() {
            created?.apply(this,arguments);
            const node=this;
            const widget=name=>node.widgets?.find(w=>w.name===name);
            const storage=widget("regions_data"), styleColors=widget("style_colors");
            function hideBackingWidgets() {
                for(const w of [storage,styleColors]) {
                    if(!w) continue;
                    // Both canvas and DOM/Vue layout engines must omit these
                    // serializable data widgets, not merely change their type.
                    w.hidden=true;w.options??={};w.options.hidden=true;
                    if(!window.LiteGraph?.vueNodesMode){w.computeSize=()=>[0,-4];w.draw=()=>{};}
                    if(w.element)w.element.style.display="none";
                    if(w.inputEl)w.inputEl.style.display="none";
                }
            }
            hideBackingWidgets();
            let regions=[], selected=-1, backgroundImage=null, drag=null;
            let revision=0;
            const root=document.createElement("div");
            root.style.cssText="display:flex;flex-direction:column;gap:8px;padding:8px;box-sizing:border-box;width:100%;height:100%;min-height:0;background:#292929;color:#ddd;font:12px sans-serif;overflow:auto;";
            const dock=document.createElement("div");
            dock.style.cssText="position:fixed;z-index:8500;width:570px;height:320px;min-height:320px;box-sizing:border-box;border:1px solid #666;border-radius:8px;box-shadow:0 8px 26px #0009;background:#292929;overflow:hidden;pointer-events:auto;transform-origin:top left;visibility:hidden;";
            const dockTitle=document.createElement("div");
            dockTitle.textContent="Layout editor";
            dockTitle.style.cssText="height:29px;box-sizing:border-box;padding:6px 9px;background:#383838;color:#ddd;font:12px sans-serif;border-bottom:1px solid #555;";
            root.style.height="calc(100% - 29px)";
            dock.append(dockTitle,root);
            document.body.append(dock);
            node._jsonBuilderDock=dock;
            node._jsonBuilderRoot=root;
            liveDocks.add(node);
            const toolbar=document.createElement("div"); toolbar.style.cssText="display:flex;gap:6px;flex-wrap:wrap;align-items:center;flex-shrink:0";
            const viewport=document.createElement("div"); viewport.style.cssText="position:relative;min-height:140px;height:270px;flex-shrink:0;background:#181818;display:flex;align-items:center;justify-content:center;overflow:hidden";
            const canvas=document.createElement("canvas"); canvas.tabIndex=0;
            canvas.style.cssText="display:block;touch-action:none;max-width:100%;max-height:100%;outline:none;cursor:crosshair";
            viewport.append(canvas);
            const panel=document.createElement("div"); panel.style.cssText="display:flex;flex-direction:column;gap:5px;min-width:0";
            const status=document.createElement("small");
            status.style.cssText="display:none";
            root.append(toolbar,viewport,panel,status);
            let lastOutput="";
            let fitDockFrame=0;
            function fitDockToContent() {
                if(fitDockFrame)return;
                fitDockFrame=requestAnimationFrame(()=>{
                    fitDockFrame=0;
                    const scale=app.canvas?.ds?.scale||1;
                    const maxHeight=Math.max(320,Math.min(980,(innerHeight-32)/scale));
                    const wanted=Math.min(maxHeight,Math.max(320,root.scrollHeight+30));
                    if(Math.abs((Number.parseFloat(dock.style.height)||0)-wanted)>1)
                        dock.style.height=`${Math.round(wanted)}px`;
                    dock.style.maxHeight=`${Math.round(maxHeight)}px`;
                    dock.style.visibility="visible";
                });
            }
            const contentObserver=new ResizeObserver(fitDockToContent);
            contentObserver.observe(root);
            contentObserver.observe(toolbar);contentObserver.observe(viewport);contentObserver.observe(panel);
            fitDockToContent();
            function fitNode() {
                const minimum=node.computeSize?.()||[540,500];
                node.setSize([Math.max(540,node.size?.[0]||0,minimum[0]),Math.min(760,Math.max(460,minimum[1]))]);
                node._widgetSlotsDirty=true;
            }
            function updateDock() {
                fitDockToContent();
                const c=app.canvas;
                if(!c?.canvas||!node.pos||!node.graph||c.graph!==node.graph){dock.style.display="none";return;}
                // Nodes 2.0: KJ's pinned-dock pattern. An absolute child of the
                // node inherits the graph transform, so panning never requires
                // screen-coordinate repositioning or a visibility flip.
                if(window.LiteGraph?.vueNodesMode&&node.id!=null){
                    let host=node._jsonBuilderHostEl;
                    if(!host?.isConnected||host.dataset.nodeId!==String(node.id))
                        host=node._jsonBuilderHostEl=document.querySelector(`[data-node-id="${node.id}"]`);
                    if(host){
                        if(dock.parentElement!==host)host.append(dock);
                        const left=-(dock.offsetWidth||570)-8;
                        if(dock.style.position!=="absolute")dock.style.position="absolute";
                        if(dock.style.left!==`${left}px`)dock.style.left=`${left}px`;
                        if(dock.style.top!=="0px")dock.style.top="0px";
                        if(dock.style.transform)dock.style.transform="";
                        if(dock.style.display)dock.style.display="";
                        return;
                    }
                }
                // Legacy canvas: share LiteGraph's own scale and offset instead
                // of clamping a body-fixed panel to the viewport every 80 ms.
                if(dock.parentElement!==document.body)document.body.append(dock);
                const r=c.canvas.getBoundingClientRect(),scale=c.ds?.scale||1,offset=c.ds?.offset||[0,0];
                const x=r.left+(node.pos[0]-(dock.offsetWidth||570)-8+offset[0])*scale;
                const y=r.top+(node.pos[1]-(window.LiteGraph?.NODE_TITLE_HEIGHT||30)+offset[1])*scale;
                const transform=`translate(${x}px,${y}px) scale(${scale})`;
                if(dock.style.position!=="fixed")dock.style.position="fixed";
                if(dock.style.left!=="0px")dock.style.left="0px";
                if(dock.style.top!=="0px")dock.style.top="0px";
                if(dock.style.transform!==transform)dock.style.transform=transform;
                if(dock.style.display)dock.style.display="";
            }
            node._jsonBuilderUpdateDock=updateDock;
            installDockWake();
            const stop=e=>e.stopPropagation();
            for(const event of ["pointerdown","mousedown","wheel","keydown"]) root.addEventListener(event,stop);
            function button(label,title,action,parent=toolbar) {
                const b=document.createElement("button"); b.textContent=label; b.title=title;
                b.style.cssText="border:1px solid #666;border-radius:4px;background:#333;color:#ddd;padding:4px 7px;cursor:pointer";
                b.onclick=action; parent.append(b); return b;
            }
            function notifyDownstream() {
                for(const linkId of node.outputs?.[0]?.links||[]) {
                    const link=node.graph?.links?.[linkId];
                    const target=node.graph?.getNodeById(link?.target_id);
                    target?._jsonBuilderChanged?.();
                }
            }
            function persist() {
                storage.value=JSON.stringify(regions);
                revision++; lastOutput=""; status.textContent="Layout changed. Queue to generate JSON.";
                node.graph?.change?.(); node.setDirtyCanvas?.(true,true); notifyDownstream();
            }
            function dimensions() {
                return [Number(widget("width")?.value)||1024,Number(widget("height")?.value)||1024];
            }
            function draw() {
                const [width,height]=dimensions(), ratio=width/height;
                const availableW=Math.max(100,viewport.clientWidth), availableH=viewport.clientHeight||270;
                let cw=Math.min(availableW,availableH*ratio), ch=cw/ratio;
                const dpr=window.devicePixelRatio||1;
                canvas.width=Math.round(cw*dpr);canvas.height=Math.round(ch*dpr);
                canvas.style.width=cw+"px";canvas.style.height=ch+"px";
                const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.fillStyle="#171717";ctx.fillRect(0,0,cw,ch);
                if(backgroundImage) {ctx.globalAlpha=.35;ctx.drawImage(backgroundImage,0,0,cw,ch);ctx.globalAlpha=1;}
                ctx.strokeStyle="#303030";ctx.lineWidth=1;
                const grid=squareGridLines(cw,ch);
                for(const x of grid.x){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,ch);ctx.stroke();}
                for(const y of grid.y){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(cw,y);ctx.stroke();}
                regions.forEach((r,i)=>{
                    const color=r.palette?.[0]||["#8AB6D6","#B99BD6","#CFB16D"][i%3];
                    const x=r.x*cw,y=r.y*ch,w=r.w*cw,h=r.h*ch;
                    ctx.fillStyle=color;ctx.globalAlpha=.12;ctx.fillRect(x,y,w,h);ctx.globalAlpha=1;
                    ctx.strokeStyle=i===selected?"#FFFFFF":color;ctx.lineWidth=i===selected?2:1;ctx.strokeRect(x,y,w,h);
                    ctx.font="12px sans-serif";ctx.fillStyle=color;ctx.fillText(`${i+1} ${r.type}`,x+3,y+14);
                    if(i===selected) {
                        ctx.fillStyle="#fff";
                        for(const [hx,hy] of [[x,y],[x+w/2,y],[x+w,y],[x,y+h/2],[x+w,y+h/2],[x,y+h],[x+w/2,y+h],[x+w,y+h]])ctx.fillRect(hx-4,hy-4,8,8);
                    }
                });
            }
            function field(label,value,onchange,multiline=false) {
                const wrap=document.createElement("label");wrap.style.cssText="display:flex;flex-direction:column;gap:3px";wrap.textContent=label;
                const input=document.createElement(multiline?"textarea":"input");input.value=value||"";
                input.style.cssText="width:100%;box-sizing:border-box;background:#222;color:#ddd;border:1px solid #555;padding:4px;min-width:0";
                if(multiline){input.rows=2;input.style.resize="vertical";}
                input.oninput=()=>{try{onchange(input.value);input.setCustomValidity("");persist();draw();}catch(e){input.setCustomValidity(e.message);input.reportValidity();}};
                wrap.append(input);panel.append(wrap);return input;
            }
            function renderPanel() {
                panel.replaceChildren(); const r=regions[selected]; if(!r) return;
                const controls=document.createElement("div");controls.style.cssText="display:flex;gap:4px;flex-wrap:wrap";panel.append(controls);
                for(const kind of ["obj","text"]) {const b=button(kind,"Region type",()=>{r.type=kind;persist();render();},controls);if(r.type===kind)b.style.borderColor="#ddd";}
                button("Duplicate","Duplicate the selected region",()=>{regions.splice(selected+1,0,structuredClone(r));selected++;persist();render();},controls);
                button("Delete","Delete the selected region",()=>{regions.splice(selected,1);selected=Math.min(selected,regions.length-1);persist();render();},controls);
                button("Back","Move one layer toward the background",()=>reorder(-1),controls);
                button("Front","Move one layer toward the foreground",()=>reorder(1),controls);
                field("Description",r.desc,v=>r.desc=v,true);
                if(r.type==="text")field("Visible text (preserved exactly)",r.text,v=>r.text=v,true);
            }
            function reorder(delta) {
                const next=selected+delta;if(next<0||next>=regions.length)return;
                [regions[next],regions[selected]]=[regions[selected],regions[next]];selected=next;persist();render();
            }
            function render(){draw();renderPanel();fitDockToContent();}
            button("Add region","Add a region at the center",()=>{regions.push({x:.25,y:.25,w:.5,h:.5,type:"obj",desc:"",text:"",relation:"",palette:[]});selected=regions.length-1;persist();render();});
            button("Reference","Load a local image as a layout guide (not sent to the model)",()=>{
                const picker=document.createElement("input");picker.type="file";picker.accept="image/*";
                picker.onchange=()=>{const file=picker.files?.[0];if(!file)return;const url=URL.createObjectURL(file),img=new Image();img.onload=()=>{backgroundImage=img;draw();URL.revokeObjectURL(url);};img.onerror=()=>URL.revokeObjectURL(url);img.src=url;};picker.click();
            });
            button("Clear reference","Remove the visual guide",()=>{backgroundImage=null;draw();});
            button("Export layout","Download editable builder settings as JSON",()=>{
                const settings=Object.fromEntries((node.widgets||[]).filter(w=>["target_model","width","height","scene","background","style_mode","style","aesthetics","lighting","medium","regions_data","style_colors"].includes(w.name)).map(w=>[w.name,w.value]));
                const blob=new Blob([JSON.stringify({version:1,settings},null,2)],{type:"application/json"});const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="json-prompt-layout.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
            });
            button("Import layout","Load editable builder settings (replaces the current layout)",()=>{
                const picker=document.createElement("input");picker.type="file";picker.accept=".json";
                picker.onchange=async()=>{try {
                    const data=JSON.parse(await picker.files[0].text());
                    if(data.version!==1||!data.settings||!Array.isArray(JSON.parse(data.settings.regions_data)))throw Error("Not a builder layout file.");
                    const s=data.settings;
                    validateRegions(JSON.parse(s.regions_data));
                    if(!["Ideogram 4","Ming Image"].includes(s.target_model)||!["photo","art_style","none"].includes(s.style_mode)||
                        ["width","height"].some(k=>!Number.isInteger(s[k])||s[k]<64||s[k]>16384)||
                        ["scene","background","style","aesthetics","lighting","medium"].some(k=>typeof s[k]!=="string"))throw Error("Invalid builder settings.");
                    const palette=JSON.parse(s.style_colors);
                    if(!Array.isArray(palette)||palette.length>16||palette.some(c=>typeof c!=="string"||!/^#[0-9a-f]{6}$/i.test(c)))throw Error("Invalid style palette.");
                    if(regions.length&&!confirm("Replace the current layout?"))return;
                    for(const name of ["target_model","width","height","scene","background","style_mode","style","aesthetics","lighting","medium","regions_data","style_colors"]){const w=widget(name);if(w)w.value=s[name];}
                    restore();persist();render();
                }catch(e){status.textContent=e.message;}};picker.click();
            });
            button("Copy JSON","Copy the last compiled JSON prompt",async()=>{try{if(lastOutput)await navigator.clipboard.writeText(lastOutput);}catch(e){status.textContent=`Could not copy JSON: ${e.message}`;}});
            button("Clear","Remove all regions",()=>{if(regions.length&&!confirm("Delete all regions?"))return;regions=[];selected=-1;persist();render();});
            function point(e){return pointInRect(canvas.getBoundingClientRect(),e.clientX,e.clientY);}
            function cursorFor(mode){return ({move:"move","resize-n":"ns-resize","resize-s":"ns-resize","resize-e":"ew-resize","resize-w":"ew-resize","resize-nw":"nwse-resize","resize-se":"nwse-resize","resize-ne":"nesw-resize","resize-sw":"nesw-resize"})[mode]||"crosshair";}
            canvas.addEventListener("pointerdown",e=>{
                if(e.button!==0)return;e.preventDefault();canvas.focus();canvas.setPointerCapture(e.pointerId);
                const p=point(e),rect=canvas.getBoundingClientRect();
                const hit=e.ctrlKey||e.metaKey?null:hitRegionHandle(regions,p,rect.width,rect.height,selected);
                if(hit){selected=hit.index;drag={start:p,original:{...regions[hit.index]},mode:hit.mode,index:hit.index};}
                else{regions.push({x:p.x,y:p.y,w:0,h:0,type:"obj",desc:"",text:"",relation:"",palette:[]});selected=regions.length-1;drag={start:p,original:{...regions[selected]},mode:"draw",index:selected};}
                render();
            });
            canvas.addEventListener("pointermove",e=>{const p=point(e),rect=canvas.getBoundingClientRect();if(!drag){const hit=hitRegionHandle(regions,p,rect.width,rect.height,selected);canvas.style.cursor=cursorFor(hit?.mode);return;}Object.assign(regions[drag.index],dragRegion(drag.original,drag.start,p,drag.mode));draw();});
            canvas.addEventListener("pointerleave",()=>{if(!drag)canvas.style.cursor="crosshair";});
            const finish=()=>{if(!drag)return;const r=regions[drag.index];if(r.w<.002||r.h<.002){regions.splice(drag.index,1);selected=-1;}drag=null;persist();render();};
            canvas.addEventListener("pointerup",finish);canvas.addEventListener("pointercancel",finish);
            canvas.addEventListener("keydown",e=>{if((e.key==="Delete"||e.key==="Backspace")&&selected>=0){e.preventDefault();regions.splice(selected,1);selected=Math.min(selected,regions.length-1);persist();render();}});
            function restore(){try{regions=validateRegions(JSON.parse(storage.value||"[]"));}catch(e){regions=[];status.textContent=`Could not restore regions: ${e.message}`;}selected=regions.length?0:-1;}
            const observer=new ResizeObserver(()=>draw());observer.observe(viewport);
            for(const name of ["target_model","width","height","scene","background","style_mode","style","aesthetics","lighting","medium"]){const w=widget(name);if(!w)continue;const cb=w.callback;w.callback=function(){const result=cb?.apply(this,arguments);persist();draw();if(name==="target_model")renderPanel();return result;};}
            const configured=node.onConfigure;node.onConfigure=function(){configured?.apply(this,arguments);hideBackingWidgets();restore();render();fitNode();wakeDocks();};
            const executed=node.onExecuted;node.onExecuted=function(message){executed?.apply(this,arguments);if(message?.text){lastOutput=message.text.join("\n");status.textContent="JSON prompt generated. Use Copy JSON to copy it.";}};
            const removed=node.onRemoved;node.onRemoved=function(){observer.disconnect();contentObserver.disconnect();if(fitDockFrame)cancelAnimationFrame(fitDockFrame);liveDocks.delete(node);delete node._jsonBuilderUpdateDock;dock.remove();removed?.apply(this,arguments);};
            node._jsonBuilderRevision=()=>revision;
            restore();render();fitNode();wakeDocks();
        };
    }
});
