import { drawCameraScene, cameraPosition, projectPoint, previewViewport } from './booru_camera_scene.js';
export const presetFields = ['vertical_view','horizontal_view','zoom','camera_angle','perspective_depth','depth_of_field','blurry_background','blurry_foreground','bokeh','soft_focus','chromatic_aberration','lens_flare','motion_blur'];
const labels = ['Vertical view','Horizontal view','Framing','Angle','Perspective / Depth','Depth of field','Blurry background','Blurry foreground','Bokeh','Soft focus','Chromatic aberration','Lens flare','Motion blur'];
const focusFields=presetFields.slice(5);
export function selectionPreview(settings) {
    return {
        x: ({Front:0,Behind:1,Side:.5,'Left side':.5,'Right side':-.5,'Front-left 45°':.25,'Front-right 45°':-.25,'Rear-left 45°':.75,'Rear-right 45°':-.75})[settings.horizontal_view] ?? 0,
        y: ({Above:.45,Below:-.45})[settings.vertical_view] ?? 0,
        z: ({'Very wide shot':1,'Wide shot':.8,'Full body':.55,'Cowboy shot':.3,'Upper body':0,Portrait:-.4,'Close-up':-.9,'Lower body':0})[settings.zoom] ?? 0,
        roll: ({'Dutch angle':20/45,Sideways:2,'Upside-down':4})[settings.camera_angle] ?? 0,
        targetHeight: ({Portrait:1.6,'Close-up':1.65,'Upper body':1.25,'Lower body':.45,'Cowboy shot':1.0})[settings.zoom] ?? .9,
    };
}
export function snapPresetAt(point,settings,view) {
    let best=null;
    const horizontal=['Front','Left side','Right side','Behind','Front-left 45°','Front-right 45°','Rear-left 45°','Rear-right 45°'];
    if(settings.horizontal_view==='Side')horizontal[1]='Side';
    for(const horizontal_view of horizontal)
        for(const vertical_view of ['None','Above','Below']) {
            const candidate={...settings,horizontal_view,vertical_view};
            const state=selectionPreview(candidate),position=cameraPosition(state.x,state.y,state.z);
            position[1]+=state.targetHeight-.9;
            const projected=projectPoint(position,view.yaw,view.pitch);
            const score=Math.hypot(projected[0]-point[0],projected[1]-point[1])+
                (horizontal_view===settings.horizontal_view?0:.001);
            if(!best||score<best.score)best={horizontal_view,vertical_view,score};
        }
    return best;
}
export const clampWeight = value => Number.isFinite(Number(value)) ? Math.max(0,Math.min(10,Number(value))) : 1;
const wrap=value=>((value+1)%2+2)%2-1;
export function magneticSnap(axis,value,force=false) {
    const stops=axis==='x'?[[0,'Front'],[.25,'Front-left 45°'],[-.25,'Front-right 45°'],[.5,'Left side'],[-.5,'Right side'],[.75,'Rear-left 45°'],[-.75,'Rear-right 45°'],[1,'Behind']]:
        [[0,'None'],[.45,'Above'],[-.45,'Below']];
    let best=null;
    for(const [position,label] of stops) {
        const delta=axis==='x'?wrap(position-value):position-value;
        if(!best||Math.abs(delta)<Math.abs(best.delta))best={position,label,delta};
    }
    return force||Math.abs(best.delta)<=.055?{value:axis==='x'?wrap(best.position):best.position,label:best.label}:{value,label:null};
}
export function pickOrbit(point,state,view,axis,near=null) {
    let best=null;
    for(let i=0;i<=400;i++) {
        const parameter=-1+i/200;
        if(near!==null && Math.abs(axis==='x'?wrap(parameter-near):parameter-near)>.2)continue;
        const candidate={...state,[axis]:parameter},p=cameraPosition(candidate.x,candidate.y,candidate.z);
        p[1]+=state.targetHeight-.9;
        const q=projectPoint(p,view.yaw,view.pitch),distance=Math.hypot(q[0]-point[0],q[1]-point[1]);
        if(!best||distance<best.distance)best={parameter,distance};
    }
    return best;
}
export function dragWeight(start,delta,shift=false) {
    return Math.round(clampWeight(start+delta*(shift ? .01 : .05))*100)/100;
}

export function installPresetList(node,data) {
    const originals=Object.fromEntries(node.widgets.map(w=>[w.name,w]));
    if(!originals.camera_angle || !node.addDOMWidget)return;
    for(const widget of node.widgets) {
        widget.type='hidden';widget.computeSize=()=>[0,-4];widget.draw=()=>{};
    }
    if(originals.panel_enabled)originals.panel_enabled.value=false;
    const root=document.createElement('div');
    root.style.cssText='width:100%;height:100%;box-sizing:border-box;padding:10px;display:flex;flex-direction:column;gap:8px;overflow:auto;background:#252b33;border-radius:8px;color:#e4eaf0;font:13px sans-serif;';
    const resolvedRandom={};
    const settings=()=>Object.fromEntries(presetFields.map(field=>[field,
        originals[field].value==='Random' ? resolvedRandom[field]??'None' : originals[field].value]));
    node._booruMigrateRandom=()=>{
        if(originals.random_camera?.value===true) {
            for(const field of presetFields)originals[field].value='Random';
            originals.random_camera.value=false;
        }
        if(focusFields.some(field=>originals[field].value==='Random')) {
            if(originals.focus_blur_random)originals.focus_blur_random.value=true;
            for(const field of focusFields)if(originals[field].value==='Random')originals[field].value='None';
        }
    };
    node._booruMigrateRandom();
    const canvas=document.createElement('canvas');
    canvas.width=640;canvas.height=300;
    const viewport=document.createElement('div');
    viewport.style.cssText='position:relative;width:100%;height:220px;min-height:180px;flex:1 0 220px;min-width:0;';
    canvas.style.cssText='display:block;width:100%;height:100%;background:#171e26;border-radius:8px;touch-action:none;cursor:grab;';
    canvas.title='Conceptual camera proxy, not a generated-image guarantee. Connected user camera text and face/torso screen direction are not geometrically resolved. Horizontal Left/Right output includes subject facing. Fixed viewpoint. Drag a circular gizmo to move one axis. Scroll changes framing.';
    const usage=document.createElement('div');
    usage.textContent='Click + drag a ring: orbit\nRelease: snap to nearest preset\nScroll: framing (down = farther)';
    usage.style.cssText='position:absolute;left:8px;bottom:8px;max-width:calc(100% - 16px);box-sizing:border-box;padding:5px 7px;border-radius:5px;background:rgba(15,21,28,.78);color:#acbfd2;font:11px/1.4 sans-serif;white-space:pre-line;pointer-events:none;user-select:none;';
    viewport.append(canvas,usage);root.append(viewport);
    const view={yaw:0,pitch:Math.PI/4};
    let previewState=null;
    let hoverAxis=null;
    const draw=()=>{
        const ctx=canvas.getContext?.('2d');if(!ctx)return;
        const width=canvas.clientWidth||640,height=canvas.clientHeight||300,dpr=globalThis.devicePixelRatio||1;
        const w=Math.round(width*dpr),h=Math.round(height*dpr);
        if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
        ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);
        const fit=previewViewport(width,height),scale=fit.scale;
        ctx.setTransform(scale*dpr,0,0,scale*dpr,fit.offsetX*dpr,fit.offsetY*dpr);
        drawCameraScene(ctx,{...(previewState||selectionPreview(settings())),hoverAxis},{...view,...fit});
    };
    const pointerPoint=event=>{
        const rect=canvas.getBoundingClientRect(),fit=previewViewport(rect.width,rect.height);
        return [(event.clientX-rect.left-fit.offsetX)/fit.scale,
            (event.clientY-rect.top-fit.offsetY)/fit.scale];
    };
    const changeSelections=values=>{
        for(const [field,value] of Object.entries(values)) {
            if(originals[field].value===value)continue;
            originals[field].value=value;originals[field].callback?.(value);
        }
        syncs.forEach(sync=>sync());node.setDirtyCanvas?.(true,true);
    };
    let inspect=null;
    canvas.addEventListener('pointerdown',event=>{
        if(event.button!==0)return;event.preventDefault();event.stopPropagation();
        const state=previewState||selectionPreview(settings()),pointer=pointerPoint(event);
        const horizontal=pickOrbit(pointer,state,view,'x'),vertical=pickOrbit(pointer,state,view,'y');
        const axis=horizontal.distance<=vertical.distance+.1?'x':'y',hit=axis==='x'?horizontal:vertical;
        if(hit.distance>12)return;
        previewState={...state,activeAxis:axis};
        inspect={axis,pointerParameter:hit.parameter,raw:state[axis]};
        canvas.style.cursor='grabbing';
        canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove',event=>{
        if(!inspect) {
            const state=previewState||selectionPreview(settings()),point=pointerPoint(event);
            const horizontal=pickOrbit(point,state,view,'x'),vertical=pickOrbit(point,state,view,'y');
            const axis=horizontal.distance<=vertical.distance+.1?'x':'y';
            const next=Math.min(horizontal.distance,vertical.distance)<=12?axis:null;
            if(next!==hoverAxis){hoverAxis=next;canvas.style.cursor=next?'grab':'default';draw();}
            return;
        }
        event.stopPropagation();
        const axis=inspect.axis,hit=pickOrbit(pointerPoint(event),previewState,view,axis,inspect.pointerParameter);
        const delta=axis==='x'?wrap(hit.parameter-inspect.pointerParameter):hit.parameter-inspect.pointerParameter;
        inspect.raw=axis==='x'?wrap(inspect.raw+delta):Math.max(-1,Math.min(1,inspect.raw+delta));
        inspect.pointerParameter=hit.parameter;
        const snap=magneticSnap(axis,inspect.raw);previewState[axis]=snap.value;
        if(snap.label) {
            changeSelections({[axis==='x'?'horizontal_view':'vertical_view']:snap.label});
            note.textContent='Snapped · Output follows the selected preset';
            note.title='Preset snapped. Output follows the lists. Drag a circular gizmo to move one axis.';
        } else {
            note.textContent='Between presets · Preview only';
            note.title='Output remains at the last selected preset. Move near a preset to snap.';
        }
        draw();
    });
    canvas.addEventListener('pointerup',()=>{
        if(inspect&&previewState) {
            const axis=inspect.axis,snap=magneticSnap(axis,inspect.raw,true);
            previewState[axis]=snap.value;
            changeSelections({[axis==='x'?'horizontal_view':'vertical_view']:snap.label});
            note.textContent='Snapped · Output follows the selected preset';
            note.title='On release, the moved axis snaps to its nearest preset.';
        }
        inspect=null;if(previewState)delete previewState.activeAxis;canvas.style.cursor='grab';draw();
    });
    for(const event of ['pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{inspect=null;if(previewState)delete previewState.activeAxis;canvas.style.cursor='grab';draw();});
    canvas.addEventListener('pointerleave',()=>{if(hoverAxis!==null){hoverAxis=null;draw();}if(!inspect)canvas.style.cursor='default';});
    canvas.addEventListener('wheel',event=>{
        event.preventDefault();event.stopPropagation();if(!event.deltaY)return;
        const choices=['Close-up','Portrait','Upper body','Cowboy shot','Full body','Wide shot','Very wide shot'];
        let index=choices.indexOf(originals.zoom.value);if(index<0)index=2;
        previewState=null;
        changeSelections({zoom:choices[Math.max(0,Math.min(choices.length-1,index+Math.sign(event.deltaY)))]});
    },{passive:false});
    const note=document.createElement('div');
    note.textContent='Horizontal: blue · Vertical: purple · Scroll: framing';
    note.title='Fixed view. Drag blue for horizontal or purple for vertical movement. Dots show snap positions while dragging. Release to snap to the nearest preset.';
    note.style.cssText='font-size:11px;color:#acbfd2;flex:0 0 18px;height:18px;min-height:18px;max-height:18px;line-height:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';root.append(note);
    const syncs=[];
    let focusHeader=null;
    presetFields.forEach((field,index)=>{
        if(focusFields.includes(field)&&field!=='depth_of_field')return;
        if(field==='depth_of_field') {
            const heading=document.createElement('div');heading.textContent='Focus / Blur';
            heading.title='Independent tag-only effects; combine foreground blur, background blur and bokeh. Preview geometry is unchanged.';
            focusHeader=document.createElement('div');
            focusHeader.style.cssText='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px 14px;flex-shrink:0;min-width:0;';
            heading.style.cssText='font-size:12px;font-weight:bold;color:#acbfd2;flex-shrink:0;';focusHeader.append(heading);root.append(focusHeader);
            const buttons=document.createElement('div');buttons.style.cssText='display:flex;flex-wrap:wrap;gap:6px;flex-shrink:0;';root.append(buttons);
            for(const effect of focusFields) {
                const button=document.createElement('button');button.type='button';button.textContent=labels[presetFields.indexOf(effect)];
                button.title='Toggle this tag-only effect. Random overrides manual selections without changing them.';
                const sync=()=>{const active=originals[effect].value==='Enabled';button.disabled=!!originals.focus_blur_random?.value;button.setAttribute('aria-pressed',String(active));button.style.cssText='padding:7px 10px;border:1px solid '+(active?'#75baff':'#526477')+';border-radius:5px;color:'+(active?'#fff':'#acbfd2')+';background:'+(active?'#255b8c':'#171d24')+';cursor:pointer;opacity:'+(button.disabled?'.55':'1')+';';};
                button.addEventListener('click',event=>{event.stopPropagation();if(button.disabled)return;originals[effect].value=originals[effect].value==='Enabled'?'None':'Enabled';originals[effect].callback?.(originals[effect].value);sync();node.setDirtyCanvas?.(true,true);});
                buttons.append(button);syncs.push(sync);sync();
            }
        }
        const isFocus=field==='depth_of_field';
        const selectionWidget=isFocus?originals.focus_blur_random:originals[field],weightWidget=isFocus?originals.focus_blur_strength:originals[field+'_strength'];
        const row=document.createElement('div');
        row.style.cssText='display:flex;align-items:center;gap:8px;flex-shrink:0;min-width:0;';
        const label=document.createElement('span');label.textContent=isFocus?'Random':labels[index];
        label.style.cssText='width:126px;flex-shrink:0;white-space:nowrap;';
        const selection=document.createElement(isFocus?'input':'select');
        if(isFocus)selection.type='checkbox';
        selection.style.cssText='flex:1;min-width:70px;background:#171d24;color:#e4eaf0;border:1px solid #526477;border-radius:5px;padding:6px;';
        if(isFocus) {
            selection.style.cssText='width:18px;height:18px;flex:0 0 18px;margin:0 auto 0 0;accent-color:#75baff;cursor:pointer;';
            label.style.cursor='pointer';
            label.addEventListener('click',()=>{if(!selection.disabled){selection.checked=!selection.checked;selection.dispatchEvent(new Event('change'));}});
        }
        selection.setAttribute('aria-label',labels[index]);
        const definition=data.input.required[field]??data.input.optional?.[field];
        selection.title=definition[1]?.tooltip || labels[index];
        if(field==='horizontal_view')selection.title+=' 45-degree options are approximate targets, not calibrated generated-image angles. The camera preview is a proxy; model output can reverse direction or rotate the head independently.';
        selection.title+=' Random samples a non-None preset on each run, excluding the previous result when multiple choices exist. Weights stay unchanged.';
        const selectionTitle=selection.title;
        for(const value of isFocus?[]:definition[0]) {
            const option=document.createElement('option');option.value=value;option.textContent=value;selection.append(option);
        }
        const number=document.createElement('input');number.type='number';number.min='0';number.max='10';number.step='.01';number.readOnly=true;
        number.style.cssText='width:76px;flex-shrink:0;box-sizing:border-box;padding:6px;background:#171d24;color:#e4eaf0;border:1px solid #526477;border-radius:5px;cursor:ew-resize;';
        number.title=(!['vertical_view','horizontal_view','camera_angle'].includes(field)
            ? 'Weight 0–10 applies to the selected tag only. '
            : 'Weight 0–10 applies to all tags and the entire natural-language instruction for this selection. ')
            +'Drag to adjust; Shift+drag for finer control. Click to type. Enter applies; Escape cancels.';
        number.setAttribute('aria-label',labels[index]+' weight');
        const sync=()=>{selection.value=selectionWidget.value;if(isFocus)selection.checked=!!selectionWidget.value;selection.title=isFocus?'Randomize the entire Focus / Blur combination on every run. Manual toggles are preserved.':selectionTitle+(selection.value==='Random'&&resolvedRandom[field] ? ' Last random result: '+resolvedRandom[field]+'.' : '');number.value=clampWeight(weightWidget.value).toFixed(2);draw();};
        const apply=value=>{weightWidget.value=Math.round(clampWeight(value)*100)/100;weightWidget.callback?.(weightWidget.value);sync();node.setDirtyCanvas?.(true,true);};
        const reset=document.createElement('button');
        reset.type='button';reset.textContent='↺';
        reset.title='Reset this weight to 2.00';
        reset.setAttribute('aria-label',(isFocus?'Focus / Blur':labels[index])+' reset weight');
        reset.style.cssText='width:26px;height:28px;flex-shrink:0;padding:0;background:#171d24;color:#acbfd2;border:1px solid #526477;border-radius:5px;font-size:18px;cursor:pointer;';
        reset.addEventListener('click',event=>{event.stopPropagation();apply(2);number.readOnly=true;number.style.cursor='ew-resize';});
        selection.addEventListener('change',()=>{previewState=null;selectionWidget.value=isFocus?selection.checked:selection.value;selectionWidget.callback?.(selectionWidget.value);syncs.forEach(sync=>sync());sync();node.setDirtyCanvas?.(true,true);});
        let drag=null;
        number.addEventListener('pointerdown',event=>{
            if(event.button!==0 || !number.readOnly)return;
            event.preventDefault();event.stopPropagation();
            drag={x:event.clientX,start:clampWeight(weightWidget.value),moved:false};number.setPointerCapture(event.pointerId);
        });
        number.addEventListener('pointermove',event=>{
            if(!drag)return;event.stopPropagation();
            if(Math.abs(event.clientX-drag.x)>2)drag.moved=true;
            if(drag.moved)apply(dragWeight(drag.start,event.clientX-drag.x,event.shiftKey));
        });
        number.addEventListener('pointerup',event=>{
            if(!drag)return;event.stopPropagation();const moved=drag.moved;drag=null;
            number.releasePointerCapture?.(event.pointerId);
            if(!moved){number.readOnly=false;number.style.cursor='text';number.focus();number.select();}
        });
        for(const event of ['pointercancel','lostpointercapture'])number.addEventListener(event,()=>{drag=null;});
        number.addEventListener('change',()=>apply(number.value));
        number.addEventListener('blur',()=>{number.readOnly=true;number.style.cursor='ew-resize';sync();});
        number.addEventListener('keydown',event=>{
            event.stopPropagation();
            if(event.key==='Enter'){apply(number.value);number.blur();}
            else if(event.key==='Escape'){sync();number.blur();}
            else if(event.key==='ArrowUp'||event.key==='ArrowDown') {
                event.preventDefault();apply(Number(number.value)+(event.key==='ArrowUp'?1:-1)*(event.shiftKey ? .1 : .01));
            }
        });
        if(isFocus){
            const strengthLabel=document.createElement('span');strengthLabel.textContent='Strength';strengthLabel.title='Shared strength for all Focus / Blur effects.';
            strengthLabel.style.cssText='margin-left:8px;white-space:nowrap;';
            label.style.cssText='width:auto;flex-shrink:0;white-space:nowrap;cursor:pointer;';
            selection.style.cssText='width:16px;height:16px;flex:0 0 16px;margin:0;accent-color:#75baff;cursor:pointer;';
            row.style.cssText='display:flex;align-items:center;gap:6px;flex-shrink:0;min-width:0;margin-left:auto;';
            number.title='Shared Focus / Blur weight (0–10). Drag to adjust; Shift+drag for finer control. Click to type.';
            number.setAttribute('aria-label','Focus / Blur strength');selection.setAttribute('aria-label','Random Focus / Blur');
            row.append(label,selection,strengthLabel,number,reset);focusHeader.append(row);
        }else {row.append(label,selection,number,reset);root.append(row);}
        syncs.push(sync);sync();
    });
    const hint=document.createElement('div');hint.textContent='Weight: drag to adjust · click to type · Shift+drag for fine control';
    hint.style.cssText='font-size:11px;color:#acbfd2;flex-shrink:0;';root.append(hint);
    const dom=node.addDOMWidget('booru_preset_list','booru-preset-list',root,{serialize:false,hideOnZoom:false});dom.serialize=false;
    // Only one output remains. Reserve its header/slot space, not the old
    // three-output panel's 130px allowance. Flex gives surplus height to the preview.
    const height=current=>Math.max(180,(current?.size?.[1]||800)-50);
    dom.computeSize=width=>[width||node.size[0],height(node)];
    dom.computeLayoutSize=current=>({minHeight:height(current||node),maxHeight:height(current||node)});
    node._booruStrengthSync=syncs;
    node._booruRandomResult=values=>{
        if(!values || typeof values!=='object')return;
        previewState=null;
        for(const field of presetFields) {
            const definition=data.input.required[field]??data.input.optional?.[field];
            if(originals[field].value==='Random' && values[field]!=='None' && values[field]!=='Random' && definition[0].includes(values[field]))resolvedRandom[field]=values[field];
        }
        if(originals.focus_blur_random?.value) {
            const enabled=focusFields.filter(field=>values[field]==='Enabled').map(field=>labels[presetFields.indexOf(field)]);
            note.textContent='Random Focus / Blur: '+(enabled.join(', ')||'None');
            note.title=note.textContent;
        }
        syncs.forEach(sync=>sync());node.setDirtyCanvas?.(true,true);
    };
    syncs.forEach(sync=>sync());
    const observer=typeof ResizeObserver==='undefined'?null:new ResizeObserver(draw);observer?.observe(canvas);
    const removed=node.onRemoved;node.onRemoved=function(){observer?.disconnect();removed?.apply(this,arguments);};
    const resized=node.onResize;node.onResize=function(){resized?.apply(this,arguments);draw();};
    node.setSize?.([Math.max(node.size[0],480),800]);draw();
}
