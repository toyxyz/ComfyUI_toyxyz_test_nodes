import {PAINT_CORNERS,MAX_PAINT_AXIS,validPaintFrame,paintFramePoint,hitPaintFrame,resizePaintFrame,transformedPaintBounds} from './couple_painter_geometry.js';
import {PainterPoseSource,parsePoseSource,parsePoseArchive} from './couple_pose_source.js';
import {PainterLayerMasks,maskWouldCycle,validateLayerMasks} from './couple_layer_masks.js';
import {MaskLayerPicker} from './couple_mask_picker.js';

const MAX_LAYERS = 32, MAX_PIXELS = 64 * 1024 * 1024;
const BACKGROUND_ID='__painter_background__';
const BASE_ID='__painter_base__',MIN_LIST_HEIGHT=160,DEFAULT_LIST_HEIGHT=260;
const DEFAULT_PANEL_WIDTH=260,MIN_CANVAS_WIDTH=180,PANEL_LAYOUT_SPACE=40;
const BACKGROUND_MODES=['region_and_background','background_only'];
const makeBackground=(prompt='',mode=BACKGROUND_MODES[0])=>({id:BACKGROUND_ID,name:'Background',color:'#000000',visible:true,prompt,mode});
const makeBase=(prompt='')=>({id:BASE_ID,name:'Base',visible:true,prompt});
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const makeCanvas = (w, h) => Object.assign(document.createElement('canvas'), {width:w, height:h});
const validColor = color => /^#[0-9a-f]{6}$/i.test(color);
const TOOL_ICONS={
    select:['M4 3l7 17 3-7 7-3L4 3Z','M14 14l6 6'],
    brush:['M14 6l4-4a2.83 2.83 0 0 1 4 4l-4 4-4-4Z','M14 6l-6 6 4 4 6-6','M8 12c-3 0-5 2-5 5 0 2-1 3-2 4 2 0 3 1 5 1 3 0 5-2 5-5 0-3-2-5-5-5Z'],
    line:['M5 19L19 5','M3 17l4 4M17 3l4 4'],
    box:['M4 5h16v14H4Z'],
    ellipse:['M21 12a9 6 0 1 0-18 0 9 6 0 1 0 18 0Z'],
    eraser:['M7 21l-4.5-4.5a2.1 2.1 0 0 1 0-3l10-10a2.1 2.1 0 0 1 3 0l6 6a2.1 2.1 0 0 1 0 3L14 21H7Z','M9 7l8 8','M7 21h15'],
    transform:['M12 3v18M3 12h18','M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3'],
    rotate:['M21 10a9 9 0 1 0-2.7 6.4','M21 3v7h-7'],
    undo:['M3 7v6h6','M21 17a9 9 0 0 0-15.5-6.5L3 13'],
    redo:['M21 7v6h-6','M3 17a9 9 0 0 1 15.5-6.5L21 13'],
    copy:['M8 8h13v13H8z','M16 8V3H3v13h5'],
    paste:['M8 5H5v16h14V5h-3','M9 3h6v4H9z','M8 12h8M8 16h6'],
};
function toolIcon(tool){
    // Local vector icons: consistent at any graph zoom, no font/CDN dependency.
    const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
    for(const [key,value] of Object.entries({viewBox:'0 0 24 24',width:'20',height:'20',fill:'none',stroke:'currentColor','stroke-width':'1.8','stroke-linecap':'round','stroke-linejoin':'round','aria-hidden':'true',focusable:'false'}))svg.setAttribute(key,value);
    svg.style.cssText='display:block;pointer-events:none;flex-shrink:0';
    for(const d of TOOL_ICONS[tool]){const path=document.createElementNS(ns,'path');path.setAttribute('d',d);svg.append(path);}return svg;
}

function nextColor(layers) {
    // Choose the most distant of randomized bright palette candidates.
    const colors=layers.map(l=>[1,3,5].map(i=>parseInt(l.color.slice(i,i+2),16)));
    let best='#ff6655', score=-1;
    for(let i=0;i<80;i++) {
        const hue=Math.random()*6, sector=Math.floor(hue), f=hue-sector;
        const rgb=[[1,f,0],[1-f,1,0],[0,1,f],[0,1-f,1],[f,0,1],[1,0,1-f]][sector].map(x=>Math.round(35+x*220));
        const distance=colors.length?Math.min(...colors.map(c=>Math.hypot(...c.map((v,j)=>v-rgb[j])))):1;
        if(distance>score){score=distance;best='#'+rgb.map(v=>v.toString(16).padStart(2,'0')).join('');}
    }
    return best;
}

export class CoupleRegionPainterUI {
    constructor(node, root) {
        this.node=node;this.root=root;this.layers=[];this.background=makeBackground();this.base=makeBase();this.selected=null;this.history=[];this.future=[];this.tool='brush';this.brushSize=40;
        this.poseSource=new PainterPoseSource(this);
        this.layerMasks=new PainterLayerMasks(this);
        root.className='toyxyz-couple-painter';
        root.style.cssText='height:100%;width:100%;display:flex;gap:6px;padding:10px;box-sizing:border-box;background:#262626;color:#ddd;font:13px sans-serif;overflow:hidden;min-height:0';
        const style=document.createElement('style');
        style.textContent='.toyxyz-couple-painter button,.toyxyz-couple-painter input,.toyxyz-couple-painter textarea{box-sizing:border-box;font:inherit;color:inherit;background:#353535;border:1px solid #666;border-radius:4px;padding:5px;min-width:0}.toyxyz-couple-painter button{cursor:pointer}.toyxyz-couple-painter button:disabled{opacity:.4;cursor:default}.toyxyz-couple-painter button[aria-pressed=true]{background:#34566e;border-color:#8dc4e7}.toyxyz-couple-painter textarea{resize:none}.toyxyz-couple-painter label{display:flex;gap:6px;align-items:center}';
        root.append(style);
        const left=document.createElement('div');left.style.cssText='flex:1;min-width:0;display:flex;flex-direction:column;gap:8px;min-height:0';
        this.toolbar=document.createElement('div');this.toolbar.style.cssText='display:flex;gap:5px;flex-wrap:wrap;align-items:center';
        const button=(title,fn,parent=this.toolbar)=>{const b=document.createElement('button');b.type='button';b.textContent=title;b.onclick=fn;parent.append(b);return b;};
        this.toolRail=document.createElement('div');this.toolRail.setAttribute('aria-label','Painting tools');this.toolRail.style.cssText='display:flex;flex-direction:column;gap:6px;flex-shrink:0;width:32px';
        this.selectButton=button('',()=>this.setTool('select'),this.toolRail);this.selectButton.append(toolIcon('select'));this.selectButton.title='Select layer · Click a visible mask to select its layer. Hidden, covered and clipped-out pixels are ignored.';this.selectButton.setAttribute('aria-label','Select layer');this.selectButton.style.cssText='width:32px;height:32px;flex-shrink:0;display:flex;align-items:center;justify-content:center';
        this.brushButton=button('',()=>this.setTool('brush'),this.toolRail);
        this.lineButton=button('',()=>this.setTool('line'),this.toolRail);
        this.boxButton=button('',()=>this.setTool('box'),this.toolRail);this.ellipseButton=button('',()=>this.setTool('ellipse'),this.toolRail);
        this.eraserButton=button('',()=>this.setTool('eraser'),this.toolRail);
        this.transformButton=button('',()=>this.setTool('transform'),this.toolRail);
        for(const [control,tool,label,tip] of [[this.brushButton,'brush','Brush','Brush · Scroll over the canvas to adjust size. Shift + scroll: fine adjustment.'],[this.lineButton,'line','Line','Line · Drag to draw a straight line. Scroll: thickness. Shift: 45° angle steps. Escape: cancel.'],[this.boxButton,'box','Box','Box · Drag to draw a filled rectangle. Shift: square. Escape: cancel.'],[this.ellipseButton,'ellipse','Ellipse','Ellipse · Drag to draw a filled ellipse. Shift: circle. Escape: cancel.'],[this.eraserButton,'eraser','Eraser','Eraser · Scroll over the canvas to adjust size. Shift + scroll: fine adjustment.'],[this.transformButton,'transform','Transform','Transform · Drag inside to move, a corner for free resize, or the round handle to rotate. Shift: keep resize proportions / snap rotation to 15°. Escape: cancel.']]){control.append(toolIcon(tool));control.title=tip;control.setAttribute('aria-label',label);control.style.cssText='width:32px;height:32px;flex-shrink:0;display:flex;align-items:center;justify-content:center';}
        this.sizeDisplay=document.createElement('span');this.sizeDisplay.style.cssText='white-space:nowrap;min-width:70px';this.sizeDisplay.title='Brush / eraser diameter in canvas pixels. Scroll over the canvas to adjust; Shift + scroll for fine adjustment.';this.toolbar.append(this.sizeDisplay);
        this.undoButton=button('',()=>this.undo());this.redoButton=button('',()=>this.redo());
        for(const [control,icon,label] of [[this.undoButton,'undo','Undo'],[this.redoButton,'redo','Redo']]){control.append(toolIcon(icon));control.title=label;control.setAttribute('aria-label',label);control.style.cssText='width:32px;height:32px;display:flex;align-items:center;justify-content:center';}
        this.centerButton=button('Center layer',()=>this.centerLayer());this.centerButton.title='Bring the entire selected painting back to the center of the canvas.';
        this.clearButton=button('Clear layer',()=>{const layer=this.current();if(!layer||this.isFixed(layer)||this.loading||this.error)return;this.finishEdit();layer._content=null;layer.canvas.getContext('2d').clearRect(0,0,this.width,this.height);this.invalidateLayer(layer);this.changed();});
        this.copyMaskButton=button('',()=>this.copyMask());this.pasteMaskButton=button('',()=>this.pasteMask());
        for(const [control,icon,label,tip] of [[this.copyMaskButton,'copy','Copy mask','Copy mask · Copy the selected layer as a black-and-white PNG. Ctrl+C outside text fields.'],[this.pasteMaskButton,'paste','Paste mask','Paste mask · Replace the selected layer from a clipboard image. White: mask; black: empty. Ctrl+V outside text fields.']]){
            control.append(toolIcon(icon));control.title=tip;control.setAttribute('aria-label',label);control.style.cssText='width:32px;height:32px;display:flex;align-items:center;justify-content:center';
        }
        this.viewport=document.createElement('div');this.viewport.style.cssText='flex:1;min-height:0;min-width:0;background:#111;display:flex;align-items:center;justify-content:center;overflow:hidden';
        const canvasArea=document.createElement('div');canvasArea.style.cssText='display:flex;gap:8px;flex:1;min-width:0;min-height:0';canvasArea.append(this.toolRail,this.viewport);
        this.canvas=makeCanvas(1024,1024);this.canvas.tabIndex=0;this.canvas.style.cssText='display:block;touch-action:none;outline:none;cursor:crosshair';
        this.canvasStage=document.createElement('div');this.canvasStage.style.cssText='position:relative;flex-shrink:0;overflow:hidden';
        // Separate DOM overlay: never painted into a mask, PNG, or workflow.
        // Local CSS coordinates inherit the same graph zoom as the canvas.
        this.brushCursor=document.createElement('div');this.brushCursor.setAttribute('aria-hidden','true');
        this.brushCursor.style.cssText='display:none;position:absolute;pointer-events:none;z-index:1;box-sizing:border-box;outline:1px solid white;border-radius:50%;box-shadow:0 0 0 2px #000;transform:translate(-50%,-50%)';
        this.transformBox=document.createElement('div');this.transformBox.setAttribute('aria-hidden','true');
        this.transformBox.style.cssText='display:none;position:absolute;pointer-events:none;z-index:2;outline:1px dashed white;box-shadow:0 0 0 1px #000';
        for(const corner of PAINT_CORNERS){const handle=document.createElement('span');handle.dataset.corner=corner;handle.style.cssText=`position:absolute;pointer-events:none;width:8px;height:8px;background:white;box-shadow:0 0 0 1px #000;left:${corner.includes('e')?'100%':'0'};top:${corner.includes('s')?'100%':'0'};transform:translate(-50%,-50%)`;this.transformBox.append(handle);}
        this.rotationLink=document.createElement('span');this.rotationLink.setAttribute('aria-hidden','true');this.rotationLink.style.cssText='display:none;position:absolute;height:1px;background:#ddd;pointer-events:none;z-index:2;transform-origin:0 0';
        this.rotationHandle=document.createElement('button');this.rotationHandle.type='button';this.rotationHandle.setAttribute('aria-label','Rotate layer');this.rotationHandle.title='Drag to rotate the layer around its center. Shift: 15° steps. Escape: cancel.';this.rotationHandle.style.cssText='display:none;position:absolute;width:18px;height:18px;border:1px solid white;border-radius:50%;background:#333;color:white;padding:2px;align-items:center;justify-content:center;transform:translate(-50%,-50%);cursor:grab;z-index:3;touch-action:none';const rotateIcon=toolIcon('rotate');rotateIcon.setAttribute('width','12');rotateIcon.setAttribute('height','12');this.rotationHandle.append(rotateIcon);this.rotationHandle.addEventListener('pointerdown',e=>{e.stopPropagation();this.beginTransform(e);});
        this.canvasStage.append(this.canvas,this.brushCursor,this.transformBox,this.rotationLink,this.rotationHandle);this.viewport.append(this.canvasStage);
        this.status=document.createElement('div');this.status.style.cssText='font-size:12px;color:#c6c6c6;min-height:30px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
        left.append(this.toolbar,canvasArea,this.status);
        this.panel=document.createElement('div');this.panel.style.cssText='width:260px;flex-shrink:0;display:flex;flex-direction:column;gap:4px;min-height:0;overflow:hidden';
        this.panelSplitter=document.createElement('div');this.panelSplitter.className='painter-panel-splitter';this.panelSplitter.tabIndex=0;
        this.panelSplitter.setAttribute('role','separator');this.panelSplitter.setAttribute('aria-orientation','vertical');this.panelSplitter.setAttribute('aria-label','Resize layers and prompt panel');
        this.panelSplitter.title='Drag to resize the layers and prompt panel. Left / Right: adjust width. Double-click: reset.';
        this.panelSplitter.style.cssText='width:8px;flex:0 0 8px;cursor:col-resize;touch-action:none;user-select:none;display:flex;align-items:center;justify-content:center;background:#353535;border-radius:4px;outline-offset:2px';
        const panelGrip=document.createElement('span');panelGrip.style.cssText='width:2px;height:34px;background:#999;border-radius:2px;pointer-events:none';this.panelSplitter.append(panelGrip);
        this.panelSplitter.addEventListener('pointerdown',e=>this.beginPanelResize(e));
        this.panelSplitter.addEventListener('pointermove',e=>this.movePanelResize(e));
        for(const type of ['pointerup','pointercancel','lostpointercapture'])this.panelSplitter.addEventListener(type,e=>{if(type==='pointerup')this.movePanelResize(e);this.endPanelResize(e,type==='pointercancel');});
        this.panelSplitter.addEventListener('dblclick',e=>{e.preventDefault();this.endPanelResize(null,true);this.setPanelWidth(DEFAULT_PANEL_WIDTH,true);});
        this.panelSplitter.addEventListener('keydown',e=>{
            if(e.key==='Escape'&&this.panelResizeDrag){e.preventDefault();e.stopPropagation();this.endPanelResize(null,true);return;}
            if(e.ctrlKey||e.metaKey||e.altKey)return;
            const {min,max}=this.panelWidthLimits(),width=this.panel.offsetWidth;
            const next={ArrowLeft:width+(e.shiftKey?40:10),ArrowRight:width-(e.shiftKey?40:10),Home:min,End:max}[e.key];
            if(next!==undefined){e.preventDefault();e.stopPropagation();this.setPanelWidth(next,true);}
        });
        // Two independent panes: scrolling the layer list never moves its
        // actions or the selected layer's settings below it.
        this.layerListPanel=document.createElement('section');this.layerListPanel.setAttribute('aria-label','Layers');
        this.layerListPanel.style.cssText='display:flex;flex-direction:column;gap:8px;flex:0 0 260px;min-height:160px;padding:8px;box-sizing:border-box;border:1px solid #484848;border-radius:5px;background:#202020;overflow:hidden';
        const actions=document.createElement('div');actions.style.cssText='display:flex;gap:5px;flex-wrap:wrap;flex-shrink:0';
        this.addButton=button('+',()=>this.addLayer(),actions);this.deleteButton=button('−',()=>this.deleteLayer(),actions);
        this.frontButton=button('↑',()=>this.reorder(1),actions);this.backButton=button('↓',()=>this.reorder(-1),actions);
        for(const [control,label] of [[this.addButton,'Add layer'],[this.deleteButton,'Delete layer'],[this.frontButton,'Move layer up (Front)'],[this.backButton,'Move layer down (Back)']]){control.title=label;control.setAttribute('aria-label',label);control.style.minWidth='32px';}
        this.list=document.createElement('div');this.list.setAttribute('aria-label','Layer list');this.list.tabIndex=0;
        this.list.style.cssText='display:flex;flex-direction:column;gap:4px;flex:1;min-height:0;overflow:auto;overscroll-behavior:contain';
        this.backgroundRow=document.createElement('div');this.backgroundRow.style.cssText='display:flex;gap:5px;align-items:center;flex-shrink:0';
        const spacer=document.createElement('span');spacer.style.cssText='width:18px;flex-shrink:0';spacer.setAttribute('aria-hidden','true');
        this.backgroundButton=button('',()=>this.select(BACKGROUND_ID),this.backgroundRow);this.backgroundButton.style.cssText='flex:1;text-align:left';this.backgroundButton.title='Background prompt · Fixed black background. Cannot be painted, moved or deleted.';
        const blackSwatch=document.createElement('span');blackSwatch.style.cssText='display:inline-block;width:12px;height:12px;background:#000;margin-right:6px;border:1px solid #999';this.backgroundButton.append(blackSwatch,document.createTextNode('Background'));this.backgroundRow.prepend(spacer);
        this.baseRow=document.createElement('div');this.baseRow.style.cssText='display:flex;gap:5px;align-items:center;flex-shrink:0';
        const baseSpacer=spacer.cloneNode();this.baseRow.append(baseSpacer);
        this.baseButton=button('Base',()=>this.select(BASE_ID),this.baseRow);this.baseButton.style.cssText='flex:1;text-align:left';
        this.baseButton.title='Base prompt · Shared with all regions and the background. Cannot be painted, moved or deleted.';
        this.layerListPanel.append(actions,this.baseRow,this.backgroundRow,this.list);
        this.listSplitter=document.createElement('div');this.listSplitter.className='layer-list-splitter';this.listSplitter.tabIndex=0;
        this.listSplitter.setAttribute('role','separator');this.listSplitter.setAttribute('aria-orientation','horizontal');this.listSplitter.setAttribute('aria-label','Resize layer list');
        this.listSplitter.title='Drag to resize the layer list. Arrow keys: adjust height. Double-click: reset.';
        this.listSplitter.style.cssText='height:8px;flex:0 0 8px;cursor:row-resize;touch-action:none;user-select:none;display:flex;align-items:center;justify-content:center;background:#353535;border-radius:4px';
        const grip=document.createElement('span');grip.style.cssText='width:34px;height:2px;background:#999;border-radius:2px;pointer-events:none';this.listSplitter.append(grip);
        this.listSplitter.addEventListener('pointerdown',e=>this.beginListResize(e));
        this.listSplitter.addEventListener('pointermove',e=>this.moveListResize(e));
        for(const type of ['pointerup','pointercancel','lostpointercapture'])this.listSplitter.addEventListener(type,e=>{if(type==='pointerup')this.moveListResize(e);this.endListResize(e,type==='pointercancel');});
        this.listSplitter.addEventListener('dblclick',e=>{e.preventDefault();this.endListResize(null,true);this.setListHeight(DEFAULT_LIST_HEIGHT,true);});
        this.listSplitter.addEventListener('keydown',e=>{
            e.stopPropagation();if(e.key==='Escape'&&this.listResizeDrag){e.preventDefault();this.endListResize(null,true);return;}
            const {min,max}=this.listHeightLimits(),height=this.layerListPanel.offsetHeight;
            const next={ArrowUp:height-10,ArrowDown:height+10,Home:min,End:max}[e.key];if(next!==undefined){e.preventDefault();this.setListHeight(next,true);}
        });
        this.layerSettings=document.createElement('section');this.layerSettings.setAttribute('aria-label','Layer settings');
        this.layerSettings.style.cssText='display:flex;flex-direction:column;gap:8px;flex:1;min-height:0;padding:8px;box-sizing:border-box;border:1px solid #484848;border-radius:5px;background:#292929;overflow:auto;overscroll-behavior:contain';
        this.colorStrengthRow=document.createElement('div');this.colorStrengthRow.style.cssText='display:flex;flex-wrap:wrap;gap:10px;align-items:center;flex-shrink:0';
        const colorRow=document.createElement('label');colorRow.textContent='Color';colorRow.style.cssText='white-space:nowrap;flex-shrink:0';this.colorInput=Object.assign(document.createElement('input'),{type:'color'});this.colorInput.style.cssText='width:52px;height:32px;padding:2px;flex-shrink:0';colorRow.append(this.colorInput);
        const strengthRow=document.createElement('label');strengthRow.textContent='Strength';strengthRow.style.cssText='white-space:nowrap;min-width:0';this.strengthInput=Object.assign(document.createElement('input'),{type:'number',min:0,max:10,step:.01});this.strengthInput.style.cssText='width:60px;flex-shrink:0';strengthRow.append(this.strengthInput);
        this.colorStrengthRow.append(colorRow,strengthRow);
        this.maskRow=document.createElement('label');this.maskRow.textContent='Mask';this.maskRow.style.cssText='flex:1;min-width:150px';
        this.maskInput=document.createElement('select');this.maskInput.setAttribute('aria-label','Layer Mask');this.maskInput.style.cssText='flex:1;min-width:0;width:100%;font:inherit;color:inherit;background:#353535;border:1px solid #666;border-radius:4px;padding:5px';
        this.maskInput.title='Clip this layer to another layer’s mask. None restores the original painting. Hidden layers can still be used as masks.';this.maskRow.append(this.maskInput);this.colorStrengthRow.append(this.maskRow);
        this.maskPicker=new MaskLayerPicker(this.maskInput,this.maskRow);
        this.maskInput.onchange=()=>{const layer=this.current(),id=this.maskInput.value;if(this.loading||this.error||!layer||this.isFixed(layer))return;if(id&&(!this.layers.some(l=>l.id===id)||maskWouldCycle(this.layers,layer.id,id))){this.renderMaskOptions();return;}this.finishEdit();if(id)layer.mask_layer_id=id;else delete layer.mask_layer_id;this.changed();};
        this.backgroundModeRow=document.createElement('label');this.backgroundModeRow.style.cssText='display:none;flex-direction:column;align-items:stretch;flex-shrink:0';this.backgroundModeRow.textContent='Background mode';
        this.backgroundModeInput=document.createElement('select');this.backgroundModeInput.setAttribute('aria-label','Background mode');this.backgroundModeInput.style.cssText='font:inherit;color:inherit;background:#353535;border:1px solid #666;border-radius:4px;padding:5px;width:100%;min-width:0';
        this.backgroundModeInput.title='region_and_background: append to each region prompt and use for uncovered background. background_only: use only for uncovered background. Leave the prompt blank to keep any incoming background settings.';
        for(const mode of BACKGROUND_MODES){const option=document.createElement('option');option.value=option.textContent=mode;this.backgroundModeInput.append(option);}this.backgroundModeRow.append(this.backgroundModeInput);
        this.backgroundModeInput.onchange=()=>{if(this.loading||this.error||!this.isBackground())return;this.background.mode=this.backgroundModeInput.value;this.changed();};
        const promptTitle=this.promptTitle=document.createElement('div');promptTitle.textContent='Region prompt';promptTitle.style.flexShrink='0';
        this.promptInput=Object.assign(document.createElement('textarea'),{placeholder:'Describe this painted region…'});this.promptInput.style.cssText='width:100%;flex:1;min-height:120px';
        this.layerSettings.append(this.colorStrengthRow,this.backgroundModeRow,promptTitle,this.promptInput);
        this.panel.append(this.layerListPanel,this.listSplitter,this.layerSettings);root.append(left,this.panelSplitter,this.panel);
        for(const [input,key] of [[this.colorInput,'color'],[this.strengthInput,'strength'],[this.promptInput,'prompt']]) {
            input.oninput=()=>{const layer=this.current();if(!layer||this.loading||this.error||(this.isFixed(layer)&&key!=='prompt')||(key==='color'&&this.poseSource.isImported(layer)))return;let value=input.value;if(key==='strength')value=clamp(Number(value)||0,0,10);layer[key]=value;if(key==='color')layer._tintDirty=true;this.commit();this.draw();if(key==='prompt'||key==='color')this.renderList();};
            input.onchange=()=>this.record();input.onblur=()=>this.record();
        }
        // Do not let graph pan, shortcuts or wheel zoom steal editor interaction.
        for(const type of ['pointerdown','pointermove','pointerup','wheel','dblclick'])root.addEventListener(type,e=>e.stopPropagation());
        this.viewport.addEventListener('wheel',e=>this.resizeBrushFromWheel(e),{passive:false});
        root.addEventListener('keydown',e=>{
            // Native weight editing preserves text Undo/settings; native queue
            // execution preserves ComfyUI's normal serialization/validation.
            // Pass through only these shortcuts, not graph movement or editing.
            if((e.ctrlKey||e.metaKey)&&!e.altKey&&!e.isComposing&&e.keyCode!==229&&!this.disposed&&(
                e.key==='Enter'||(e.target===this.promptInput&&['ArrowUp','ArrowDown'].includes(e.key)&&!this.loading&&!this.error&&!this.promptInput.readOnly&&!this.promptInput.disabled)
            ))return;
            e.stopPropagation();if(e.key==='Escape'&&this.shapeDrag){e.preventDefault();this.endShape(null,true);}else if(e.key==='Escape'&&this.transformDrag){e.preventDefault();this.endTransform(null,true);}else if(e.target===this.canvas&&(e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?this.redo():this.undo();}else if(!this.isTextField(e.target)&&(e.ctrlKey||e.metaKey)&&!e.shiftKey&&!e.altKey&&e.key.toLowerCase()==='c'){e.preventDefault();this.copyMask();}
        });
        root.addEventListener('paste',e=>this.handleMaskPaste(e));
        this.canvas.addEventListener('pointerdown',e=>{this.updateBrushCursor(e);this.tool==='select'?this.pickCanvasLayer(e):this.tool==='transform'?this.beginTransform(e):['line','box','ellipse'].includes(this.tool)?this.beginShape(e):this.beginStroke(e);});
        this.canvas.addEventListener('pointerenter',e=>this.updateBrushCursor(e));
        this.canvas.addEventListener('pointerleave',()=>this.hideBrushCursor());
        this.canvas.addEventListener('pointermove',e=>{this.updateBrushCursor(e);this.moveTransform(e);this.moveShape(e);this.moveStroke(e);});
        for(const type of ['pointerup','pointercancel','lostpointercapture'])this.canvas.addEventListener(type,e=>{if(type==='pointerup'){this.moveTransform(e);this.moveShape(e);}this.endTransform(e,type==='pointercancel');this.endShape(e,type!=='pointerup');this.endStroke(e);});
        this.blurCursor=()=>{this.hideBrushCursor();this.endShape(null,true);this.endPanelResize(null,true);};window.addEventListener('blur',this.blurCursor);
        this.observer=new ResizeObserver(()=>{this.fitPanelWidth();this.fit();this.fitListHeight();});this.observer.observe(root);this.observer.observe(this.viewport);this.observer.observe(this.panel);
        this.setBrushSize(this.brushSize);this.setTool('brush');this.ready=this.load();
    }
    widget(name){return this.node.widgets.find(w=>w.name===name);}
    current(){return this.selected===BACKGROUND_ID?this.background:this.selected===BASE_ID?this.base:this.layers.find(l=>l.id===this.selected);}
    isBackground(layer=this.current()){return layer===this.background;}
    isBase(layer=this.current()){return layer===this.base;}
    isFixed(layer=this.current()){return this.isBackground(layer)||this.isBase(layer);}
    layerLabel(layer=this.current()){return !layer?'Add a layer to paint':this.isFixed(layer)?layer.name:layer.prompt.trim().replace(/\s+/gu,' ')||this.poseSource.label(layer)||'Empty prompt';}
    renderMaskOptions(){
        const layer=this.current();this.maskInput.replaceChildren();const none=document.createElement('option');none.value='';none.textContent='None';this.maskInput.append(none);
        if(layer&&!this.isFixed(layer))for(const candidate of [...this.layers].reverse()){if(candidate===layer)continue;const option=document.createElement('option'),label=candidate.prompt.trim().replace(/\s+/gu,' ')||this.poseSource.label(candidate)||`Layer ${this.layers.indexOf(candidate)+1}`;option.value=candidate.id;option.textContent=label.length>60?label.slice(0,60)+'…':label;option.title=label;option.disabled=maskWouldCycle(this.layers,layer.id,candidate.id);this.maskInput.append(option);}
        this.maskInput.value=this.layers.some(l=>l.id===layer?.mask_layer_id)?layer.mask_layer_id:'';
        this.maskInput.disabled=this.loading||!!this.error||!layer||this.isFixed(layer);
        this.maskPicker.sync(new Map(this.layers.map(layer=>[layer.id,layer.color])));
    }
    isTextField(target){return !!target?.closest?.('input,textarea,select,[role=combobox],[contenteditable]:not([contenteditable=false])');}
    clipboardContext(){
        const layer=this.current();
        if(this.disposed||this.loading||this.error||!layer||this.isFixed(layer))throw Error('Select a painted region layer first.');
        this.endStroke();this.endTransform();this.endShape();this.commit();
        return {layer,token:this.loadToken,width:this.width,height:this.height,state:this.widget('paint_data').value};
    }
    clipboardFailure(action,error){if(!this.disposed){this.status.textContent=`${action} mask failed: ${error.message}`;this.status.title=this.status.textContent;}return false;}
    clipboardMaskCanvas(layer){
        // Copy this layer's original clipped alpha, including covered portions,
        // not the composited RGB preview or another layer's effective mask.
        const canvas=makeCanvas(this.width,this.height),ctx=canvas.getContext('2d');
        const pixels=this.poseSource.maskCanvas(layer).getContext('2d').getImageData(0,0,this.width,this.height);
        for(let i=0;i<pixels.data.length;i+=4){const alpha=pixels.data[i+3];pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=alpha;pixels.data[i+3]=255;}
        ctx.putImageData(pixels,0,0);return canvas;
    }
    async copyMask(){
        if(this.clipboardBusy)return false;
        try{
            const {layer}=this.clipboardContext();
            if(!navigator.clipboard?.write||!globalThis.ClipboardItem)throw Error('Clipboard image copying requires localhost or HTTPS and browser clipboard permission.');
            this.clipboardBusy=true;this.renderPanel();const canvas=this.clipboardMaskCanvas(layer);
            // Start write inside the user gesture; PNG encoding can resolve later.
            const png=new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(Error('Cannot encode the mask PNG.')),'image/png'));
            await navigator.clipboard.write([new ClipboardItem({'image/png':png})]);
            if(!this.disposed){this.status.textContent='Mask copied as a black-and-white PNG.';this.status.title='';}return true;
        }catch(error){return this.clipboardFailure('Copy',error);}
        finally{this.clipboardBusy=false;if(!this.disposed)this.renderPanel();}
    }
    async pasteMask(blob=null){
        if(this.clipboardBusy)return false;let bitmap;
        try{
            const context=this.clipboardContext();this.clipboardBusy=true;this.renderPanel();
            if(!blob){
                if(!navigator.clipboard?.read)throw Error('Click the canvas and press Ctrl+V, or use localhost / HTTPS with clipboard permission.');
                const items=await navigator.clipboard.read();
                for(const item of items){const type=item.types.includes('image/png')?'image/png':item.types.find(t=>t.startsWith('image/'));if(type){blob=await item.getType(type);break;}}
            }
            if(!blob||!blob.type.startsWith('image/'))throw Error('The clipboard contains no mask image. Copy an image, not a file path or text.');
            if(blob.size>96*1024*1024)throw Error('Clipboard image is too large (96 MB limit).');
            bitmap=await createImageBitmap(blob);
            if(bitmap.width<1||bitmap.height<1||bitmap.width>MAX_PAINT_AXIS||bitmap.height>MAX_PAINT_AXIS||bitmap.width*bitmap.height>MAX_PIXELS)throw Error('Clipboard image exceeds the 8192-pixel / 64-million-pixel limit.');
            if(this.disposed||this.loading||this.error||context.token!==this.loadToken||this.current()!==context.layer||context.width!==this.width||context.height!==this.height||context.state!==this.widget('paint_data').value||this.stroke||this.transformDrag||this.shapeDrag)throw Error('The selected layer or painting changed while reading the clipboard. Paste again.');
            const canvas=makeCanvas(this.width,this.height),ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0,this.width,this.height);
            const pixels=ctx.getImageData(0,0,this.width,this.height);
            for(let i=0;i<pixels.data.length;i+=4){
                const alpha=Math.round((.2126*pixels.data[i]+.7152*pixels.data[i+1]+.0722*pixels.data[i+2])*pixels.data[i+3]/255);
                pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=255;pixels.data[i+3]=alpha;
            }
            ctx.putImageData(pixels,0,0);
            this.record();const layer=context.layer;layer.canvas=canvas;layer._content=null;layer._tint=null;this.invalidateLayer(layer);this.changed();
            this.status.textContent='Mask pasted into the selected layer.';this.status.title='';return true;
        }catch(error){return this.clipboardFailure('Paste',error);}
        finally{bitmap?.close();this.clipboardBusy=false;if(!this.disposed)this.renderPanel();}
    }
    handleMaskPaste(event){
        // Preserve native text pasting, but keep clipboard events local to this
        // editor so ComfyUI does not paste graph nodes instead of the mask.
        event.stopPropagation();if(event.defaultPrevented||this.isTextField(event.target))return;
        const item=[...(event.clipboardData?.items??[])].find(item=>item.kind==='file'&&item.type.startsWith('image/'));
        const blob=item?.getAsFile();if(!blob)return;event.preventDefault();this.pasteMask(blob);
    }
    panelWidthLimits(){const max=Math.max(0,this.root.clientWidth-PANEL_LAYOUT_SPACE-MIN_CANVAS_WIDTH);return {min:Math.min(DEFAULT_PANEL_WIDTH,max),max};}
    fitPanelWidth(){
        if(this.disposed)return;const {min,max}=this.panelWidthLimits(),width=clamp(this.panelWidth??DEFAULT_PANEL_WIDTH,min,max);
        this.panel.style.width=`${width}px`;
        for(const [key,value] of Object.entries({'aria-valuemin':min,'aria-valuemax':max,'aria-valuenow':width}))this.panelSplitter.setAttribute(key,String(value));
    }
    setPanelWidth(width,persist=false){
        const {min,max}=this.panelWidthLimits();this.panelWidth=clamp(Math.round(width),min,max);this.fitPanelWidth();this.fit();
        if(persist){this.node.properties??={};this.node.properties.couple_painter_panel_width=this.panelWidth;this.node.setDirtyCanvas?.(true,true);}
    }
    beginPanelResize(event){
        if(event.button!==0||this.disposed||this.panelResizeDrag)return;event.preventDefault();event.stopPropagation();this.endListResize();this.endStroke();this.endTransform();this.endShape();
        this.panelResizeDrag={pointer:event.pointerId,startX:event.clientX,width:this.panel.offsetWidth,before:this.panelWidth};
        this.panelSplitter.focus({preventScroll:true});this.panelSplitter.setPointerCapture(event.pointerId);
    }
    movePanelResize(event){
        const drag=this.panelResizeDrag;if(!drag||event.pointerId!==drag.pointer)return;event.preventDefault();event.stopPropagation();
        // Pointer coordinates are screen pixels; panel widths are local CSS
        // pixels. Compensate for ComfyUI's graph zoom without changing masks.
        const scale=this.root.getBoundingClientRect().width/this.root.offsetWidth;
        if(scale>0)this.setPanelWidth(drag.width-(event.clientX-drag.startX)/scale);
    }
    endPanelResize(event,cancel=false){
        const drag=this.panelResizeDrag;if(!drag||(event&&event.pointerId!==drag.pointer))return;this.panelResizeDrag=null;
        if(cancel){this.panelWidth=drag.before;this.fitPanelWidth();this.fit();}else this.setPanelWidth(this.panelWidth??drag.width,true);
        if(this.panelSplitter.hasPointerCapture(drag.pointer))this.panelSplitter.releasePointerCapture(drag.pointer);
    }
    listHeightLimits(){return {min:MIN_LIST_HEIGHT,max:Math.max(MIN_LIST_HEIGHT,this.panel.clientHeight-16-180)};}
    fitListHeight(){
        if(this.disposed)return;const {min,max}=this.listHeightLimits(),height=clamp(this.listHeight??DEFAULT_LIST_HEIGHT,min,max);
        this.layerListPanel.style.flexBasis=`${height}px`;
        for(const [key,value] of Object.entries({'aria-valuemin':min,'aria-valuemax':max,'aria-valuenow':height}))this.listSplitter.setAttribute(key,String(value));
    }
    setListHeight(height,persist=false){
        const {min,max}=this.listHeightLimits();this.listHeight=clamp(Math.round(height),min,max);this.fitListHeight();
        if(persist){this.node.properties??={};this.node.properties.couple_painter_list_height=this.listHeight;this.node.setDirtyCanvas?.(true,true);}
    }
    beginListResize(event){
        if(event.button!==0||this.disposed||this.listResizeDrag)return;event.preventDefault();event.stopPropagation();this.endStroke();this.endTransform();this.endShape();
        this.listResizeDrag={pointer:event.pointerId,startY:event.clientY,height:this.layerListPanel.offsetHeight,before:this.listHeight};
        this.listSplitter.focus();this.listSplitter.setPointerCapture(event.pointerId);
    }
    moveListResize(event){
        const drag=this.listResizeDrag;if(!drag||event.pointerId!==drag.pointer)return;event.preventDefault();event.stopPropagation();
        const scale=this.panel.getBoundingClientRect().height/this.panel.clientHeight;
        if(scale>0)this.setListHeight(drag.height+(event.clientY-drag.startY)/scale);
    }
    endListResize(event,cancel=false){
        const drag=this.listResizeDrag;if(!drag||(event&&event.pointerId!==drag.pointer))return;this.listResizeDrag=null;
        if(cancel){this.listHeight=drag.before;this.fitListHeight();}else this.setListHeight(this.listHeight??drag.height,true);
        if(this.listSplitter.hasPointerCapture(drag.pointer))this.listSplitter.releasePointerCapture(drag.pointer);
    }
    invalidateLayer(layer,bounds){layer._mask=null;layer._tintDirty=true;layer._bounds=bounds;layer._frame=null;}
    contentLimit(w,h,layer=null,layers=this.layers,canvasW=this.width,canvasH=this.height){
        const pixels=canvasW*canvasH*Math.max(1,layers.length)+layers.reduce((n,l)=>n+(l===layer?0:(l._content?.canvas.width||0)*(l._content?.canvas.height||0)),0)+w*h;
        if(!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||w>MAX_PAINT_AXIS||h>MAX_PAINT_AXIS||pixels>MAX_PIXELS)throw Error('Editable painting limit: 8192 pixels per side / 64 million total pixels. Center the layer or reduce its size before painting.');
    }
    projectContent(layer,bounds){
        const ctx=layer.canvas.getContext('2d');ctx.clearRect(0,0,layer.canvas.width,layer.canvas.height);
        if(layer._content){const c=layer._content;ctx.drawImage(c.canvas,c.x,c.y);}
        this.invalidateLayer(layer,bounds);
    }
    layerBounds(layer=this.current()){
        if(!layer||this.isFixed(layer))return null;if(layer._bounds!==undefined)return layer._bounds;
        const source=layer._content?.canvas??layer.canvas,w=source.width,h=source.height;
        const {data}=source.getContext('2d').getImageData(0,0,w,h);
        let left=w,top=h,right=-1,bottom=-1;
        for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(data[(y*w+x)*4+3]){left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=y;}
        return layer._bounds=right<0?null:{x:left+(layer._content?.x??0),y:top+(layer._content?.y??0),w:right-left+1,h:bottom-top+1};
    }
    layerFrame(layer=this.current()){if(!layer||this.isFixed(layer))return null;if(layer._frame)return layer._frame;const bounds=this.layerBounds(layer);return bounds?{...bounds,angle:0}:null;}
    dimensions(){return ['width','height'].map(name=>clamp(Math.round(Number(this.widget(name).value)||1024),64,4096));}
    limit(count,w=this.width,h=this.height){if(count>MAX_LAYERS||w*h*Math.max(1,count)>MAX_PIXELS)throw Error('Limit: 32 layers / 64 million layer pixels. Reduce layers or resolution.');}
    snapshotLayer(l){
        const item={id:l.id,name:l.name,color:l.color,prompt:l.prompt,strength:l.strength,visible:l.visible,mask:l._mask??(l._mask=l.canvas.toDataURL('image/png'))};if(l.mask_layer_id)item.mask_layer_id=l.mask_layer_id;
        // The backend still consumes only the canvas-sized visible mask. The
        // optional content preserves the unclipped artwork in saved workflows.
        if(l._content){const c=l._content;item.content={x:c.x,y:c.y,width:c.canvas.width,height:c.canvas.height,mask:c.mask??(c.mask=c.canvas.toDataURL('image/png'))};}if(l._frame)item.transform_frame={...l._frame};if(l.pose_source)item.pose_source={...l.pose_source};return item;
    }
    snapshot(){const data={version:1,width:this.width,height:this.height,selected:this.selected,background:{prompt:this.background.prompt,mode:this.background.mode},base:{prompt:this.base.prompt},layers:this.layers.map(l=>this.snapshotLayer(l))};const source=this.poseSource.snapshot();if(source)data.pose_source=source;if(this.poseSource.archives.size)data.pose_archive=[...this.poseSource.archives.values()];return data;}
    commit(){if(this.loading||this.error)return;this.widget('paint_data').value=JSON.stringify(this.snapshot());this.node.setDirtyCanvas?.(true,true);}
    record(){if(this.loading||this.error)return;const value=JSON.stringify(this.snapshot());if(value===this.history.at(-1))return;this.history.push(value);if(this.history.length>25)this.history.shift();this.future=[];this.updateHistory();}
    changed(){this.commit();this.record();this.renderPanel();this.draw();}
    updateHistory(){this.undoButton.disabled=this.loading||this.history.length<2;this.redoButton.disabled=this.loading||!this.future.length;}
    async load(raw=this.widget('paint_data').value,resetHistory=true) {
        this.endListResize(null,true);this.endStroke();this.endTransform();this.endShape(null,true);
        const token=this.loadToken=(this.loadToken||0)+1;this.loading=true;this.error=null;this.draw();
        try {
            const data=typeof raw==='string'?(raw?JSON.parse(raw):null):raw;
            if(data&&(data.version!==1||!Array.isArray(data.layers)||!Number.isInteger(data.width)||!Number.isInteger(data.height)||data.width<64||data.height<64||data.width>4096||data.height>4096))throw Error('Invalid saved painter data.');
            const background=data&&'background' in data?data.background:{prompt:'',mode:BACKGROUND_MODES[0]};
            if(!background||typeof background.prompt!=='string'||!BACKGROUND_MODES.includes(background.mode))throw Error('Invalid saved background prompt or mode.');
            const base=data&&'base' in data?data.base:{prompt:''};
            if(!base||typeof base.prompt!=='string')throw Error('Invalid saved base prompt.');
            const sourceState=await parsePoseSource(data?.pose_source),sourceArchive=parsePoseArchive(data?.pose_archive);
            const [w,h]=sourceState?[sourceState.payload.width,sourceState.payload.height]:this.dimensions();
            this.limit(data?.layers.length||1,w,h);
            let sourcePixels=w*h*Math.max(1,data?.layers.length||0);
            for(const item of data?.layers||[])if(item?.content){
                const c=item.content,cw=Math.max(1,Math.round(c.width*w/data.width)),ch=Math.max(1,Math.round(c.height*h/data.height));
                if(!Number.isSafeInteger(c.x)||!Number.isSafeInteger(c.y)||!Number.isSafeInteger(Math.round(c.x*w/data.width))||!Number.isSafeInteger(Math.round(c.y*h/data.height))||!Number.isInteger(c.width)||!Number.isInteger(c.height)||c.width<1||c.height<1||c.width>MAX_PAINT_AXIS||c.height>MAX_PAINT_AXIS||cw>MAX_PAINT_AXIS||ch>MAX_PAINT_AXIS||!c.mask?.startsWith('data:image/png;base64,'))throw Error('Invalid saved painting content.');
                sourcePixels+=cw*ch;if(sourcePixels>MAX_PIXELS)throw Error('Saved painting exceeds the editable pixel limit.');
            }
            const ids=new Set(),sourceKeys=new Set();
            const layers=await Promise.all((data?.layers||[]).map(async item=>{
                if(!item||typeof item.id!=='string'||[BACKGROUND_ID,BASE_ID].includes(item.id)||ids.has(item.id)||!validColor(item.color)||typeof item.prompt!=='string'||typeof item.visible!=='boolean'||!Number.isFinite(item.strength)||item.strength<0||item.strength>10||!item.mask?.startsWith('data:image/png;base64,'))throw Error('Invalid saved layer.');
                if(item.transform_frame!==undefined&&!validPaintFrame(item.transform_frame))throw Error('Invalid saved transform frame.');
                if(item.pose_source!==undefined&&(!this.poseSource.isImported(item)||!sourceState||item.pose_source.source_id!==sourceState.payload.source_id||!sourceState.byId.has(item.pose_source.body_id)))throw Error('Protected pose layer has no matching saved body source.');
                if(item.pose_source){const key=JSON.stringify([item.pose_source.source_id,item.pose_source.body_id]);if(sourceKeys.has(key))throw Error('Duplicate layers for the same protected body.');sourceKeys.add(key);}
                ids.add(item.id);const image=new Image();image.src=item.mask;await image.decode();
                if(image.width!==data.width||image.height!==data.height)throw Error('Saved layer resolution mismatch.');
                const canvas=makeCanvas(w,h);canvas.getContext('2d').drawImage(image,0,0,w,h);
                const layer={...item,canvas,_bounds:undefined,_mask:w===data.width&&h===data.height?item.mask:null};
                if(item.content){
                    const c=item.content,source=new Image();source.src=c.mask;await source.decode();if(source.width!==c.width||source.height!==c.height)throw Error('Saved painting content resolution mismatch.');
                    const content=makeCanvas(Math.max(1,Math.round(c.width*w/data.width)),Math.max(1,Math.round(c.height*h/data.height)));content.getContext('2d').drawImage(source,0,0,content.width,content.height);
                    layer._content={canvas:content,x:Math.round(c.x*w/data.width),y:Math.round(c.y*h/data.height),mask:w===data.width&&h===data.height?c.mask:null};
                    // Always derive the output mask from authoritative artwork,
                    // including when an externally edited workflow has a stale mask.
                    this.projectContent(layer);
                }
                // A display resize preserves the frame. Actual nonuniform
                // bitmap resizing introduces shear, so re-fit that new artwork.
                const sx=w/data.width,sy=h/data.height;
                layer._frame=item.transform_frame&&Math.abs(sx-sy)<1e-10?{...item.transform_frame,x:item.transform_frame.x*sx,y:item.transform_frame.y*sy,w:item.transform_frame.w*sx,h:item.transform_frame.h*sy}:null;
                if(layer._frame&&!validPaintFrame(layer._frame))throw Error('Resized transform frame exceeds the editable limit.');
                return layer;
            }));
            validateLayerMasks(layers);
            if(token!==this.loadToken||this.disposed)return;
            this.poseSource.restore(sourceState,sourceArchive);
            if(sourceState){this.widget('width').value=w;this.widget('height').value=h;}
            this.width=w;this.height=h;this.canvas.width=w;this.canvas.height=h;this.layers=layers;this.background=makeBackground(background.prompt,background.mode);this.base=makeBase(base.prompt);this.selected=[BACKGROUND_ID,BASE_ID].includes(data?.selected)||layers.some(l=>l.id===data?.selected)?data.selected:layers.at(-1)?.id??BACKGROUND_ID;
            this.loading=false;if(!data)this.addLayer(false);
            const listHeight=this.node.properties?.couple_painter_list_height;this.listHeight=Number.isFinite(listHeight)?clamp(Math.round(listHeight),MIN_LIST_HEIGHT,10000):DEFAULT_LIST_HEIGHT;this.fitListHeight();
            const panelWidth=this.node.properties?.couple_painter_panel_width;this.panelWidth=Number.isFinite(panelWidth)?clamp(Math.round(panelWidth),DEFAULT_PANEL_WIDTH,10000):DEFAULT_PANEL_WIDTH;this.fitPanelWidth();
            if(resetHistory){this.history=[];this.future=[];}this.commit();if(resetHistory)this.record();this.renderPanel();this.draw();this.fit();
        }catch(e){if(token===this.loadToken){this.loading=false;this.error=e;this.status.textContent=`Painter error: ${e.message}`;this.updateHistory();}throw e;}
    }
    async serialize(){await this.ready;if(this.error)throw this.error;this.poseBinding?.detachIfUnlinked();this.endPanelResize();this.endListResize();this.endStroke();this.endTransform();this.endShape();this.resizeCanvas();this.commit();return this.widget('paint_data').value;}
    addLayer(record=true){
        if(this.loading||this.error)return;try{this.limit(this.layers.length+1);const pixels=this.width*this.height*(this.layers.length+1)+this.layers.reduce((n,l)=>n+(l._content?.canvas.width||0)*(l._content?.canvas.height||0),0);if(pixels>MAX_PIXELS)throw Error('Adding a layer exceeds the editable pixel limit.');}catch(e){this.status.textContent=e.message;return;}
        this.finishEdit();const index=this.layers.findIndex(layer=>layer.id===this.selected);
        const layer={id:crypto.randomUUID(),name:`Region ${this.layers.length+1}`,color:nextColor(this.layers),prompt:'',strength:1,visible:true,canvas:makeCanvas(this.width,this.height)};
        // Storage is back-to-front; the list displays the reverse order.
        this.layers.splice(index<0?this.layers.length:index+1,0,layer);this.selected=layer.id;if(record)this.changed();else this.renderPanel();
    }
    deleteLayer(){if(this.loading||this.error||this.isFixed()||this.poseSource.isImported(this.current()))return;this.finishEdit();const i=this.layers.findIndex(l=>l.id===this.selected);if(i<0)return;const removed=this.layers.splice(i,1)[0];for(const layer of this.layers)if(layer.mask_layer_id===removed.id)delete layer.mask_layer_id;this.selected=this.layers[Math.min(i,this.layers.length-1)]?.id??BACKGROUND_ID;this.changed();}
    reorder(direction){if(this.loading||this.error||this.isFixed())return;this.finishEdit();const i=this.layers.findIndex(l=>l.id===this.selected),j=i+direction;if(i<0||j<0||j>=this.layers.length)return;[this.layers[i],this.layers[j]]=[this.layers[j],this.layers[i]];this.changed();}
    finishEdit(){this.endStroke();this.endTransform();this.endShape();this.record();}
    select(id){if(this.loading||this.error)return;this.finishEdit();this.selected=id;this.commit();this.renderPanel();}
    async undo(){if(this.loading||this.error)return;this.finishEdit();if(this.history.length<2)return;this.future.push(this.history.pop());const data=JSON.parse(this.history.at(-1));this.widget('width').value=data.width;this.widget('height').value=data.height;this.ready=this.load(data,false);await this.ready;if(this.poseBinding)await this.poseBinding.syncNow(true);this.updateHistory();}
    async redo(){if(this.loading||this.error||!this.future.length)return;const value=this.future.pop(),data=JSON.parse(value);this.history.push(value);this.widget('width').value=data.width;this.widget('height').value=data.height;this.ready=this.load(data,false);await this.ready;if(this.poseBinding)await this.poseBinding.syncNow(true);this.updateHistory();}
    resizeCanvas(record=true){
        if(this.loading||this.error)return;
        // Source masks share the Pose camera's exact output grid. Do not
        // stretch artwork twice when a painter widget is changed and the
        // live source immediately restores its resolution.
        if(record&&this.poseSource.state){this.widget('width').value=this.poseSource.state.payload.width;this.widget('height').value=this.poseSource.state.payload.height;}
        const [w,h]=this.dimensions();if(w===this.width&&h===this.height)return;
        const sx=w/this.width,sy=h/this.height;
        try{this.limit(this.layers.length,w,h);let pixels=w*h*Math.max(1,this.layers.length);for(const l of this.layers)if(l._content){const cw=Math.max(1,Math.round(l._content.canvas.width*sx)),ch=Math.max(1,Math.round(l._content.canvas.height*sy));pixels+=cw*ch;if(cw>MAX_PAINT_AXIS||ch>MAX_PAINT_AXIS||pixels>MAX_PIXELS)throw Error('Resize exceeds the editable painting limit.');}}
        catch(e){this.widget('width').value=this.width;this.widget('height').value=this.height;this.status.textContent=e.message;return;}
        if(Math.abs(sx-sy)<1e-10&&this.layers.some(l=>l._frame&&!validPaintFrame({...l._frame,x:l._frame.x*sx,y:l._frame.y*sy,w:l._frame.w*sx,h:l._frame.h*sy}))){this.widget('width').value=this.width;this.widget('height').value=this.height;this.status.textContent='Resized transform frame exceeds the editable limit.';return;}
        this.endStroke();this.endTransform();this.endShape();
        for(const layer of this.layers){const frame=layer._frame,canvas=makeCanvas(w,h);if(layer._content){const c=layer._content,source=makeCanvas(Math.max(1,Math.round(c.canvas.width*sx)),Math.max(1,Math.round(c.canvas.height*sy)));source.getContext('2d').drawImage(c.canvas,0,0,source.width,source.height);layer._content={canvas:source,x:Math.round(c.x*sx),y:Math.round(c.y*sy)};layer.canvas=canvas;this.projectContent(layer);}else{canvas.getContext('2d').drawImage(layer.canvas,0,0,w,h);layer.canvas=canvas;this.invalidateLayer(layer);}if(frame&&Math.abs(sx-sy)<1e-10)layer._frame={...frame,x:frame.x*sx,y:frame.y*sy,w:frame.w*sx,h:frame.h*sy};layer._tint=null;}
        this.width=w;this.height=h;this.canvas.width=w;this.canvas.height=h;if(record)this.changed();else{this.commit();this.renderPanel();this.draw();}this.fit();
    }
    fit(){if(!this.width)return;const w=this.viewport.clientWidth,h=this.viewport.clientHeight,scale=Math.min(w/this.width,h/this.height);this.canvas.style.width=`${this.width*scale}px`;this.canvas.style.height=`${this.height*scale}px`;this.updateTransformOverlay();this.updateBrushCursor();}
    setBrushSize(value){this.brushSize=clamp(Math.round(Number(value)||1),1,512);this.sizeDisplay.textContent=`Size: ${this.brushSize} px`;if(this.stroke)this.stroke.size=this.brushSize;if(this.shapeDrag?.tool==='line'){const drag=this.shapeDrag;drag.size=this.brushSize;drag.bounds=this.lineBounds(drag.start,drag.bounds.end,drag.size);this.draw();}this.updateBrushCursor();}
    resizeBrushFromWheel(event){
        // Consume wheel only over the drawing viewport, never over the layer
        // list/settings. Diameter is measured in canvas pixels, not graph zoom.
        event.preventDefault();event.stopPropagation();
        if(this.disposed||this.loading||this.error||!['brush','line','eraser'].includes(this.tool)||event.ctrlKey||!Number.isFinite(event.deltaY)||!event.deltaY)return;
        const delta=event.deltaY*(event.deltaMode===1?40:event.deltaMode===2?100:1);
        if(Math.sign(delta)!==Math.sign(this.wheelRemainder)||this.wheelFine!==event.shiftKey)this.wheelRemainder=0;
        this.wheelFine=event.shiftKey;this.wheelRemainder+=delta;
        const steps=Math.trunc(this.wheelRemainder/100);this.wheelRemainder-=steps*100;
        if(steps)this.setBrushSize(this.brushSize-steps*(event.shiftKey?1:4));
        this.updateBrushCursor(event);
    }
    setTool(tool){if(!['select','brush','line','box','ellipse','eraser','transform'].includes(tool))return;this.endStroke();this.endTransform();this.endShape();this.tool=tool;this.wheelRemainder=0;for(const [control,name] of [[this.selectButton,'select'],[this.brushButton,'brush'],[this.lineButton,'line'],[this.boxButton,'box'],[this.ellipseButton,'ellipse'],[this.eraserButton,'eraser'],[this.transformButton,'transform']])control.setAttribute('aria-pressed',tool===name);this.sizeDisplay.style.opacity=['brush','line','eraser'].includes(tool)?'1':'.4';this.centerButton.style.display=tool==='transform'?'':'none';this.updateTransformOverlay();this.updateBrushCursor();}
    hideBrushCursor(){this.cursorPointer=null;this.brushCursor.style.display='none';this.canvas.style.cursor=['select','transform'].includes(this.tool)?'default':'crosshair';}
    updateBrushCursor(event){
        if(event)this.cursorPointer={clientX:event.clientX,clientY:event.clientY};
        if(this.tool==='select'){this.brushCursor.style.display='none';this.canvas.style.cursor='default';return;}
        const pointer=this.cursorPointer,layer=this.current(),r=this.canvas.getBoundingClientRect();
        if(!pointer||this.disposed||this.loading||this.error||this.isFixed(layer)||!layer?.visible||!this.width||r.width<=0||r.height<=0||pointer.clientX<r.left||pointer.clientX>r.right||pointer.clientY<r.top||pointer.clientY>r.bottom){
            this.brushCursor.style.display='none';this.canvas.style.cursor=this.tool==='transform'?'default':'crosshair';return;
        }
        if(this.tool==='transform'){const hit=this.transformDrag?.handle??this.transformHit(this.point(pointer));this.brushCursor.style.display='none';if(hit==='move'||hit==='rotate')this.canvas.style.cursor=this.transformDrag?'grabbing':'grab';else if(PAINT_CORNERS.includes(hit)){const angle=(this.layerFrame()?.angle??0)+(hit==='nw'||hit==='se'?Math.PI/4:-Math.PI/4),index=(Math.round(angle/(Math.PI/4))%4+4)%4;this.canvas.style.cursor=['ew-resize','nwse-resize','ns-resize','nesw-resize'][index];}else this.canvas.style.cursor='default';return;}
        if(['box','ellipse'].includes(this.tool)){this.brushCursor.style.display='none';this.canvas.style.cursor='crosshair';return;}
        const x=(pointer.clientX-r.left)/r.width,y=(pointer.clientY-r.top)/r.height,size=this.stroke?.size??this.brushSize;
        const localWidth=Number.parseFloat(this.canvas.style.width)||this.canvas.clientWidth;
        const localHeight=Number.parseFloat(this.canvas.style.height)||this.canvas.clientHeight;
        Object.assign(this.brushCursor.style,{display:'block',left:`${x*localWidth}px`,top:`${y*localHeight}px`,width:`${size*localWidth/this.width}px`,height:`${size*localHeight/this.height}px`});
        this.canvas.style.cursor='none';
    }
    pickCanvasLayer(event){
        if(event.button!==0||this.tool!=='select'||this.loading||this.error||this.disposed)return;
        event.preventDefault();this.canvas.focus({preventScroll:true});const point=this.point(event),x=Math.floor(point.x),y=Math.floor(point.y);
        if(!Number.isFinite(x)||!Number.isFinite(y)||x<0||y<0||x>=this.width||y>=this.height)return;
        // Pick by effective alpha, not RGB: same-color and black bodies stay
        // distinct. Only read the clicked pixel, never the whole mask image.
        const rendered=this.layerMasks.render();
        for(let index=this.layers.length-1;index>=0;index--){const layer=this.layers[index];if(!layer.visible)continue;
            if(!rendered.get(layer).canvas.getContext('2d').getImageData(x,y,1,1).data[3])continue;
            this.select(layer.id);this.revealSelectedLayer();return;
        }
    }
    revealSelectedLayer(){
        const row=this.list.querySelector('button[aria-pressed=true]')?.parentElement;if(!row)return;
        const bounds=this.list.getBoundingClientRect(),item=row.getBoundingClientRect(),scale=bounds.height/this.list.offsetHeight;if(!(scale>0))return;
        if(item.top<bounds.top)this.list.scrollTop+=(item.top-bounds.top)/scale;else if(item.bottom>bounds.bottom)this.list.scrollTop+=(item.bottom-bounds.bottom)/scale;
    }
    point(e){const r=this.canvas.getBoundingClientRect();return {x:(e.clientX-r.left)*this.width/r.width,y:(e.clientY-r.top)*this.height/r.height};}
    rotationAnchor(bounds){const frame={angle:0,...bounds},sx=(Number.parseFloat(this.canvas.style.width)||this.width)/this.width,sy=(Number.parseFloat(this.canvas.style.height)||this.height)/this.height,w=this.width*sx,h=this.height*sy;let point=paintFramePoint(frame,0,-frame.h/2-24/sy);if(point.x*sx<10||point.x*sx>w-10||point.y*sy<10||point.y*sy>h-10)point=paintFramePoint(frame,0,-frame.h/2+24/sy);return {x:clamp(point.x*sx,10,Math.max(10,w-10))/sx,y:clamp(point.y*sy,10,Math.max(10,h-10))/sy};}
    transformHit(point){const localW=Number.parseFloat(this.canvas.style.width)||this.width,localH=Number.parseFloat(this.canvas.style.height)||this.height,frame=this.layerFrame();if(frame){const anchor=this.rotationAnchor(frame);if(Math.hypot((point.x-anchor.x)*localW/this.width,(point.y-anchor.y)*localH/this.height)<=11)return 'rotate';}return hitPaintFrame(frame,point,7*this.width/localW,7*this.height/localH);}
    updateTransformOverlay(){
        const layer=this.current(),bounds=this.tool==='transform'&&!this.loading&&!this.error&&layer?.visible?this.layerFrame(layer):null;
        this.centerButton.disabled=!bounds||this.disposed;
        if(!bounds||this.disposed){this.transformBox.style.display=this.rotationHandle.style.display=this.rotationLink.style.display='none';return;}
        const sx=(Number.parseFloat(this.canvas.style.width)||this.canvas.clientWidth)/this.width,sy=(Number.parseFloat(this.canvas.style.height)||this.canvas.clientHeight)/this.height;
        const c=Math.cos(bounds.angle),s=Math.sin(bounds.angle);
        Object.assign(this.transformBox.style,{display:'block',left:`${bounds.x*sx}px`,top:`${bounds.y*sy}px`,width:`${bounds.w*sx}px`,height:`${bounds.h*sy}px`,transformOrigin:'50% 50%',transform:`matrix(${c},${s*sy/sx},${-s*sx/sy},${c},0,0)`});
        const anchor=this.rotationAnchor(bounds),top=paintFramePoint(bounds,0,-bounds.h/2),x=anchor.x*sx,y=anchor.y*sy,fromX=top.x*sx,fromY=top.y*sy;
        Object.assign(this.rotationHandle.style,{display:'flex',left:`${x}px`,top:`${y}px`,cursor:this.transformDrag?.handle==='rotate'?'grabbing':'grab'});
        Object.assign(this.rotationLink.style,{display:'block',left:`${fromX}px`,top:`${fromY}px`,width:`${Math.hypot(x-fromX,y-fromY)}px`,transform:`rotate(${Math.atan2(y-fromY,x-fromX)}rad)`});
    }
    beginTransform(e){
        if(e.button!==0||this.loading||this.error||this.transformDrag)return;const layer=this.current();if(!layer?.visible)return;
        const point=this.point(e),handle=this.transformHit(point),bounds=this.layerBounds(layer),frame=this.layerFrame(layer);if(!handle||!bounds||!frame)return;
        this.finishEdit();e.preventDefault();this.canvas.focus();this.canvas.setPointerCapture(e.pointerId);
        // Each drag renders from one immutable original, not the previous frame:
        // no cumulative blur during resize, and integer moves preserve pixels.
        const source=this.captureContent(layer,bounds);
        const center={x:frame.x+frame.w/2,y:frame.y+frame.h/2};this.transformDrag={pointer:e.pointerId,layer,handle,start:point,bounds:{...bounds},frame:{...frame},current:{...frame},center,startAngle:Math.atan2(point.y-center.y,point.x-center.x),angle:frame.angle,source,beforeMask:layer._mask,beforeContent:layer._content,beforeFrame:layer._frame?{...layer._frame}:null};this.updateBrushCursor();
    }
    moveTransform(e){
        const drag=this.transformDrag;if(!drag||e.pointerId!==drag.pointer)return;e.preventDefault();
        const point=this.point(e),dx=point.x-drag.start.x,dy=point.y-drag.start.y;
        let frame;
        if(drag.handle==='rotate'){const delta=Math.atan2(point.y-drag.center.y,point.x-drag.center.x)-drag.startAngle;let angle=drag.frame.angle+delta;if(e.shiftKey)angle=Math.round(angle/(Math.PI/12))*Math.PI/12;angle=Math.atan2(Math.sin(angle),Math.cos(angle));frame={...drag.frame,angle};}
        else frame=drag.handle==='move'?{...drag.frame,x:drag.frame.x+Math.round(dx),y:drag.frame.y+Math.round(dy)}:resizePaintFrame(drag.frame,drag.handle,dx,dy,!e.shiftKey);
        if(['x','y','w','h'].every(k=>Math.abs(frame[k]-drag.current[k])<1e-10)&&Math.abs(frame.angle-drag.current.angle)<1e-12)return;
        try{this.renderFrameTransform(drag,frame);drag.current=frame;drag.angle=frame.angle;this.draw();if(drag.handle==='rotate')this.status.textContent+=` · ${Math.round(frame.angle*180/Math.PI)}°`;}catch(error){this.status.textContent=error.message;}
    }
    captureContent(layer,bounds){const c=layer._content,source=makeCanvas(bounds.w,bounds.h);source.getContext('2d').drawImage(c?.canvas??layer.canvas,bounds.x-(c?.x??0),bounds.y-(c?.y??0),bounds.w,bounds.h,0,0,bounds.w,bounds.h);return source;}
    renderTransform(drag,bounds,angle=0){
        if(!Number.isSafeInteger(bounds.x)||!Number.isSafeInteger(bounds.y))throw Error('Invalid painting position.');
        this.contentLimit(bounds.w,bounds.h,drag.layer);
        const source=!angle&&bounds.w===drag.source.width&&bounds.h===drag.source.height?drag.source:makeCanvas(bounds.w,bounds.h);
        if(source!==drag.source){const ctx=source.getContext('2d');ctx.imageSmoothingQuality='high';if(angle){const original=drag.bounds;ctx.translate(original.x+original.w/2-bounds.x,original.y+original.h/2-bounds.y);ctx.rotate(angle);ctx.drawImage(drag.source,-original.w/2,-original.h/2);}else ctx.drawImage(drag.source,0,0,bounds.w,bounds.h);}
        drag.layer._content={canvas:source,x:bounds.x,y:bounds.y};this.projectContent(drag.layer,{...bounds});
    }
    renderFrameTransform(drag,frame){
        if(!validPaintFrame(frame))throw Error('Invalid painting transform frame.');
        const old=drag.frame,bounds=transformedPaintBounds(drag.bounds,old,frame);
        if(!Number.isSafeInteger(bounds.x)||!Number.isSafeInteger(bounds.y))throw Error('Invalid painting position.');
        this.contentLimit(bounds.w,bounds.h,drag.layer);
        const translation=frame.angle===old.angle&&frame.w===old.w&&frame.h===old.h;
        const source=translation?drag.source:makeCanvas(bounds.w,bounds.h);
        if(!translation){const ctx=source.getContext('2d');ctx.imageSmoothingQuality='high';ctx.translate(frame.x+frame.w/2-bounds.x,frame.y+frame.h/2-bounds.y);ctx.rotate(frame.angle);ctx.scale(frame.w/old.w,frame.h/old.h);ctx.rotate(-old.angle);ctx.translate(-old.x-old.w/2,-old.y-old.h/2);ctx.drawImage(drag.source,drag.bounds.x,drag.bounds.y);}
        drag.layer._content={canvas:source,x:bounds.x,y:bounds.y};this.projectContent(drag.layer);drag.layer._frame={...frame};
    }
    centerLayer(){
        if(this.loading||this.error)return;this.finishEdit();const layer=this.current(),bounds=this.layerBounds(layer);if(!layer?.visible||!bounds)return;
        const frame=this.layerFrame(layer),dx=Math.round((this.width-bounds.w)/2)-bounds.x,dy=Math.round((this.height-bounds.h)/2)-bounds.y;
        try{this.renderFrameTransform({layer,bounds,frame,source:this.captureContent(layer,bounds)},{...frame,x:frame.x+dx,y:frame.y+dy});this.changed();}catch(error){this.status.textContent=error.message;}
    }
    endTransform(e,cancel=false){
        const drag=this.transformDrag;if(!drag||(e&&e.pointerId!==drag.pointer))return;this.transformDrag=null;
        if(cancel){drag.layer._content=drag.beforeContent;if(drag.beforeContent)this.projectContent(drag.layer,drag.bounds);else{const ctx=drag.layer.canvas.getContext('2d');ctx.clearRect(0,0,this.width,this.height);ctx.drawImage(drag.source,drag.bounds.x,drag.bounds.y);this.invalidateLayer(drag.layer,drag.bounds);}drag.layer._mask=drag.beforeMask;drag.layer._frame=drag.beforeFrame;}
        if(this.canvas.hasPointerCapture(drag.pointer))this.canvas.releasePointerCapture(drag.pointer);
        if(cancel){this.commit();this.draw();}else this.changed();
    }
    shapeBounds(start,end,square=false){let dx=end.x-start.x,dy=end.y-start.y;if(square){const size=Math.max(Math.abs(dx),Math.abs(dy));dx=(dx<0?-1:1)*size;dy=(dy<0?-1:1)*size;}return {x:Math.min(start.x,start.x+dx),y:Math.min(start.y,start.y+dy),w:Math.abs(dx),h:Math.abs(dy)};}
    lineBounds(start,end,size,snap=false){
        if(snap){const dx=end.x-start.x,dy=end.y-start.y,length=Math.hypot(dx,dy),angle=Math.round(Math.atan2(dy,dx)/(Math.PI/4))*Math.PI/4;end={x:start.x+Math.cos(angle)*length,y:start.y+Math.sin(angle)*length};}
        const pad=size/2;return {x:Math.min(start.x,end.x)-pad,y:Math.min(start.y,end.y)-pad,w:Math.abs(end.x-start.x)+size,h:Math.abs(end.y-start.y)+size,start,end,size};
    }
    fillShape(context,tool,bounds,color){if(!bounds?.w||!bounds.h)return;context.beginPath();if(tool==='line'){if(bounds.start.x===bounds.end.x&&bounds.start.y===bounds.end.y)return;context.strokeStyle=color;context.lineWidth=bounds.size;context.lineCap='round';context.moveTo(bounds.start.x,bounds.start.y);context.lineTo(bounds.end.x,bounds.end.y);context.stroke();return;}context.fillStyle=color;if(tool==='ellipse')context.ellipse(bounds.x+bounds.w/2,bounds.y+bounds.h/2,bounds.w/2,bounds.h/2,0,0,Math.PI*2);else context.rect(bounds.x,bounds.y,bounds.w,bounds.h);context.fill();}
    beginShape(event){
        if(event.button!==0||!['line','box','ellipse'].includes(this.tool)||this.loading||this.error||this.disposed||this.shapeDrag)return;
        const layer=this.current();if(!layer?.visible||this.isFixed(layer))return;this.finishEdit();event.preventDefault();this.canvas.focus();this.canvas.setPointerCapture(event.pointerId);
        const point=this.point(event);this.shapeDrag={pointer:event.pointerId,layer,tool:this.tool,start:point,size:this.brushSize,bounds:this.tool==='line'?this.lineBounds(point,point,this.brushSize):this.shapeBounds(point,point)};
    }
    moveShape(event){const drag=this.shapeDrag;if(!drag||event.pointerId!==drag.pointer)return;event.preventDefault();drag.bounds=drag.tool==='line'?this.lineBounds(drag.start,this.point(event),drag.size,event.shiftKey):this.shapeBounds(drag.start,this.point(event),event.shiftKey);this.draw();}
    endShape(event,cancel=false){
        const drag=this.shapeDrag;if(!drag||(event&&event.pointerId!==drag.pointer))return;this.shapeDrag=null;drag.layer._shapeTint=null;
        if(this.canvas.hasPointerCapture(drag.pointer))this.canvas.releasePointerCapture(drag.pointer);
        if(cancel||!drag.bounds.w||!drag.bounds.h||(drag.tool==='line'&&drag.bounds.start.x===drag.bounds.end.x&&drag.bounds.start.y===drag.bounds.end.y)){this.draw();return;}
        try{
            const {layer,tool,bounds}=drag;let content=layer._content;
            if(content){
                // Extend preserved transform content only within the paintable
                // canvas, retaining any earlier artwork outside its boundaries.
                const left=Math.max(0,Math.floor(bounds.x)-1),top=Math.max(0,Math.floor(bounds.y)-1),right=Math.min(this.width,Math.ceil(bounds.x+bounds.w)+1),bottom=Math.min(this.height,Math.ceil(bounds.y+bounds.h)+1);
                if(right>left&&bottom>top){const x=Math.min(content.x,left),y=Math.min(content.y,top),w=Math.max(content.x+content.canvas.width,right)-x,h=Math.max(content.y+content.canvas.height,bottom)-y;this.contentLimit(w,h,layer);if(x!==content.x||y!==content.y||w!==content.canvas.width||h!==content.canvas.height){const canvas=makeCanvas(w,h);canvas.getContext('2d').drawImage(content.canvas,content.x-x,content.y-y);content=layer._content={canvas,x,y};}}
            }
            const ctx=(content?.canvas??layer.canvas).getContext('2d');ctx.save();
            if(content){ctx.translate(-content.x,-content.y);ctx.beginPath();ctx.rect(0,0,this.width,this.height);ctx.clip();}
            this.fillShape(ctx,tool,bounds,'#ffffff');ctx.restore();if(content){content.mask=null;this.projectContent(layer);}else this.invalidateLayer(layer);this.changed();
        }catch(error){this.draw();this.status.textContent=error.message;}
    }
    beginStroke(e){if(e.button!==0||!['brush','eraser'].includes(this.tool)||this.loading||this.error)return;const layer=this.current();if(!layer||!layer.visible||this.isFixed(layer))return;this.finishEdit();e.preventDefault();this.canvas.focus();this.canvas.setPointerCapture(e.pointerId);this.stroke={pointer:e.pointerId,layer,point:this.point(e),tool:this.tool,size:this.brushSize};this.paintSegment(this.stroke.point);}
    paintSegment(point,redraw=true){
        const s=this.stroke;if(!s)return;let content=s.layer._content;
        if(content&&s.tool!=='eraser'){
            const pad=s.size/2+1,left=Math.max(0,Math.floor(Math.min(s.point.x,point.x)-pad)),top=Math.max(0,Math.floor(Math.min(s.point.y,point.y)-pad));
            const right=Math.min(this.width,Math.ceil(Math.max(s.point.x,point.x)+pad)),bottom=Math.min(this.height,Math.ceil(Math.max(s.point.y,point.y)+pad));
            if(right>left&&bottom>top){const x=Math.min(content.x,left),y=Math.min(content.y,top),w=Math.max(content.x+content.canvas.width,right)-x,h=Math.max(content.y+content.canvas.height,bottom)-y;
                try{this.contentLimit(w,h,s.layer);}catch(error){s.point=point;this.status.textContent=error.message;return;}
                if(x!==content.x||y!==content.y||w!==content.canvas.width||h!==content.canvas.height){const canvas=makeCanvas(w,h);canvas.getContext('2d').drawImage(content.canvas,content.x-x,content.y-y);content=s.layer._content={canvas,x,y};}
            }
        }
        const ctx=(content?.canvas??s.layer.canvas).getContext('2d');ctx.save();
        if(content){ctx.translate(-content.x,-content.y);ctx.beginPath();ctx.rect(0,0,this.width,this.height);ctx.clip();}
        ctx.globalCompositeOperation=s.tool==='eraser'?'destination-out':'source-over';ctx.strokeStyle=ctx.fillStyle='#ffffff';ctx.lineWidth=s.size;ctx.lineCap=ctx.lineJoin='round';ctx.beginPath();ctx.moveTo(s.point.x,s.point.y);ctx.lineTo(point.x,point.y);ctx.stroke();ctx.beginPath();ctx.arc(point.x,point.y,s.size/2,0,Math.PI*2);ctx.fill();ctx.restore();s.point=point;
        if(content){content.mask=null;if(redraw)this.projectContent(s.layer);else this.invalidateLayer(s.layer);}else this.invalidateLayer(s.layer);if(redraw)this.draw();
    }
    moveStroke(e){if(!this.stroke||e.pointerId!==this.stroke.pointer)return;e.preventDefault();const points=e.getCoalescedEvents?.();for(const p of points?.length?points:[e])this.paintSegment(this.point(p),false);if(this.stroke.layer._content)this.projectContent(this.stroke.layer);this.requestDraw();}
    endStroke(e){if(!this.stroke||(e&&e.pointerId!==this.stroke.pointer))return;const pointer=this.stroke.pointer;this.stroke=null;if(this.canvas.hasPointerCapture(pointer))this.canvas.releasePointerCapture(pointer);this.changed();}
    requestDraw(){if(this.disposed||this.drawFrame)return;this.drawFrame=requestAnimationFrame(()=>{this.drawFrame=0;if(!this.disposed)this.draw();});}
    draw(){
        if(this.drawFrame){cancelAnimationFrame(this.drawFrame);this.drawFrame=0;}
        this.updateTransformOverlay();this.updateBrushCursor();const ctx=this.canvas.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,this.canvas.width,this.canvas.height);
        const rendered=this.layerMasks.render();for(const layer of this.layers)if(layer.visible)ctx.drawImage(rendered.get(layer).canvas,0,0);
        if(this.error)return;this.status.textContent=this.loading?'Loading layers…':`${this.width} × ${this.height} · ${this.layers.length} layers`;
    }
    renderList(){const scrollTop=this.list.scrollTop;this.list.replaceChildren();for(const layer of [...this.layers].reverse()){const row=document.createElement('div');row.style.cssText='display:flex;gap:5px;align-items:center;flex-shrink:0';const visible=Object.assign(document.createElement('input'),{type:'checkbox',checked:layer.visible,title:'Include this layer in preview and outputs'});visible.onchange=()=>{if(this.loading)return;this.finishEdit();layer.visible=visible.checked;this.changed();};const select=document.createElement('button');select.type='button';select.style.cssText='flex:1;text-align:left;overflow:hidden;white-space:nowrap;text-overflow:ellipsis';select.setAttribute('aria-pressed',layer.id===this.selected);const swatch=document.createElement('span');swatch.style.cssText=`display:inline-block;width:12px;height:12px;background:${layer.color};margin-right:6px;border:1px solid #999`;select.title=layer.prompt||'Enter a region prompt';select.append(swatch,document.createTextNode(this.layerLabel(layer)));select.onclick=()=>{this.select(layer.id);this.canvas.focus({preventScroll:true});};row.append(visible,select);this.list.append(row);}this.list.scrollTop=scrollTop;}
    renderPanel(){
        const layer=this.current(),background=this.isBackground(layer),base=this.isBase(layer),fixed=this.isFixed(layer),busy=!!(this.loading||this.error);
        this.renderList();this.backgroundButton.setAttribute('aria-pressed',String(background));
        this.renderMaskOptions();
        this.baseButton.setAttribute('aria-pressed',String(base));
        for(const [input,key] of [[this.colorInput,'color'],[this.strengthInput,'strength'],[this.promptInput,'prompt']]){
            input.disabled=!layer||busy||(fixed&&key!=='prompt');input.value=layer?.[key]??(key==='color'?'#ffffff':'');
        }
        this.colorStrengthRow.style.display=fixed?'none':'flex';
        this.backgroundModeRow.style.display=background?'flex':'none';
        this.backgroundModeInput.value=this.background.mode;this.backgroundModeInput.disabled=!background||busy;
        this.promptTitle.textContent=background?'Background prompt':base?'Base prompt':'Region prompt';
        this.promptInput.placeholder=background?'Describe the background…':base?'Enter a shared base prompt…':'Describe this painted region…';
        const index=this.layers.indexOf(layer);
        this.deleteButton.disabled=busy||!layer||fixed||this.poseSource.isImported(layer);this.deleteButton.title=this.poseSource.isImported(layer)?'Delete this body in 3D Pose Editor. Its original mask is protected.':'Delete layer';
        if(this.poseSource.isImported(layer)){this.colorInput.disabled=true;this.colorInput.title='Body color follows 3D Pose Editor. Only extra painting is editable.';}else this.colorInput.title='Layer color';
        this.backButton.disabled=busy||index<=0;this.frontButton.disabled=busy||index<0||index>=this.layers.length-1;
        this.addButton.disabled=busy||this.layers.length>=MAX_LAYERS;
        this.selectButton.disabled=busy||!this.layers.length;
        for(const control of [this.brushButton,this.lineButton,this.boxButton,this.ellipseButton,this.eraserButton,this.transformButton,this.clearButton])control.disabled=busy||!layer||fixed;
        for(const control of [this.copyMaskButton,this.pasteMaskButton])control.disabled=busy||this.clipboardBusy||!layer||fixed;
        this.updateHistory();this.updateTransformOverlay();this.updateBrushCursor();
    }
    dispose(){this.maskPicker.dispose();this.poseBinding?.dispose();this.endPanelResize(null,true);this.endListResize(null,true);this.disposed=true;if(this.drawFrame){cancelAnimationFrame(this.drawFrame);this.drawFrame=0;}this.loadToken++;this.observer.disconnect();window.removeEventListener('blur',this.blurCursor);this.hideBrushCursor();this.transformBox.style.display=this.rotationHandle.style.display=this.rotationLink.style.display='none';this.stroke=this.transformDrag=this.shapeDrag=null;this.layers=[];this.history=[];this.future=[];}
}
