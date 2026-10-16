import {validPaintFrame,MAX_PAINT_AXIS} from './couple_painter_geometry.js';
import {maskWouldCycle} from './couple_layer_masks.js';
const MAX_PIXELS=64*1024*1024,MAX_LAYERS=32;
const canvas=(w,h)=>Object.assign(document.createElement('canvas'),{width:w,height:h});
export const poseKey=owner=>JSON.stringify([owner.source_id,owner.body_id]);
const color=value=>typeof value==='string'&&/^#[0-9a-f]{6}$/i.test(value);
const identity=owner=>owner&&typeof owner.source_id==='string'&&owner.source_id.length>0&&owner.source_id.length<=256&&typeof owner.body_id==='string'&&owner.body_id.length>0&&owner.body_id.length<=80;
const png=value=>typeof value==='string'&&value.startsWith('data:image/png;base64,')&&value.length<=96*1024*1024;
const validSize=value=>Number.isInteger(value)&&value>=64&&value<=4096;
async function imageOf(uri,w,h){if(!png(uri))throw Error('Invalid pose-source image.');const image=new Image();image.src=uri;await image.decode();if(image.width!==w||image.height!==h)throw Error('Pose-source image resolution mismatch.');return image;}

export async function parsePoseSource(payload){
    if(payload==null)return null;
    if(payload.version!==1||!validSize(payload.width)||!validSize(payload.height)||typeof payload.source_id!=='string'||!payload.source_id||payload.source_id.length>256||!Array.isArray(payload.bodies)||payload.bodies.length>256||!png(payload.id_map))throw Error('Invalid pose-region source.');
    const ids=new Set(),labels=new Set(),bodies=payload.bodies.map(body=>{if(!identity({source_id:payload.source_id,body_id:body.id})||ids.has(body.id)||!Number.isInteger(body.label)||body.label<1||body.label>256||labels.has(body.label)||!color(body.color)||typeof body.name!=='string'||body.name.length>160)throw Error('Invalid pose-source body.');ids.add(body.id);labels.add(body.label);return {id:body.id,label:body.label,name:body.name,color:body.color};});
    const image=await imageOf(payload.id_map,payload.width,payload.height),c=canvas(payload.width,payload.height),ctx=c.getContext('2d');ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,c.width,c.height).data,map=new Uint32Array(c.width*c.height);
    const bounds=new Map(bodies.map(body=>[body.label,{x:payload.width,y:payload.height,right:-1,bottom:-1}]));
    for(let i=0;i<map.length;i++){const label=pixels[i*4]+(pixels[i*4+1]<<8)+(pixels[i*4+2]<<16);if(label&&!labels.has(label))throw Error('Pose-source map contains an unknown body ID.');map[i]=label;if(label){const b=bounds.get(label),x=i%payload.width,y=Math.floor(i/payload.width);b.x=Math.min(b.x,x);b.y=Math.min(b.y,y);b.right=Math.max(b.right,x);b.bottom=Math.max(b.bottom,y);}}
    const clean={version:1,source_id:payload.source_id,width:payload.width,height:payload.height,bodies,id_map:payload.id_map};
    return {payload:clean,labels:map,bounds,seeds:new Map(),byId:new Map(bodies.map(body=>[body.id,body])),signature:JSON.stringify(clean)};
}

export function parsePoseArchive(records=[]){
    if(!Array.isArray(records)||records.length>32)throw Error('Invalid pose-layer archive.');
    const map=new Map();for(const item of records){const layer=item?.layer;if(!validSize(item?.width)||!validSize(item?.height)||!identity(layer?.pose_source)||!png(layer?.mask)||typeof layer.prompt!=='string'||!color(layer.color)||typeof layer.visible!=='boolean'||!Number.isFinite(layer.strength)||layer.strength<0||layer.strength>10)throw Error('Invalid archived pose layer.');map.set(poseKey(layer.pose_source),item);}return map;
}

export class PainterPoseSource {
    constructor(ui){this.ui=ui;this.state=null;this.archives=new Map();}
    restore(state,archives){this.state=state;this.archives=archives;}
    isImported(layer){return identity(layer?.pose_source);}
    body(layer){return this.state&&layer?.pose_source?.source_id===this.state.payload.source_id?this.state.byId.get(layer.pose_source.body_id):null;}
    label(layer){return this.body(layer)?.name;}
    snapshot(){return this.state?this.state.payload:null;}
    detach(){
        const ui=this.ui;if(!this.state&&!this.archives.size&&!ui.layers.some(layer=>this.isImported(layer)))return false;
        ui.finishEdit();
        ui.layers=ui.layers.filter(layer=>!this.isImported(layer));this.state=null;this.archives.clear();
        for(const layer of ui.layers){layer._tintSource=null;layer._tintDirty=true;}
        if(!ui.isFixed()&&!ui.layers.some(layer=>layer.id===ui.selected))ui.selected=ui.layers.at(-1)?.id??ui.background.id;
        // Keep manual painting history, but never resurrect a disconnected
        // body layer, protected ID map or archived body via Undo / Redo.
        const clean=value=>{const data=JSON.parse(value);data.layers=data.layers.filter(layer=>!this.isImported(layer));delete data.pose_source;delete data.pose_archive;if(![ui.base.id,ui.background.id].includes(data.selected)&&!data.layers.some(layer=>layer.id===data.selected))data.selected=data.layers.at(-1)?.id??ui.background.id;return JSON.stringify(data);};
        const history=ui.history.map(clean);ui.history=history.filter((value,index)=>index===0||value!==history[index-1]);ui.future=ui.future.map(clean);
        ui.commit();ui.record();ui.renderPanel();ui.draw();ui.fit();return true;
    }
    seedPatch(layer){
        const state=this.state,body=this.body(layer);if(!body)return null;
        if(state.seeds.has(body.label))return state.seeds.get(body.label);
        const b=state.bounds.get(body.label);let patch=null;
        if(b.right>=b.x){
            const w=b.right-b.x+1,h=b.bottom-b.y+1,c=canvas(w,h),ctx=c.getContext('2d'),pixels=ctx.createImageData(w,h),sourceW=state.payload.width;
            for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(state.labels[(b.y+y)*sourceW+b.x+x]===body.label){const i=(y*w+x)*4;pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=pixels.data[i+3]=255;}
            ctx.putImageData(pixels,0,0);patch={canvas:c,x:b.x,y:b.y};
        }
        state.seeds.set(body.label,patch);return patch;
    }
    drawSeed(layer,ctx){
        const patch=this.seedPatch(layer);if(!patch)return;
        const sx=this.ui.width/this.state.payload.width,sy=this.ui.height/this.state.payload.height;
        ctx.save();ctx.imageSmoothingEnabled=false;ctx.drawImage(patch.canvas,patch.x*sx,patch.y*sy,patch.canvas.width*sx,patch.canvas.height*sy);ctx.restore();
    }
    maskCanvas(layer){
        if(!this.body(layer))return layer.canvas;
        // Source protection is local to its layer, not a priority above every
        // other layer. Union seed + extra paint, then use normal stack order.
        // Build seed pixels once when the Pose map changes. The hot drawing
        // path uses drawSeed directly: no full-image readback or allocations.
        const ui=this.ui,c=canvas(ui.width,ui.height),ctx=c.getContext('2d');ctx.drawImage(layer.canvas,0,0);this.drawSeed(layer,ctx);return c;
    }
    async archivedLayer(record,w,h){
        const item=record.layer,image=await imageOf(item.mask,record.width,record.height),c=canvas(w,h);c.getContext('2d').drawImage(image,0,0,w,h);
        const layer={...item,canvas:c,_mask:null,_bounds:undefined,_frame:null};
        if(item.content){const saved=item.content;const sw=Math.max(1,Math.round(saved.width*w/record.width)),sh=Math.max(1,Math.round(saved.height*h/record.height));const src=await imageOf(saved.mask,saved.width,saved.height),content=canvas(sw,sh);content.getContext('2d').drawImage(src,0,0,sw,sh);layer._content={canvas:content,x:Math.round(saved.x*w/record.width),y:Math.round(saved.y*h/record.height)};this.ui.projectContent(layer);}
        if(item.transform_frame&&w/record.width===h/record.height){const f=item.transform_frame,s=w/record.width;layer._frame={...f,x:f.x*s,y:f.y*s,w:f.w*s,h:f.h*s};if(!validPaintFrame(layer._frame))throw Error('Invalid archived transform frame.');}return layer;
    }
    async apply(payload,guard=()=>true){
        const ui=this.ui,signature=JSON.stringify(payload),state=this.state?.signature===signature?this.state:await parsePoseSource(payload),{width:w,height:h,source_id:sourceId,bodies}=state.payload;
        const initial=ui.layers.length===1&&!this.state&&ui.history.length===1&&ui.layers[0].prompt===''&&!this.isImported(ui.layers[0])&&!ui.layerBounds(ui.layers[0])?ui.layers[0]:null;
        const ordinary=ui.layers.filter(layer=>!this.isImported(layer)&&layer!==initial);if(ordinary.length+bodies.length>MAX_LAYERS||w*h*Math.max(1,ordinary.length+bodies.length)>MAX_PIXELS)throw Error('Pose source exceeds 32 layers / 64 million layer pixels. Reduce bodies or resolution.');
        const current=new Map(ui.layers.filter(layer=>this.isImported(layer)).map(layer=>[poseKey(layer.pose_source),layer])),restored=new Map();
        // Preflight the whole incoming scene (including off-canvas artwork)
        // before decoding archives or mutating the current layers/widgets.
        let pixels=w*h*Math.max(1,ordinary.length+bodies.length);
        const check=(content,frame,oldW,oldH)=>{
            const sx=w/oldW,sy=h/oldH;
            if(frame&&!validPaintFrame(frame))throw Error('Invalid archived transform frame.');
            if(content){const cw=Math.max(1,Math.round(content.width*sx)),ch=Math.max(1,Math.round(content.height*sy));pixels+=cw*ch;if(!Number.isSafeInteger(content.x)||!Number.isSafeInteger(content.y)||!Number.isInteger(content.width)||!Number.isInteger(content.height)||content.width<1||content.height<1||content.width>MAX_PAINT_AXIS||content.height>MAX_PAINT_AXIS||cw>MAX_PAINT_AXIS||ch>MAX_PAINT_AXIS||!Number.isSafeInteger(Math.round(content.x*sx))||!Number.isSafeInteger(Math.round(content.y*sy))||pixels>MAX_PIXELS)throw Error('Pose source exceeds the editable painting pixel limit.');}
            if(frame&&Math.abs(sx-sy)<1e-10&&!validPaintFrame({...frame,x:frame.x*sx,y:frame.y*sy,w:frame.w*sx,h:frame.h*sy}))throw Error('Pose-source resize exceeds the transform-frame limit.');
        };
        const live=layer=>check(layer._content?{...layer._content,width:layer._content.canvas.width,height:layer._content.canvas.height}:null,layer._frame,ui.width,ui.height);
        ordinary.forEach(live);
        for(const body of bodies){const key=JSON.stringify([sourceId,body.id]),layer=current.get(key),archive=this.archives.get(key);if(layer)live(layer);else if(archive)check(archive.layer.content,archive.layer.transform_frame,archive.width,archive.height);}
        for(const body of bodies){const key=JSON.stringify([sourceId,body.id]);if(!current.has(key)&&this.archives.has(key))restored.set(key,await this.archivedLayer(this.archives.get(key),w,h));}
        if(!guard())return false;
        const nextKeys=new Set(bodies.map(body=>JSON.stringify([sourceId,body.id])));
        for(const [key,layer] of current)if(!nextKeys.has(key)){this.archives.delete(key);this.archives.set(key,{width:ui.width,height:ui.height,layer:ui.snapshotLayer(layer)});}
        while(this.archives.size>32)this.archives.delete(this.archives.keys().next().value);
        ui.layers=ui.layers.filter(layer=>layer!==initial&&(!this.isImported(layer)||nextKeys.has(poseKey(layer.pose_source))));
        if(ui.width!==w||ui.height!==h){ui.widget('width').value=w;ui.widget('height').value=h;ui.resizeCanvas(false);if(ui.width!==w||ui.height!==h)throw Error('Pose-source resolution could not be applied.');}
        for(const body of bodies){const key=JSON.stringify([sourceId,body.id]);let layer=current.get(key);if(!layer){layer=restored.get(key)??{id:'pose:'+key,name:body.name,color:body.color,prompt:'',strength:1,visible:true,canvas:canvas(w,h),pose_source:{source_id:sourceId,body_id:body.id}};ui.layers.push(layer);}if(layer.color!==body.color)layer._tintDirty=true;layer.color=body.color;layer.name=body.name;this.archives.delete(key);}
        // Hidden layers must not keep an old ID map / seed-cache alive solely
        // through their tint cache token after the upstream pose changes.
        if(this.state!==state)for(const layer of ui.layers)layer._tintSource=null;
        // A restored body may complete a previously dangling Mask chain.
        // Drop only that cyclic body reference instead of blocking live poses.
        for(const layer of ui.layers)if(this.isImported(layer)&&maskWouldCycle(ui.layers,layer.id,layer.mask_layer_id))delete layer.mask_layer_id;
        this.state=state;if(!ui.layers.some(layer=>layer.id===ui.selected)&&!ui.isFixed())ui.selected=ui.layers.at(-1)?.id??ui.background.id;
        ui.commit();ui.renderPanel();ui.draw();ui.fit();return true;
    }
}
