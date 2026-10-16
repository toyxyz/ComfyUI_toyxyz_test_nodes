// Clipping uses layer alpha before visibility/stack occlusion, never changes
// stored painting, and stays on the canvas/GPU path during brush gestures.
export function maskWouldCycle(layers,layerId,targetId){
    const byId=new Map(layers.map(layer=>[layer.id,layer])),seen=new Set([layerId]);
    while(targetId&&byId.has(targetId)){if(seen.has(targetId))return true;seen.add(targetId);targetId=byId.get(targetId).mask_layer_id;}
    return false;
}
export function validateLayerMasks(layers){
    for(const layer of layers){const id=layer.mask_layer_id;if(id==null||id==='')continue;if(typeof id!=='string'||maskWouldCycle(layers,layer.id,id))throw Error('Invalid layer Mask reference: a layer cannot mask itself or form a cycle.');}
}
export class PainterLayerMasks {
    constructor(ui){this.ui=ui;this.version=0;}
    render(){
        const ui=this.ui,byId=new Map(ui.layers.map(layer=>[layer.id,layer])),memo=new Map(),visiting=new Set();
        const make=()=>Object.assign(document.createElement('canvas'),{width:ui.width,height:ui.height});
        const resolve=layer=>{
            if(memo.has(layer))return memo.get(layer);
            if(visiting.has(layer))throw Error('Layer Mask cycle.');visiting.add(layer);
            const source=ui.poseSource.body(layer)?ui.poseSource.state:null;
            if(!layer._tint||layer._tintDirty||layer._tintSource!==source){
                const tint=layer._tint??(layer._tint=make()),ctx=tint.getContext('2d');ctx.globalCompositeOperation='source-over';ctx.clearRect(0,0,ui.width,ui.height);ctx.drawImage(layer.canvas,0,0);ui.poseSource.drawSeed(layer,ctx);ctx.globalCompositeOperation='source-in';ctx.fillStyle=layer.color;ctx.fillRect(0,0,ui.width,ui.height);layer._tintDirty=false;layer._tintSource=source;layer._tintVersion=++this.version;
            }
            let raw=layer._tint,version=layer._tintVersion;
            if(ui.shapeDrag?.layer===layer){raw=layer._shapeTint??(layer._shapeTint=make());if(raw.width!==ui.width||raw.height!==ui.height){raw.width=ui.width;raw.height=ui.height;}const ctx=raw.getContext('2d');ctx.clearRect(0,0,ui.width,ui.height);ctx.drawImage(layer._tint,0,0);ui.fillShape(ctx,ui.shapeDrag.tool,ui.shapeDrag.bounds,layer.color);version=++this.version;}
            const target=byId.get(layer.mask_layer_id);let result={canvas:raw,version};
            if(target){
                const mask=resolve(target),key=`${version}:${target.id}:${mask.version}:${ui.width}x${ui.height}`;
                if(layer._maskTintKey!==key){const tint=layer._maskTint??(layer._maskTint=make());if(tint.width!==ui.width||tint.height!==ui.height){tint.width=ui.width;tint.height=ui.height;}const ctx=tint.getContext('2d');ctx.globalCompositeOperation='source-over';ctx.clearRect(0,0,ui.width,ui.height);ctx.drawImage(raw,0,0);ctx.globalCompositeOperation='destination-in';ctx.drawImage(mask.canvas,0,0);ctx.globalCompositeOperation='source-over';layer._maskTintKey=key;layer._maskTintVersion=++this.version;}
                result={canvas:layer._maskTint,version:layer._maskTintVersion};
            }else{layer._maskTint=null;layer._maskTintKey=null;}
            visiting.delete(layer);memo.set(layer,result);return result;
        };
        for(const layer of ui.layers)if(layer.visible)resolve(layer);
        return memo;
    }
}
