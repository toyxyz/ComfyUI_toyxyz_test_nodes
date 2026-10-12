// Frontend previews follow connected editors without queuing model inference.
// The normal ComfyUI graph transports the same ID map on actual execution.
const bindings=new Set(),captures=new WeakMap();
const sourceStamp=source=>{const editor=source?._pose3d;return `${editor?.regionRevision??0}:${editor?.widget('width')?.value}x${editor?.widget('height')?.value}`;};
export function linkedPoseEditor(node){
    let graph=node.graph,slot=node.inputs?.find(input=>input.name==='region_painter'),linkId=slot?.link;const seen=new Set();
    while(linkId!=null){const link=graph?.getLink?.(linkId)??graph?.links?.[linkId],source=graph?.getNodeById?.(link?.origin_id);if(!source||seen.has(source))return null;seen.add(source);if(source._pose3d)return source;if(!source.isVirtualNode)return null;graph=source.graph??graph;linkId=source.inputs?.[0]?.link;}
    return null;
}
export function notifyPoseRegions(node){for(const binding of bindings)if(linkedPoseEditor(binding.node)===node)binding.schedule();}
export function bindPoseRegions(node,ui){
    const binding={node,ui,timer:0,promise:null,disposed:false,last:null,
        syncResolutionWidgets(){
            if(this.disposed||ui.disposed)return false;const editor=linkedPoseEditor(node)?._pose3d;if(!editor||editor.disposed)return false;
            const values=['width','height'].map(name=>editor.widget(name)?.value);if(!values.every(value=>Number.isInteger(value)&&value>=64&&value<=4096))return false;
            let changed=false;['width','height'].forEach((name,index)=>{const widget=ui.widget(name);if(widget.value!==values[index]){widget.value=values[index];changed=true;}});
            if(changed)node.setDirtyCanvas?.(true,true);return true;
        },
        schedule(){if(this.disposed)return;this.syncResolutionWidgets();clearTimeout(this.timer);this.timer=setTimeout(()=>this.syncNow().catch(error=>{if(!ui.disposed)ui.status.textContent=`Pose source: ${error.message}`;}),120);},
        async syncNow(force=false){
            if(this.disposed||ui.disposed||ui.error)return;if(this.promise){await this.promise;return this.syncNow(force);}
            this.syncResolutionWidgets();
            const source=linkedPoseEditor(node),editor=source?._pose3d;if(!editor||editor.disposed)return;
            if(!force&&(ui.stroke||ui.shapeDrag||ui.transformDrag||ui.loading)){this.schedule();return;}
            this.promise=(async()=>{
                await editor.ready;await ui.ready;if(ui.error)return;if(editor.loading||ui.loading){this.schedule();return;}
                this.syncResolutionWidgets();
                if(force)ui.finishEdit();
                const stamp=sourceStamp(source),token=ui.loadToken,key=`${stamp}:${token}`,paintState=ui.widget('paint_data').value;
                if(!force&&this.last?.source===source&&this.last.key===key)return;
                let cached=captures.get(editor);if(!cached||cached.stamp!==stamp){cached={stamp,payload:editor.captureRegionPainter()};captures.set(editor,cached);}
                const applied=await ui.poseSource.apply(cached.payload,()=>!this.disposed&&!ui.disposed&&!ui.error&&!editor.loading&&linkedPoseEditor(node)===source&&sourceStamp(source)===stamp&&ui.loadToken===token&&ui.widget('paint_data').value===paintState&&!ui.stroke&&!ui.shapeDrag&&!ui.transformDrag&&!ui.loading);
                if(applied)this.last={source,key:`${stamp}:${ui.loadToken}`};else this.schedule();
            })();
            try{await this.promise;}finally{this.promise=null;}
        },
        dispose(){this.disposed=true;clearTimeout(this.timer);clearInterval(this.poll);bindings.delete(this);},
    };
    bindings.add(binding);ui.poseBinding=binding;
    // Also catch reroute reconnections and workflow/Undo reloads, which do not
    // necessarily generate an edit notification from the upstream scene.
    binding.poll=setInterval(()=>{binding.syncResolutionWidgets();const source=linkedPoseEditor(node);if(source&&(binding.last?.source!==source||binding.last.key!==`${sourceStamp(source)}:${ui.loadToken}`))binding.schedule();},400);
    binding.schedule();return binding;
}
