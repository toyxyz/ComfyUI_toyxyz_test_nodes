import {app} from '../../scripts/app.js';
import {CoupleRegionPainterUI} from './util/couple_region_painter_ui.js';
import {bindPoseRegions} from './util/pose_region_bridge.js';
const ensurePoseInput=node=>{if(!node.inputs?.some(input=>input.name==='region_painter'))node.addInput('region_painter','TOYXYZ_REGION_PAINTER');};

app.registerExtension({name:'toyxyz.couple_region_painter',async beforeRegisterNodeDef(NodeType,data){
    if(data.name!=='ComfyCoupleRegionPainter')return;
    const created=NodeType.prototype.onNodeCreated;
    NodeType.prototype.onNodeCreated=function(){
        created?.apply(this,arguments);
        ensurePoseInput(this);const root=document.createElement('div'),ui=this._couplePainter=new CoupleRegionPainterUI(this,root),binding=bindPoseRegions(this,ui);
        // Let ComfyUI allocate the node's remaining height to this editor.
        // A fixed max height prevented vertical node resizing from reaching
        // the viewport. Its ResizeObserver only fits CSS display dimensions;
        // width/height widgets still control the real mask/output resolution.
        const dom=this.addDOMWidget('region_painter','div',root,{serialize:false,hideOnZoom:false,getMinHeight:()=>360});dom.serialize=false;
        const storage=this.widgets.find(w=>w.name==='paint_data');
        storage.hidden=true;storage.options??={};storage.options.hidden=true;storage.computeSize=()=>[0,-4];
        if(storage.element)storage.element.style.display='none';if(storage.inputEl)storage.inputEl.style.display='none';
        storage.serializeValue=async()=>{await binding.syncNow(true);return ui.serialize();};
        for(const name of ['width','height']){const widget=this.widgets.find(w=>w.name===name),previous=widget.callback;widget.callback=(...args)=>{previous?.apply(widget,args);if(!binding.syncResolutionWidgets())ui.resizeCanvas();binding.last=null;binding.schedule();};}
        const connections=this.onConnectionsChange;this.onConnectionsChange=function(...args){const result=connections?.apply(this,args);binding.last=null;binding.schedule();return result;};
        const configured=this.onConfigure;this.onConfigure=function(){configured?.apply(this,arguments);ensurePoseInput(this);ui.ready=ui.load();ui.ready.catch(()=>{});binding.last=null;binding.schedule();if(this.size[0]<820)this.setSize([820,this.size[1]]);};
        const removed=this.onRemoved;this.onRemoved=function(){ui.dispose();removed?.apply(this,arguments);};
        ui.ready.catch(()=>{});this.setSize([900,Math.max(660,this.computeSize?.()[1]||0)]);
    };
}});
