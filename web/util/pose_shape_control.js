// Local MHR rest-surface editing. No Python package or inference is needed.
const SHAPE_SCHEMA='MHR.identity45.v1';
let basisPromise;
export function validMHRShape(shape){
    return shape?.schema===SHAPE_SCHEMA&&Array.isArray(shape.coefficients)&&shape.coefficients.length===45&&
        shape.coefficients.every(value=>Number.isFinite(value)&&Math.abs(value)<=64);
}
export async function loadMHRShapeBasis(){
    if(!basisPromise)basisPromise=(async()=>{
        const response=await fetch(new URL('../vendor/pose3d/mhr_identity45.bin.gz',import.meta.url));
        if(!response.ok)throw Error(`Shape data: HTTP ${response.status}`);
        const stream=new Blob([await response.arrayBuffer()]).stream().pipeThrough(new DecompressionStream('gzip'));
        const bytes=await new Response(stream).arrayBuffer();
        if(bytes.byteLength!==45*18439*3*4)throw Error('Invalid MHR identity basis size.');
        const data=new Float32Array(bytes);
        if(!data.every(Number.isFinite))throw Error('Invalid MHR identity basis values.');
        return data;
    })();
    try{return await basisPromise;}catch(error){basisPromise=undefined;throw error;}
}

function element(tag,parent,css,text){const item=document.createElement(tag);if(css)item.style.cssText=css;if(text)item.textContent=text;parent?.append(item);return item;}
const buttonStyle='background:#333;color:#ddd;border:1px solid #606060;border-radius:4px;padding:4px 7px;cursor:pointer;font:inherit';
const inputStyle='background:#202020;color:#eee;border:1px solid #575757;border-radius:3px;box-sizing:border-box;font:inherit';

export class PoseModelTree {
    constructor(parent,onAdd,onDelete,onScene){
        const panelStyle='flex:0 0 auto;min-height:0;display:flex;flex-direction:column;gap:7px;border:1px solid #505050;border-radius:5px;padding:8px;box-sizing:border-box;background:#242424;overflow:hidden';
        this.root=element('aside',parent,panelStyle+';width:260px;min-width:230px;max-width:42%');
        element('div',this.root,'font-weight:600','Models');
        const actions=element('div',this.root,'display:flex;gap:5px');
        this.addButton=element('button',actions,buttonStyle+';flex:1','Add body');this.addButton.onclick=onAdd;
        this.deleteButton=element('button',actions,buttonStyle+';flex:1','Delete body');this.deleteButton.onclick=onDelete;
        this.deleteButton.title='Delete the selected body (Undo restores it)';
        this.tree=element('div',this.root,'flex:1;min-height:0;overflow-y:auto;overscroll-behavior:contain;padding-right:3px');this.tree.setAttribute('role','tree');this.tree.setAttribute('aria-label','Models');
        this.sceneRow=element('div',this.tree,'display:flex;gap:3px;align-items:center');
        this.expand=element('button',this.sceneRow,buttonStyle,'▾');this.expand.title='Collapse / expand models';this.expand.setAttribute('aria-expanded','true');
        this.sceneButton=element('button',this.sceneRow,buttonStyle+';text-align:left;flex:1','Scene');this.sceneButton.setAttribute('role','treeitem');
        this.branch=element('div',this.tree,'margin-left:13px;padding:5px 0 0 10px;border-left:1px solid #666');this.branch.setAttribute('role','group');
        this.expand.onclick=()=>{const open=this.branch.hidden;this.branch.hidden=!open;this.expand.textContent=open?'▾':'▸';this.expand.setAttribute('aria-expanded',String(open));};
        this.sceneButton.onclick=onScene;this.setActions(false,false);
    }
    setActions(loading,selected){
        this.addButton.disabled=loading;this.deleteButton.disabled=loading||!selected;
        for(const button of [this.addButton,this.deleteButton]){button.style.opacity=button.disabled?'.45':'1';button.style.cursor=button.disabled?'default':'pointer';}
    }
}

export class PoseShapeControl {
    constructor(tree,onChange,onFinish,onSelect,onColorChange){
        this.onChange=onChange;this.onFinish=onFinish;this.onSelect=onSelect;this.onColorChange=onColorChange;
        // Per-body properties retain their immutable rest surface and coefficients
        // even when another model is selected. Only +/- expands the subtree.
        for(const key of ['root','tree','branch','sceneButton','expand'])this[key]=tree[key];
        this.entry=element('div',this.branch,'margin-bottom:6px');
        this.modelRow=element('div',this.entry,'display:flex;align-items:center;gap:4px');
        this.modelButton=element('button',this.modelRow,buttonStyle+';flex:1;min-width:0;text-align:left','◈ Human mesh');this.modelButton.setAttribute('role','treeitem');
        this.parameterToggle=element('button',this.modelRow,buttonStyle+';flex:0 0 26px;padding:4px 0','+');
        this.parameterToggle.onclick=event=>{event.stopPropagation();this.setExpanded(!this.parametersExpanded);};
        this.modelButton.onclick=()=>this.onSelect?.();
        this.inspector=element('div',this.entry,'margin-top:8px;padding:3px 0 0 3px');this.inspector.setAttribute('aria-label','Body parameters');
        this.rows=[];this.select(true);this.setExpanded(false);
    }
    setExpanded(expanded){
        this.parametersExpanded=expanded;this.inspector.hidden=!expanded;
        this.parameterToggle.textContent=expanded?'−':'+';
        this.parameterToggle.setAttribute('aria-expanded',String(expanded));
        this.parameterToggle.title=expanded?'Collapse body parameters':'Expand body parameters';
        this.parameterToggle.setAttribute('aria-label',this.parameterToggle.title);
    }
    select(selected){
        this.modelSelected=selected;this.modelButton.setAttribute('aria-selected',String(selected));
        this.modelButton.style.backgroundColor=selected?'#36536b':'#333';this.modelButton.style.borderColor=selected?'#83a9c9':'#606060';
    }
    bind(mesh,data,basis,error){
        this.mesh=mesh;this.modelKind=data.kind;this.basis=basis;this.rows=[];this.inspector.replaceChildren();
        this.modelButton.textContent=`◈ Human mesh (${data.kind??'Body'})`;
        this.defaults=data.shape?.coefficients?.slice();this.values=this.defaults?.slice();
        this.base=mesh?Float32Array.from(mesh.geometry.attributes.position.array):null;
        this.enabled=data.kind==='MHR'&&validMHRShape(data.shape)&&!!basis&&this.base?.length===18439*3;
        const colorRow=element('label',this.inspector,'display:flex;align-items:center;gap:8px;margin-bottom:10px');
        element('span',colorRow,'flex:1','Body color');
        this.colorPicker=element('input',colorRow,inputStyle+';width:56px;height:28px;padding:2px;cursor:pointer');this.colorPicker.type='color';this.colorPicker.value='#d6a85f';
        this.colorPicker.setAttribute('aria-label','Body color');this.colorPicker.title='Body material color in the preview and images output';
        this.colorPicker.oninput=()=>this.onColorChange?.(this.colorPicker.value,false);
        this.colorPicker.onchange=()=>this.onColorChange?.(this.colorPicker.value,true);
        if(!this.enabled){element('div',this.inspector,'color:#d9b897;line-height:1.5',error?`Shape controls unavailable: ${error}`:'Shape controls require the MHR body. Reset pose to switch to MHR.');return;}
        const actions=element('div',this.inspector,'display:flex;gap:5px;flex-wrap:wrap;margin-bottom:7px');
        const reset=element('button',actions,buttonStyle,'Reset all');reset.title='Restore the default shape without changing the pose';reset.onclick=()=>this.setValues(this.defaults,true);
        const neutral=element('button',actions,buttonStyle,'Neutral');neutral.title='Set all 45 MHR identity coefficients to zero';neutral.onclick=()=>this.setValues(Array(45).fill(0),true);
        for(const [title,start,count,open] of [['Body',0,20,true],['Head',20,20,false],['Hands',40,5,false]]){
            const details=element('details',this.inspector,'border-top:1px solid #444;padding-top:5px;margin-bottom:8px');details.open=open;
            element('summary',details,'cursor:pointer;font-weight:600;padding:3px 0 7px',`${title} (${count})`);
            const groupReset=element('button',details,buttonStyle+';font-size:11px;margin-bottom:5px',`Reset ${title.toLowerCase()}`);
            groupReset.onclick=()=>{const values=this.values.slice();for(let index=start;index<start+count;index++)values[index]=this.defaults[index];this.setValues(values,true);};
            for(let index=start;index<start+count;index++){
                const row=element('div',details,'margin-bottom:7px;padding:3px;border-radius:3px');
                const top=element('div',row,'display:flex;align-items:center;gap:4px;margin-bottom:2px');
                const label=element('label',top,'flex:1',`${title} ${String(index-start+1).padStart(2,'0')}`);label.title=`MHR identity coefficient ${index}. Typical range: -3 to 3; imported values are preserved.`;
                const number=element('input',top,inputStyle+';width:61px;padding:3px');number.type='number';number.min='-3';number.max='3';number.step='0.01';number.setAttribute('aria-label',label.textContent);
                const resetOne=element('button',top,buttonStyle+';padding:2px 5px','↺');resetOne.title='Reset this parameter';resetOne.onclick=()=>this.setOne(index,this.defaults[index],true);
                const slider=element('input',row,'width:100%;margin:0;accent-color:#96b9d7;cursor:ew-resize');slider.type='range';slider.min='-3';slider.max='3';slider.step='0.01';slider.setAttribute('aria-label',`${label.textContent} slider`);
                slider.oninput=()=>this.setOne(index,Number(slider.value),false);slider.onchange=()=>this.finish();slider.ondblclick=()=>this.setOne(index,this.defaults[index],true);
                number.oninput=()=>{if(number.value!==''&&number.validity.valid)this.setOne(index,Number(number.value),false,number);};
                number.onchange=()=>{const value=number.valueAsNumber;if(Number.isFinite(value))this.setOne(index,value,true);else{this.syncRows();this.finish();}};
                number.onkeydown=event=>{if(event.key==='Enter'){number.blur();event.stopPropagation();}};
                this.rows.push({index,row,number,slider});
            }
        }
        this.syncRows();
    }
    syncRows(exclude){
        for(const {index,row,number,slider} of this.rows){const value=this.values[index],limit=Math.max(3,Math.ceil(Math.abs(value)));for(const input of [number,slider]){input.min=String(-limit);input.max=String(limit);}slider.value=String(value);if(number!==exclude)number.value=value.toFixed(2);row.style.backgroundColor=Math.abs(value-this.defaults[index])>.00001?'#303b43':'';}
    }
    setOne(index,value,finish=false,exclude){
        if(!this.enabled||!Number.isFinite(value))return;
        const limit=Math.max(3,Math.ceil(Math.abs(this.values[index]))),values=this.values.slice();values[index]=Math.max(-limit,Math.min(limit,value));this.setValues(values,finish,exclude);
    }
    setValues(values,finish=false,exclude){
        const shape={schema:SHAPE_SCHEMA,coefficients:Array.from(values)};if(!this.enabled||!validMHRShape(shape))return false;
        this.values=shape.coefficients;this.syncRows(exclude);this.apply();this.onChange?.();if(finish)this.finish();return true;
    }
    apply(){
        if(!this.enabled)return;
        const position=this.mesh.geometry.attributes.position,output=position.array,stride=output.length;
        // Rebuild from an immutable baseline: no cumulative drift through undo,
        // repeated slider edits, reloads or simultaneous head/hand changes.
        output.set(this.base);
        for(let component=0;component<45;component++){
            const delta=this.values[component]-this.defaults[component];if(Math.abs(delta)<1e-9)continue;
            const offset=component*stride;for(let i=0;i<stride;i++)output[i]+=this.basis[offset+i]*delta;
        }
        position.needsUpdate=true;this.mesh.geometry.computeVertexNormals();this.mesh.geometry.computeBoundingSphere();
    }
    finish(){if(this.enabled)this.onFinish?.();}
    snapshot(){const shape={schema:SHAPE_SCHEMA,coefficients:this.values?.slice()};return this.modelKind==='MHR'&&validMHRShape(shape)?shape:undefined;}
    restore(shape){if(this.modelKind!=='MHR'||!this.defaults)return;if(shape&&!validMHRShape(shape))return;this.values=shape?.coefficients.slice()??this.defaults.slice();this.syncRows();this.apply();}
    assertRenderable(){
        // A missing/failed local basis must not erase a saved shape or silently
        // render the unchanged template as though the edited shape were applied.
        if(this.modelKind==='MHR'&&!this.enabled&&this.values?.some((value,index)=>Math.abs(value-this.defaults[index])>1e-7))
            throw Error('MHR shape data is unavailable. Reload the editor before rendering the saved shape.');
    }
}
