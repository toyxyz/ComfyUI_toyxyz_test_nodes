import {app} from "../../scripts/app.js";
import {api} from "../../scripts/api.js";
import * as T from "./vendor/three/three.module.min.js";
import {OrbitControls} from "./vendor/three/OrbitControls.js";
import {TransformControls} from "./vendor/three/TransformControls.js";
import {PoseLightControl} from "./util/pose_light_control.js";
import {PoseOpenPose} from "./util/pose_openpose.js";
import {PoseRenderPasses} from "./util/pose_render_passes.js";
import {PoseModelTree,PoseShapeControl,loadMHRShapeBasis,validMHRShape} from "./util/pose_shape_control.js";
import {buildSAMPosePrompt,samBodyFingerprint} from "./util/pose_sam_import.js";

const names=["pelvis","left_hip","right_hip","spine1","left_knee","right_knee","spine2","left_ankle","right_ankle","spine3","left_foot","right_foot","neck","left_collar","right_collar","head","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"];
const parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];
const v=(x,y,z)=>new T.Vector3(x,y,z);
const fingerNames=["index","middle","pinky","ring","thumb"];
const embeddedBodies=new Map();
const DEFAULT_BODY_COLOR='#d6a85f';
const validBodyColor=color=>typeof color==='string'&&/^#[0-9a-f]{6}$/i.test(color);
const BODY_FIELDS=['gender','source','bodyMesh','rootControl','bodyRoot','bones','pelvisBone','proceduralRotations','proceduralInputs','material','proxyParts','rootMarker','markers','handles','poleLinks','chains','openPose','shapeControl'];
const finiteVector=(value,size)=>Array.isArray(value)&&value.length===size&&value.every(Number.isFinite);
// Remove the former first widget before LiteGraph assigns positional values.
// Work directly on workflow data before links/slots are restored by the graph.
export function migratePose3DWorkflow(data){
    for(const node of data?.nodes??[]){
        if(node.type!=='ToyxyzPose3DEditor')continue;
        if(['female','male'].includes(node.widgets_values?.[0]))node.widgets_values.splice(0,1);
        const removed=(node.inputs??[]).flatMap((input,index)=>input.name==='body'?[index]:[]);
        if(!removed.length)continue;
        const deleted=new Set();
        data.links=(data.links??[]).filter(link=>{
            const target=Array.isArray(link)?link[3]:link.target_id,slot=Array.isArray(link)?link[4]:link.target_slot;
            if(target!==node.id)return true;
            if(removed.includes(slot)){deleted.add(Array.isArray(link)?link[0]:link.id);return false;}
            const next=slot-removed.filter(index=>index<slot).length;
            if(Array.isArray(link))link[4]=next;else link.target_slot=next;
            return true;
        });
        for(const other of data.nodes)for(const output of other.outputs??[])if(output.links)output.links=output.links.filter(id=>!deleted.has(id));
        node.inputs=node.inputs.filter(input=>input.name!=='body');
    }
    return data;
}
// Workflows saved before the OpenPose output was introduced retain render's
// original slot and links; add only the missing output after configuration.
export function ensureOpenPoseOutput(node){
    const imageOutput=node.outputs?.[0];
    if(imageOutput?.name==='render'){
        imageOutput.name='images';
        if(imageOutput.label==='render')imageOutput.label='images';
    }
    for(const name of ['openpose','depth','normal'])if(!node.outputs?.some(output=>output.name===name))node.addOutput(name,'IMAGE');
}
function ensureSAMInput(node){if(!node.inputs?.some(input=>input.name==='sam3d_body_model'))node.addInput('sam3d_body_model','SAM3D_BODY_MODEL');}
async function loadDefaultBody(gender){
    const key=`mhr_${gender}`;
    if(!embeddedBodies.has(key))embeddedBodies.set(key,(async()=>{
        const response=await fetch(new URL(`./vendor/pose3d/mhr_${gender}.json.gz`,import.meta.url));
        if(!response.ok)throw Error(`Default body asset: HTTP ${response.status}`);
        const stream=new Blob([await response.arrayBuffer()]).stream().pipeThrough(new DecompressionStream('gzip'));
        return JSON.parse(await new Response(stream).text());
    })());
    try{return await embeddedBodies.get(key);}catch(error){embeddedBodies.delete(key);throw error;}
}

export function solveTwoBone(root,target,pole,a,b,fallbackBend=null,fallbackAxis=null){
    const axis=target.clone().sub(root), distance=axis.length();
    if(distance<1e-8)axis.copy(fallbackAxis??v(0,-1,0)).normalize();else axis.divideScalar(distance);
    // Exact extension must not introduce an artificial elbow/knee bend on snap.
    const d=T.MathUtils.clamp(distance,Math.max(Math.abs(a-b),1e-8),a+b);
    const bend=pole.clone().sub(root).addScaledVector(axis,-pole.clone().sub(root).dot(axis));
    if(bend.lengthSq()<1e-12&&fallbackBend)bend.copy(fallbackBend).addScaledVector(axis,-fallbackBend.dot(axis));
    if(bend.lengthSq()<1e-8){bend.set(0,0,1).addScaledVector(axis,-axis.z);if(bend.lengthSq()<1e-8)bend.set(1,0,0);}
    bend.normalize();
    const along=(a*a-b*b+d*d)/(2*d),height=Math.sqrt(Math.max(0,a*a-along*along));
    return {joint:root.clone().addScaledVector(axis,along).addScaledVector(bend,height),end:root.clone().addScaledVector(axis,d)};
}

function defaultRig(body){
    const wide=body==="male"?1.12:1;
    const points=[[0,.92,0],[-.105,.9,0],[.105,.9,0],[0,1.08,0],[-.105,.5,.015],[.105,.5,.015],[0,1.24,0],[-.105,.1,0],[.105,.1,0],[0,1.42,0],[-.105,.055,.15],[.105,.055,.15],[0,1.56,0],[-.09*wide,1.43,0],[.09*wide,1.43,0],[0,1.7,0],[-.2*wide,1.43,0],[.2*wide,1.43,0],[-.49*wide,1.41,0],[.49*wide,1.41,0],[-.75*wide,1.4,0],[.75*wide,1.4,0]];
    const ns=[...names], ps=[...parents];
    for(const side of ["left","right"]){const s=side==="left"?-1:1, wrist=side==="left"?20:21;
        for(let f=0;f<5;f++){let parent=wrist;for(let k=0;k<3;k++){const len=f===4?.055:.075;const p=[s*(.75*wide+.035+len*(k+1)),1.4+(f===4?-.035:0),.035*(f-2)];ns.push(`${side}_${fingerNames[f]}${k+1}`);ps.push(parent);parent=points.length;points.push(p);}}
    }
    return {joints:points,parents:ps,names:ns};
}

export class Pose3DEditorUI {
    constructor(node,root){
        this.node=node;this.root=root;this.widget=n=>node.widgets.find(w=>w.name===n);this.gender='female';this.disposed=false;this.loading=true;this.bodies=[];this.nextBodyId=0;this.bodyAssets=new Map();
        root.style.cssText="width:100%;height:620px;display:flex;flex-direction:column;gap:6px;background:#292929;color:#ddd;padding:8px;box-sizing:border-box;font:12px sans-serif";
        this.bar=document.createElement("div");this.bar.style.cssText="display:flex;gap:5px;flex-wrap:wrap;align-items:center";root.append(this.bar);
        this.button("Reset pose",()=>this.reset());
        this.button("Reset view",()=>{this.camera.position.set(0,1.2,5.2);this.orbit.target.set(0,.95,0);this.orbit.update();this.commit();});
        this.button("Undo",()=>this.undo());this.button("Redo",()=>this.redo());
        this.getPoseButton=this.button('Get pose',()=>this.choosePoseImage());this.getPoseButton.title='Import body shape and pose from one image using SAM 3D Body';
        this.poseFile=document.createElement('input');this.poseFile.type='file';this.poseFile.accept='image/png,image/jpeg,image/webp';this.poseFile.hidden=true;root.append(this.poseFile);
        this.poseFile.onchange=()=>{const file=this.poseFile.files?.[0],body=this.poseFileBody;this.poseFile.value='';if(file&&body)this.getPoseFromImage(file,body).catch(error=>{if(!this.disposed)this.note.textContent=`Get pose failed: ${error.message}`;});};
        this.poseExecuting=event=>{const job=this.poseJob,data=event.detail;if(!job?.promptId||data?.prompt_id!==job.promptId)return;if(data.node==='pose_import')this.note.textContent='Get pose: Estimating body and hands…';else if(data.node)this.note.textContent='Get pose: Loading image / model…';};
        api.addEventListener?.('executing',this.poseExecuting);
        this.floorButton=this.button("Floor",()=>{this.setFloorVisible(!this.floor.visible);this.commit();this.record();});
        this.floorButton.title="Show or hide the floor in the preview and images output";
        this.button("Rotate",()=>this.setMode("rotate"));this.button("Move",()=>this.setMode("translate"));
        this.note=document.createElement("div");this.note.style.color="#b9c5d0";root.append(this.note);
        this.workspace=document.createElement('div');this.workspace.style.cssText='display:flex;gap:8px;flex:1;min-height:0;overflow:hidden';root.append(this.workspace);
        this.modelTree=new PoseModelTree(this.workspace,()=>{this.ready=this.addBody().catch(error=>{this.note.textContent=`Add body failed: ${error.message}`;throw error;});this.ready.catch(()=>{});},()=>this.deleteBody(),()=>this.selectScene());
        this.view=document.createElement("div");this.view.style.cssText="flex:1;min-width:0;min-height:240px;position:relative;overflow:hidden";this.workspace.append(this.view);
        this.info=document.createElement("div");this.info.textContent="Click joint / IK target / pole · W: move · E: rotate · Left drag: orbit · Middle drag: pan · Scroll: zoom";root.append(this.info);
        this.renderer=new T.WebGLRenderer({antialias:true,preserveDrawingBuffer:true,alpha:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=T.PCFSoftShadowMap;this.renderer.outputColorSpace=T.SRGBColorSpace;this.renderer.toneMapping=T.ACESFilmicToneMapping;this.view.append(this.renderer.domElement);
        this.scene=new T.Scene();this.scene.background=new T.Color("#000000");this.sceneBodies=new T.Group();this.scene.add(this.sceneBodies);this.camera=new T.PerspectiveCamera(36,1,.01,100);this.camera.position.set(0,1.2,5.2);
        this.orbit=new OrbitControls(this.camera,this.renderer.domElement);this.orbit.mouseButtons.LEFT=T.MOUSE.ROTATE;this.orbit.mouseButtons.MIDDLE=-1;this.orbit.target.set(0,.95,0);this.orbit.update();this.orbit.addEventListener("change",()=>{if(!this.loading)this.commit();});
        this.gizmo=new TransformControls(this.camera,this.renderer.domElement);this.gizmo.setSize(.7);this.gizmo.setSpace("local");this.scene.add(this.gizmo.getHelper());
        this.gizmo.addEventListener("dragging-changed",e=>{this.orbit.enabled=!e.value;if(!e.value){this.commit();this.record();}});
        this.gizmo.addEventListener("objectChange",()=>this.onTransformChange());
        this.scene.add(new T.HemisphereLight(0xc9dcff,0x253044,1.7));const key=this.keyLight=new T.DirectionalLight(0xffffff,3.2);key.position.set(-2,4,3);key.target.position.set(0,.95,0);key.castShadow=true;key.shadow.mapSize.set(2048,2048);key.shadow.camera.left=-2;key.shadow.camera.right=2;key.shadow.camera.top=3;key.shadow.camera.bottom=-2;key.shadow.bias=-.0001;this.scene.add(key,key.target);
        this.lightControl=new PoseLightControl(this.view,this.camera,()=>key.position.clone().sub(key.target.position).normalize(),direction=>{this.setLightDirection(direction);this.commit();},()=>this.record());
        this.renderPasses=new PoseRenderPasses();
        const rim=new T.DirectionalLight(0x87aaff,2);rim.position.set(2,2,-3);this.scene.add(rim);
        this.floor=new T.Mesh(new T.PlaneGeometry(200,200),new T.MeshStandardMaterial({color:0x303844,roughness:.85}));this.floor.rotation.x=-Math.PI/2;this.floor.position.y=-.01;this.floor.receiveShadow=true;this.scene.add(this.floor);
        this.grid=new T.GridHelper(4,20,0x657587,0x394451);this.grid.position.y=.001;this.scene.add(this.grid);
        this.setFloorVisible(true);
        this.ray=new T.Raycaster();
        const canvas=this.renderer.domElement;canvas.tabIndex=0;let start=null,suppressedGizmo=null;
        let middlePan=null;
        const restoreGizmo=()=>{if(suppressedGizmo!==null){this.gizmo.enabled=suppressedGizmo;this.gizmo.axis=null;suppressedGizmo=null;}};
        // Capture before TransformControls' bubbling listener. A different IK
        // handle must be selectable even under the previous control's picker.
        canvas.addEventListener('pointerdown',event=>{
            if(event.button!==0||this.gizmo.dragging)return;
            const object=this.pickObjectAt(event.clientX,event.clientY);
            if(object?.userData.chain&&object!==this.selected){suppressedGizmo=this.gizmo.enabled;this.gizmo.enabled=false;this.gizmo.axis=null;}
        },true);
        canvas.addEventListener("pointerdown",e=>{start=e.button===0?[e.clientX,e.clientY,this.gizmo.dragging]:null;if(e.button===1){middlePan=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);e.preventDefault();e.stopImmediatePropagation();}});
        canvas.addEventListener("pointermove",e=>{if(!middlePan||(e.buttons&4)===0)return;const dx=e.clientX-middlePan[0],dy=e.clientY-middlePan[1];middlePan=[e.clientX,e.clientY];const distance=this.camera.position.distanceTo(this.orbit.target),scale=2*distance*Math.tan(T.MathUtils.degToRad(this.camera.fov*.5))/(this.renderer.domElement.clientHeight||1),right=new T.Vector3().setFromMatrixColumn(this.camera.matrixWorld,0),up=new T.Vector3().setFromMatrixColumn(this.camera.matrixWorld,1),delta=right.multiplyScalar(-dx*scale).add(up.multiplyScalar(dy*scale));this.camera.position.add(delta);this.orbit.target.add(delta);this.orbit.update();e.preventDefault();e.stopImmediatePropagation();});
        canvas.addEventListener("pointerup",e=>{if(e.button===1){middlePan=null;e.preventDefault();e.stopImmediatePropagation();}});
        canvas.addEventListener("pointerup",e=>{
            if(e.button!==0)return;restoreGizmo();const pressed=start;start=null;
            if(!pressed||pressed[2]||Math.hypot(e.clientX-pressed[0],e.clientY-pressed[1])>4||this.gizmo.axis)return;
            const object=this.pickObjectAt(e.clientX,e.clientY);if(object){this.selectObject(object);this.commit();}
        });
        canvas.addEventListener('pointercancel',()=>{restoreGizmo();start=null;middlePan=null;});
        canvas.addEventListener("keydown",e=>{if(e.key.toLowerCase()==="w")this.setMode("translate");if(e.key.toLowerCase()==="e")this.setMode("rotate");e.stopPropagation();});
        for(const name of ["pointerdown","wheel","keydown"])root.addEventListener(name,e=>e.stopPropagation());
        this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(this.view);
        this.ready=this.load();this.loop=()=>{if(this.disposed)return;this.updateAllMarkers();this.renderer.render(this.scene,this.camera);this.lightControl.draw();this.frame=requestAnimationFrame(this.loop);};this.loop();
    }
    button(text,action){const b=document.createElement("button");b.textContent=text;b.onclick=action;this.bar.append(b);return b;}
    updateGetPoseButton(){if(this.getPoseButton)this.getPoseButton.disabled=this.loading||!!this.poseJob||this.selectedBody?.source!=='MHR'||!this.selectedBody?.shapeControl.enabled;}
    choosePoseImage(){this.updateGetPoseButton();if(this.getPoseButton.disabled)return;this.poseFileBody=this.selectedBody;this.poseFile.click();}
    async getPoseFromImage(file,body=this.selectedBody){
        if(this.loading||this.poseJob||this.disposed)return;
        if(body?.source!=='MHR'||!body.shapeControl.enabled||!this.bodies.includes(body))throw Error('Select an editable MHR body first.');
        const job=this.poseJob={id:crypto.randomUUID(),body,signature:samBodyFingerprint(this.bodySnapshot(body)),loadId:this.loadId};this.updateGetPoseButton();
        try{
            this.note.textContent='Get pose: Uploading image…';
            const form=new FormData();form.append('image',file,`pose_${job.id}.${file.name.split('.').pop()}`);form.append('type','input');form.append('overwrite','false');
            const upload=await api.fetchApi('/upload/image',{method:'POST',body:form});const uploaded=await upload.json();if(!upload.ok)throw Error(uploaded.error??`Image upload: HTTP ${upload.status}`);
            const image=uploaded.subfolder?`${uploaded.subfolder}/${uploaded.name}`:uploaded.name;
            const prompt=await buildSAMPosePrompt(this.node,image,job.id);
            this.note.textContent='Get pose: Queued — loading model / estimating body…';
            const response=await api.fetchApi('/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,client_id:api.clientId,extra_data:{toyxyz_get_pose:true}})});
            const queued=await response.json();if(!response.ok)throw Error(JSON.stringify(queued.error??queued.node_errors??queued));
            job.promptId=queued.prompt_id;const deadline=Date.now()+20*60*1000;
            while(!this.disposed&&this.poseJob===job){
                const response=await api.fetchApi(`/history/${encodeURIComponent(job.promptId)}`);if(!response.ok)throw Error(`History: HTTP ${response.status}`);
                const history=(await response.json())[job.promptId],result=history?.outputs?.pose_import?.toyxyz_sam_pose?.[0];
                if(result){
                    if(result.request_id!==job.id)throw Error('Received a mismatched pose result.');
                    if(this.loadId!==job.loadId||!this.bodies.includes(body)||samBodyFingerprint(this.bodySnapshot(body))!==job.signature)throw Error('The target body changed during inference. Result discarded; try again.');
                    this.applySAMPose(body,result);this.note.textContent='Get pose: Body shape and pose applied.';return;
                }
                const messages=history?.status?.messages??[],failure=messages.find(([type])=>type==='execution_error'||type==='execution_interrupted');
                if(failure)throw Error(failure[1]?.exception_message??'Pose inference interrupted.');
                if(history?.status?.completed)throw Error('Pose inference finished without an import result.');
                if(Date.now()>deadline)throw Error('Pose inference timed out. Check the ComfyUI execution queue.');
                await new Promise(resolve=>setTimeout(resolve,1000));
            }
        }finally{if(this.poseJob===job)this.poseJob=null;this.updateGetPoseButton();}
    }
    refreshChainLengths(){this.rootControl.updateMatrixWorld(true);for(const chain of this.chains){const [a,b,c]=chain.ids.map(i=>this.bones[i].getWorldPosition(v(0,0,0)));chain.lengths=[a.distanceTo(b),b.distanceTo(c)];}}
    applySAMPose(body,data){
        const positiveScale=s=>finiteVector(s,3)&&s.every(x=>x>1e-6&&x<=1000);
        if(!this.bodies.includes(body)||body.source!=='MHR'||!body.shapeControl.enabled||data?.version!==1||data.source!=='MHR'||!validMHRShape(data.shape)||data.joints?.length!==body.bones.length||data.parents?.length!==body.bones.length||!data.joints.every(j=>finiteVector(j.position,3)&&finiteVector(j.quaternion,4)&&j.quaternion.some(x=>x!==0)&&positiveScale(j.scale))||!data.parents.every((parent,i)=>parent===body.bones.indexOf(body.bones[i].parent)))throw Error('Incompatible SAM MHR pose.');
        this.record();const before=this.bodySnapshot(body),selected=this.selectedBody===body;
        try{this.withBody(body,()=>{
            if(selected)this.gizmo.detach();this.shapeControl.restore(data.shape);
            data.joints.forEach((joint,index)=>{const bone=this.bones[index];if(bone.parent.isBone)bone.position.fromArray(joint.position);bone.quaternion.fromArray(joint.quaternion).normalize();bone.scale.fromArray(joint.scale);});
            // Preserve the scene's placement: native root translation contains
            // estimated camera positioning, not the editor root controller.
            for(const rule of this.proceduralRotations)rule.offset=new T.Quaternion();
            const imported=this.proceduralRotations.map(rule=>this.bones[rule.bone].quaternion.clone());this.updateProceduralRotations();
            this.proceduralRotations.forEach((rule,i)=>{rule.offset=this.bones[rule.bone].quaternion.clone().invert().multiply(imported[i]);});
            this.updateProceduralRotations();this.refreshChainLengths();
            for(const chain of this.chains){chain.fkFrame=null;chain.bendLocal=null;chain.axisLocal=null;this.syncFKChain(chain,true);chain.enabled=true;}
            this.updateMarkers();
        });this.commit();this.record();if(selected)this.selectObject(body.rootControl);
        }catch(error){this.restoreBody(body,before);this.commit();throw error;}
    }
    pickObjectAt(clientX,clientY){
        if(this.loading)return null;
        this.updateAllMarkers();this.scene.updateMatrixWorld(true);this.camera.updateMatrixWorld(true);
        const rect=this.renderer.domElement.getBoundingClientRect();if(!rect.width||!rect.height)return null;
        this.ray.setFromCamera({x:2*(clientX-rect.left)/rect.width-1,y:1-2*(clientY-rect.top)/rect.height},this.camera);
        const controls=this.selectedBody?[...this.selectedBody.markers.children,...this.selectedBody.handles.children,this.selectedBody.rootMarker]:[];
        const hits=this.ray.intersectObjects(controls,false).filter(hit=>{
            for(let object=hit.object;object;object=object.parent)if(!object.visible)return false;return true;
        });
        // Handles are drawn over joints (depthTest=false, renderOrder=4).
        // Picking must follow that visible overlay order, not mesh depth alone.
        hits.sort((a,b)=>b.object.renderOrder-a.object.renderOrder||a.distance-b.distance);
        if(hits.length)return hits[0].object.userData.object??hits[0].object;
        // Only a body-surface click resolves to the character's Root controller.
        // IK handles also carry poseBody ownership but must remain the target.
        const mesh=this.ray.intersectObjects(this.bodies.map(body=>body.bodyMesh).filter(Boolean),false)[0]?.object;
        return mesh?.userData.poseBody?.rootControl??null;
    }
    setFloorVisible(visible){
        this.floor.visible=visible;this.grid.visible=visible;
        this.floorButton.textContent=visible?'Floor: On':'Floor: Off';
        this.floorButton.setAttribute('aria-pressed',String(visible));
        this.floorButton.style.backgroundColor=visible?'#36536b':'';
    }
    setLightDirection(direction){this.keyLight.position.copy(this.keyLight.target.position).addScaledVector(direction.clone().normalize(),5);this.lightControl?.draw();}
    setBodyColor(color){
        if(!validBodyColor(color)||!this.material)return false;
        this.material.color.set(color);this.rootMarker?.material.color.copy(this.material.color);
        if(this.shapeControl.colorPicker)this.shapeControl.colorPicker.value=`#${this.material.color.getHexString()}`;
        return true;
    }
    useBody(body){for(const key of BODY_FIELDS)this[key]=body?.[key];this.bodyContext=body;}
    withBody(body,action){
        const previous=Object.fromEntries(BODY_FIELDS.map(key=>[key,this[key]])),context=this.bodyContext;
        this.useBody(body);try{return action();}finally{Object.assign(this,previous);this.bodyContext=context;}
    }
    updateControllerVisibility(){
        for(const body of this.bodies){
            const visible=body===this.selectedBody;
            for(const object of [body.markers,body.handles,body.poleLinks,body.rootMarker])object.visible=visible;
        }
    }
    activateBody(body){
        if(!body||!this.bodies.includes(body))return;
        this.activeBody=body;this.selectedBody=body;this.useBody(body);
        for(const item of this.bodies)item.shapeControl.select(item===body);
        this.updateControllerVisibility();
        this.modelTree.sceneButton.setAttribute('aria-selected','false');this.modelTree.setActions(this.loading,true);
        this.updateGetPoseButton();
        this.note.textContent=`${body.source} · articulated hands`;
    }
    selectScene(){
        this.selectedBody=null;this.selected=null;this.gizmo.detach();
        this.updateGetPoseButton();
        for(const body of this.bodies)body.shapeControl.select(false);
        this.updateControllerVisibility();
        this.modelTree.sceneButton.setAttribute('aria-selected','true');this.modelTree.setActions(this.loading,false);this.commit();
    }
    async prepareBody(saved={}){
        const gender=['female','male'].includes(saved.body)?saved.body:'female';let data={available:false};
        if(['SMPL','SMPL-H'].includes(saved.source)){
            const response=await api.fetchApi(`/toyxyz/pose3d/body?gender=${gender}`);data=await response.json();if(data.error)throw Error(data.error);
        }
        if(!data.available)data=await loadDefaultBody(gender);
        let basis,error;
        if(data.kind==='MHR')try{basis=await loadMHRShapeBasis();}catch(cause){error=cause.message;}
        const key=`${data.kind}:${gender}`,prepared={gender,data,basis,error,validation:this.bodyAssets.get(key)?.validation};this.bodyAssets.set(key,prepared);return prepared;
    }
    createBody(prepared,saved={},id){
        this.gender=prepared.gender;this.build(prepared.data);
        if(!id)id=`body-${++this.nextBodyId}`;
        else this.nextBodyId=Math.max(this.nextBodyId,Number(id.match(/^body-(\d+)$/)?.[1])||0);
        const body={id,prepared,...Object.fromEntries(BODY_FIELDS.map(key=>[key,this[key]]))};
        const controls=new PoseShapeControl(this.modelTree,()=>this.commit(),()=>this.record(),()=>{this.activateBody(body);this.selectObject(body.rootControl);this.commit();},(color,finish)=>{this.withBody(body,()=>this.setBodyColor(color));this.commit();if(finish)this.record();});
        body.shapeControl=controls;this.shapeControl=controls;this.bodyContext=body;
        controls.bind(body.bodyMesh,prepared.data,prepared.basis,prepared.error);
        controls.modelButton.textContent=`◈ Human mesh ${id.replace(/^body-/,'')} (${body.source})`;
        this.bodies.push(body);
        for(const object of [body.rootControl,...body.bones,...body.handles.children,...(body.bodyMesh?[body.bodyMesh]:[])])object.userData.poseBody=body;
        prepared.validation={gender:body.gender,source:body.source,bones:body.bones.map(b=>({name:b.name})),chains:body.chains.map(c=>({name:c.name})),proceduralCount:body.proceduralRotations.length};
        this.setBodyColor(validBodyColor(saved.body_color)?saved.body_color:DEFAULT_BODY_COLOR);
        if(validMHRShape(saved.shape))controls.restore(saved.shape);
        if(this.validBodyState(saved,body))this.restoreBody(body,saved);
        else if(finiteVector(saved.root_transform?.position,3)&&finiteVector(saved.root_transform?.quaternion,4)&&saved.root_transform.quaternion.some(value=>value!==0)){
            body.rootControl.position.fromArray(saved.root_transform.position);body.rootControl.quaternion.fromArray(saved.root_transform.quaternion).normalize();
        }
        this.updateMarkers();this.updateControllerVisibility();return body;
    }
    destroyBody(body){
        const geometries=new Set(),materials=new Set();
        for(const root of [body.rootControl,body.markers,body.poleLinks]){
            root.removeFromParent();root.traverse(object=>{
                if(object.geometry)geometries.add(object.geometry);
                for(const material of Array.isArray(object.material)?object.material:[object.material])if(material)materials.add(material);
                object.skeleton?.dispose();
            });
        }
        geometries.forEach(item=>item.dispose());materials.forEach(item=>item.dispose());body.shapeControl.entry.remove();
        this.bodies=this.bodies.filter(item=>item!==body);
    }
    clearBodies(){this.gizmo.detach();for(const body of [...this.bodies])this.destroyBody(body);this.activeBody=null;this.selectedBody=null;this.selected=null;this.useBody(null);}
    async load(saved){
        this.loading=true;const loadId=this.loadId=(this.loadId||0)+1;this.modelTree.setActions(true,false);
        try{
            if(!saved){try{saved=JSON.parse(this.widget('pose_data').value);}catch{}}
            saved??={};const records=Array.isArray(saved.bodies)?saved.bodies:[saved];
            if(records.length>256||records.some(item=>!item||typeof item!=='object'))throw Error('Invalid body list.');
            const ids=records.map(item=>item.id).filter(id=>id!==undefined);
            if(ids.some(id=>typeof id!=='string'||id.length>80)||new Set(ids).size!==ids.length)throw Error('Invalid body identifiers.');
            const prepared=await Promise.all(records.map(item=>this.prepareBody(item)));
            if(this.disposed||loadId!==this.loadId)return;
            this.clearBodies();records.forEach((item,index)=>this.createBody(prepared[index],item,item.id));
            this.restoreSceneSettings(saved);
            const selected=saved.selected_body_id===null?null:this.bodies.find(body=>body.id===saved.selected_body_id)??this.bodies[0];
            if(selected){this.activateBody(selected);this.selectObject(selected.rootControl);}else this.selectScene();
            this.resize();
        }catch(error){if(loadId===this.loadId)this.note.textContent=`Body load failed: ${error.message}`;throw error;}
        finally{if(loadId===this.loadId&&!this.disposed){this.loading=false;this.modelTree.setActions(false,!!this.selectedBody);}}
        this.commit();this.history=[this.snapshot()];this.future=[];this.updateGetPoseButton();
    }
    async addBody(){
        if(this.loading||this.disposed)return false;
        if(this.bodies.length>=256){this.note.textContent='The scene supports up to 256 bodies.';return false;}
        this.record();const loadId=this.loadId;this.loading=true;this.modelTree.setActions(true,!!this.selectedBody);
        try{
            const prepared=await this.prepareBody();if(this.disposed||loadId!==this.loadId)return false;
            const x=this.bodies.length?Math.max(...this.bodies.map(body=>body.rootControl.position.x))+1.2:0;
            const color=new T.Color().setHSL(Math.random(),.55+Math.random()*.25,.5+Math.random()*.15);
            const body=this.createBody(prepared,{body_color:`#${color.getHexString()}`});body.rootControl.position.x=x;this.activateBody(body);this.selectObject(body.rootControl);this.updateMarkers();
            if(this.modelTree.branch.hidden)this.modelTree.expand.click();
        }finally{if(loadId===this.loadId&&!this.disposed){this.loading=false;this.modelTree.setActions(false,!!this.selectedBody);}}
        this.commit();this.record();this.updateGetPoseButton();return true;
    }
    deleteBody(){
        const body=this.selectedBody;if(this.loading||!body)return false;
        this.record();const index=this.bodies.indexOf(body);this.gizmo.detach();this.destroyBody(body);
        const next=this.bodies[Math.min(index,this.bodies.length-1)];
        if(next){this.activateBody(next);this.selectObject(next.rootControl);}else{this.activeBody=null;this.useBody(null);this.selectScene();this.note.textContent='No bodies · Add body to begin';}
        this.commit();this.record();return true;
    }
    clear(group){for(const object of [...group.children]){group.remove(object);object.traverse(o=>{o.geometry?.dispose();if(o.material)o.material.dispose();});}}
    build(data){
        this.bodyMesh=null;
        this.markers=new T.Group();this.handles=new T.Group();this.poleLinks=new T.Group();this.scene.add(this.markers,this.poleLinks);
        const rig=data.available?data:defaultRig(this.gender);this.source=data.available?data.kind:"procedural";const ns=rig.names||[...names,...(rig.joints.length===24?["left_hand","right_hand"]:[...['left','right'].flatMap(side=>fingerNames.flatMap(f=>[1,2,3].map(k=>`${side}_${f}${k}`)))])];
        // Keep skin indices and legacy joint arrays intact. This extra parent bone
        // transforms the entire rig, mesh and IK handles about the feet together.
        this.rootControl=new T.Bone();this.rootControl.name='Root controller';this.rootControl.userData.rootController=true;this.sceneBodies.add(this.rootControl);this.rootControl.add(this.handles);
        this.bodyRoot=new T.Group();if(data.available)this.bodyRoot.position.y=-(data.groundY??Math.min(...data.vertices.filter((_,i)=>i%3===1)));this.rootControl.add(this.bodyRoot);this.bones=rig.joints.map((_,i)=>{const bone=new T.Bone();bone.name=ns[i];bone.userData.index=i;return bone;});
        rig.joints.forEach((p,i)=>{
            const parent=rig.parents[i],bone=this.bones[i],local=rig.localTransforms?.[i];
            if(local){bone.position.fromArray(local.position);bone.quaternion.fromArray(local.quaternion).normalize();bone.scale.setScalar(local.scale??1);}
            else{bone.position.fromArray(p);if(parent>=0)bone.position.sub(v(...rig.joints[parent]));}
            if(parent>=0)this.bones[parent].add(bone);else this.bodyRoot.add(bone);
        });
        this.pelvisBone=this.bones[data.pelvisIndex??ns.findIndex(name=>name.toLowerCase()==='pelvis')];
        this.proceduralRotations=(data.proceduralRotations??[]).map(rule=>({...rule,bind:this.bones[rule.bone].quaternion.clone()}));
        this.proceduralInputs=new Map();
        for(const rule of this.proceduralRotations)for(const axis of rule.axes)for(const [index] of axis)
            if(!this.proceduralInputs.has(index))this.proceduralInputs.set(index,{inverseBind:this.bones[index].quaternion.clone().invert(),angles:new T.Euler()});
        this.material=new T.MeshPhysicalMaterial({side:data.kind==='MHR'?T.DoubleSide:T.FrontSide,color:DEFAULT_BODY_COLOR,roughness:.38,metalness:.04,clearcoat:.2,clearcoatRoughness:.38});this.proxyParts=[];
        this.rootMarker=new T.Mesh(new T.RingGeometry(.27,.30,64),new T.MeshBasicMaterial({color:this.material.color.clone(),side:T.DoubleSide,depthTest:false,depthWrite:false}));
        this.rootMarker.name='Root controller circle';this.rootMarker.rotation.x=-Math.PI/2;this.rootMarker.position.y=.006;this.rootMarker.renderOrder=2;this.rootMarker.userData.object=this.rootControl;this.rootControl.add(this.rootMarker);
        if(data.available){const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(data.vertices,3));geo.setIndex(data.faces);geo.setAttribute('skinIndex',new T.Uint16BufferAttribute(data.indices,4));geo.setAttribute('skinWeight',new T.Float32BufferAttribute(data.weights,4));geo.computeVertexNormals();const mesh=new T.SkinnedMesh(geo,this.material);this.bodyRoot.add(mesh);this.bodyRoot.updateMatrixWorld(true);mesh.bind(new T.Skeleton(this.bones));mesh.normalizeSkinWeights();mesh.frustumCulled=false;mesh.castShadow=true;mesh.receiveShadow=true;this.bodyMesh=mesh;}
        else{
            for(let i=1;i<this.bones.length;i++){const parent=this.bones[i].parent;const radius=i>=22?.013:([4,5].includes(i)?.065:[18,19,20,21].includes(i)?.042:[3,6,9,12].includes(i)?.105:.055);const mesh=new T.Mesh(new T.CapsuleGeometry(radius,1,6,12),this.material);parent.add(mesh);mesh.castShadow=true;mesh.receiveShadow=true;this.proxyParts.push({mesh,child:this.bones[i],radius});}
            const ellipsoid=(index,scale,offset)=>{const m=new T.Mesh(new T.SphereGeometry(1,32,24),this.material);m.scale.fromArray(scale);m.position.fromArray(offset);m.castShadow=true;m.receiveShadow=true;this.bones[index].add(m);};
            ellipsoid(0,[this.gender==='female'?.17:.16,.14,.12],[0,0,0]);ellipsoid(6,[this.gender==='male'?.22:.19,.22,.12],[0,.025,0]);ellipsoid(15,[.085,.115,.09],[0,.045,.005]);for(const i of [10,11])ellipsoid(i,[.052,.04,.105],[0,0,.01]);
        }
        const controls=data.controlJoints?new Set(data.controlJoints):null;
        this.bones.forEach((bone,i)=>{if(controls&&!controls.has(i))return;const pelvis=bone===this.pelvisBone,color=pelvis?0x57e0af:0xffffff,finger=controls?fingerNames.some(name=>bone.name.includes(name)):i>=22,radius=pelvis?.025:(finger?.007:.0125);const sphere=new T.Mesh(new T.SphereGeometry(radius,10,8),new T.MeshBasicMaterial({color,depthTest:false}));sphere.renderOrder=3;sphere.userData.object=bone;this.markers.add(sphere);});
        this.bodyRoot.updateMatrixWorld(true);this.chains=[];
        // Neutral bodies face +Z: elbow poles go behind (-Z), knee poles in front (+Z).
        const chains=data.ikChains?data.ikChains.map(c=>[c.name,c.bones.map(b=>ns.indexOf(b)),c.poleOffset]):[['Left arm',[16,18,20],[0,0,-.35]],['Right arm',[17,19,21],[0,0,-.35]],['Left leg',[1,4,7],[0,0,.5]],['Right leg',[2,5,8],[0,0,.5]]];
        for(const [name,ids,poleOffset] of chains){const end=this.bones[ids[2]].getWorldPosition(v(0,0,0)),joint=this.bones[ids[1]].getWorldPosition(v(0,0,0)),start=this.bones[ids[0]].getWorldPosition(v(0,0,0));const chain={name,ids,lengths:[start.distanceTo(joint),joint.distanceTo(end)],enabled:true,lastTarget:end.clone()};for(const [kind,position,color] of [['target',end,0x57e0af],['pole',joint.clone().add(v(...poleOffset)),0xd78aff]]){const handle=new T.Mesh(new T.SphereGeometry(.035,14,10),new T.MeshBasicMaterial({color,depthTest:false}));handle.position.copy(position);handle.renderOrder=4;handle.userData={chain,kind};handle.name=`${name} ${kind}`;this.handles.add(handle);chain[kind]=handle;}
            const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.BufferAttribute(new Float32Array(6),3));
            chain.poleLink=new T.Line(geometry,new T.LineBasicMaterial({color:0xd78aff,linewidth:1,transparent:true,opacity:.7,depthTest:false,depthWrite:false}));
            chain.poleLink.name=`${name} pole connection`;chain.poleLink.renderOrder=2;chain.poleLink.frustumCulled=false;this.poleLinks.add(chain.poleLink);chain.poleDistance=v(...poleOffset).length();this.chains.push(chain);this.rememberChainPose(chain);this.syncFKChain(chain,true);}
        this.openPose=new PoseOpenPose(this.bones,this.bodyMesh,this.source,data.openpose);
        this.updateMarkers();
    }
    selectObject(object){if(object.userData.poseBody&&object.userData.poseBody!==this.selectedBody)this.activateBody(object.userData.poseBody);const chain=object.userData.chain;if(chain&&!chain.enabled){this.updateProceduralRotations();this.rootControl.updateMatrixWorld(true);this.syncFKChain(chain,true);}if(object===this.pelvisBone)this.preparePelvisIK();this.selected=object;this.gizmo.attach(object);if(object.userData.rootController||object.userData.chain||object===this.pelvisBone){this.gizmo.setMode('translate');this.gizmo.setSpace('world');}else{this.gizmo.setMode('rotate');this.gizmo.setSpace('local');}this.info.textContent=`${object.name} · White: rotate only · Green/Purple: move · IK end / Root: W move / E rotate`;}
    preparePelvisIK(){
        // Re-arm FK-edited (or old saved inactive) limbs BEFORE pelvis motion,
        // matching the current pose rather than moving the targets with it.
        this.updateProceduralRotations();this.rootControl.updateMatrixWorld(true);
        for(const chain of this.chains)if(!chain.enabled){this.syncFKChain(chain,true);chain.enabled=true;}
    }
    setMode(mode){const selected=this.selected;if(!selected)return;const chain=selected.userData.chain;if(chain){if(selected.userData.kind==='pole')mode='translate';if(mode==='rotate'){this.gizmo.attach(this.bones[chain.ids[2]]);this.gizmo.setSpace('local');}else{this.gizmo.attach(selected);this.gizmo.setSpace('world');}}else if(selected.userData.rootController){this.gizmo.attach(selected);this.gizmo.setSpace(mode==='rotate'?'local':'world');}else if(selected===this.pelvisBone){mode='translate';this.gizmo.attach(selected);this.gizmo.setSpace('world');}else{mode='rotate';this.gizmo.attach(selected);this.gizmo.setSpace('local');}this.gizmo.setMode(mode);}
    followPole(chain){chain.pole.position.add(chain.target.position.clone().sub(chain.lastTarget));chain.lastTarget.copy(chain.target.position);}
    onTransformChange(){
        const object=this.gizmo.object;if(!object||!this.chains)return;
        if(object.userData.chain)object.userData.chain.enabled=true;
        else if(object.isBone&&!object.userData.rootController){
            this.updateProceduralRotations();this.rootControl.updateMatrixWorld(true);
            for(const chain of this.chains){
                // A rotated shoulder/clavicle/spine also changes its descendant
                // limb. End-joint rotation alone never rotates the pole.
                let affects=object===this.bones[chain.ids[1]];
                for(let ancestor=this.bones[chain.ids[0]];ancestor&&ancestor!==this.rootControl;ancestor=ancestor.parent)
                    if(ancestor===object)affects=true;
                // Pelvis translation retains the existing planted-IK behavior.
                if(!affects||(object===this.pelvisBone&&this.gizmo.mode==='translate'))continue;
                if(chain.enabled)chain.poleDistance=Math.max(.01,chain.pole.getWorldPosition(v(0,0,0)).distanceTo(this.bones[chain.ids[1]].getWorldPosition(v(0,0,0))));
                chain.enabled=false;this.syncFKChain(chain,true);
            }
        }
        this.solveIK();this.commit();
    }
    chainFrame(chain){
        return {points:chain.ids.map(i=>this.handles.worldToLocal(this.bones[i].getWorldPosition(v(0,0,0)))),rotation:this.handles.getWorldQuaternion(new T.Quaternion()).invert().multiply(this.bones[chain.ids[0]].getWorldQuaternion(new T.Quaternion()))};
    }
    rememberChainPose(chain){
        chain.fkFrame=this.chainFrame(chain);
        const [root,joint,end]=chain.fkFrame.points,axis=end.clone().sub(root);
        if(axis.lengthSq()<1e-12)axis.copy(chain.axisLocal?.clone().applyQuaternion(chain.fkFrame.rotation)??joint.clone().sub(root));
        axis.normalize();
        const bend=chain.pole.position.clone().sub(root);bend.addScaledVector(axis,-bend.dot(axis));
        if(bend.lengthSq()>1e-12)chain.bendLocal=bend.normalize().applyQuaternion(chain.fkFrame.rotation.clone().invert());
        chain.axisLocal=axis.applyQuaternion(chain.fkFrame.rotation.clone().invert());
    }
    syncFKChain(chain,force=false){
        const frame=this.chainFrame(chain),previous=chain.fkFrame;
        if(!force&&previous&&frame.points.every((p,i)=>p.distanceToSquared(previous.points[i])<1e-16)&&1-Math.abs(frame.rotation.dot(previous.rotation))<1e-14)return;
        const [root,joint,end]=frame.points,axis=end.clone().sub(root);
        if(axis.lengthSq()<1e-12)axis.copy(chain.axisLocal?.clone().applyQuaternion(frame.rotation)??joint.clone().sub(root));
        axis.normalize();
        const bend=joint.clone().sub(root);bend.addScaledVector(axis,-bend.dot(axis));
        // At full extension the bend plane is undefined. Carry the last stable
        // plane in the upper bone's frame, including FK twist, instead of +Z.
        if(bend.length()<(chain.lengths[0]+chain.lengths[1])*1e-4){
            bend.copy(chain.bendLocal?.clone().applyQuaternion(frame.rotation)??chain.pole.position.clone().sub(root));
            bend.addScaledVector(axis,-bend.dot(axis));
        }
        if(bend.lengthSq()<1e-12){bend.set(0,0,1).addScaledVector(axis,-axis.z);if(bend.lengthSq()<1e-12)bend.set(1,0,0);}
        chain.target.position.copy(end);chain.lastTarget.copy(end);
        chain.pole.position.copy(joint).addScaledVector(bend.normalize(),chain.poleDistance);
        this.rememberChainPose(chain);
    }
    updateProceduralRotations(){
        // MHR's twist supports have pose-driven counter-rotations. Leaving them
        // frozen in parent space inverts triangles near hips/shoulders on bends.
        // Native Momentum XYZ is extrinsic: Three's equivalent order is ZYX.
        for(const [index,input] of this.proceduralInputs??[])input.angles.setFromQuaternion(input.inverseBind.clone().multiply(this.bones[index].quaternion),'ZYX');
        for(const rule of this.proceduralRotations??[]){
            const angles=rule.axes.map(axis=>axis.reduce((sum,[index,component,weight])=>sum+this.proceduralInputs.get(index).angles[['x','y','z'][component]]*weight,0));
            this.bones[rule.bone].quaternion.copy(rule.bind).multiply(new T.Quaternion().setFromEuler(new T.Euler(...angles,'ZYX')));
            if(rule.offset)this.bones[rule.bone].quaternion.multiply(rule.offset);
        }
    }
    solveIK(){
        if(!this.rootControl)return;this.rootControl.updateMatrixWorld(true);
        for(const chain of this.chains){
            if(!chain.enabled)continue;this.followPole(chain);
            const [a,b,c]=chain.ids.map(i=>this.bones[i]),root=a.getWorldPosition(v(0,0,0)),endRotation=c.getWorldQuaternion(new T.Quaternion()),upperRotation=a.getWorldQuaternion(new T.Quaternion());
            const result=solveTwoBone(root,chain.target.getWorldPosition(v(0,0,0)),chain.pole.getWorldPosition(v(0,0,0)),...chain.lengths,chain.bendLocal?.clone().applyQuaternion(upperRotation),chain.axisLocal?.clone().applyQuaternion(upperRotation));
            // An already-solved chain (especially a saved neutral pose) needs
            // no extra quaternion round-trip. Avoid numerical drift on reload.
            if(b.getWorldPosition(v(0,0,0)).distanceToSquared(result.joint)<1e-16&&c.getWorldPosition(v(0,0,0)).distanceToSquared(result.end)<1e-16){this.rememberChainPose(chain);continue;}
            const aim=(bone,child,target)=>{const origin=bone.getWorldPosition(v(0,0,0)),current=child.getWorldPosition(v(0,0,0)).sub(origin).normalize(),direction=target.clone().sub(origin).normalize();const delta=new T.Quaternion().setFromUnitVectors(current,direction);const world=delta.multiply(bone.getWorldQuaternion(new T.Quaternion()));const parent=bone.parent.getWorldQuaternion(new T.Quaternion()).invert();bone.quaternion.copy(parent.multiply(world));bone.updateWorldMatrix(false,true);};
            aim(a,b,result.joint);aim(b,c,result.end);
            // Translation of an IK target must not inherit the solver's change
            // of parent rotation at the wrist/ankle.
            c.quaternion.copy(c.parent.getWorldQuaternion(new T.Quaternion()).invert().multiply(endRotation));
            this.bodyRoot.updateMatrixWorld(true);this.rememberChainPose(chain);
        }
    }
    updateMarkers(){if(!this.bones)return;this.updateProceduralRotations();this.rootControl.updateMatrixWorld(true);this.markers.children.forEach(m=>m.position.copy(m.userData.object.getWorldPosition(v(0,0,0))));for(const {mesh,child,radius} of this.proxyParts){const length=child.position.length();mesh.position.copy(child.position).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(v(0,1,0),child.position.clone().normalize());mesh.scale.set(1,Math.max(.001,length)/(1+2*radius),1);}for(const chain of this.chains){if(!chain.enabled)this.syncFKChain(chain);const joint=this.bones[chain.ids[1]].getWorldPosition(v(0,0,0)),pole=chain.pole.getWorldPosition(v(0,0,0)),points=chain.poleLink.geometry.attributes.position;points.setXYZ(0,joint.x,joint.y,joint.z);points.setXYZ(1,pole.x,pole.y,pole.z);points.needsUpdate=true;}}
    updateAllMarkers(){for(const body of this.bodies)this.withBody(body,()=>this.updateMarkers());}
    bodySnapshot(body){return this.withBody(body,()=>{
        this.updateProceduralRotations();
        return {version:1,id:body.id,body:this.gender,source:this.source,body_color:`#${this.material.color.getHexString()}`,shape:this.shapeControl.snapshot(),root_transform:{position:this.rootControl.position.toArray(),quaternion:this.rootControl.quaternion.toArray()},joints:this.bones.map(b=>({name:b.name,position:b.position.toArray(),quaternion:b.quaternion.toArray(),scale:b.scale.toArray()})),procedural_offsets:this.proceduralRotations.map(rule=>(rule.offset??new T.Quaternion()).toArray()),ik:this.chains.map(c=>({name:c.name,enabled:c.enabled,target:c.target.position.toArray(),pole:c.pole.position.toArray()})),openpose:this.openPose.project(this.camera,this.widget('width').value,this.widget('height').value)};
    });}
    snapshot(){
        const bodies=this.bodies.map(body=>this.bodySnapshot(body));
        const common={version:1,selected_body_id:this.selectedBody?.id??null,floor_visible:this.floor.visible,camera:{position:this.camera.position.toArray(),target:this.orbit.target.toArray()},lighting:{direction:this.keyLight.position.clone().sub(this.keyLight.target.position).normalize().toArray()}};
        // Preserve the original one-body format for existing workflows/callers.
        return bodies.length===1?{...bodies[0],...common}:{...common,bodies,openpose:{version:2,width:this.widget('width').value,height:this.widget('height').value,people:bodies.map(body=>body.openpose)}};
    }
    validBodyState(data,body=this.bodyContext){
        if(data?.joints?.some(j=>j?.scale!==undefined&&(!finiteVector(j.scale,3)||!j.scale.every(s=>s>1e-6&&s<=1000))))return false;
        if(data?.procedural_offsets!==undefined&&(!Array.isArray(data.procedural_offsets)||data.procedural_offsets.length!==(body?.proceduralRotations?.length??body?.proceduralCount??0)||!data.procedural_offsets.every(q=>finiteVector(q,4)&&q.some(x=>x!==0))))return false;
        return !!body&&data?.version===1&&data.body===body.gender&&data.source===body.source&&(data.body_color===undefined||validBodyColor(data.body_color))&&(data.shape===undefined||validMHRShape(data.shape))&&(data.root_transform===undefined||(finiteVector(data.root_transform?.position,3)&&finiteVector(data.root_transform?.quaternion,4)&&data.root_transform.quaternion.some(x=>x!==0)))&&Array.isArray(data.joints)&&data.joints.length===body.bones.length&&data.joints.every((j,i)=>j?.name===body.bones[i].name&&finiteVector(j.position,3)&&finiteVector(j.quaternion,4)&&j.quaternion.some(x=>x!==0))&&Array.isArray(data.ik)&&data.ik.length===body.chains.length&&data.ik.every((c,i)=>c?.name===body.chains[i].name&&typeof c.enabled==='boolean'&&finiteVector(c.target,3)&&finiteVector(c.pole,3));
    }
    validPose(data){
        if(data?.version!==1||!finiteVector(data.camera?.position,3)||!finiteVector(data.camera?.target,3)||(data.floor_visible!==undefined&&typeof data.floor_visible!=='boolean')||(data.lighting!==undefined&&(!finiteVector(data.lighting?.direction,3)||!data.lighting.direction.some(x=>x!==0))))return false;
        const description=state=>this.bodies.find(body=>body.id===state.id&&body.source===state.source&&body.gender===state.body)??this.bodyAssets.get(`${state.source}:${state.body}`)?.validation;
        if(!Array.isArray(data.bodies))return this.validBodyState(data,description(data)??this.bodyContext);
        const ids=data.bodies.map(body=>body?.id);
        return data.bodies.length<=256&&ids.every(id=>typeof id==='string'&&id.length>0&&id.length<=80)&&new Set(ids).size===ids.length&&(data.selected_body_id===null||ids.includes(data.selected_body_id))&&data.bodies.every(body=>this.validBodyState(body,description(body)));
    }
    restoreBody(body,data){this.withBody(body,()=>{
        this.setBodyColor(data.body_color??DEFAULT_BODY_COLOR);this.shapeControl.restore(data.shape);
        this.rootControl.position.fromArray(data.root_transform?.position??[0,0,0]);this.rootControl.quaternion.fromArray(data.root_transform?.quaternion??[0,0,0,1]).normalize();
        this.proceduralRotations.forEach((rule,index)=>{rule.offset=new T.Quaternion().fromArray(data.procedural_offsets?.[index]??[0,0,0,1]).normalize();});
        data.joints?.forEach((joint,index)=>{this.bones[index].position.fromArray(joint.position);this.bones[index].quaternion.fromArray(joint.quaternion).normalize();this.bones[index].scale.fromArray(joint.scale??Array(3).fill(body.prepared.data.localTransforms?.[index]?.scale??1));});
        data.ik?.forEach((chain,index)=>{this.chains[index].enabled=chain.enabled;this.chains[index].target.position.fromArray(chain.target);this.chains[index].pole.position.fromArray(chain.pole);this.chains[index].lastTarget.copy(this.chains[index].target.position);});
        this.updateProceduralRotations();this.refreshChainLengths();for(const chain of this.chains){this.rememberChainPose(chain);chain.poleDistance=Math.max(.01,chain.pole.position.distanceTo(chain.fkFrame.points[1]));}
        this.solveIK();this.updateMarkers();
    });}
    restoreSceneSettings(data){
        if(finiteVector(data.camera?.position,3)&&finiteVector(data.camera?.target,3)){this.camera.position.fromArray(data.camera.position);this.orbit.target.fromArray(data.camera.target);this.orbit.update();}
        if(finiteVector(data.lighting?.direction,3)&&data.lighting.direction.some(x=>x!==0))this.setLightDirection(v(...data.lighting.direction));
        this.setFloorVisible(data.floor_visible??true);
    }
    restore(data){
        if(!this.validPose(data))throw Error('Invalid saved pose scene.');
        const records=Array.isArray(data.bodies)?data.bodies:[{...data,id:data.id??this.bodyContext?.id??`body-${++this.nextBodyId}`}];
        const ids=new Set(records.map(item=>item.id));this.gizmo.detach();
        for(const body of [...this.bodies])if(!ids.has(body.id))this.destroyBody(body);
        for(const record of records){
            let body=this.bodies.find(item=>item.id===record.id);
            if(body&&(body.source!==record.source||body.gender!==record.body)){this.destroyBody(body);body=null;}
            if(!body){const prepared=this.bodyAssets.get(`${record.source}:${record.body}`);if(!prepared)throw Error('Saved body asset is unavailable. Reload the editor.');body=this.createBody(prepared,record,record.id);}
            else this.restoreBody(body,record);
        }
        this.bodies.sort((a,b)=>records.findIndex(record=>record.id===a.id)-records.findIndex(record=>record.id===b.id));
        for(const body of this.bodies)this.modelTree.branch.append(body.shapeControl.entry);
        this.restoreSceneSettings(data);
        const selected=data.selected_body_id===null?null:this.bodies.find(body=>body.id===data.selected_body_id)??this.bodies[0];
        if(selected){this.activateBody(selected);this.selectObject(selected.rootControl);}else{this.activeBody=this.bodies[0]??null;this.useBody(this.activeBody);this.selectScene();}
    }
    commit(){if(this.loading)return;this.widget('pose_data').value=JSON.stringify(this.snapshot());this.widget('render_data').value='';this.node.setDirtyCanvas?.(true,true);}
    record(){if(this.loading)return;const data=this.snapshot();if(JSON.stringify(data)===JSON.stringify(this.history?.at(-1)))return;this.history??=[];this.history.push(data);if(this.history.length>50)this.history.shift();this.future=[];}
    undo(){if(this.loading||this.history?.length<2)return;this.restore(this.history.at(-2));this.future.push(this.history.pop());this.commit();}
    redo(){if(this.loading||!this.future?.length)return;const data=this.future.at(-1);this.restore(data);this.future.pop();this.history.push(data);this.commit();}
    resize(){if(this.disposed)return;const width=this.widget('width').value,height=this.widget('height').value,availableW=this.view.clientWidth||500,availableH=this.view.clientHeight||440;const w=Math.min(availableW,availableH*width/height),h=w*height/width;this.camera.aspect=width/height;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h);this.renderer.domElement.style.margin='auto';}
    capture(){if(this.loading)throw Error('Wait for the pose body to finish loading.');for(const body of this.bodies)body.shapeControl.assertRenderable();this.updateAllMarkers();this.commit();const width=this.widget('width').value,height=this.widget('height').value;const visible=[...this.bodies.flatMap(body=>[body.markers,body.handles,body.poleLinks,body.rootMarker]),this.grid,this.gizmo.getHelper()];const saved=visible.map(o=>o.visible);const size=this.renderer.getSize(new T.Vector2()),pixel=this.renderer.getPixelRatio();try{visible.forEach(o=>o.visible=false);this.renderer.setPixelRatio(1);this.renderer.setSize(width,height,false);this.renderer.render(this.scene,this.camera);const png=this.renderer.domElement.toDataURL('image/png');const maps=this.renderPasses.capture(this.renderer,this.scene,this.camera,this.bodies.map(body=>body.bodyRoot),this.floor);this.widget('render_data').value=JSON.stringify({version:1,render:png,...maps});return png;}finally{visible.forEach((o,i)=>o.visible=saved[i]);this.renderer.setPixelRatio(pixel);this.renderer.setSize(size.x,size.y,false);this.renderer.render(this.scene,this.camera);}}
    reset(){
        if(this.loading||!this.selectedBody)return;
        const scene=this.snapshot(),shape=this.shapeControl.snapshot();
        const reset={id:this.selectedBody.id,...(shape?{source:this.source,body:this.gender,shape}:{}),body_color:`#${this.material.color.getHexString()}`};
        if(Array.isArray(scene.bodies)){reset.root_transform={position:this.rootControl.position.toArray(),quaternion:[0,0,0,1]};scene.bodies=scene.bodies.map(body=>body.id===reset.id?reset:body);}
        else Object.assign(scene,reset,{joints:undefined,ik:undefined,root_transform:undefined,source:shape?this.source:undefined});
        this.widget('pose_data').value=JSON.stringify(scene);this.ready=this.load(scene);
    }
    dispose(){this.disposed=true;api.removeEventListener?.('executing',this.poseExecuting);cancelAnimationFrame(this.frame);this.resizeObserver.disconnect();this.clearBodies();this.lightControl.dispose();this.renderPasses.dispose();this.orbit.dispose();this.gizmo.dispose();this.scene.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});this.renderer.dispose();}
}

app.registerExtension({name:'toyxyz.pose3d',beforeConfigureGraph(data){migratePose3DWorkflow(data);},async beforeRegisterNodeDef(NodeType,data){
    if(data.name!=='ToyxyzPose3DEditor')return;
    const configure=NodeType.prototype.configure;
    if(configure)NodeType.prototype.configure=function(info,...args){
        if(['female','male'].includes(info?.widgets_values?.[0]))info={...info,widgets_values:info.widgets_values.slice(1)};
        return configure.call(this,info,...args);
    };
    const created=NodeType.prototype.onNodeCreated;
    NodeType.prototype.onNodeCreated=function(){
        created?.apply(this,arguments);ensureOpenPoseOutput(this);ensureSAMInput(this);
        const root=document.createElement('div');let ui;
        try{ui=this._pose3d=new Pose3DEditorUI(this,root);}catch(e){root.textContent=`3D editor failed: ${e.message}`;return;}
        const dom=this.addDOMWidget('pose_editor','div',root,{serialize:false,hideOnZoom:false,getMinHeight:()=>620,getMaxHeight:()=>620});dom.serialize=false;
        for(const name of ['pose_data','render_data']){
            const widget=this.widgets.find(w=>w.name===name);widget.hidden=true;widget.options??={};widget.options.hidden=true;widget.computeSize=()=>[0,-4];
            widget.serializeValue=async()=>{await ui.ready;if(name==='render_data')ui.capture();else ui.commit();return widget.value;};
        }
        for(const name of ['width','height']){
            const widget=this.widgets.find(w=>w.name===name),previous=widget.callback;
            widget.callback=(...args)=>{previous?.apply(widget,args);ui.resize();ui.commit();};
        }
        const configured=this.onConfigure;this.onConfigure=function(){configured?.apply(this,arguments);ensureOpenPoseOutput(this);ensureSAMInput(this);if(this.size?.[0]<900)this.setSize([900,this.size[1]]);ui.ready=ui.load();};
        const removed=this.onRemoved;this.onRemoved=function(){ui.dispose();removed?.apply(this,arguments);};
        this.setSize([940,Math.max(820,this.computeSize?.()[1]||0)]);
    };
}});
