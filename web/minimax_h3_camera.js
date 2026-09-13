import { app } from "../../scripts/app.js";
import * as THREE from "./vendor/three/three.module.min.js";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { TransformControls } from "./vendor/three/TransformControls.js";
import { createBackgroundGrid, addHumanFaceMarker } from "./h3_orientation_helpers.js";
import { channels, clone, defaultSubject, normalizeScene, sampleEntity, evaluate, resolution, ratios, framing, sub, matVec, transpose, subjectGeometry } from "./h3_scene_math.js";

const DEG = 180/Math.PI;
const KEY_COLORS = {hold:"#e8b15c",linear:"#64b5f6",smooth:"#70cf9b"};
const TIMELINE_HEIGHT=145, TIMELINE_LANE=24;
// Studio-style, view-relative lighting. Keep the formula aligned with shade_surface
// in the CPU renderer, so the editor and output camera both show solid geometry.
function subjectMaterial(color) {
    return new THREE.ShaderMaterial({
        uniforms:{baseColor:{value:new THREE.Color(color)}},side:THREE.DoubleSide,
        vertexShader:`varying vec3 surfaceNormal;
            void main(){surfaceNormal=normalMatrix*normal;
            gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
        fragmentShader:`uniform vec3 baseColor; varying vec3 surfaceNormal;
            void main(){vec3 n=normalize(surfaceNormal);if(!gl_FrontFacing)n=-n;
            vec3 key=normalize(vec3(-0.45,0.65,1.0));
            vec3 fill=normalize(vec3(0.7,0.1,0.5));
            vec3 halfVector=normalize(key+vec3(0.0,0.0,1.0));
            float light=0.22+0.65*max(dot(n,key),0.0)+0.10*max(dot(n,fill),0.0);
            float highlight=0.10*pow(max(dot(n,halfVector),0.0),24.0);
            gl_FragColor=vec4(baseColor*light+vec3(highlight),1.0);
            #include <colorspace_fragment>
            }`,
    });
}
function el(tag, attrs={}, parent=null) {
    const node=document.createElement(tag);
    for (const [key,value] of Object.entries(attrs)) {
        if (key==="text") node.textContent=value;
        else if (key==="class") node.className=value;
        else node[key]=value;
    }
    parent?.append(node); return node;
}
function styles() {
    if (document.getElementById("h3-scene-style")) return;
    el("style",{id:"h3-scene-style",text:`
    .h3-scene{height:100%;box-sizing:border-box;background:#171c23;color:#d6e0eb;font:12px Arial,sans-serif;padding:10px;display:flex;flex-direction:column;gap:8px;overflow:hidden}
    .h3-scene *{box-sizing:border-box}.h3-scene button,.h3-scene select,.h3-scene input,.h3-scene textarea{font:inherit;color:inherit;background:#272f39;border:1px solid #44505e;border-radius:4px;min-width:0;padding:5px}
    .h3-scene button{cursor:pointer}.h3-scene button.active{background:#256290}.h3-scene button:hover{border-color:#7cc8ff}.h3-scene input[type=checkbox]{width:auto}.h3-scene input[type=number]{width:65px}
    .h3-scene .bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.h3-scene .main{display:grid;grid-template-columns:145px minmax(220px,1fr) 205px;gap:8px;flex:1;min-height:300px}
    .h3-scene .list,.h3-scene .inspector{background:#1e242d;border:1px solid #36414f;border-radius:5px;padding:8px;overflow:auto}
    .h3-scene .list button{width:100%;text-align:left;margin-bottom:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3-scene .viewport{position:relative;min-width:0;min-height:0;background:#0e1419;overflow:hidden;border:1px solid #36414f;border-radius:5px}
    .h3-scene .viewport-toolbar{position:absolute;z-index:2;left:8px;right:8px;top:8px;display:flex;align-items:flex-start;justify-content:space-between;gap:8px;pointer-events:none}
    .h3-scene .transform-tools{display:flex;flex-wrap:wrap;gap:4px;min-width:0}
    .h3-scene .viewport-toolbar button,.h3-scene .viewport-toolbar select{pointer-events:auto}
    .h3-scene .camera-view-toggle{flex:none;white-space:nowrap}
    .h3-scene .camera-view[hidden]{display:none}
    .h3-scene .viewport>canvas{display:block;width:100%;height:100%;touch-action:none}.h3-scene .camera-view{position:absolute;right:8px;top:8px;width:26%;max-width:240px;max-height:40%;border:1px solid #64bcf4;background:#0e1419;pointer-events:none}
    .h3-scene .camera-view canvas{display:block;width:100%;height:100%;object-fit:contain}.h3-scene .view-label{position:absolute;left:8px;bottom:8px;max-width:95%;background:#151d25db;padding:5px;pointer-events:none}
    .h3-scene .inspector label{display:flex;justify-content:space-between;align-items:center;gap:5px;margin:5px 0}.h3-scene .inspector label>select,.h3-scene .inspector label>input:not([type=checkbox]){width:115px}.h3-scene .inspector .vector{display:flex;gap:4px}.h3-scene .inspector .vector input{width:calc(33.33% - 3px)}
    .h3-scene .section-title{color:#8dcafa;margin:9px 0 5px}
    .h3-scene .timeline-scroll{height:147px;min-height:147px;flex:0 0 147px;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;background:#10161e;border:1px solid #3b4859}
    .h3-scene .timeline{display:block;position:sticky;top:0;width:100%;height:145px;background:#10161e;touch-action:none}
    .h3-scene .timeline-spacer{pointer-events:none}
    .h3-scene .timeline-timing{margin-left:auto;justify-content:flex-end}
    .h3-scene .timeline-timing>span{white-space:nowrap}
    .h3-scene .status{min-height:15px;color:#b2c6da;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    `},document.head);
}

export class H3SceneEditor {
    constructor(node,root,widget) {
        styles(); this.node=node;this.root=root;this.widget=widget;root.className="h3-scene";
        this.data=normalizeScene(widget?.value);this.frame=0;this.selected="camera";this.autoKey=true;
        this.cameraViewVisible=node.properties?.h3_camera_view_visible!==false;
        this.selectedKey=null;this.transformButtons=new Map();
        this.undo=[];this.redo=[];this.meshes=new Map();this.disposed=false;this.playing=false;
        this.build();
        // Native select popups do not inherit the canvas's CSS transform.
        // Match the prompter's option-font compensation without resizing the control.
        this.nativeSelectScaleHandler=event=>{
            const select=event.target?.closest?.("select");
            if(this.disposed||!select||!this.root.contains(select))return;
            const scale=Math.max(1,Math.min(4,this.root.getBoundingClientRect().width/Math.max(1,this.root.offsetWidth)));
            const font=Math.max(10,Number.parseFloat(getComputedStyle(select).fontSize)||12);
            for(const option of select.querySelectorAll("option,optgroup"))option.style.fontSize=`${font*scale}px`;
        };
        for(const type of ["pointerdown","keydown","focusin"])this.root.addEventListener(type,this.nativeSelectScaleHandler,true);
        try {this.setup3D();} catch(error) {this.dispose3D();this.status.textContent=`3D preview unavailable: ${error.message}. Scene data and backend output remain available.`;}
        this.refresh(); this.save();
        this.lastTime=performance.now();this.tick=this.tick.bind(this);this.raf=requestAnimationFrame(this.tick);
    }
    button(parent,label,title,fn) {const b=el("button",{text:label,title},parent);b.onclick=fn;return b;}
    select(parent,values,value,title,fn) {
        const s=el("select",{title},parent);
        for (const entry of values) {const [v,name]=Array.isArray(entry)?entry:[entry,entry];el("option",{value:v,text:name},s);}
        s.value=value;s.onchange=()=>fn(s.value);return s;
    }
    entity(id=this.selected) {return id==="camera"?this.data.camera:this.data.subjects.find(s=>s.id===id);}
    build() {
        const bar=el("div",{class:"bar"},this.root);
        this.button(bar,"+ Subject","Add a shaded sphere-head / box-body subject with arms",()=>{
            this.remember();const s=clone(defaultSubject);s.id=`subject_${Date.now()}_${this.data.subjects.length}`;s.name=`Subject ${this.data.subjects.length+1}`;
            s.position[0]=this.data.subjects.length*.8;s.color=["#38a6c9","#e8a34a","#d176ac","#81c878"][this.data.subjects.length%4];
            this.data.subjects.push(s);this.selected=s.id;this.commit();
        });
        this.button(bar,"Delete","Delete the selected subject (camera cannot be deleted)",()=>{
            if(this.selected==="camera")return;this.remember();this.data.subjects=this.data.subjects.filter(s=>s.id!==this.selected);
            if(this.data.camera.target===this.selected)this.data.camera.target=this.data.subjects[0]?.id??"";
            this.selected="camera";this.commit();
        });
        this.button(bar,"Undo","Undo scene editing",()=>this.history(false));this.button(bar,"Redo","Redo scene editing",()=>this.history(true));
        this.button(bar,"Reset view","Reset only the editor viewpoint",()=>{this.observer?.position.set(6,5,8);this.controls?.target.set(0,1,0);this.controls?.update();});
        const main=el("div",{class:"main"},this.root);
        this.list=el("div",{class:"list"},main);this.viewport=el("div",{class:"viewport"},main);
        this.canvas=el("canvas",{title:"Drag empty space to orbit; middle/right-drag to pan; scroll the wheel to zoom. Click a subject/camera to select."},this.viewport);
        this.inset=el("div",{class:"camera-view",title:"Output camera view. No editor guides are rendered."},this.viewport);
        this.outputCanvas=el("canvas",{},this.inset);this.viewLabel=el("div",{class:"view-label"},this.viewport);
        this.viewportToolbar=el("div",{class:"viewport-toolbar"},this.viewport);
        const tools=el("div",{class:"transform-tools"},this.viewportToolbar);
        for(const mode of ["translate","rotate","scale"])this.transformButtons.set(mode,this.button(tools,mode[0].toUpperCase()+mode.slice(1),`Use the ${mode} gizmo`,()=>{
            if(this.selected==="camera"&&mode==="scale"){this.status.textContent="Camera scale is fixed. Change FOV or use Translate instead.";return;}
            this.gizmo?.setMode(mode);
            this.updateTransformButtons();
        }));
        this.updateTransformButtons();
        this.select(tools,[["world","World"],["local","Local"]],"world","Gizmo coordinate space",v=>this.gizmo?.setSpace(v));
        this.cameraViewToggle=this.button(this.viewportToolbar,"Camera view","Toggle the camera-view preview; rendered outputs are unchanged",()=>this.setCameraViewVisible(!this.cameraViewVisible));
        this.cameraViewToggle.className="camera-view-toggle";
        this.setCameraViewVisible(this.cameraViewVisible,false);
        this.inspector=el("div",{class:"inspector"},main);
        const transport=el("div",{class:"bar"},this.root);
        this.playButton=this.button(transport,"Play","Play the timeline",()=>{this.playing=!this.playing;this.playTime=this.frame;this.playButton.textContent=this.playing?"Pause":"Play";this.lastTime=performance.now();this.publishPlayhead();});
        this.button(transport,"|◀","Go to frame zero",()=>this.seek(0));
        this.button(transport,"+ Key","Key the selected object's current transform",()=>{this.remember();this.writeKey(this.entity(),sampleEntity(this.entity(),this.frame,this.data.interpolation));this.commit();});
        this.button(transport,"− Key","Remove the selected object's key at this frame",()=>{this.remember();this.entity().keys=this.entity().keys.filter(k=>k.frame!==this.frame);this.commit();});
        const auto=el("label",{title:"Create a key when editing an animated object at an unkeyed frame"},transport);
        const check=el("input",{type:"checkbox",checked:true},auto);auto.append(" Auto key");check.onchange=()=>this.autoKey=check.checked;
        this.timelineConfig=el("div",{class:"bar"},transport);
        const timing=el("div",{class:"bar timeline-timing"},transport);
        this.frameInput=el("input",{type:"number",min:0,max:this.data.frames-1,value:0,title:"Current frame"},timing);this.frameInput.onchange=()=>this.seek(Number(this.frameInput.value));
        this.timeLabel=el("span",{},timing);
        this.durationConfig=el("div",{class:"bar"},timing);
        this.timelineScroll=el("div",{class:"timeline-scroll",title:"Scroll vertically to browse object tracks. Timeline height stays fixed."},this.root);
        this.timeline=el("canvas",{class:"timeline",title:"Drag the playhead to scrub. Drag a diamond to move that object's keyframe. Scroll to browse tracks."},this.timelineScroll);
        this.timelineSpacer=el("div",{class:"timeline-spacer","aria-hidden":"true"},this.timelineScroll);
        this.timelineScroll.onscroll=()=>this.drawTimeline();
        this.timeline.onpointerdown=e=>this.timelineDown(e);
        const config=el("div",{class:"bar"},this.root);
        this.config=config;this.renderConfig();
        const footer=el("div",{class:"bar"},this.root);
        this.status=el("div",{class:"status",text:"Proxy geometry only. Hold camera jumps create cuts; other keys remain continuous."},footer);
        for(const event of ["pointerdown","pointerup","wheel","keydown"])this.root.addEventListener(event,e=>e.stopPropagation());
    }
    renderConfig() {
        this.config.replaceChildren();this.timelineConfig.replaceChildren();this.durationConfig.replaceChildren();
        const number=(label,key,min,max,step,parent=this.config,title=label)=>{
            el("span",{text:label,title},parent);const input=el("input",{type:"number",min,max,step,value:this.data[key],title},parent);
            input.onchange=()=>{
                this.remember();const oldEnd=this.data.frames-1;this.data[key]=Number(input.value);
                if(key==="requested_duration"){
                    const next=normalizeScene({...this.data,camera:{...this.data.camera,keys:[]},subjects:[]});
                    const ratio=(next.frames-1)/oldEnd;
                    for(const entity of [this.data.camera,...this.data.subjects])for(const k of entity.keys)k.frame=Math.round(k.frame*ratio);
                    this.frame=Math.round(this.frame*ratio);
                }
                this.data=normalizeScene(this.data);this.frame=Math.min(this.frame,this.data.frames-1);this.playTime=this.frame;this.commit();
            };
        };
        number("Duration (s)","requested_duration",.1,60,.1,this.durationConfig);
        el("span",{text:"Key interpolation"},this.timelineConfig);
        this.keyInterpolation=this.select(this.timelineConfig,["smooth","linear","hold"],"smooth","Move the playhead onto a key. Applies from the previous key to this key; Hold jumps at this key.",v=>{
            const key=this.keyAtPlayhead();
            if(!key||key.frame===0)return;this.remember();key.interpolation=v;this.commit();
        });
        const legend=el("span",{title:"Incoming interpolation: previous key → selected key. White outline marks selection."},this.timelineConfig);
        legend.style.fontSize="10px";
        for(const [mode,color] of Object.entries(KEY_COLORS)){
            const label=el("span",{text:`◆ ${mode[0].toUpperCase()+mode.slice(1)} `},legend);label.style.color=color;
        }
        this.updateKeyInterpolation();
        this.select(this.config,Object.keys(ratios),this.data.aspect_ratio,"Output aspect ratio (rounded to multiples of 32)",v=>{this.remember();this.data.aspect_ratio=v;this.commit();});
        number("MP","megapixels",.01,4,.01,this.config,"Megapixels. Lower resolution renders faster and can speed up video generation.");
        const promptToggle=this.button(this.config,"Use camera prompt","Send procedural camera motion to the connected prompter when enabled. Explicit user instructions take priority. Camera render and the separate camera_prompt output are unchanged.",()=>{
            this.remember();this.data.use_camera_prompt=!this.data.use_camera_prompt;this.commit();
        });
        promptToggle.classList.toggle("active",this.data.use_camera_prompt);
        promptToggle.setAttribute("aria-pressed",String(this.data.use_camera_prompt));
        const refvidToggle=this.button(this.config,"refvid","Send the 3D render as a video reference to the connected prompter. Off sends only camera text when Use camera prompt is enabled. Local previews and the separate camera_render output are unchanged.",()=>{
            this.remember();this.data.refvid=!this.data.refvid;this.commit();
        });
        refvidToggle.classList.toggle("active",this.data.refvid);
        refvidToggle.setAttribute("aria-pressed",String(this.data.refvid));
        const gridToggle=this.button(this.config,"Floor grid","Show the floor and grid in both previews and the rendered sequence sent to the prompter.",()=>{
            this.remember();this.data.show_grid=!this.data.show_grid;this.commit();
        });
        gridToggle.classList.toggle("active",this.data.show_grid);
        gridToggle.setAttribute("aria-pressed",String(this.data.show_grid));
        const backgroundToggle=this.button(this.config,"Background grid","World-oriented spherical grid for reading camera rotation. Included in both previews and the rendered reference; independent of Floor grid.",()=>{
            this.remember();this.data.show_background_grid=!this.data.show_background_grid;this.commit();
        });
        backgroundToggle.classList.toggle("active",this.data.show_background_grid);
        backgroundToggle.setAttribute("aria-pressed",String(this.data.show_background_grid));
        const [w,h]=resolution(this.data);el("span",{text:`${w}×${h} · ${this.data.frames}f / ${(this.data.frames/this.data.fps).toFixed(3)}s · 24 fps`,title:"Actual output: 24 fps, frame count rounded up to 17n + 5 (same as the prompter)"},this.config);
    }
    setup3D() {
        this.renderer=new THREE.WebGLRenderer({canvas:this.canvas,antialias:true});
        this.outputRenderer=new THREE.WebGLRenderer({canvas:this.outputCanvas,antialias:true});
        for(const r of [this.renderer,this.outputRenderer]){r.setPixelRatio(1);r.setClearColor(0x0e1419);}
        this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0x0e1419);
        this.backgroundGrid=createBackgroundGrid();this.backgroundGrid.visible=this.data.show_background_grid;this.scene.add(this.backgroundGrid);
        this.observer=new THREE.PerspectiveCamera(45,1,.01,1000);this.observer.position.set(6,5,8);this.observer.layers.enable(1);
        this.camera=new THREE.PerspectiveCamera(40,1,.01,500);
        this.controls=new OrbitControls(this.observer,this.canvas);this.controls.mouseButtons.MIDDLE=THREE.MOUSE.PAN;this.controls.target.set(0,1,0);this.controls.update();
        this.floorGrid=new THREE.Group();this.scene.add(this.floorGrid);this.floorGrid.visible=this.data.show_grid;
        const floor=new THREE.Mesh(new THREE.PlaneGeometry(40,40),new THREE.MeshBasicMaterial({color:0x1a2126,side:THREE.DoubleSide}));floor.rotation.x=-Math.PI/2;this.floorGrid.add(floor);
        const grid=new THREE.GridHelper(40,40,0x3b4d57,0x3b4d57);grid.position.y=.0005;grid.material.transparent=false;grid.material.opacity=1;this.floorGrid.add(grid);
        this.rig=new THREE.Group();this.rig.userData.entityId="camera";this.scene.add(this.rig);
        // The apex is the camera position; the broad square face points along
        // the camera's local -Z viewing axis. Only eight outer edges are drawn.
        const pyramidGeometry=new THREE.ConeGeometry(.36,.55,4);
        pyramidGeometry.rotateY(Math.PI/4);pyramidGeometry.rotateX(Math.PI/2);pyramidGeometry.translate(0,0,-.275);
        const icon=new THREE.Mesh(pyramidGeometry,new THREE.MeshBasicMaterial({color:0xf5bb63,transparent:true,opacity:.18,depthWrite:false,side:THREE.DoubleSide}));
        icon.add(new THREE.LineSegments(new THREE.EdgesGeometry(pyramidGeometry),new THREE.LineBasicMaterial({color:0xf5bb63})));
        this.rig.add(icon);
        this.rig.traverse(o=>o.layers.set(1));
        this.targetLine=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),
            new THREE.LineBasicMaterial({color:0xa8d7ed,transparent:true,opacity:.7,depthWrite:false}));
        this.targetLine.layers.set(1);this.targetLine.frustumCulled=false;this.scene.add(this.targetLine);
        this.gizmo=new TransformControls(this.observer,this.canvas);this.gizmo.setSize(.75);
        const helper=this.gizmo.getHelper();helper.traverse(o=>o.layers.set(1));this.scene.add(helper);this.gizmo.getRaycaster().layers.enable(1);
        this.gizmo.addEventListener("dragging-changed",e=>{this.controls.enabled=!e.value;this.dragging=e.value;});
        this.gizmo.addEventListener("mouseDown",()=>{this.remember();this.playing=false;this.playButton.textContent="Play";});
        this.gizmo.addEventListener("objectChange",()=>this.gizmoChange());
        this.gizmo.addEventListener("mouseUp",()=>{this.save();this.updatePath();this.renderInspector();this.drawTimeline();});
        this.canvasPointerDown=e=>{this.pickStart=e.button===0?[e.clientX,e.clientY]:null;};
        this.canvasPointerUp=e=>{
            if(e.button!==0||this.dragging||this.gizmo.axis||!this.pickStart||Math.hypot(e.clientX-this.pickStart[0],e.clientY-this.pickStart[1])>4)return;
            const box=this.canvas.getBoundingClientRect(),pointer=new THREE.Vector2((e.clientX-box.left)/box.width*2-1,1-(e.clientY-box.top)/box.height*2);
            const ray=new THREE.Raycaster();ray.layers.enable(1);ray.setFromCamera(pointer,this.observer);
            const hits=ray.intersectObjects([this.rig,...this.meshes.values()],true);
            let object=hits[0]?.object;while(object&&!object.userData.entityId)object=object.parent;
            if(object){this.selected=object.userData.entityId;this.refresh();}
        };
        this.canvas.addEventListener("pointerdown",this.canvasPointerDown);
        this.canvas.addEventListener("pointerup",this.canvasPointerUp);
        this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(this.viewport);this.resizeObserver.observe(this.viewportToolbar);this.resize();
    }
    setCameraViewVisible(visible,persist=true) {
        this.cameraViewVisible=!!visible;this.inset.hidden=!this.cameraViewVisible;
        this.cameraViewToggle.classList.toggle("active",this.cameraViewVisible);
        this.cameraViewToggle.setAttribute("aria-pressed",String(this.cameraViewVisible));
        if(persist){this.node.properties??={};this.node.properties.h3_camera_view_visible=this.cameraViewVisible;this.node.setDirtyCanvas?.(true,true);}
        this.resize();
    }
    updateTransformButtons() {
        const active=this.gizmo?.mode??"translate";
        for(const [mode,button] of this.transformButtons){button.classList.toggle("active",mode===active);button.setAttribute("aria-pressed",String(mode===active));}
    }
    resize() {
        if(!this.renderer||!this.outputRenderer||!this.observer||!this.camera)return;
        const w=Math.max(1,this.viewport.clientWidth),h=Math.max(1,this.viewport.clientHeight);
        this.renderer.setSize(w,h,false);this.observer.aspect=w/h;this.observer.updateProjectionMatrix();
        const top=this.viewportToolbar.offsetHeight+16;this.inset.style.top=`${top}px`;
        const [rw,rh]=resolution(this.data);
        const iw=Math.max(32,Math.min(w*.26,240,Math.max(32,Math.min(h*.4,180,h-top-8))*rw/rh)),ih=iw*rh/rw;
        this.inset.style.width=`${iw}px`;this.inset.style.height=`${ih}px`;
        if(this.cameraViewVisible)this.outputRenderer.setSize(Math.round(iw),Math.round(ih),false);
        this.camera.aspect=rw/rh;this.camera.updateProjectionMatrix();this.drawTimeline();
    }
    remember() {this.undo.push(JSON.stringify(this.data));if(this.undo.length>60)this.undo.shift();this.redo=[];}
    history(forward) {
        const from=forward?this.redo:this.undo,to=forward?this.undo:this.redo;if(!from.length)return;
        to.push(JSON.stringify(this.data));this.data=normalizeScene(from.pop());this.frame=Math.min(this.frame,this.data.frames-1);this.commit();
    }
    save() {this.widget.value=JSON.stringify(this.data);this.node.setDirtyCanvas?.(true,true);
        window.dispatchEvent(new CustomEvent("h3-camera-scene-changed",{detail:{nodeId:this.node.id,node:this.node}}));}
    commit() {this.data=normalizeScene(this.data);this.save();this.refresh();}
    load(value) {this.data=normalizeScene(value);this.frame=0;this.playTime=0;this.selected="camera";this.selectedKey=null;this.undo=[];this.redo=[];this.setCameraViewVisible(this.node.properties?.h3_camera_view_visible!==false,false);this.refresh();}
    writeKey(entity,state) {
        // Preserve an edited key's incoming mode; inserting within a segment
        // inherits that segment's destination mode, not the previous key's mode.
        const interpolation=entity.keys.find(k=>k.frame>=this.frame)?.interpolation??entity.keys.at(-1)?.interpolation??this.data.interpolation;
        const key={frame:this.frame,interpolation,...Object.fromEntries(channels.filter(k=>k in state).map(k=>[k,clone(state[k])]))};
        if(entity.id==="camera")key.aim=state.aim??entity.aim;
        entity.keys=entity.keys.filter(k=>k.frame!==this.frame);entity.keys.push(key);entity.keys.sort((a,b)=>a.frame-b.frame);
        this.selectedKey={id:entity.id,frame:key.frame};
    }
    editChannels(values,remember=true) {
        const entity=this.entity();if(!entity)return false;
        if(!this.autoKey&&entity.keys.length&&!entity.keys.some(k=>k.frame===this.frame)){
            this.status.textContent="Add a key at this frame, or enable Auto key before editing.";this.applyState();return false;
        }
        if(remember)this.remember();
        const state={...sampleEntity(entity,this.frame,this.data.interpolation),...values};
        if(entity.keys.length||this.frame>0||this.autoKey)this.writeKey(entity,state);
        else for(const k of [...channels,...(entity.id==="camera"?["aim"]:[])])if(k in values)entity[k]=clone(values[k]);
        return true;
    }
    gizmoChange() {
        if(this.applying)return;
        const object=this.gizmo.object;if(!object)return;
        let values={position:object.position.toArray(),rotation:[object.rotation.x*DEG,object.rotation.y*DEG,object.rotation.z*DEG]};
        const preceding=sampleEntity(this.entity(),this.frame,this.data.interpolation);
        values.rotation=values.rotation.map((v,i)=>v+360*Math.round((preceding.rotation[i]-v)/360));
        if(this.selected==="camera"){
            const c=this.data.camera;
            if(this.gizmo.mode==="rotate") values.aim="free";
            else delete values.rotation;
            if(c.mode==="orbit"&&this.gizmo.mode==="translate"){
                const state=evaluate(this.data,this.frame),v=sub(values.position,state.anchor),r=Math.hypot(...v);
                let az=Math.atan2(v[0],v[2])*DEG;az+=360*Math.round((state.camera.azimuth-az)/360);
                values={azimuth:az,elevation:Math.asin(v[1]/Math.max(r,1e-8))*DEG,radius:Math.max(.01,r)};
            }
        } else values.scale=object.scale.toArray();
        if(this.editChannels(values,false)){
            // Apply the same finite/range constraints as saved scenes and the CPU
            // renderer, without rebuilding the gizmo during an active drag.
            this.data=normalizeScene(this.data);this.save();this.applyState();
        }
    }
    refresh() {
        if(!this.entity())this.selected="camera";
        if(this.selectedKey&&(this.selectedKey.id!==this.selected||!this.entity()?.keys.some(k=>k.frame===this.selectedKey.frame)))this.selectedKey=null;
        this.renderList();this.renderInspector();this.renderConfig();this.syncMeshes();this.updatePath();this.applyState();this.resize();this.drawTimeline();
    }
    keyAtPlayhead() {
        // The selected object's key at the cursor is authoritative, including
        // scrubbing, frame-number entry and linked prompter playback/seek.
        return this.entity()?.keys.find(k=>k.frame===this.frame);
    }
    updateKeyInterpolation() {
        if(!this.keyInterpolation)return;
        const key=this.keyAtPlayhead();
        this.keyInterpolation.disabled=!key||key.frame===0;
        this.keyInterpolation.value=key?.interpolation??this.data.interpolation;
        this.keyInterpolation.title=!key?"Move the playhead onto a key of the selected object to view or edit its interpolation.":key.frame===0?"The frame-zero key has no preceding interval. Select a later key.":"Applies from the previous key (or initial pose) to the key at the playhead. Hold jumps at this key.";
        this.keyInterpolation.style.borderColor=KEY_COLORS[this.keyInterpolation.value];
    }
    renderList() {
        this.list.replaceChildren();el("div",{class:"section-title",text:"SCENE"},this.list);
        for(const e of [this.data.camera,...this.data.subjects]){
            const b=this.button(this.list,e.name,`Select ${e.name}`,()=>{this.selected=e.id;this.refresh();});
            if(e.id===this.selected)b.classList.add("active");if(e.color)b.style.borderLeft=`4px solid ${e.color}`;
        }
    }
    renderInspector() {
        this.targetHeightInputs=null;
        this.channelInputs=[];
        this.aimSelect=null;this.rotationInputs=[];
        this.inspector.replaceChildren();const entity=this.entity();if(!entity)return;
        const current=sampleEntity(entity,this.frame,this.data.interpolation);
        const label=(name)=>{const l=el("label",{},this.inspector);el("span",{text:name},l);return l;};
        const name=el("input",{value:entity.name,title:"Stable subject name used in camera prose"},label("Name"));name.onchange=()=>{this.remember();entity.name=name.value;this.commit();};
        if(entity.id!=="camera"){
            this.select(label("Shape"),[["human","Human"],["box","Box"],["sphere","Sphere"]],entity.shape,"Proxy shape. Preserves transforms, color and keys; applies to the whole timeline.",v=>{this.remember();entity.shape=v;this.commit();});
            const color=el("input",{type:"color",value:entity.color,title:"Proxy render color"},label("Color"));color.onchange=()=>{this.remember();entity.color=color.value;this.commit();};
        }else{
            this.select(label("Path"),["free","orbit"],entity.mode,"Free XYZ translation or unwrapped target-relative orbit",v=>this.changeMode(v));
            this.aimSelect=this.select(label("Aim"),[["target","Track target"],["free","Free rotation"]],current.aim,"Keyframed aim. Applies from this key onward; position/rotation use incoming interpolation.",v=>this.changeAim(v));
            this.select(label("Target"),[["","Scene origin"],...this.data.subjects.map(s=>[s.id,s.name])],entity.target,"Primary framing target",v=>{this.remember();entity.target=v;this.commit();});
            const row=label("Target height");
            row.style.display="grid";row.style.gridTemplateColumns="minmax(0,1fr) 64px";row.firstChild.style.gridColumn="1 / -1";
            const slider=el("input",{type:"range",min:0,max:1,step:.01,value:current.target_height,title:"Target height: 0 = base, 1 = top. Keyframable; follows the subject's scale and rotation."},row);
            slider.style.width="100%";slider.style.minWidth="50px";
            const height=el("input",{type:"number",min:0,max:1,step:.01,value:Number(current.target_height.toFixed(3)),title:"Normalized target height (0–1). Uses the current key's incoming interpolation."},row);
            height.style.width="64px";
            this.targetHeightInputs={slider,height};
            slider.disabled=height.disabled=!this.data.subjects.some(s=>s.id===entity.target);
            let editing=false;
            const setHeight=value=>{
                const h=Math.max(0,Math.min(1,Number(value)||0));
                if(!this.editChannels({target_height:h},!editing)){
                    slider.value=String(sampleEntity(entity,this.frame,this.data.interpolation).target_height);height.value=slider.value;return false;
                }
                editing=true;this.playing=false;this.playButton.textContent="Play";
                slider.value=String(h);height.value=String(h);
                this.applyState();this.drawTimeline();this.publishPlayhead();return true;
            };
            const finish=()=>{if(editing){editing=false;this.commit();}};
            slider.oninput=()=>setHeight(slider.value);slider.onchange=finish;slider.onblur=finish;
            height.onchange=()=>{if(setHeight(height.value))finish();};
        }
        for(const key of ["position","rotation",...(entity.id==="camera"?[]:["scale"])]){
            if(entity.id==="camera"&&entity.mode==="orbit"&&key==="position")continue;
            el("div",{class:"section-title",text:`${key.toUpperCase()} · X / Y / Z`},this.inspector);
            const row=el("div",{class:"vector"},this.inspector);
            current[key].forEach((v,i)=>{
                const inp=el("input",{type:"number",step:key==="rotation"?1:.05,value:Number(v.toFixed(3)),title:`${key} ${"XYZ"[i]}${key==="rotation"?" (degrees, YXZ order)":""}`},row);
                inp.disabled=entity.id==="camera"&&current.aim==="target"&&key==="rotation"&&i<2;
                if(entity.id==="camera"&&key==="rotation")this.rotationInputs.push(inp);
                this.channelInputs.push({input:inp,key,index:i});
                if(key==="scale"){inp.min=.01;inp.title+=" (minimum 0.01; negative scale is not supported)";}
                inp.onfocus=()=>{this.playing=false;this.playButton.textContent="Play";};
                inp.onchange=()=>{const vec=[...sampleEntity(this.entity(),this.frame,this.data.interpolation)[key]];vec[i]=Number(inp.value);if(this.editChannels({[key]:vec}))this.commit();};
            });
        }
        // Orbit channels remain in scene/key data and are edited through the viewport.
        if(entity.id==="camera")for(const key of ["fov"]){
            const n=el("input",{type:"number",step:1,value:Number(current[key].toFixed(3)),title:key},label(key));
            this.channelInputs.push({input:n,key});
            n.onfocus=()=>{this.playing=false;this.playButton.textContent="Play";};
            n.onchange=()=>{if(this.editChannels({[key]:Number(n.value)}))this.commit();};
        }
    }
    changeAim(aim) {
        const entity=this.entity();if(entity?.id!=="camera")return;
        const current=sampleEntity(entity,this.frame,this.data.interpolation);
        if(current.aim===aim)return;
        // Release tracking from the visible bearing. When enabling tracking,
        // key its destination bearing so incoming free rotation can reach it.
        const scene=aim==="target"?{...this.data,camera:{...entity,aim:"target",keys:entity.keys.map(k=>({...k,aim:"target"}))}}:this.data;
        const state=evaluate(scene,this.frame),m=new THREE.Matrix4();
        m.set(...state.basis[0],0,...state.basis[1],0,...state.basis[2],0,0,0,0,1);
        const e=new THREE.Euler().setFromRotationMatrix(m,"YXZ");
        const rotation=[e.x*DEG,e.y*DEG,e.z*DEG].map((v,i)=>v+360*Math.round((current.rotation[i]-v)/360));
        if(this.editChannels({aim,rotation}))this.commit();
    }
    changeMode(mode) {
        const c=this.data.camera;if(c.mode===mode)return;this.remember();
        const frames=[0,...c.keys.map(k=>k.frame)];const states=frames.map(f=>evaluate(this.data,f));
        let previous=0;
        const converted=states.map((s,i)=>{
            const snap=sampleEntity(c,frames[i],this.data.interpolation);
            if(mode==="free") snap.position=s.position;
            else {const v=sub(s.position,s.anchor),r=Math.hypot(...v);let az=Math.atan2(v[0],v[2])*DEG;if(i)az+=360*Math.round((previous-az)/360);previous=az;Object.assign(snap,{radius:Math.max(.01,r),azimuth:az,elevation:Math.asin(v[1]/Math.max(r,1e-8))*DEG});}
            return snap;
        });
        for(const k of channels)if(k in converted[0])c[k]=clone(converted[0][k]);
        c.keys=c.keys.map((k,i)=>({...k,...Object.fromEntries(channels.filter(k=>k in converted[i+1]).map(k=>[k,clone(converted[i+1][k])]))}));
        c.mode=mode;this.commit();this.status.textContent="Path mode changed: keyed positions retained; the path between keys may change. Check the preview.";
    }
    syncMeshes() {
        if(!this.scene)return;
        const ids=new Set(this.data.subjects.map(s=>s.id));
        for(const [id,g] of this.meshes)if(!ids.has(id)){this.disposeObject(g);this.scene.remove(g);this.meshes.delete(id);}
        for(const s of this.data.subjects){
            let group=this.meshes.get(s.id);
            if(group&&group.userData.shape!==s.shape){
                if(this.gizmo.object===group)this.gizmo.detach();
                this.disposeObject(group);this.scene.remove(group);this.meshes.delete(s.id);group=null;
            }
            if(!group){
                group=new THREE.Group();group.userData.entityId=s.id;group.userData.shape=s.shape;group.rotation.order="YXZ";
                const geometry=subjectGeometry(s);
                for(const box of geometry.boxes){
                    const mesh=new THREE.Mesh(new THREE.BoxGeometry(...box.max.map((v,i)=>v-box.min[i])),subjectMaterial(s.color));
                    mesh.position.fromArray(box.max.map((v,i)=>(v+box.min[i])/2));mesh.userData.subjectSurface=true;group.add(mesh);
                }
                for(const sphere of geometry.spheres){
                    const mesh=new THREE.Mesh(new THREE.SphereGeometry(sphere.radius,64,32),subjectMaterial(s.color));mesh.position.fromArray(sphere.center);mesh.userData.subjectSurface=true;group.add(mesh);
                }
                if(s.shape==="human")addHumanFaceMarker(group);
                const front=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(0,.02,0),.7,0xfbd87d,.15,.09);front.traverse(o=>o.layers.set(1));group.add(front);
                this.scene.add(group);this.meshes.set(s.id,group);
            }
            for(const mesh of group.children)if(mesh.userData.subjectSurface)mesh.material.uniforms.baseColor.value.set(s.color);
        }
        this.rig.rotation.order="YXZ";
        if(this.selected==="camera"&&this.gizmo.mode==="scale")this.gizmo.setMode("translate");
        this.updateTransformButtons();
        this.gizmo.attach(this.selected==="camera"?this.rig:this.meshes.get(this.selected));
    }
    updatePath() {
        if(!this.scene)return;
        if(this.pathLine){this.scene.remove(this.pathLine);this.disposeObject(this.pathLine);}
        const points=[],steps=Math.min(240,this.data.frames-1);
        const times=new Set([0,...[this.data.camera,...this.data.subjects].flatMap(e=>e.keys.map(k=>k.frame))]);
        for(let i=0;i<=steps;i++)times.add(i/steps*(this.data.frames-1));
        const sorted=[...times].sort((a,b)=>a-b);
        for(let i=1;i<sorted.length;i++)points.push(new THREE.Vector3(...evaluate(this.data,sorted[i-1]).position),new THREE.Vector3(...evaluate(this.data,sorted[i],true).position));
        // Never draw a travel line through an instantaneous Hold cut.
        this.pathLine=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xd3a957}));
        this.pathLine.layers.set(1);this.scene.add(this.pathLine);
    }
    applyState() {
        if(this.floorGrid)this.floorGrid.visible=this.data.show_grid;
        if(this.backgroundGrid)this.backgroundGrid.visible=this.data.show_background_grid;
        const state=evaluate(this.data,this.frame);this.frameInput.value=this.frame;this.frameInput.max=this.data.frames-1;
        const selectedState=this.selected==="camera"?state.camera:state.subjects.find(s=>s.id===this.selected);
        for(const {input,key,index} of this.channelInputs??[]){
            if(selectedState&&document.activeElement!==input){const v=index==null?selectedState[key]:selectedState[key][index];input.value=String(Number(v.toFixed(3)));}
        }
        if(this.aimSelect){this.aimSelect.value=state.camera.aim;this.rotationInputs.forEach((input,i)=>input.disabled=state.camera.aim==="target"&&i<2);}
        if(this.targetHeightInputs){
            const {slider,height}=this.targetHeightInputs;
            if(document.activeElement!==slider)slider.value=String(state.camera.target_height);
            if(document.activeElement!==height)height.value=String(Number(state.camera.target_height.toFixed(3)));
        }
        this.timeLabel.textContent=`${(this.frame/this.data.fps).toFixed(3)}s / frame ${this.data.frames-1}`;
        const [w,h]=resolution(this.data);this.viewLabel.textContent=`Camera view · ${framing(state,w/h)} · XYZ ${state.position.map(v=>v.toFixed(2)).join(', ')}`;
        if(!this.scene)return;this.applying=true;
        for(const s of state.subjects){const g=this.meshes.get(s.id);if(!g)continue;g.position.fromArray(s.position);g.rotation.set(...s.rotation.map(v=>v/DEG),"YXZ");g.scale.fromArray(s.scale);}
        this.camera.position.fromArray(state.position);
        const m=new THREE.Matrix4();m.set(...state.basis[0],0,...state.basis[1],0,...state.basis[2],0,0,0,0,1);
        this.camera.quaternion.setFromRotationMatrix(m);this.camera.fov=state.camera.fov;this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.camera.updateMatrixWorld(true);
        this.rig.position.copy(this.camera.position);this.rig.quaternion.copy(this.camera.quaternion);this.applying=false;
        this.targetLine.visible=state.subjects.some(s=>s.id===state.camera.target);
        const linePositions=this.targetLine.geometry.attributes.position;
        linePositions.setXYZ(0,...state.position);linePositions.setXYZ(1,...state.anchor);linePositions.needsUpdate=true;
    }
    publishPlayhead(origin=this) {
        window.dispatchEvent(new CustomEvent("h3-camera-playhead",{detail:{node:this.node,frame:this.frame,fps:this.data.fps,origin}}));
    }
    seek(frame,origin=this) {
        this.playing=false;this.playButton.textContent="Play";
        this.selectedKey=null;
        this.frame=Math.max(0,Math.min(this.data.frames-1,Math.round(Number(frame)||0)));
        this.playTime=this.frame;this.applyState();this.drawTimeline();this.renderInspector();
        this.publishPlayhead(origin);
    }
    drawTimeline() {
        this.updateKeyInterpolation();
        const c=this.timeline,ctx=c.getContext("2d"),rows=[this.data.camera,...this.data.subjects];
        const h=TIMELINE_HEIGHT,lane=TIMELINE_LANE,maxScroll=Math.max(0,25+rows.length*lane-h);
        // A sticky viewport canvas keeps the ruler/playhead visible; the spacer
        // provides a native scrollbar without allocating a canvas for every row.
        this.timelineSpacer.style.height=`${maxScroll}px`;
        if(this.timelineScroll.scrollTop>maxScroll)this.timelineScroll.scrollTop=maxScroll;
        const scroll=this.timelineScroll.scrollTop,w=Math.max(300,c.clientWidth||800);
        c.style.height=`${h}px`;c.width=w;c.height=h;
        const x=f=>125+f/(this.data.frames-1)*(w-140);
        ctx.fillStyle="#10161e";ctx.fillRect(0,0,w,h);ctx.font="11px Arial";
        for(let i=0;i<=5;i++){const f=(this.data.frames-1)*i/5;ctx.fillStyle="#9baabd";ctx.fillText(`${(f/this.data.fps).toFixed(2)}s`,x(f)-12,12);ctx.strokeStyle="#263443";ctx.beginPath();ctx.moveTo(x(f),18);ctx.lineTo(x(f),h);ctx.stroke();}
        ctx.save();ctx.beginPath();ctx.rect(0,25,w,h-25);ctx.clip();
        rows.forEach((e,i)=>{
            const y=25+lane*(i+.5)-scroll;
            if(y+lane/2<=25||y-lane/2>=h)return;
            ctx.fillStyle=e.id===this.selected?"#8dccff":"#aeb9c6";ctx.fillText(e.name.slice(0,18),6,y+4);
            for(const k of e.keys){const selected=this.selectedKey?.id===e.id&&this.selectedKey.frame===k.frame;
                const radius=selected?7:5;ctx.fillStyle=KEY_COLORS[k.interpolation??this.data.interpolation];ctx.beginPath();ctx.moveTo(x(k.frame),y-radius);ctx.lineTo(x(k.frame)+radius,y);ctx.lineTo(x(k.frame),y+radius);ctx.lineTo(x(k.frame)-radius,y);ctx.closePath();ctx.fill();
                if(selected){ctx.strokeStyle="#ffffff";ctx.lineWidth=2;ctx.stroke();ctx.lineWidth=1;}
            }
        });
        ctx.restore();
        ctx.strokeStyle="#ef6666";ctx.beginPath();ctx.moveTo(x(this.frame),16);ctx.lineTo(x(this.frame),h);ctx.stroke();
        ctx.fillStyle="#ef6666";ctx.beginPath();ctx.moveTo(x(this.frame)-8,16);ctx.lineTo(x(this.frame)+8,16);ctx.lineTo(x(this.frame),29);ctx.closePath();ctx.fill();
        this.timelineGeometry={w,h,lane,rows,scroll};
    }
    timelineDown(e) {
        this.playing=false;this.playButton.textContent="Play";
        const c=this.timeline,{w,h,lane,rows}=this.timelineGeometry;
        const coords=event=>{const b=c.getBoundingClientRect();return [(event.clientX-b.left)*w/b.width,(event.clientY-b.top)*h/b.height];};
        const [x,y]=coords(e),row=y>=25&&y<h?rows[Math.floor((y-25+this.timelineScroll.scrollTop)/lane)]:null;
        const toFrame=x=>Math.max(0,Math.min(this.data.frames-1,Math.round((x-125)/(w-140)*(this.data.frames-1))));
        const playheadX=125+this.frame/(this.data.frames-1)*(w-140);
        const handle=y>=13&&y<=31&&Math.abs(x-playheadX)<=12;
        let key=!handle&&row?.keys.filter(k=>Math.abs(125+k.frame/(this.data.frames-1)*(w-140)-x)<9)
            .sort((a,b)=>Math.abs(a.frame-toFrame(x))-Math.abs(b.frame-toFrame(x)))[0];
        if(x<120&&row&&!handle){this.selected=row.id;this.selectedKey=null;this.refresh();return;}
        const originalFrame=key?.frame,originalRedo=this.redo;let dragged=false;
        if(key){this.selected=row.id;this.selectedKey={id:row.id,frame:key.frame};this.frame=key.frame;this.playTime=key.frame;this.refresh();this.publishPlayhead();}
        else this.seek(handle?this.frame:toFrame(x));
        c.setPointerCapture(e.pointerId);
        const move=event=>{const [mx]=coords(event);
            if(key){if(!dragged&&Math.abs(mx-x)<3)return;
                const f=Math.max(0,Math.min(this.data.frames-1,originalFrame+Math.round((mx-x)/(w-140)*(this.data.frames-1))));
                // Never expose duplicate-time keys to interpolation or destroy
                // the existing key on drop. Crossing an occupied frame is fine.
                if(row.keys.some(k=>k!==key&&k.frame===f)){this.status.textContent=`Frame ${f} already has a key. Drop on an empty frame.`;return;}
                if(f===key.frame)return;
                if(!dragged){this.remember();dragged=true;}
                key.frame=f;this.selectedKey={id:row.id,frame:f};row.keys.sort((a,b)=>a.frame-b.frame);this.frame=f;this.playTime=f;this.applyState();this.drawTimeline();this.publishPlayhead();
            }else this.seek(toFrame(mx));};
        const up=event=>{
            c.removeEventListener("pointermove",move);c.removeEventListener("pointerup",up);c.removeEventListener("pointercancel",up);
            if(c.hasPointerCapture(event.pointerId))c.releasePointerCapture(event.pointerId);
            if(key&&dragged){
                if(event.type==="pointercancel"){key.frame=originalFrame;this.selectedKey={id:row.id,frame:originalFrame};this.frame=originalFrame;this.playTime=originalFrame;this.undo.pop();this.redo=originalRedo;row.keys.sort((a,b)=>a.frame-b.frame);this.refresh();this.publishPlayhead();}
                else{row.keys.sort((a,b)=>a.frame-b.frame);this.commit();}
            }
            else this.renderInspector();
        };
        c.addEventListener("pointermove",move);c.addEventListener("pointerup",up);c.addEventListener("pointercancel",up);
    }
    tick(now) {
        if(this.disposed)return;
        if(this.playing){const previous=this.frame;this.playTime=(this.playTime??this.frame)+(now-this.lastTime)/1000*this.data.fps;if(this.playTime>this.data.frames-1)this.playTime=0;this.frame=Math.floor(this.playTime);this.applyState();this.drawTimeline();if(previous!==this.frame)this.publishPlayhead();}
        else this.playTime=this.frame;
        this.lastTime=now;
        if(this.renderer&&this.root.isConnected){this.controls.update();this.renderer.render(this.scene,this.observer);if(this.cameraViewVisible)this.outputRenderer.render(this.scene,this.camera);}
        this.raf=requestAnimationFrame(this.tick);
    }
    executed(message) {this.status.textContent=message?.camera_info?.[0]??"Render complete.";}
    disposeObject(object) {object.traverse(o=>{o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material?.dispose();});}
    dispose3D() {
        // Initialization can fail between the two contexts or midway through
        // scene setup. Release partial resources and leave the data editor usable.
        for(const cleanup of [
            ()=>this.resizeObserver?.disconnect(),
            ()=>this.canvas?.removeEventListener("pointerdown",this.canvasPointerDown),
            ()=>this.canvas?.removeEventListener("pointerup",this.canvasPointerUp),
            ()=>this.controls?.dispose(),()=>this.gizmo?.dispose(),
            ()=>{if(this.scene)this.disposeObject(this.scene);},
            ...[this.renderer,this.outputRenderer].flatMap(r=>[()=>r?.dispose(),()=>r?.forceContextLoss?.()]),
        ]){try{cleanup();}catch(error){console.warn("H3 Camera preview cleanup:",error);}}
        for(const key of ["renderer","outputRenderer","scene","observer","camera","controls","gizmo","rig","floorGrid","backgroundGrid","targetLine","pathLine","resizeObserver","canvasPointerDown","canvasPointerUp"])this[key]=null;
        this.meshes?.clear();this.dragging=false;this.applying=false;
    }
    dispose() {this.disposed=true;if(this.timelineScroll)this.timelineScroll.onscroll=null;for(const type of ["pointerdown","keydown","focusin"])this.root.removeEventListener(type,this.nativeSelectScaleHandler,true);cancelAnimationFrame(this.raf);this.dispose3D();}
}

app.registerExtension({
    name:"toyxyz.MinimaxH3Camera",
    async beforeRegisterNodeDef(nodeType,nodeData) {
        if(nodeData.name!=="MinimaxH3Camera")return;
        const created=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(){
            created?.apply(this,arguments);
            const widget=this.widgets?.find(w=>w.name==="scene_data");if(!widget)return;
            widget.hidden=true;widget.options??={};widget.options.hidden=true;
            if(!window.LiteGraph?.vueNodesMode){widget.computeSize=()=>[0,-4];widget.draw=()=>{};}
            if(widget.element)widget.element.style.display="none";
            if(widget.inputEl)widget.inputEl.style.display="none";
            const root=document.createElement("div");
            const dom=this.addDOMWidget("h3_scene_editor","h3_scene_editor",root,{getValue:()=>"",setValue:()=>{},getMinHeight:()=>650,getMaxHeight:()=>Math.max(650,(this.size?.[1]??740)-85)});dom.serialize=false;
            this.h3SceneEditor=new H3SceneEditor(this,root,widget);this.setSize([Math.max(this.size?.[0]??0,1050),Math.max(this.size?.[1]??0,750)]);
        };
        const configured=nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure=function(){configured?.apply(this,arguments);this.h3SceneEditor?.load(this.widgets?.find(w=>w.name==="scene_data")?.value);
            // Reorder by identity, preserving existing output objects and their links.
            const ports=[["prompter_camera","MINIMAX_H3_CAMERA"],["camera_render","IMAGE"],["camera_prompt","STRING"]];
            for(let i=(this.outputs?.length||0)-1;i>=0;i--)if(!ports.some(([name])=>name===this.outputs[i].name))this.removeOutput(i);
            for(const [name,type] of ports)if(!this.outputs?.some(o=>o.name===name))this.addOutput(name,type);
            this.outputs=ports.map(([name,type])=>Object.assign(this.outputs.find(o=>o.name===name),{type}));
            // Legacy five-port nodes placed prose before the render. Preserve
            // those output objects and update link indices to the current order.
            this._h3SyncCameraOutputLinks=()=>this.outputs.forEach((output,index)=>{for(const id of output.links||[]){
                const link=this.graph?.links?.get?.(id)||this.graph?.links?.[id];
                if(link)link.origin_slot=index;
            }});
            this._h3SyncCameraOutputLinks();
            this._widgetSlotsDirty=true;
            this.setDirtyCanvas?.(true,true);
        };
        const graphConfigured=nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured=function(){
            graphConfigured?.apply(this,arguments);
            this._h3SyncCameraOutputLinks?.();
        };
        const executed=nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted=function(message){executed?.apply(this,arguments);this.h3SceneEditor?.executed(message);};
        const removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved=function(){this.h3SceneEditor?.dispose();removed?.apply(this,arguments);};
    },
});
