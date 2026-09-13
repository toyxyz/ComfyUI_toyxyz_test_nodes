// Keep this coordinate/interpolation contract aligned with nodes/minimax_h3_camera.py.
export const ratios = {"1:1": 1, "2:3": 2/3, "3:2": 1.5, "3:4": .75, "4:3": 4/3, "9:16": 9/16, "16:9": 16/9, "21:9": 21/9};
export const channels = ["position", "rotation", "scale", "fov", "azimuth", "elevation", "radius", "target_height"];
export const subjectHeight = 1.8;
// Front is local +Z. Render-only helpers, not anatomical/framing bounds.
export const humanFaceBoxes = [
    {min:[-.12,1.66,.175],max:[.12,1.705,.22]},
    {min:[-.025,1.49,.175],max:[.025,1.68,.22]},
];
// Matching proxy dimensions in nodes/minimax_h3_camera.py: torso and two arms.
export const subjectBoxes = [
    {min:[-.25,0,-.15],max:[.25,1.4,.15]},
    {min:[-.435,.60,-.11],max:[-.295,1.35,.11]},
    {min:[.295,.60,-.11],max:[.435,1.35,.11]},
];
export function subjectGeometry(subject) {
    if(subject.shape==="box")return {height:1,boxes:[{min:[-.5,0,-.5],max:[.5,1,.5]}],spheres:[]};
    if(subject.shape==="sphere")return {height:1,boxes:[],spheres:[{center:[0,.5,0],radius:.5}]};
    return {height:subjectHeight,boxes:subjectBoxes,spheres:[{center:[0,1.6,0],radius:.2}]};
}
export function subjectBounds(subject) {
    const {boxes,spheres}=subjectGeometry(subject);
    return [...boxes,...spheres.map(s=>({min:s.center.map(v=>v-s.radius),max:s.center.map(v=>v+s.radius)}))]
        .flatMap(b=>[b.min[0],b.max[0]].flatMap(x=>[b.min[1],b.max[1]].flatMap(y=>[b.min[2],b.max[2]].map(z=>[x,y,z]))));
}
export const defaultCamera = {id: "camera", name: "Camera", mode: "free", aim: "target", target: "subject_1", target_height: 1/subjectHeight, position: [0, 1.3, 4], rotation: [0, 0, 0], fov: 40, azimuth: 0, elevation: 0, radius: 4, keys: []};
export const defaultSubject = {id: "subject_1", name: "Subject 1", shape:"human", color: "#38a6c9", position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1], keys: []};
export const defaultScene = {version: 3, requested_duration: 5, frames: 124, fps: 24, aspect_ratio: "16:9", megapixels: .1, use_camera_prompt: true, refvid: true, show_grid: true, show_background_grid: true, interpolation: "smooth", camera: defaultCamera, subjects: [defaultSubject]};
export const clone = v => JSON.parse(JSON.stringify(v));
const num = (v, fallback, lo=-1e6, hi=1e6) => v == null || !Number.isFinite(Number(v)) ? fallback : Math.max(lo, Math.min(hi, Number(v)));
// Python round() uses ties-to-even; match the prompter backend at half frames too.
export function roundFrame(v) {const i=Math.floor(v);return v-i===.5 ? i+i%2 : Math.round(v);}
export function alignedFrameCount(seconds) {
    const frames=Math.max(5,roundFrame(seconds*24));return frames+((5-frames%17+17)%17);
}
export function normalizeScene(raw) {
    if (typeof raw === "string") { try {raw = JSON.parse(raw);} catch {raw = {};} }
    raw = raw && typeof raw === "object" ? raw : {};
    const out = clone(defaultScene);
    const outgoingKeys = num(raw.version, 0) < 3;
    const legacy=!("requested_duration" in raw)&&("frames" in raw),oldFPS=num(raw.fps,24,1,120);
    const seconds=legacy?num(raw.frames,121,2,1441)/oldFPS:5;
    out.requested_duration=Math.max(.1,Math.min(60,num(raw.requested_duration,seconds,.1,60)));
    out.frames=alignedFrameCount(out.requested_duration);out.fps=24;
    out.megapixels = num(raw.megapixels, .1, .01, 4);
    out.use_camera_prompt = raw.use_camera_prompt === undefined || raw.use_camera_prompt === true;
    out.refvid = raw.refvid === undefined || raw.refvid === true;
    out.show_grid = raw.show_grid === undefined || raw.show_grid === true;
    out.show_background_grid = raw.show_background_grid === undefined || raw.show_background_grid === true;
    out.aspect_ratio = Object.hasOwn(ratios, raw.aspect_ratio) ? raw.aspect_ratio : defaultScene.aspect_ratio;
    out.interpolation = ["linear", "smooth", "hold"].includes(raw.interpolation) ? raw.interpolation : "smooth";
    function entity(raw, base, id) {
        raw = raw && typeof raw === "object" ? raw : {};
        const e = clone(base); e.id = id; e.name = String(raw.name ?? base.name).slice(0, 100);
        for (const k of channels) {
            if (!(k in base)) continue;
            e[k] = Array.isArray(base[k]) ? base[k].map((v, i) => num(raw[k]?.[i], v, k === "scale" ? .01 : -1e4, 1e4))
                : num(raw[k], base[k], k === "radius" ? .01 : k === "fov" ? 1 : -1e5, k === "fov" ? 175 : 1e5);
        }
        if (id === "camera") {
            for (const [k, values] of Object.entries({mode:["free","orbit"], aim:["free","target"]})) e[k] = values.includes(raw[k]) ? raw[k] : base[k];
            const legacyHeight=({feet:0,body:1,head:1.6}[raw.target_part]??1)/subjectHeight;
            e.target_height=num(raw.target_height,legacyHeight,0,1);
            e.target = String(raw.target ?? base.target);
        } else {
            e.color = /^#[\da-f]{6}$/i.test(raw.color) ? raw.color : base.color;
            e.shape=["human","box","sphere"].includes(raw.shape)?raw.shape:"human";
        }
        const keys = new Map();
        for (const k of Array.isArray(raw.keys) ? raw.keys : []) {
            if (!k || typeof k !== "object") continue;
            const sourceFrame=num(k.frame,0,0,1e6);
            const frame=Math.max(0,Math.min(out.frames-1,legacy?roundFrame(sourceFrame*24/oldFPS):Math.trunc(sourceFrame)));
            const snapshot = entity({...e, ...k, keys: []}, base, id);
            const interpolation = ["linear", "smooth", "hold"].includes(k.interpolation) ? k.interpolation : out.interpolation;
            keys.set(frame, {frame, interpolation, ...Object.fromEntries(channels.filter(k => k in e).map(k => [k, snapshot[k]]))});
            if(id==="camera"&&["target","free"].includes(k.aim))keys.get(frame).aim=k.aim;
        }
        e.keys = [...keys.values()].sort((a,b) => a.frame-b.frame);
        // V1/V2 stored outgoing modes. V3 stores the segment ending at each key.
        // Shift only once; preserve the implicit frame-zero segment as well.
        if (outgoingKeys) {
            const modes=e.keys.map(k=>k.interpolation);
            e.keys.forEach((k,i)=>{k.interpolation=i>0?modes[i-1]:k.frame>0?out.interpolation:modes[i];});
        }
        return e;
    }
    const seen = new Set(["camera"]);
    out.subjects = (Array.isArray(raw.subjects) ? raw.subjects : [defaultSubject]).map((s,i) => {
        let id = String(s?.id ?? `subject_${i+1}`); while (seen.has(id)) id += "_copy"; seen.add(id);
        return entity(s, defaultSubject, id);
    });
    out.camera = entity(raw.camera, defaultCamera, "camera");
    if (out.camera.target && !out.subjects.some(s => s.id === out.camera.target)) out.camera.target = out.subjects[0]?.id ?? "";
    return out;
}
export function resolution(scene) {
    const r = ratios[scene.aspect_ratio], p = scene.megapixels*1e6;
    return [r, 1/r].map(v => Math.max(32, Math.floor(Math.sqrt(p*v)/32+.5)*32));
}
export function sampleEntity(entity, frame, interpolation="smooth", before=false) {
    let keys = entity.keys;
    const out = clone(entity); delete out.keys;
    if("aim" in entity)out.aim=keys.filter(k=>(k.frame<frame||(!before&&k.frame===frame)||frame===0&&k.frame===0)&&["target","free"].includes(k.aim)).at(-1)?.aim??entity.aim;
    if (!keys.length) return out;
    if (keys[0].frame > 0) keys = [{...out, frame:0}, ...keys];
    if (frame <= keys[0].frame || frame > keys.at(-1).frame || (frame===keys.at(-1).frame&&!before)) return {...out, ...clone(frame <= keys[0].frame ? keys[0] : keys.at(-1))};
    const i = keys.findIndex((k,j) => j < keys.length-1 && keys[j+1].frame >= frame);
    const a = keys[i], b = keys[i+1], h = b.frame-a.frame, t = (frame-a.frame)/h;
    const mode = b.interpolation ?? interpolation;
    for (const k of channels) {
        if (!(k in entity)) continue;
        const component = index => {
            const get = v => index == null ? v[k] : v[k][index];
            const x = get(a), y = get(b);
            if (mode === "hold") return t >= 1 && !before ? y : x;
            if (mode === "linear") return x+(y-x)*t;
            const slope = j => {
                if (j <= 0 || j >= keys.length-1) return 0;
                const l=keys[j-1], m=keys[j], r=keys[j+1];
                if ((m.interpolation ?? interpolation)==="hold" || (r.interpolation ?? interpolation)==="hold") return 0;
                const dl=(get(m)-get(l))/(m.frame-l.frame), dr=(get(r)-get(m))/(r.frame-m.frame);
                return dl*dr > 0 ? 2*dl*dr/(dl+dr) : 0;
            };
            return (2*t**3-3*t*t+1)*x+(t**3-2*t*t+t)*h*slope(i)+(-2*t**3+3*t*t)*y+(t**3-t*t)*h*slope(i+1);
        };
        out[k] = Array.isArray(entity[k]) ? entity[k].map((_,j) => component(j)) : component(null);
    }
    return out;
}
export const add = (a,b) => a.map((v,i) => v+b[i]);
export const sub = (a,b) => a.map((v,i) => v-b[i]);
export const mul = (a,s) => a.map(v => v*s);
export const dot = (a,b) => a.reduce((v,x,i) => v+x*b[i], 0);
export const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const unit = v => mul(v,1/Math.max(1e-12, Math.hypot(...v)));
export const matVec = (m,v) => m.map(row => dot(row,v));
export const transpose = m => m[0].map((_,i) => m.map(row => row[i]));
export function rotationMatrix(deg) {
    const [x,y,z] = deg.map(v => v*Math.PI/180), cx=Math.cos(x),cy=Math.cos(y),cz=Math.cos(z),sx=Math.sin(x),sy=Math.sin(y),sz=Math.sin(z);
    return [[cy*cz+sy*sx*sz,-cy*sz+sy*sx*cz,sy*cx],[cx*sz,cx*cz,-sx],[-sy*cz+cy*sx*sz,sy*sz+cy*sx*cz,cy*cx]];
}
export function evaluate(scene, frame, before=false) {
    const subjects = scene.subjects.map(s => sampleEntity(s,frame,scene.interpolation,before));
    const camera = sampleEntity(scene.camera,frame,scene.interpolation,before);
    const target = subjects.find(s => s.id === camera.target), height=(target?subjectGeometry(target).height:subjectHeight)*camera.target_height;
    const anchor = target ? add(target.position,matVec(rotationMatrix(target.rotation),[0,height*target.scale[1],0])) : [0,1,0];
    let position = [...camera.position];
    if (camera.mode === "orbit") {
        const a=camera.azimuth*Math.PI/180,e=camera.elevation*Math.PI/180;
        position=add(anchor,mul([Math.sin(a)*Math.cos(e),Math.sin(e),Math.cos(a)*Math.cos(e)],camera.radius));
    }
    let basis=rotationMatrix(camera.rotation);
    if (camera.aim === "target") {
        let f=unit(sub(anchor,position)); if (Math.hypot(...f)<.5) f=[0,0,-1];
        if (Math.abs(f[1])>.999999) f=unit(add(f,[0,0,-1e-5]));
        const right=unit(cross(f,[0,1,0])),up=unit(cross(right,f)),roll=camera.rotation[2]*Math.PI/180;
        basis=transpose([add(mul(right,Math.cos(roll)),mul(up,Math.sin(roll))),sub(mul(up,Math.cos(roll)),mul(right,Math.sin(roll))),mul(f,-1)]);
    }
    return {camera,subjects,position,basis,anchor,show_grid:scene.show_grid ?? true,show_background_grid:scene.show_background_grid ?? true};
}
// Match the renderer's exact left-limit cut detection, including tracked targets.
export function cameraCutFrames(scene) {
    const times=[...new Set([scene.camera,...scene.subjects].flatMap(e=>e.keys.map(k=>k.frame)).filter(f=>f>0))].sort((a,b)=>a-b);
    return times.filter(f=>{
        const a=evaluate(scene,f,true),b=evaluate(scene,f);
        const changed=[...a.position,...a.basis.flat(),a.camera.fov].some((v,i)=>
            Math.abs(v-[...b.position,...b.basis.flat(),b.camera.fov][i])>1e-8);
        if(!changed||a.camera.aim===b.camera.aim)return changed;
        const key=scene.camera.keys.find(k=>k.frame===f);
        if((key?.interpolation??scene.interpolation)==="hold")return true;
        // Changing Aim alone must not invent a Shot cut. Actual simultaneous
        // Hold jumps in position/target/rotation/FOV still retain their cuts.
        const opticalInputs=s=>[...s.position,...s.anchor,...s.camera.rotation,s.camera.fov];
        return opticalInputs(a).some((v,i)=>Math.abs(v-opticalInputs(b)[i])>1e-8);
    });
}
export function project(points,state,aspect=1) {
    const t=Math.tan(state.camera.fov*Math.PI/360),inverse=transpose(state.basis);
    return points.map(p=>{const q=matVec(inverse,sub(p,state.position)),d=-q[2];return [q[0]/Math.max(d,1e-6)/t/aspect,q[1]/Math.max(d,1e-6)/t,d];});
}
export function framing(state,aspect) {
    const s=state.subjects.find(s=>s.id===state.camera.target);
    if (!s) return "No primary target";
    const pts=[0,.5,.9,1.1,1.4,1.6,1.8].map(h=>add(s.position,matVec(rotationMatrix(s.rotation),[0,h*s.scale[1],0])));
    const q=project(pts,state,aspect),inside=q.map(p=>Math.abs(p[0])<=1&&Math.abs(p[1])<=1&&p[2]>.01);
    // Keep this scale/crop decision aligned with Python framing(). Use unclipped
    // bounds: frame-edge placement must not turn a small subject into a close-up.
    const bounds=project(subjectBounds(s).map(p=>add(s.position,matVec(rotationMatrix(s.rotation),p.map((v,i)=>v*s.scale[i])))),state,aspect);
    const xs=bounds.map(p=>p[0]),ys=bounds.map(p=>p[1]);
    const front=bounds.every(p=>p[2]>.01);
    const outside=bounds.every(p=>p[2]<=.01)||(front&&(Math.min(...xs)>1||Math.max(...xs)<-1||Math.min(...ys)>1||Math.max(...ys)<-1));
    const occupancy=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))/2;
    const edges=[[Math.min(...xs)<-1,"left"],[Math.max(...xs)>1,"right"],[Math.min(...ys)<-1,"bottom"],[Math.max(...ys)>1,"top"]].filter(([clipped])=>clipped).map(([,edge])=>edge);
    const size=occupancy<.2?"Extreme wide shot":occupancy<.55?"Wide shot":"Full shot";
    let label;
    if(outside)label="Cropped / offscreen target";
    else if(!front)label="Partial view / lens-plane intersection";
    else if(!edges.length||occupancy<=1)label=size;
    else if(s.shape&&s.shape!=="human")label="Close-up / partial object crop";
    else if (!inside.some(Boolean)) label="Cropped / offscreen target";
    else if (inside.every(Boolean)) {
        label=size;
    }
    else if (inside[0]&&!inside[4]) label="Feet / lower body crop";
    else if (inside[6]&&inside[1]) label="Medium full shot";
    else if (inside[6]&&inside[2]) label="Medium shot";
    else if (inside[5]&&!inside[2]&&!inside[0]) label=inside[3]?"Medium close-up":"Close-up";
    else label="Partial body view";
    if(front&&!outside&&edges.length)label+=` · ${edges.join("/")} edge-cropped`;
    return `${label} · height ${state.position[1].toFixed(2)} · distance ${Math.hypot(...sub(state.position,state.anchor)).toFixed(2)}`;
}
