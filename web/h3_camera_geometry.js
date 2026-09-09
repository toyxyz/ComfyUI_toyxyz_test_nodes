// Shared geometry for the scene view and the actual square camera image.
export const parts = [
  [0,2.78,0,.48,.52,.42], [0,2.28,0,.82,.48,.38], [0,1.83,0,.72,.38,.34],
  [0,1.43,0,.68,.34,.36], [-.53,2.25,0,.22,.62,.24], [.53,2.25,0,.22,.62,.24],
  [-.57,1.68,0,.2,.5,.22], [.57,1.68,0,.2,.5,.22],
  [-.23,.91,0,.26,.72,.3], [.23,.91,0,.26,.72,.3],
  [-.23,.32,0,.22,.46,.26], [.23,.32,0,.22,.46,.26],
  [-.23,.06,.09,.25,.12,.44], [.23,.06,.09,.25,.12,.44],
  [0,2.49,0,.2,.14,.22], [0,2.79,.235,.13,.10,.08], // nose: front is +Z
  [0,2.89,.235,.38,.085,.05], // horizontal front marker at eye level
  [-.57,1.35,0,.20,.16,.24],[.57,1.35,0,.20,.16,.24], // hands
];
const featureColors = {15:[239,189,98],16:[239,189,98]};
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const mul=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>mul(a,1/Math.max(1e-9,Math.hypot(...a)));
export function basis(pose) {
  const forward=unit(sub(pose.target,pose.position));
  const az=(pose.azimuth??Math.atan2(pose.position[0]-pose.target[0],pose.position[2]-pose.target[2])*180/Math.PI)*Math.PI/180;
  const right=[Math.cos(az),0,-Math.sin(az)];
  const up=cross(right,forward),r=(pose.roll||0)*Math.PI/180,c=Math.cos(r),s=Math.sin(r);
  return {forward,right:right.map((v,i)=>c*v-s*up[i]),up:up.map((v,i)=>s*right[i]+c*v)};
}
const ranges={extreme_close_up:[2.66,2.90],big_close_up:[2.56,3.00],close_up:[2.48,3.04],medium_close_up:[2.02,3.04],
  medium_shot:[1.63,3.04],cowboy_shot:[.91,3.04],medium_full_shot:[.55,3.04],
  full_shot:[0,3.04],wide_shot:[0,3.04],extreme_wide_shot:[0,3.04]};
export const compositions={center:[0,0],left:[-1/3,0],right:[1/3,0],top:[0,1/3],bottom:[0,-1/3],
  top_left:[-1/3,1/3],top_right:[1/3,1/3],bottom_left:[-1/3,-1/3],bottom_right:[1/3,-1/3]};
export const cameraLevels={ground:.12,knee:.65,hip:1.43,chest:2.28,eye:2.89,above_head:3.6,elevated:6,aerial:15};
export function subjectOffsets(config) {
  const n=({two:2,three:3,group:4})[config.subject_framing]||1;
  return Array.from({length:n},(_,i)=>[([0,1,-1,2][i])*1.6,0,0]);
}
export function framingConfig(config) {
  return {shot_size:Object.hasOwn(ranges,config.shot_size)?config.shot_size:'full_shot'};
}
export function framingPoints(config) {
  const {shot_size}=framingConfig(config);
  const [bottom,top]=ranges[shot_size];
  const points=[];
  parts.forEach(([x,y,z,w,h,d],index)=>{
    const lo=Math.max(bottom,y-h/2),hi=Math.min(top,y+h/2);
    if(lo>hi)return;
    for(const px of [x-w/2,x+w/2])for(const py of [lo,hi])for(const pz of [z-d/2,z+d/2])points.push([px,py,pz]);
  });
  return {points:subjectOffsets(config).flatMap(offset=>points.map(p=>add(p,offset))),anchor:[0,(bottom+top)/2,0],shot_size};
}
export function solveCamera(config,previous=null) {
  const turns={rotate_left_180:-180,rotate_right_180:180,rotate_left_360:-360,rotate_right_360:360};
  const azimuth=config.direction in turns ? (previous?.azimuth||0)+turns[config.direction] : ({front:0,front_left_45:-45,left_profile:-90,rear_left_45:-135,rear:180,
    rear_right_45:135,right_profile:90,front_right_45:45})[config.direction]||0;
  const elevation=({extreme_low:-60,low_angle:-18,eye_level:0,high_angle:30,extreme_high:70,overhead:89.5})[config.angle]||0;
  const roll=[-90,-45,-30,-15,0,15,30,45,90,180].includes(Number(config.roll))?Number(config.roll):0;
  const {points,anchor,shot_size}=framingPoints(config);
  if(['ots','oth','pov'].includes(config.viewpoint)) {
    const level=Object.hasOwn(cameraLevels,config.camera_level)?config.camera_level:config.viewpoint==='oth'?'hip':'eye';
    const pose=solveLevelCamera({...config,camera_level:level},azimuth,elevation,roll,points,anchor,shot_size,3.2);
    pose.viewpoint=config.viewpoint;
    pose.viewpointProxy=true;
    if(config.viewpoint!=='pov')pose.subjectOffsets.push([-.95,0,1.6,180]);
    return pose;
  }
  if(Object.hasOwn(cameraLevels,config.camera_level))return solveLevelCamera(config,azimuth,elevation,roll,points,anchor,shot_size);
  const [screenX,screenY]=compositions[config.composition]||compositions.center;
  const target=anchor, az=azimuth*Math.PI/180, el=elevation*Math.PI/180;
  const radial=[Math.sin(az)*Math.cos(el),Math.sin(el),Math.cos(az)*Math.cos(el)];
  const tangent=Math.tan(40*Math.PI/360);
  const evaluate=distance=>{
    const pose={position:add(anchor,mul(radial,distance)),target:anchor,tangent,roll};
    const b=basis(pose), projected=points.map(p=>{
      const v=sub(p,pose.position),z=dot(v,b.forward);
      return [dot(v,b.right)/(z*tangent),dot(v,b.up)/(z*tangent),z];
    });
    return {pose,minX:Math.min(...projected.map(p=>p[0])),maxX:Math.max(...projected.map(p=>p[0])),
      minY:Math.min(...projected.map(p=>p[1])),maxY:Math.max(...projected.map(p=>p[1])),near:Math.min(...projected.map(p=>p[2]))};
  };
  const occupancy=shot_size==='wide_shot'?.62:shot_size==='extreme_wide_shot'?.20:.90;
  let lo=.35,hi=subjectOffsets(config).length>1?128:30;
  for(let i=0;i<50;i++) {const mid=(lo+hi)/2,r=evaluate(mid);
    if(r.near<.1||r.maxX-r.minX>2*Math.min(occupancy,.95-Math.abs(screenX))||r.maxY-r.minY>2*Math.min(occupancy,.95-Math.abs(screenY))) lo=mid; else hi=mid;}
  // Match backend: do not approach merely because an overhead body foreshortens.
  if(shot_size==='wide_shot'||shot_size==='extreme_wide_shot')hi=Math.max(hi,3.04/(2*tangent*occupancy));
  const r=evaluate(hi);
  return {...r.pose,target,anchor,shiftX:(r.minX+r.maxX)/2-screenX,shiftY:(r.minY+r.maxY)/2-screenY,azimuth,elevation,distance:hi,
    subjectOffsets:subjectOffsets(config),orbit_route:config.direction in turns ? config.direction : config.orbit_route||'shortest'};
}
function solveLevelCamera(config,azimuth,requestedElevation,roll,points,anchor,shotSize,minRadius=.1) {
  // Shot size controls nominal distance, never a forced anatomical aim.
  const nominal=solveCamera({...config,viewpoint:'external',camera_level:'auto',angle:'eye_level',composition:'center'});
  const height=cameraLevels[config.camera_level],el=requestedElevation*Math.PI/180,az=azimuth*Math.PI/180;
  const distance=Math.max(nominal.distance,minRadius),radius=distance*Math.cos(el);
  const target=[0,height-distance*Math.sin(el),0],position=[Math.sin(az)*radius,height,Math.cos(az)*radius];
  const [sx,sy]=compositions[config.composition]||[0,0];
  return {position,target,anchor:target,azimuth,roll,tangent:nominal.tangent,effectiveLevel:config.camera_level,
    explicitLevel:true,distance,elevation:requestedElevation,requestedElevation,levelAdjusted:false,
    subjectOffsets:subjectOffsets(config),shiftX:-sx,shiftY:-sy,
    orbit_route:config.direction?.startsWith('rotate_')?config.direction:config.orbit_route||'shortest'};
}
export function orbitDelta(a,b,route='shortest') {
  const turns={rotate_left_180:-180,rotate_right_180:180,rotate_left_360:-360,rotate_right_360:360};
  if(route in turns)return turns[route];
  const positive=((b-a)%360+360)%360;
  if(positive<1e-8)return 0; // A repeated endpoint is a hold, not an implicit full turn.
  if(route==='right')return positive;
  if(route==='left')return positive-360;
  return positive>=180?positive-360:positive;
}
export function interpolateCamera(a,b,t) {
  t=Math.max(0,Math.min(1,t)); const s=t*t*t*(t*(t*6-15)+10);
  const mix=(x,y)=>x+(y-x)*s;
  const delta=orbitDelta(a.azimuth,b.azimuth,b.orbit_route);
  const azimuth=a.azimuth+delta*s;
  let elevation=mix(a.elevation,b.elevation),distance=mix(a.distance,b.distance);
  const target=a.target.map((v,i)=>mix(v,b.target[i]));
  const anchor=a.anchor.map((v,i)=>mix(v,b.anchor[i]));
  if(a.explicitLevel||b.explicitLevel){
    const height=mix(a.position[1],b.position[1]);
    target[1]=height-distance*Math.sin(elevation*Math.PI/180);anchor[1]=target[1];
  }
  const az=azimuth*Math.PI/180,el=elevation*Math.PI/180;
  return {subjectOffsets:takeOffsets([a,b]),target,anchor,position:add(anchor,mul([Math.sin(az)*Math.cos(el),Math.sin(el),Math.cos(az)*Math.cos(el)],distance)),
    tangent:mix(a.tangent,b.tangent),shiftX:mix(a.shiftX,b.shiftX),shiftY:mix(a.shiftY,b.shiftY),roll:mix(a.roll||0,b.roll||0),azimuth,elevation,distance};
}
function unwrapAngles(nodes) {
  const values=[nodes[0].pose.azimuth];
  for(let i=1;i<nodes.length;i++)values.push(values[i-1]+orbitDelta(values[i-1],nodes[i].pose.azimuth,nodes[i].pose.orbit_route));
  return values;
}
function hermite(values,times,index,t) {
  const dt=Math.max(1e-6,times[index+1]-times[index]);
  const slope=i=>{
    if(i<=0||i>=values.length-1)return 0;
    const h0=Math.max(1e-6,times[i]-times[i-1]),h1=Math.max(1e-6,times[i+1]-times[i]);
    const d0=(values[i]-values[i-1])/h0,d1=(values[i+1]-values[i])/h1;
    if(d0*d1<=0)return 0;
    const w0=2*h1+h0,w1=h1+2*h0;
    return (w0+w1)/(w0/d0+w1/d1);
  };
  const u=Math.max(0,Math.min(1,t)),u2=u*u,u3=u2*u;
  return (2*u3-3*u2+1)*values[index]+(u3-2*u2+u)*dt*slope(index)
    +(-2*u3+3*u2)*values[index+1]+(u3-u2)*dt*slope(index+1);
}
// A complete Shot path shares waypoint velocity across adjacent Moves. Only the
// reversing or held scalar stops at its waypoint; other axes keep moving.
function takeOffsets(poses) {return [...new Map(poses.flatMap(p=>p.subjectOffsets||[[0,0,0]]).map(o=>[JSON.stringify(o),o])).values()];}
export function interpolateCameraPath(nodes,index,t) {
  if(nodes.length<2)return nodes[0]?.pose;
  index=Math.max(0,Math.min(nodes.length-2,index));
  const times=nodes.map(n=>n.time),azimuths=unwrapAngles(nodes);
  const scalar=key=>hermite(nodes.map(n=>n.pose[key]),times,index,t);
  const vector=key=>[0,1,2].map(axis=>hermite(nodes.map(n=>n.pose[key][axis]),times,index,t));
  const azimuth=hermite(azimuths,times,index,t),target=vector('target'),anchor=vector('anchor');
  let distance=scalar('distance'),elevation=scalar('elevation'),radius=distance*Math.cos(elevation*Math.PI/180),height=target[1]+distance*Math.sin(elevation*Math.PI/180);
  if(nodes.some(n=>n.pose.explicitLevel)){
    height=hermite(nodes.map(n=>n.pose.position[1]),times,index,t);
    target[1]=height-distance*Math.sin(elevation*Math.PI/180);anchor[1]=target[1];
  }
  const az=azimuth*Math.PI/180,position=[Math.sin(az)*radius,height,Math.cos(az)*radius];
  return {viewpointProxy:nodes.some(n=>n.pose.viewpointProxy),levelAdjusted:nodes.some(n=>n.pose.levelAdjusted),subjectOffsets:takeOffsets(nodes.map(n=>n.pose)),target,anchor,position,tangent:scalar('tangent'),shiftX:scalar('shiftX'),shiftY:scalar('shiftY'),roll:scalar('roll'),azimuth,elevation,distance};
}
export function projectPoint(p,pose) {
  const b=basis(pose),v=sub(p,pose.position),z=dot(v,b.forward);
  return [dot(v,b.right)/(z*pose.tangent)-(pose.shiftX||0),dot(v,b.up)/(z*pose.tangent)-(pose.shiftY||0),z];
}
export function modelFaces(pose,size) {
  const screen=p=>{const q=projectPoint(p,pose);return [(q[0]+1)*size/2,(1-q[1])*size/2,q[2]];};
  const faces=[];
  (pose.subjectOffsets||[[0,0,0]]).forEach(offset=>parts.forEach(([x,y,z,wx,hy,dz],index)=>{
    const facing=offset[3]===180?-1:1;
    const [cx,cy,cz]=add([x*facing,y,z*facing],offset);
    const vs=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
      .map(([a,c,d])=>[cx+a*wx/2,cy+c*hy/2,cz+d*dz/2]);
    for(const ids of [[0,3,2,1],[4,5,6,7],[0,4,7,3],[1,2,6,5],[3,7,6,2],[0,1,5,4]]){
      const vertices=ids.map(i=>vs[i]),normal=unit(cross(sub(vertices[1],vertices[0]),sub(vertices[2],vertices[0])));
      if(dot(normal,sub(pose.position,vertices[0]))<=0)continue;
      const ps=vertices.map(screen);if(ps.some(p=>p[2]<.05))continue;
      faces.push({ps,depth:ps.reduce((s,p)=>s+p[2],0)/4,light:.55+.45*Math.max(0,dot(normal,unit([-.5,1,1]))),index});
    }
  }));
  return faces;
}

// Per-pixel depth: a large head face must never paint over a nearer facial marker
// just because the face's average distance sorts ahead of the small marker.
export function rasterizeFaces(faces,size,pixels=new Uint8ClampedArray(size*size*4),depth=new Float64Array(size*size),wireWidth=0) {
  pixels.fill(0); depth.fill(0);
  const edge=(a,b,x,y)=>(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]);
  for(const face of faces) {
    const color=(featureColors[face.index]||[50,137,170]).map(c=>Math.round(c*face.light));
    // Shade only the quad perimeter, not the triangulation diagonal. Outlines
    // participate in the same depth test as the surface, so hidden edges stay hidden.
    const edges=face.ps.map((a,i)=>{
      const b=face.ps[(i+1)%4];
      return {a,b,length:Math.hypot(b[0]-a[0],b[1]-a[1])};
    });
    for(const triangle of [[0,1,2],[0,2,3]]) {
      const [a,b,c]=triangle.map(i=>face.ps[i]);
      const area=edge(a,b,c[0],c[1]); if(Math.abs(area)<1e-10)continue;
      const minX=Math.max(0,Math.floor(Math.min(a[0],b[0],c[0]))),maxX=Math.min(size-1,Math.ceil(Math.max(a[0],b[0],c[0])));
      const minY=Math.max(0,Math.floor(Math.min(a[1],b[1],c[1]))),maxY=Math.min(size-1,Math.ceil(Math.max(a[1],b[1],c[1])));
      for(let y=minY;y<=maxY;y++)for(let x=minX;x<=maxX;x++) {
        const u=edge(b,c,x+.5,y+.5)/area,v=edge(c,a,x+.5,y+.5)/area,t=1-u-v;
        if(u< -1e-9||v< -1e-9||t< -1e-9)continue;
        // Reciprocal depth interpolates linearly after perspective projection.
        const inverseZ=u/a[2]+v/b[2]+t/c[2],offset=y*size+x;
        if(inverseZ<=depth[offset])continue;
        depth[offset]=inverseZ;
        const pixel=offset*4;
        const outline=wireWidth>0 && edges.some(e=>e.length>1e-9 && Math.abs(edge(e.a,e.b,x+.5,y+.5))/e.length<wireWidth*.5);
        pixels[pixel]=outline?255:color[0];pixels[pixel+1]=outline?255:color[1];pixels[pixel+2]=outline?255:color[2];pixels[pixel+3]=255;
      }
    }
  }
  return pixels;
}

const renderBuffers=new WeakMap();
function scene(ctx,rect,pose,active,path) {
  const [x,y,w]=rect;
  const screen=p=>{const q=projectPoint(p,pose);return [x+(q[0]+1)*w/2,y+(1-q[1])*w/2,q[2]];};
  const line=(a,c,color)=>{const p=screen(a),q=screen(c);if(p[2]<.05||q[2]<.05)return;
    ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();};
  ctx.save();ctx.beginPath();ctx.rect(x,y,w,w);ctx.clip();ctx.fillStyle='#111922';ctx.fillRect(x,y,w,w);
  for(let i=-10;i<=10;i++){line([i,0,-10],[i,0,10],'#263540');line([-10,0,i],[10,0,i],'#263540');}
  // Render above display resolution so thin features survive distance scaling.
  const size=Math.min(1200,Math.max(1,Math.ceil(w*Math.max(2,ctx.getTransform().a))));
  let buffers=renderBuffers.get(ctx);
  if(!buffers){buffers=new Map();renderBuffers.set(ctx,buffers);}
  const key=active?'scene':'output';
  let buffer=buffers.get(key);
  if(!buffer||buffer.size!==size){
    const canvas=ctx.canvas.ownerDocument.createElement('canvas');canvas.width=size;canvas.height=size;
    const context=canvas.getContext('2d');
    buffer={size,canvas,context,image:context.createImageData(size,size),depth:new Float64Array(size*size)};
    buffers.set(key,buffer);
  }
  rasterizeFaces(modelFaces(pose,size),size,buffer.image.data,buffer.depth,size/w);
  buffer.context.putImageData(buffer.image,0,0);
  ctx.drawImage(buffer.canvas,x,y,w,w);
  if(active){
    if(path)for(let i=1;i<path.length;i++)line(path[i-1].position,path[i].position,'#9c83ff');
    const cb=basis(active),depth=Math.min(active.distance,2),center=add(active.position,mul(cb.forward,depth));
    const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,c])=>add(center,add(mul(cb.right,(a+active.shiftX)*depth*active.tangent),mul(cb.up,(c+active.shiftY)*depth*active.tangent))));
    corners.forEach((p,i)=>{line(active.position,p,'#eaa469');line(p,corners[(i+1)%4],'#eaa469');});
    line(active.position,active.target,'#ffdc72');
    const p=screen(active.position),q=screen(active.target);ctx.fillStyle='#ffc663';ctx.fillRect(p[0]-4,p[1]-4,8,8);
    ctx.beginPath();ctx.arc(q[0],q[1],4,0,Math.PI*2);ctx.fill();
  }else{const q=screen(pose.target);ctx.strokeStyle='#ffd46d';ctx.beginPath();ctx.moveTo(q[0]-5,q[1]);ctx.lineTo(q[0]+5,q[1]);ctx.moveTo(q[0],q[1]-5);ctx.lineTo(q[0],q[1]+5);ctx.stroke();}
  ctx.restore();
}
export function drawCameraPreview(canvas,pose,path=[]) {
  const size=Math.max(240,canvas.clientWidth||400),dpr=Math.min(3,window.devicePixelRatio||1);
  if(canvas.width!==Math.round(size*dpr)||canvas.height!==Math.round(size*dpr)){canvas.width=Math.round(size*dpr);canvas.height=Math.round(size*dpr);}
  const ctx=canvas.getContext('2d');if(!ctx)return;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.lineWidth=1;
  const radius=Math.max(7,pose.distance*1.8);
  const overview={subjectOffsets:pose.subjectOffsets,position:[radius,.7*radius,radius],target:[0,1.5,0],tangent:.65};
  scene(ctx,[0,0,size],overview,pose,path);
  const w=size*.47,x=size-w-10,y=30;
  scene(ctx,[x,y,w],pose);ctx.strokeStyle='#65b9ff';ctx.strokeRect(x,y,w,w);
  ctx.fillStyle='#d8edff';ctx.font='10px sans-serif';ctx.fillText('CAMERA OUTPUT 1:1',x+6,y+14);
  if(pose.levelAdjusted){ctx.fillStyle='#ffc663';ctx.font='11px sans-serif';ctx.fillText('Level + crop: best-fit angle (panel angle adjusted).',10,size-40);}
  if(pose.viewpointProxy){ctx.fillStyle='#ffc663';ctx.font='11px sans-serif';ctx.fillText('Viewpoint proxy: fixed example layout, not user scene.',10,size-55);}
  if([pose,...path].some(p=>p.position[1]<0)) {
    ctx.fillStyle='#ffc663';ctx.font='11px sans-serif';
    ctx.fillText('Warning: camera path below ground.',10,size-26);
    ctx.fillText('Use a less low angle or a tighter shot.',10,size-11);
  }
}
