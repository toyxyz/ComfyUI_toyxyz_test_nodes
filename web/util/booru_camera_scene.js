// Software perspective projection of a 3D proxy scene. No actual scene geometry is known.
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const mul=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>mul(a,1/(Math.hypot(...a)||1));
// Fit the maximum (Very wide) orbit plus the camera body, not an empty
// 640x300 letterbox. Keep this transform identical for rendering and picking.
export function previewViewport(width,height) {
    const scale=Math.min(width,height)/344;
    return {scale,offsetX:width/2-320*scale,offsetY:height/2-150*scale,
        left:320-width/(2*scale),right:320+width/(2*scale),
        top:150-height/(2*scale),bottom:150+height/(2*scale)};
}
export function cameraPosition(x,y,z) {
    const az=x*Math.PI, el=y*Math.PI/2, radius=2.4+z*.9;
    return [Math.sin(az)*Math.cos(el)*radius,.9+Math.sin(el)*radius,Math.cos(az)*Math.cos(el)*radius];
}
export function cameraOrbitPaths(state,steps=96) {
    const radius=2.4+state.z*.9,az=state.x*Math.PI,el=state.y*Math.PI/2;
    const horizontal=[],vertical=[];
    for(let i=0;i<=steps;i++) {
        const t=i/steps*Math.PI*2;
        horizontal.push([Math.sin(t)*Math.cos(el)*radius,.9+Math.sin(el)*radius,Math.cos(t)*Math.cos(el)*radius]);
        vertical.push([Math.sin(az)*Math.cos(t)*radius,.9+Math.sin(t)*radius,Math.cos(az)*Math.cos(t)*radius]);
    }
    return {horizontal,vertical};
}
export function projectPoint(point,yaw=0,pitch=Math.PI/4) {
    const right=[Math.cos(yaw),0,-Math.sin(yaw)];
    const up=[-Math.sin(yaw)*Math.sin(pitch),Math.cos(pitch),-Math.cos(yaw)*Math.sin(pitch)];
    const depth=[Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)];
    const p=add(point,[0,-.9,0]);
    const scale=540/Math.max(1,12.5-dot(p,depth));
    return [320+dot(p,right)*scale,150-dot(p,up)*scale];
}
export function drawCameraScene(ctx,state,view) {
    const project=p=>projectPoint(p,view.yaw,view.pitch);
    const line=(a,b,color='#35536a',dashed=false)=>{
        const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.setLineDash(dashed?[4,5]:[]);
        ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([]);
    };
    // Paths follow the actual orbit controls at the current distance.
    for(let i=-2;i<=2;i++) {line([i,0,-2],[i,0,2],'#253947');line([-2,0,i],[2,0,i],'#253947');}
    const camera=cameraPosition(state.x,state.y,state.z);
    const targetHeight=Number.isFinite(state.targetHeight)?state.targetHeight:.9;
    camera[1]+=targetHeight-.9;
    const paths=cameraOrbitPaths(state);
    const horizontalColor=state.activeAxis==='x'?'#b6e5ff':state.hoverAxis==='x'?'#80caff':'#4d98cc';
    const verticalColor=state.activeAxis==='y'?'#e6ccff':state.hoverAxis==='y'?'#cc9cff':'#a176cb';
    for(const [points,color] of [
        [paths.horizontal,horizontalColor],
        [paths.vertical,verticalColor]]) {
        ctx.strokeStyle=color;ctx.setLineDash([4,6]);ctx.beginPath();
        points.forEach((p,i)=>{const q=project(add(p,[0,targetHeight-.9,0]));if(i===0)ctx.moveTo(...q);else ctx.lineTo(...q);});
        ctx.stroke();ctx.setLineDash([]);
    }
    if(state.activeAxis) {
        const stops=state.activeAxis==='x'?[0,.25,-.25,.5,-.5,.75,-.75,1]:[0,.45,-.45];
        const color=state.activeAxis==='x'?horizontalColor:verticalColor;
        for(const stop of stops) {
            const position=cameraPosition(state.activeAxis==='x'?stop:state.x,state.activeAxis==='y'?stop:state.y,state.z);
            position[1]+=targetHeight-.9;
            const point=project(position);
            ctx.beginPath();ctx.arc(point[0],point[1],4,0,Math.PI*2);
            ctx.fillStyle=color;ctx.fill();
            ctx.strokeStyle='#14202b';ctx.lineWidth=1.5;ctx.stroke();
        }
        ctx.lineWidth=1;
    }
    const faces=[];
    const depth=[Math.sin(view.yaw)*Math.cos(view.pitch),Math.sin(view.pitch),Math.cos(view.yaw)*Math.cos(view.pitch)];
    const box=(center,size,color,basis=[[1,0,0],[0,1,0],[0,0,1]])=>{
        const vertices=[];
        for(let i=0;i<8;i++) {
            let p=center;
            for(let axis=0;axis<3;axis++)p=add(p,mul(basis[axis],((i>>axis)&1?1:-1)*size[axis]/2));
            vertices.push(p);
        }
        for(const indices of [[0,2,6,4],[1,5,7,3],[0,4,5,1],[2,3,7,6],[0,1,3,2],[4,6,7,5]]) {
            const points=indices.map(i=>vertices[i]);
            const faceCenter=points.reduce((sum,p)=>add(sum,mul(p,.25)),[0,0,0]);
            let normal=unit(cross(add(points[1],mul(points[0],-1)),add(points[2],mul(points[0],-1))));
            if(dot(normal,add(faceCenter,mul(center,-1)))<0)normal=mul(normal,-1);
            // Ambient + fixed upper-front key light; normals are world-space.
            const brightness=.45+.55*Math.max(0,dot(normal,unit([-.6,1,.8])));
            const rgb=[1,3,5].map(start=>Math.round(parseInt(color.slice(start,start+2),16)*brightness));
            faces.push({points,color:`rgb(${rgb.join(',')})`,depth:dot(faceCenter,depth)});
        }
    };
    box([0,1.65,0],[.3,.3,.3],'#b4b4b4');box([0,1.1,0],[.48,.7,.25],'#b4b4b4');
    box([-.14,.38,0],[.17,.75,.2],'#b4b4b4');box([.14,.38,0],[.17,.75,.2],'#b4b4b4');
    box([-.35,1.08,0],[.14,.65,.2],'#b4b4b4');box([.35,1.08,0],[.14,.65,.2],'#b4b4b4');
    const target=[0,targetHeight,0],forward=unit(add(target,mul(camera,-1)));
    let right=unit(cross(forward,Math.abs(forward[1])>.99?[0,0,1]:[0,1,0]));
    let up=unit(cross(right,forward));
    const roll=state.roll*Math.PI/4,r=right;
    right=add(mul(r,Math.cos(roll)),mul(up,Math.sin(roll)));
    up=add(mul(up,Math.cos(roll)),mul(r,-Math.sin(roll)));
    box(camera,[.38,.25,.3],'#ffb86b',[right,up,forward]);
    const lens=add(camera,mul(forward,.22));line(camera,target,'#ffc978');
    const corners=[[-1,-1],[-1,1],[1,1],[1,-1]].map(([x,y])=>
        add(add(add(camera,mul(forward,.8)),mul(right,x*.35)),mul(up,y*.22)));
    corners.forEach((p,i)=>{line(lens,p,'#cb995c');line(p,corners[(i+1)%4],'#cb995c');});
    const ground=[camera[0],0,camera[2]];line(camera,ground,'#ffc978',true);
    // Painter-sort all mannequin/camera faces together for proper proxy occlusion.
    faces.sort((a,b)=>a.depth-b.depth);
    for(const face of faces) {
        ctx.fillStyle=face.color;ctx.beginPath();
        face.points.forEach((p,i)=>{const q=project(p);if(i===0)ctx.moveTo(...q);else ctx.lineTo(...q);});
        ctx.closePath();ctx.fill();
    }
    // Orientation gizmo is independent of camera output; only X/Y/Z text remains.
    const origin=project([0,.9,0]),anchor=[(view.right??640)-52,(view.top??0)+52];
    for(const [axis,point,color] of [['X',[1,.9,0],'#f07878'],['Y',[0,1.9,0],'#85d798'],['Z',[0,.9,1],'#79b8ff']]) {
        const end=project(point),dx=(end[0]-origin[0])*.45,dy=(end[1]-origin[1])*.45;
        ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(...anchor);ctx.lineTo(anchor[0]+dx,anchor[1]+dy);ctx.stroke();
        const angle=Math.atan2(dy,dx),x=anchor[0]+dx,y=anchor[1]+dy;
        ctx.beginPath();ctx.moveTo(x,y);
        ctx.lineTo(x-6*Math.cos(angle-.45),y-6*Math.sin(angle-.45));
        ctx.lineTo(x-6*Math.cos(angle+.45),y-6*Math.sin(angle+.45));ctx.closePath();ctx.fill();
        ctx.font='12px sans-serif';ctx.fillText(axis,x+5,y-4);
    }
    ctx.lineWidth=1;
}
