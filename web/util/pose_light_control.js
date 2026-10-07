import * as T from '../vendor/three/three.module.min.js';

// A compact shaded sphere shows the key-light source in camera coordinates.
// Its canvas is independent of the scene canvas, so it never enters captures.
export class PoseLightControl {
    constructor(view, camera, getDirection, onChange, onFinish) {
        this.camera=camera;this.getDirection=getDirection;this.onChange=onChange;this.onFinish=onFinish;
        this.defaultDirection=getDirection().clone();this.signature='';this.drag=null;
        this.root=document.createElement('div');
        this.root.style.cssText='position:absolute;right:8px;top:8px;width:80px;z-index:5;user-select:none;text-align:center;color:#d8dce3;font:11px sans-serif';
        this.canvas=document.createElement('canvas');this.canvas.width=160;this.canvas.height=160;
        this.canvas.style.cssText='display:block;width:80px;height:80px;touch-action:none;cursor:grab;border-radius:50%;background:#20262e99';
        this.canvas.tabIndex=0;this.canvas.setAttribute('aria-label','Key light direction');
        this.canvas.title='Drag to aim the key light. Double-click to reset.';
        this.context=this.canvas.getContext('2d');
        const label=document.createElement('div');label.textContent='Light · drag';label.style.marginTop='2px';
        this.root.append(this.canvas,label);view.append(this.root);
        this.abort=new AbortController();const options={signal:this.abort.signal};
        this.canvas.addEventListener('pointerdown',event=>{
            if(event.button!==0)return;
            const local=getDirection().clone().applyQuaternion(camera.quaternion.clone().invert());
            const rect=this.canvas.getBoundingClientRect();
            this.drag={id:event.pointerId,x:event.clientX,y:event.clientY,width:rect.width,height:rect.height,yaw:Math.atan2(local.x,local.z),pitch:Math.asin(T.MathUtils.clamp(local.y,-1,1)),camera:camera.quaternion.clone()};
            if(event.isTrusted)this.canvas.setPointerCapture(event.pointerId);
            this.canvas.style.cursor='grabbing';event.preventDefault();event.stopPropagation();
        },options);
        this.canvas.addEventListener('pointermove',event=>{
            if(!this.drag||event.pointerId!==this.drag.id)return;
            const d=this.drag,yaw=d.yaw+(event.clientX-d.x)*Math.PI*2/d.width;
            const pitch=T.MathUtils.clamp(d.pitch-(event.clientY-d.y)*Math.PI/d.height,-Math.PI*.495,Math.PI*.495);
            const direction=new T.Vector3(Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)).applyQuaternion(d.camera);
            onChange(direction);this.draw();event.preventDefault();event.stopPropagation();
        },options);
        const finish=event=>{
            if(!this.drag||event.pointerId!==this.drag.id)return;
            this.drag=null;this.canvas.style.cursor='grab';onFinish();event.preventDefault();event.stopPropagation();
        };
        this.canvas.addEventListener('pointerup',finish,options);
        this.canvas.addEventListener('pointercancel',finish,options);
        this.canvas.addEventListener('lostpointercapture',finish,options);
        this.canvas.addEventListener('dblclick',event=>{onChange(this.defaultDirection.clone());onFinish();this.draw();event.preventDefault();event.stopPropagation();},options);
        for(const name of ['pointerdown','pointerup','pointermove','wheel','keydown'])this.root.addEventListener(name,event=>event.stopPropagation(),options);
        this.draw();
    }
    draw() {
        const direction=this.getDirection().clone().applyQuaternion(this.camera.quaternion.clone().invert()).normalize();
        const signature=direction.toArray().map(x=>x.toFixed(5)).join(',');if(signature===this.signature)return;this.signature=signature;
        const context=this.context,size=this.canvas.width,center=size/2,radius=size*.36;
        context.clearRect(0,0,size,size);const image=context.createImageData(size,size);
        for(let y=0;y<size;y++)for(let x=0;x<size;x++){
            const nx=(x-center)/radius,ny=(center-y)/radius,squared=nx*nx+ny*ny;if(squared>1)continue;
            const nz=Math.sqrt(1-squared),diffuse=Math.max(0,nx*direction.x+ny*direction.y+nz*direction.z);
            const highlight=Math.pow(diffuse,24)*.28,shade=.2+.65*diffuse;
            const offset=(y*size+x)*4;
            image.data[offset]=Math.min(255,205*shade+255*highlight);
            image.data[offset+1]=Math.min(255,190*shade+255*highlight);
            image.data[offset+2]=Math.min(255,160*shade+255*highlight);image.data[offset+3]=255;
        }
        context.putImageData(image,0,0);
        context.strokeStyle='#91a0b1';context.lineWidth=1.5;context.beginPath();context.arc(center,center,radius,0,Math.PI*2);context.stroke();
        const x=center+direction.x*radius,y=center-direction.y*radius;
        context.strokeStyle='#ffe19a';context.lineWidth=2;context.setLineDash(direction.z<0?[4,4]:[]);
        context.beginPath();context.moveTo(center,center);context.lineTo(x,y);context.stroke();context.setLineDash([]);
        context.beginPath();context.arc(x,y,5,0,Math.PI*2);
        if(direction.z>=0){context.fillStyle='#ffe19a';context.fill();}else context.stroke();
    }
    dispose(){this.abort.abort();this.root.remove();}
}
