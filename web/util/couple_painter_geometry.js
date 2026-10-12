const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
export const PAINT_CORNERS=['nw','ne','sw','se'];
export const MAX_PAINT_AXIS=8192;

export function movePaintBounds(bounds,dx,dy,width,height) {
    return {...bounds,x:Math.round(bounds.x+dx),y:Math.round(bounds.y+dy)};
}

export function resizePaintBounds(bounds,corner,dx,dy,width,height,free=true) {
    if(!PAINT_CORNERS.includes(corner))return {...bounds};
    const sx=corner.includes('e')?1:-1,sy=corner.includes('s')?1:-1;
    const ax=sx>0?bounds.x:bounds.x+bounds.w,ay=sy>0?bounds.y:bounds.y+bounds.h;
    // Source artwork is independent of the output canvas. Crossing its edges
    // must not clip the editable bitmap or change the opposite anchor.
    const maxW=MAX_PAINT_AXIS,maxH=MAX_PAINT_AXIS;
    const rawW=bounds.w+sx*dx,rawH=bounds.h+sy*dy;
    let w,h;
    if(free) {
        w=clamp(Math.round(rawW),1,maxW);h=clamp(Math.round(rawH),1,maxH);
    }else {
        // Project the pointer delta onto the original diagonal. Keep the
        // opposite corner fixed, preserve aspect ratio and never flip/crop.
        const requested=(bounds.w*rawW+bounds.h*rawH)/(bounds.w*bounds.w+bounds.h*bounds.h);
        const scale=clamp(requested,Math.max(1/bounds.w,1/bounds.h),Math.min(maxW/bounds.w,maxH/bounds.h));
        w=clamp(Math.round(bounds.w*scale),1,maxW);h=clamp(Math.round(bounds.h*scale),1,maxH);
    }
    return {x:sx>0?ax:ax-w,y:sy>0?ay:ay-h,w,h};
}

// Bounding canvas for rotation about the artwork's center, not the output
// canvas center. Keep off-canvas artwork and never clamp/crop its rotated size.
export function rotatePaintBounds(bounds,angle) {
    const cosine=Math.abs(Math.cos(angle)),sine=Math.abs(Math.sin(angle));
    const w=bounds.w*(cosine<1e-12?0:cosine)+bounds.h*(sine<1e-12?0:sine);
    const h=bounds.w*(sine<1e-12?0:sine)+bounds.h*(cosine<1e-12?0:cosine);
    const cx=bounds.x+bounds.w/2,cy=bounds.y+bounds.h/2;
    const x=Math.floor(cx-w/2+1e-10),y=Math.floor(cy-h/2+1e-10);
    return {x,y,w:Math.max(1,Math.ceil(cx+w/2-1e-10)-x),h:Math.max(1,Math.ceil(cy+h/2-1e-10)-y)};
}

export function hitPaintBounds(bounds,point,padX,padY) {
    if(!bounds)return null;
    let nearest=null,distance=Infinity;
    for(const corner of PAINT_CORNERS) {
        const x=corner.includes('e')?bounds.x+bounds.w:bounds.x,y=corner.includes('s')?bounds.y+bounds.h:bounds.y;
        const dx=Math.abs(point.x-x)/padX,dy=Math.abs(point.y-y)/padY;
        if(dx<=1&&dy<=1&&dx*dx+dy*dy<distance){nearest=corner;distance=dx*dx+dy*dy;}
    }
    if(nearest)return nearest;
    return point.x>=bounds.x&&point.x<=bounds.x+bounds.w&&point.y>=bounds.y&&point.y<=bounds.y+bounds.h?'move':null;
}

// Transform-frame geometry is distinct from the axis-aligned raster envelope.
// Its width/height and local axes remain stable while the artwork rotates.
export function validPaintFrame(frame) {
    return !!frame&&['x','y','w','h','angle'].every(key=>Number.isFinite(frame[key]))&&
        Math.abs(frame.x)<=Number.MAX_SAFE_INTEGER/4&&Math.abs(frame.y)<=Number.MAX_SAFE_INTEGER/4&&
        frame.w>0&&frame.h>0&&frame.w<=MAX_PAINT_AXIS&&frame.h<=MAX_PAINT_AXIS&&Math.abs(frame.angle)<=Math.PI+1e-10;
}
export function paintFramePoint(frame,x,y) {
    const c=Math.cos(frame.angle),s=Math.sin(frame.angle);
    return {x:frame.x+frame.w/2+c*x-s*y,y:frame.y+frame.h/2+s*x+c*y};
}
export function paintFrameLocal(frame,point) {
    const dx=point.x-frame.x-frame.w/2,dy=point.y-frame.y-frame.h/2,c=Math.cos(frame.angle),s=Math.sin(frame.angle);
    return {x:c*dx+s*dy,y:-s*dx+c*dy};
}
export function hitPaintFrame(frame,point,padX,padY) {
    if(!frame)return null;const local=paintFrameLocal(frame,point);
    return hitPaintBounds({x:-frame.w/2,y:-frame.h/2,w:frame.w,h:frame.h},local,padX,padY);
}
export function resizePaintFrame(frame,corner,dx,dy,free=true) {
    if(!PAINT_CORNERS.includes(corner))return {...frame};
    const c=Math.cos(frame.angle),s=Math.sin(frame.angle);
    const resized=resizePaintBounds({x:0,y:0,w:frame.w,h:frame.h},corner,c*dx+s*dy,-s*dx+c*dy,0,0,free);
    const sx=corner.includes('e')?1:-1,sy=corner.includes('s')?1:-1;
    const anchor=paintFramePoint(frame,-sx*frame.w/2,-sy*frame.h/2),ox=sx*resized.w/2,oy=sy*resized.h/2;
    return {x:anchor.x+c*ox-s*oy-resized.w/2,y:anchor.y+s*ox+c*oy-resized.h/2,w:resized.w,h:resized.h,angle:frame.angle};
}
export function mapPaintFramePoint(point,from,to) {
    const p=paintFrameLocal(from,point);
    return paintFramePoint(to,p.x*to.w/from.w,p.y*to.h/from.h);
}
export function transformedPaintBounds(bounds,from,to) {
    const points=PAINT_CORNERS.map(corner=>mapPaintFramePoint({x:bounds.x+(corner.includes('e')?bounds.w:0),y:bounds.y+(corner.includes('s')?bounds.h:0)},from,to));
    const x=Math.floor(Math.min(...points.map(p=>p.x))+1e-10),y=Math.floor(Math.min(...points.map(p=>p.y))+1e-10);
    return {x,y,w:Math.max(1,Math.ceil(Math.max(...points.map(p=>p.x))-1e-10)-x),h:Math.max(1,Math.ceil(Math.max(...points.map(p=>p.y))-1e-10)-y)};
}
