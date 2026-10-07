// Canonical editor geometry is always top-left x/y plus width/height in 0..1.
export function pointInRect(rect, clientX, clientY) {
    return {x: Math.max(0, Math.min(1, (clientX-rect.left)/rect.width)),
            y: Math.max(0, Math.min(1, (clientY-rect.top)/rect.height))};
}
export function dragRegion(original, start, point, mode) {
    const dx = point.x-start.x, dy = point.y-start.y;
    if (mode === "move") return {...original,
        x: Math.max(0,Math.min(1-original.w,original.x+dx)),
        y: Math.max(0,Math.min(1-original.h,original.y+dy))};
    if (mode.startsWith("resize-")) {
        let left=original.x, top=original.y, right=left+original.w, bottom=top+original.h;
        const edge=mode.slice(7);
        if(edge.includes("w"))left=Math.max(0,Math.min(1,original.x+dx));
        if(edge.includes("e"))right=Math.max(0,Math.min(1,original.x+original.w+dx));
        if(edge.includes("n"))top=Math.max(0,Math.min(1,original.y+dy));
        if(edge.includes("s"))bottom=Math.max(0,Math.min(1,original.y+original.h+dy));
        if(right<left)[left,right]=[right,left];
        if(bottom<top)[top,bottom]=[bottom,top];
        return {...original,x:left,y:top,w:right-left,h:bottom-top};
    }
    const x1 = mode === "draw" ? start.x : original.x;
    const y1 = mode === "draw" ? start.y : original.y;
    const x2 = Math.max(0,Math.min(1,point.x)), y2 = Math.max(0,Math.min(1,point.y));
    return {...original,x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1),h:Math.abs(y2-y1)};
}

// KJ-style direct manipulation: selected edge/corner handles resize, the box
// interior moves it. Pixel radii keep handles usable at every output resolution.
export function hitRegionHandle(regions, point, width, height, activeIndex = -1, radius = 10) {
    const rx=radius/Math.max(1,width), ry=radius/Math.max(1,height);
    const order=[...regions.keys()].filter(i=>i!==activeIndex).reverse();
    if(activeIndex>=0&&activeIndex<regions.length)order.unshift(activeIndex);
    for(const i of order) {
        const r=regions[i], left=r.x, top=r.y, right=left+r.w, bottom=top+r.h;
        const near=(x,y)=>Math.abs(point.x-x)<=rx&&Math.abs(point.y-y)<=ry;
        const corners=[[left,top,"nw"],[right,top,"ne"],[left,bottom,"sw"],[right,bottom,"se"]];
        const corner=corners.find(([x,y])=>near(x,y));
        if(corner)return {index:i,mode:`resize-${corner[2]}`};
        if(point.x>=left-rx&&point.x<=right+rx) {
            if(Math.abs(point.y-top)<=ry)return {index:i,mode:"resize-n"};
            if(Math.abs(point.y-bottom)<=ry)return {index:i,mode:"resize-s"};
        }
        if(point.y>=top-ry&&point.y<=bottom+ry) {
            if(Math.abs(point.x-left)<=rx)return {index:i,mode:"resize-w"};
            if(Math.abs(point.x-right)<=rx)return {index:i,mode:"resize-e"};
        }
        if(point.x>=left&&point.x<=right&&point.y>=top&&point.y<=bottom)return {index:i,mode:"move"};
    }
    return null;
}
export function hitRegion(regions, point) {
    for (let i=regions.length-1;i>=0;i--) {
        const r=regions[i];
        if (point.x>=r.x && point.x<=r.x+r.w && point.y>=r.y && point.y<=r.y+r.h) return i;
    }
    return -1;
}

// Use one pixel spacing on both axes so the guide grid never appears stretched
// when the target image uses a portrait or landscape aspect ratio.
export function squareGridLines(width, height, divisions = 10) {
    const step = Math.min(width, height) / divisions;
    if (!Number.isFinite(step) || step <= 0) return {step: 0, x: [], y: []};
    const positions = length => Array.from({length: Math.floor(length / step)}, (_, i) => (i + 1) * step);
    return {step, x: positions(width), y: positions(height)};
}

export function validateRegions(value) {
    if (!Array.isArray(value) || value.length > 128) throw Error("Layout must contain at most 128 regions.");
    return value.map(r => {
        if (!r || typeof r !== "object" || ["x","y","w","h"].some(k => typeof r[k] !== "number" || !Number.isFinite(r[k])) ||
            r.x < 0 || r.y < 0 || r.w < .001 || r.h < .001 || r.x+r.w > 1.000001 || r.y+r.h > 1.000001 ||
            !["obj","text"].includes(r.type ?? "obj")) throw Error("Invalid region geometry or type.");
        for (const key of ["desc","text","relation"]) if (r[key] != null && typeof r[key] !== "string") throw Error("Region descriptions must be text.");
        const palette=r.palette ?? [];
        if (!Array.isArray(palette) || palette.length > 16 || palette.some(c => typeof c !== "string" || !/^#[0-9a-f]{6}$/i.test(c))) throw Error("Invalid region palette.");
        return {x:r.x,y:r.y,w:r.w,h:r.h,type:r.type??"obj",desc:r.desc??"",text:r.text??"",relation:r.relation??"",palette:[...palette],locked:!!r.locked};
    });
}
