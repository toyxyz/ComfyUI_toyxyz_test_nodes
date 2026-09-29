// Camera proxy: the orbit is illustrative, not a scene-aware framing calculation.
import { drawCameraScene } from './booru_camera_scene.js';
export function installCameraPanel(node) {
    const widgets = Object.fromEntries(node.widgets.map(w => [w.name, w]));
    if (!widgets.pos_x || !node.addDOMWidget) return;
    for (const widget of node.widgets) {
        widget.type = 'hidden'; widget.computeSize = () => [0, -4]; widget.draw = () => {};
    }
    widgets.panel_enabled.value = true;
    const root = document.createElement('div');
    root.style.cssText = 'width:100%;height:100%;min-width:0;min-height:0;overflow:auto;background:#222a33;color:#e5edf5;border-radius:10px;padding:12px;box-sizing:border-box;font:13px sans-serif;display:flex;flex-direction:column;gap:9px;';
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;gap:10px;flex-shrink:0;';
    const title = document.createElement('strong'); title.textContent = 'Camera composition'; title.style.flex = '1';
    const enabled = document.createElement('input'); enabled.type = 'checkbox'; enabled.setAttribute('aria-label', 'Enable camera');
    enabled.title = 'Enable or disable camera tags without changing your source tags.';
    const reset = document.createElement('button'); reset.textContent = 'Reset';
    reset.title = 'Reset position, elevation, distance and roll. Keep tag strengths.';
    header.append(title, enabled, reset); root.append(header);
    const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 300;
    canvas.style.cssText = 'display:block;width:100%;height:190px;min-height:160px;flex:1 0 190px;background:#171e26;border-radius:8px;touch-action:none;';
    canvas.title = 'Left drag: camera position. Right drag or Alt+drag: inspect the 3D scene. Scroll: distance. Shift+scroll: roll. Proxy geometry only.';
    root.append(canvas);
    const help = document.createElement('div');
    help.textContent = 'Left drag: camera / Right drag or Alt+drag: inspect / Scroll: distance / Shift+scroll: roll'; help.style.flexShrink = '0'; root.append(help);
    const view = {yaw:0,pitch:Math.PI/4};
    const controls = [];
    const clamp = (v, min, max) => Math.max(min, Math.min(max, Number(v) || 0));
    const value = name => Number(widgets[name]?.value) || 0;
    const set = (name, next) => {
        const widget = widgets[name]; widget.value = next; widget.callback?.(next);
        sync(); node.setDirtyCanvas?.(true, true);
    };
    const makeRow = (name, label, min, max) => {
        const row = document.createElement('div'); row.style.cssText = 'display:flex;align-items:center;gap:10px;flex-shrink:0;min-width:0;';
        const text = document.createElement('span'); text.textContent = label; text.style.cssText = 'width:126px;flex-shrink:0;white-space:nowrap;';
        const slider = document.createElement('input'); slider.type = 'range'; slider.min = min; slider.max = max; slider.step = '.01';
        slider.style.cssText = 'flex:1;min-width:30px;accent-color:#72b8ff;'; slider.setAttribute('aria-label', label);
        const number = document.createElement('input'); number.type = 'number'; number.min = min; number.max = max; number.step = '.01';
        number.style.cssText = 'width:70px;background:#151c24;color:#e5edf5;border:1px solid #536779;border-radius:5px;padding:5px;';
        number.setAttribute('aria-label', label + ' numeric input');
        const tooltip = `${label}: ${min} to ${max}. Arrow keys: 0.01; Shift+arrow: 0.10. Enter applies; Escape cancels.`;
        slider.title = tooltip; number.title = tooltip;
        if (name === 'pos_z') {
            slider.title = number.title = tooltip + ' Lower values move closer; higher values move farther away.';
        }
        const apply = v => set(name, Math.round(clamp(v, min, max) * 100) / 100);
        slider.addEventListener('input', () => apply(slider.value)); number.addEventListener('change', () => apply(number.value));
        number.addEventListener('keydown', event => {
            if (event.key === 'Escape') { sync(); number.blur(); }
            if (event.key === 'Enter') { apply(number.value); number.blur(); }
            if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
                event.preventDefault(); apply(Number(number.value) + (event.key === 'ArrowUp' ? 1 : -1) * (event.shiftKey ? .1 : .01));
            }
        });
        row.append(text, slider, number); root.append(row); controls.push({name, slider, number});
    };
    for (const [name, label] of [['pos_x','Horizontal (X)'],['pos_y','Elevation (Y)'],['pos_z','Distance (Z)'],['roll','Roll (R)']]) makeRow(name,label,-1,1);
    const buttons = document.createElement('div'); buttons.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;flex-shrink:0;';
    for (const [label, x] of [['Front',0],['Left',.5],['Right',-.5],['Rear',1]]) {
        const button = document.createElement('button'); button.textContent = label;
        button.title = `Set ${label.toLowerCase()} view relative to the subject. Reset elevation only.`;
        button.addEventListener('click', () => { set('pos_x', x); set('pos_y',0); }); buttons.append(button);
    }
    root.append(buttons);
    const note = document.createElement('div'); note.textContent = 'Conceptual proxy. Tag blending does not guarantee exact angles. Left/right are subject-relative.';
    note.style.cssText = 'font-size:11px;color:#a7bacd;flex-shrink:0;overflow-wrap:anywhere;'; root.append(note);
    const caption = document.createElement('div'); caption.textContent = 'Tag strengths (0–10)'; caption.style.flexShrink = '0'; root.append(caption);
    for (const [name,label] of [['horizontal_view_strength','Horizontal weight'],['vertical_view_strength','Elevation weight'],['zoom_strength','Zoom weight'],['camera_angle_strength','Roll weight']]) makeRow(name,label,0,10);
    function draw() {
        const ctx = canvas.getContext('2d'); if (!ctx) return;
        const width = canvas.clientWidth || 640, height = canvas.clientHeight || 300;
        const pixelRatio = globalThis.devicePixelRatio || 1;
        const backingWidth = Math.round(width * pixelRatio), backingHeight = Math.round(height * pixelRatio);
        if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
            canvas.width = backingWidth; canvas.height = backingHeight;
        }
        ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,canvas.width,canvas.height);
        const scale = Math.min(width / 640, height / 300);
        ctx.setTransform(scale * pixelRatio,0,0,scale * pixelRatio,
            (width - 640 * scale) / 2 * pixelRatio, (height - 300 * scale) / 2 * pixelRatio);
        drawCameraScene(ctx,{x:value('pos_x'),y:value('pos_y'),z:value('pos_z'),roll:value('roll')},view);
    }
    function sync() {
        enabled.checked = widgets.camera_enabled.value !== false;
        for (const {name,slider,number} of controls) { slider.value=value(name); number.value=value(name).toFixed(2); }
        draw();
    }
    enabled.addEventListener('change', () => set('camera_enabled', enabled.checked));
    reset.addEventListener('click', () => { for (const key of ['pos_x','pos_y','pos_z','roll']) set(key,0); });
    let drag = null;
    canvas.addEventListener('pointerdown', event => {
        if (event.button !== 0 && event.button !== 2) return;
        event.preventDefault();
        drag = {x:event.clientX,y:event.clientY,az:value('pos_x'),el:value('pos_y'),inspect:event.button===2||event.altKey,yaw:view.yaw,pitch:view.pitch}; canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
        if (!drag) return;
        const rect=canvas.getBoundingClientRect();
        if (drag.inspect) {
            view.yaw=drag.yaw+(event.clientX-drag.x)/rect.width*6;
            view.pitch=clamp(drag.pitch+(event.clientY-drag.y)/rect.height*2,-1.2,1.2);
            draw();return;
        }
        let az = drag.az + (event.clientX-drag.x)/rect.width*2;
        az = ((az+1)%2+2)%2-1;
        set('pos_x',Math.round(az*100)/100); set('pos_y',Math.round(clamp(drag.el-(event.clientY-drag.y)/rect.height*2,-1,1)*100)/100);
    });
    for (const event of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(event, () => {drag=null;});
    canvas.addEventListener('contextmenu',event=>event.preventDefault());
    canvas.addEventListener('wheel', event => {
        event.preventDefault(); event.stopPropagation();
        const key=event.shiftKey?'roll':'pos_z';
        const direction=event.shiftKey?-1:1;
        set(key,Math.round(clamp(value(key)+direction*Math.sign(event.deltaY)*.05,-1,1)*100)/100);
    },{passive:false});
    const dom = node.addDOMWidget('booru_camera_panel','booru-camera-panel',root,{serialize:false,hideOnZoom:false});
    dom.serialize = false;
    const panelHeight = current => Math.max(220, (current?.size?.[1] || 740) - 110);
    dom.computeSize = width => [width || node.size[0], panelHeight(node)];
    dom.computeLayoutSize = current => {
        const height = panelHeight(current || node);
        return {minHeight: height, maxHeight: height};
    };
    const onResize = node.onResize;
    node.onResize = function () {
        onResize?.apply(this, arguments);
        draw(); this.setDirtyCanvas?.(true,true);
    };
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(draw);
    observer?.observe(canvas);
    const onRemoved = node.onRemoved;
    node.onRemoved = function () { observer?.disconnect(); onRemoved?.apply(this, arguments); };
    node._booruStrengthSync = [sync]; sync();
    node.setSize?.([Math.max(node.size[0],560),Math.max(node.size[1],1000)]);
}
