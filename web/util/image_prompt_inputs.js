const widgetOrder = ['prompt','llm_model','seed','prompt_type','enhance','edited_prompt'];
function compareInputs(a,b) {
    const key = input => input.name === 'preset' ? [0,0]
        : input.name === 'builder' ? [0,1]
        : /^image_([1-9]|10)$/.test(input.name) ? [1,Number(input.name.slice(6))]
        : input.name === 'image' ? [2,0]
        : widgetOrder.includes(input.name) ? [3,widgetOrder.indexOf(input.name)] : [4,input.name];
    const x=key(a),y=key(b);
    return x[0]-y[0] || (typeof x[1]==='number' ? x[1]-y[1] : String(x[1]).localeCompare(String(y[1])));
}

function reorderInputs(node) {
    const previous=[...(node.inputs ?? [])],ordered=[...previous].sort(compareInputs);
    if (ordered.every((input,i)=>input===previous[i])) return;
    const links=node.graph?.links;
    // Capture BEFORE changing the array. Modern ComfyUI input.link is a getter
    // based on the slot's CURRENT index, not a link attached to the object.
    const moves=previous.flatMap((input,oldSlot)=>{
        const id=input.link,link=id==null ? null : links?.get?.(id) ?? links?.[id];
        const newSlot=ordered.indexOf(input);
        return link && oldSlot!==newSlot ? [{link,oldSlot,newSlot}] : [];
    });
    // Modern link stores reject swaps into occupied targets. Park moving links
    // at unused indices synchronously, then restore them at their final slots.
    // The same link objects/IDs, origin endpoints and reroutes remain intact.
    const allLinks=links?.values ? [...links.values()] : Object.values(links ?? {});
    const spare=Math.max(previous.length,...allLinks.filter(l=>node.id==null||l.target_id===node.id).map(l=>l.target_slot+1));
    moves.forEach(({link},i)=>{link.target_slot=spare+i;});
    node.inputs.splice(0,node.inputs.length,...ordered);
    moves.forEach(({link,newSlot})=>{link.target_slot=newSlot;});
    const floating=node.graph?.floatingLinks;
    const floatingLinks=floating?.values ? [...floating.values()] : Object.values(floating ?? {});
    for(const link of floatingLinks)if(link.target_id===node.id&&previous[link.target_slot])link.target_slot=ordered.indexOf(previous[link.target_slot]);
    // Refresh frontend slot geometry after a change in array order.
    if(node.setSize && node.size)node.setSize([...node.size]);
}

export function syncImageInputs(node) {
    if (node._syncingImageInputs) return;
    node._syncingImageInputs = true;
    try {
        let legacy = node.inputs?.find(x => x.name === "image");
        const first = node.inputs?.find(x => x.name === "image_1");
        if (legacy && (!first || (first.link == null && legacy.link != null))) {
            if (first) node.removeInput(node.inputs.indexOf(first));
            legacy.name = "image_1";
            if (legacy.label) legacy.label = "image_1";
        }
        // Keep conflicting connected legacy inputs visible rather than dropping a link.
        for (let i=(node.inputs?.length ?? 0)-1;i>=0;i--) {
            if (node.inputs[i].name === "image" && node.inputs[i].link == null) node.removeInput(i);
        }
        const numbered = () => (node.inputs ?? []).filter(x => /^image_([1-9]|10)$/.test(x.name));
        const connected = numbered().filter(x => x.link != null);
        const highest = Math.max(0,...connected.map(x => Number(x.name.slice(6))));
        const visible = Math.min(10,highest+1);
        // Remove only trailing unused inputs. Interior holes retain their identities.
        for (let i=(node.inputs?.length ?? 0)-1;i>=0;i--) {
            const input=node.inputs[i];
            if (/^image_([1-9]|10)$/.test(input.name) && Number(input.name.slice(6))>visible && input.link==null) node.removeInput(i);
        }
        for (let i=1;i<=visible;i++) {
            if (!numbered().some(x => x.name === `image_${i}`)) node.addInput(`image_${i}`,"IMAGE");
        }
        reorderInputs(node);
        // Output index 0 remains prompt; image_N always uses output index N.
        // Never remove a linked output, even if its input has been disconnected.
        if (node.outputs && node.addOutput && node.removeOutput) {
            const lastLinked = Math.max(0,...node.outputs.map((output,index) =>
                output.links?.length ? index : 0));
            const outputCount = Math.max(visible,lastLinked);
            while (node.outputs.length > outputCount+1) node.removeOutput(node.outputs.length-1);
            for (let i=node.outputs.length;i<=outputCount;i++) {
                node.addOutput(i === 0 ? "prompt" : `image_${i}`,i === 0 ? "STRING" : "IMAGE");
            }
        }
        node.setDirtyCanvas?.(true,true);
    } finally { node._syncingImageInputs=false; }
}

export function scheduleImageInputs(node, isConfiguring = () => false) {
    if (node._imageInputSyncPending || node._syncingImageInputs) return;
    node._imageInputSyncPending=true;
    const run = () => {
        // Graph loading restores connections and widget inputs in several
        // phases. Do not prune/sort partially hydrated sockets between phases.
        if(isConfiguring()) {setTimeout(run,16);return;}
        node._imageInputSyncPending=false;
        syncImageInputs(node);
    };
    queueMicrotask(run);
}
