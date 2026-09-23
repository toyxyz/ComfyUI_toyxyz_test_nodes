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
        // Dynamic inputs are appended by LiteGraph. Keep the preset first and
        // references together, including after loading older mixed-order graphs.
        const rank = input => input.name === "preset" ? 0
            : /^image_([1-9]|10)$/.test(input.name) ? Number(input.name.slice(6))
            : input.name === "image" ? 11 : 12;
        node.inputs?.sort((a,b) => rank(a)-rank(b));
        // Links store a numeric target slot; moving only the visual inputs would
        // otherwise reconnect existing wires to the wrong input on serialization.
        node.inputs?.forEach((input,index) => {
            if (input.link == null) return;
            const links = node.graph?.links;
            const link = links?.get?.(input.link) ?? links?.[input.link];
            if (link) link.target_slot = index;
        });
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

export function scheduleImageInputs(node) {
    if (node._imageInputSyncPending) return;
    node._imageInputSyncPending=true;
    queueMicrotask(() => {node._imageInputSyncPending=false;syncImageInputs(node);});
}
