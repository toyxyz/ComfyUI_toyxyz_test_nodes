// Keep edit state scoped to the selected writer mode. Loading a saved workflow
// establishes a baseline; only an actual subsequent mode change clears it.
export function watchPromptMode(node, onReset) {
    const mode = node.widgets?.find(w => w.name === "prompt_type");
    const override = node.widgets?.find(w => w.name === "edited_prompt");
    let previous = mode?.value;
    let revision = 0;
    const callback = mode?.callback;
    if (mode) mode.callback = function(value, ...args) {
        const result = callback?.call(this,value,...args);
        if (value !== previous) {
            previous = value;
            revision++;
            if (override) override.value = "";
            node.properties ??= {};
            delete node.properties.prompt_undo;
            delete node.properties.generated_prompt;
            onReset();
            node.setDirtyCanvas?.(true,true);
        }
        return result;
    };
    return {
        revision: () => revision,
        configured: () => { previous = mode?.value; revision++; },
    };
}
