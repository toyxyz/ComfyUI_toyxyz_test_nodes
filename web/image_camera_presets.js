import {app} from "../../scripts/app.js";

// v1 widget order was [mixed preset, optional shot]; v2 is [shot_size, angle].
export function migratePresetWidgets(values) {
    if (!Array.isArray(values) || values.length < 2) return null;
    const shots = new Set(["Extreme wide shot", "Wide shot", "Full body shot", "Cowboy shot", "Medium shot", "Medium close-up", "Close-up", "Extreme close-up"]);
    const [preset, shot] = values;
    const legacy = shot === "From preset / user prompt" || shots.has(shot);
    if (!legacy) {
        const renamed = values.map(v => v === "From user prompt" ? "None" : v);
        return renamed.some((v, i) => v !== values[i]) ? renamed : null;
    }
    return [shots.has(shot) ? shot : shots.has(preset) ? preset : "None",
        shots.has(preset) || preset === "From user prompt" ? "None" : preset];
}

export function migratePresetPorts(node, type) {
    if (type === "ToyxyzImagePrompter") {
        for (const input of node.inputs ?? []) {
            if (input.name === "camera") input.name = "preset";
            if (input.label === "camera") input.label = "preset";
        }
    } else if (type === "ToyxyzImageCameraPresets") {
        for (const output of node.outputs ?? []) {
            if (output.name === "camera") output.name = "preset";
            if (output.name === "camera_prompt") output.name = "preset_prompt";
            if (output.label === "camera") output.label = "preset";
            if (output.label === "camera_prompt") output.label = "preset_prompt";
        }
        if (node.title === "image camera presets") node.title = "image prompter preset";
    }
}

export function configureStyleFilter(node, groups, restore = false) {
    const category = node.widgets?.find(w => w.name === "style_category");
    const style = node.widgets?.find(w => w.name === "style");
    if (!category || !style) return;
    const all = ["None", ...Object.values(groups).flat()];
    if (restore && category.value !== "All" && style.value !== "None" && !groups[category.value]?.includes(style.value)) {
        category.value = Object.keys(groups).find(key => groups[key].includes(style.value)) ?? "All";
    }
    if (category.value !== "All" && !groups[category.value]) category.value = "All";
    style.options ??= {};
    style.options.values = category.value === "All" ? all : ["None", ...groups[category.value]];
    if (!style.options.values.includes(style.value)) style.value = "None";
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "toyxyz.ImageCameraPresets",
    beforeRegisterNodeDef(nodeType, data) {
        if (!["ToyxyzImageCameraPresets", "ToyxyzImagePrompter"].includes(data.name)) return;
        const groups = data.input?.required?.style?.[1]?.style_categories ?? {};
        if (data.name === "ToyxyzImageCameraPresets") {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = created?.apply(this, arguments);
                const category = this.widgets?.find(w => w.name === "style_category");
                if (category) {
                    const callback = category.callback;
                    category.callback = (...args) => {
                        callback?.apply(category, args);
                        configureStyleFilter(this, groups);
                    };
                }
                configureStyleFilter(this, groups, true);
                return result;
            };
        }
        const original = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            original?.apply(this, arguments);
            migratePresetPorts(this, data.name);
            if (data.name !== "ToyxyzImageCameraPresets") return;
            const migrated = migratePresetWidgets(info?.widgets_values);
            if (migrated) {
                for (const [i, name] of ["shot_size", "angle"].entries()) {
                    const widget = this.widgets?.find(w => w.name === name);
                    if (widget) widget.value = migrated[i];
                }
            }
            // v3 had [shot_size, angle, style]; v4 inserts category before style.
            const values = info?.widgets_values;
            if (Array.isArray(values) && values.length <= 3) {
                const style = this.widgets?.find(w => w.name === "style");
                const category = this.widgets?.find(w => w.name === "style_category");
                if (style) style.value = values[2] ?? "None";
                if (category) category.value = Object.keys(groups).find(key => groups[key].includes(values[2])) ?? "All";
            }
            configureStyleFilter(this, groups, true);
        };
    },
});
