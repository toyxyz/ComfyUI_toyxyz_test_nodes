import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { readEditResponse } from "./util/image_prompt_response.js";

// Status labels are refreshed without replacing a user's unsupported legacy
// local-model selection silently. The backend accepts either old status label.
const type = "ToyxyzImagePrompter";
const modelName = "Qwen3.8-27B Uncensored Q4_K_M";
const known = new Set([modelName, `${modelName} — Installed`, `${modelName} — Not installed`,
    `${modelName} — 설치됨`, `${modelName} — 설치 안 됨`]);
let pending;
function installEditor(node) {
    if (node._promptEditor) return;
    node._promptEditor = true;
    const override = node.widgets?.find(w => w.name === "edited_prompt");
    if (!override) return;
    override.type = "hidden";
    override.computeSize = () => [0, -4];
    if (override.inputEl) override.inputEl.style.display = "none";
    const box = document.createElement("div");
    box.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;width:100%;height:100%;";
    const output = document.createElement("textarea");
    output.readOnly = true;
    output.placeholder = "Generated prompt";
    output.style.cssText = "flex:1;min-height:90px;width:100%;box-sizing:border-box;resize:none;background:#222;color:#ddd;";
    const status = document.createElement("small");
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;gap:6px;flex-wrap:wrap";
    box.append(output, controls, status);
    node.addDOMWidget("prompt_editor", "div", box, {serialize:false, hideOnZoom:false});
    let busy = false;
    const show = text => {
        output.value = text ?? "";
        node.properties ??= {};
        node.properties.generated_prompt = output.value;
        status.textContent = override.value ? "Edited output active. Input changes do not replace it; use Regenerate from inputs." : "Generated output";
        node.setDirtyCanvas?.(true,true);
    };
    const button = (label, tooltip, action) => {
        const b = document.createElement("button"); b.textContent = label;
        b.title = tooltip;
        b.onclick = action; controls.append(b); return b;
    };
    button("Edit Prompt", "Edit the current output with Qwen while preserving details unrelated to your request. Generate a prompt first. The edited text becomes the output on the next workflow run; image generation is not started automatically.", () => {
        if (busy || !output.value) return;
        const dialog = document.createElement("dialog");
        const input = document.createElement("textarea");
        input.placeholder = "Describe what to change. Other details will be preserved.";
        input.style.cssText = "display:block;width:min(520px,75vw);height:130px;margin-bottom:8px";
        const ok = document.createElement("button"); ok.textContent = "OK";
        ok.title = "Apply your edit request to the current prompt. The output display updates when editing succeeds.";
        const cancel = document.createElement("button"); cancel.textContent = "Cancel";
        cancel.title = "Close this dialog and keep the current prompt. If editing is in progress, its result is discarded; server inference may continue.";
        const error = document.createElement("div");
        dialog.append(input,ok,cancel,error); document.body.append(dialog); dialog.showModal(); input.focus();
        dialog.addEventListener("close",()=>dialog.remove());
        cancel.onclick = () => dialog.close();
        ok.onclick = async () => {
            if (!input.value.trim() || busy) return;
            busy = true; ok.disabled = true; input.disabled = true;
            const original = output.value;
            const originalOverride = override.value;
            status.textContent = "Editing with Qwen…";
            try {
                const value = name => node.widgets?.find(w=>w.name===name)?.value;
                const response = await api.fetchApi("/toyxyz/image-prompter/edit", {
                    method:"POST",headers:{"Content-Type":"application/json"},
                    body:JSON.stringify({current_prompt:original,instruction:input.value,llm_model:value("llm_model"),seed:value("seed")})
                });
                const result = await readEditResponse(response);
                // Closing the dialog discards the response; a concurrent execution wins.
                if (!dialog.open || output.value !== original || override.value !== originalOverride) return;
                node.properties.prompt_undo = original;
                override.value = result.prompt;
                show(result.prompt);
                dialog.close();
            } catch (e) { error.textContent = e.message; }
            finally { busy = false; ok.disabled = false; input.disabled = false; show(output.value); }
        };
    });
    button("Undo", "Restore the prompt from before the last edit (one step). The restored text becomes the output on the next workflow run.", () => {
        if (busy || !node.properties?.prompt_undo) return;
        override.value = node.properties.prompt_undo;
        delete node.properties.prompt_undo;
        show(override.value);
    });
    button("Regenerate from inputs", "Clear the edited-output override. Run the workflow to use the original prompt, image, and preset settings again. This button does not queue execution. Identical inputs may use cached results; change the seed for a new result.", () => {
        if (busy) return;
        override.value = "";
        status.textContent = "Edit override cleared. Queue the workflow to generate from inputs.";
        node.setDirtyCanvas?.(true,true);
    });
    const executed = node.onExecuted;
    node.onExecuted = function(message) {
        executed?.apply(this,arguments);
        if (message?.text) show(message.text.join("\n"));
    };
    const configure = node.onConfigure;
    node.onConfigure = function() { configure?.apply(this,arguments); show(override.value || node.properties?.generated_prompt || ""); };
    show(override.value || node.properties?.generated_prompt || "");
    node.setSize([Math.max(node.size[0],440),Math.max(node.size[1],620)]);
}
function migratePromptType(node) {
    const widget = node.widgets?.find(w => w.name === "prompt_type" || w.name === "target_model");
    if (widget) {
        widget.name = "prompt_type";
        if (widget.value === "krea") widget.value = "normal";
    }
    // Converted widgets can have named input slots as well as widget values.
    for (const input of node.inputs ?? []) {
        if (input.name === "target_model") input.name = "prompt_type";
        if (input.widget?.name === "target_model") input.widget.name = "prompt_type";
    }
}
async function refresh(nodes) {
    if (!nodes.length) return;
    try {
        pending ??= api.fetchApi(`/object_info/${type}`).then(async response => {
            if (!response.ok) throw new Error(`Model status: HTTP ${response.status}`);
            return response.json();
        }).finally(() => { pending = undefined; });
        const info = await pending;
        const values = info[type]?.input?.required?.llm_model?.[0];
        if (!Array.isArray(values) || !values.length) return;
        for (const node of nodes) {
            const widget = node.widgets?.find(w => w.name === "llm_model");
            if (!widget) continue;
            widget.options.values = [...values];
            if (known.has(widget.value)) widget.value = values[0];
            node.setDirtyCanvas?.(true, true);
        }
    } catch (error) {
        console.warn("image prompter: could not refresh model installation status", error);
    }
}

app.registerExtension({
    name: "toyxyz.image_prompter.model_status",
    nodeCreated(node) {
        if (node.comfyClass === type || node.type === type) {
            migratePromptType(node);
            installEditor(node);
            void refresh([node]);
        }
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== type) return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            migratePromptType(this);
            return result;
        };
    },
    setup() {
        api.addEventListener("execution_success", () => {
            void refresh((app.graph?._nodes ?? []).filter(node => node.comfyClass === type || node.type === type));
        });
    },
});
