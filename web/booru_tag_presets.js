import { app } from "../../scripts/app.js";
import { installPresetList } from "./util/booru_preset_list.js";

app.registerExtension({
    name: "toyxyz.BooruPresetStrength",
    beforeRegisterNodeDef(nodeType, data) {
        if (data.name !== "ToyxyzBooruTagPresets") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            installPresetList(this,data);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result=executed?.apply(this,arguments);
            this._booruRandomResult?.(message?.camera_settings?.[0]);
            return result;
        };
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            // Camera remains slot zero, retaining its existing connections.
            for (let i=(this.outputs?.length??0)-1;i>0;i--) this.removeOutput?.(i);
            if (this.outputs?.[0]) this.outputs[0].name = "camera";
            if (this.title === "booru tag preset") this.title = "booru tag camera";
            const panel = this.widgets?.find(w => w.name === "panel_enabled");
            const vertical = this.widgets?.find(w => w.name === "vertical_view");
            if (["Above 45°", "Bird's-eye view"].includes(vertical?.value)) vertical.value = "Above";
            if (["Below 45°", "Worm's-eye view"].includes(vertical?.value)) vertical.value = "Below";
            if (panel) panel.value = false;
            // Compact older camera-panel layouts once; later resized layouts stay intact.
            this.properties ??= {};
            if (this.properties.booruPanelLayoutVersion !== 5) {
                this.setSize?.([Math.max(this.size[0],480),800]);
                this.properties.booruPanelLayoutVersion = 5;
            }
            this._booruMigrateRandom?.();
            this._booruStrengthSync?.forEach(sync => sync());
            return result;
        };
    },
});
