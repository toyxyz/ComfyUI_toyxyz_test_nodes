import { app } from "../../scripts/app.js";
import { addAreaGraphWidget } from "./widgets/graph_widget.js";
import { addNumberInput } from "./util/util.js";

const _ID = "VisualAreaMask";
const _AREA_DEFAULTS = [0.0, 0.0, 1.0, 1.0, 1.0];


function updateWidgetValues(node) {
  if (!node.properties["area_values"][node.index]) {
    node.properties["area_values"][node.index] = [];
  }
  const areaValues = node.properties["area_values"][node.index];
  [..._AREA_DEFAULTS].forEach((value, i) => {
    const newValue = areaValues[i] || value;
    node.properties["area_values"][node.index][i] = newValue;
    node.widgets[i + 4].value = newValue;
  });
}

function updateAreaIdAndInputs(node) {
  const countDynamicInputs = node.widgets.find(w => w.name === "area_number").value;
  const newMaxIdx = Math.max(countDynamicInputs - 1, 0);
  const areaIdWidget = node.widgets.find(w => w.name === "area_id");
  areaIdWidget.options.max = newMaxIdx;
  areaIdWidget.value = newMaxIdx;
  node.index = newMaxIdx;
  updateWidgetValues(node);
  node.properties["area_values"] = node.properties["area_values"].slice(0, countDynamicInputs);
  node?.graph?.setDirtyCanvas(true);
}

function updateOutputs(node) {
  const targetNumber = node.widgets.find(w => w.name === "area_number").value;
  if (!node.outputs) {
    node.outputs = [];
  }
  while (node.outputs.length > targetNumber) {
    node.removeOutput(node.outputs.length - 1);
  }
  while (node.outputs.length < targetNumber) {
    node.addOutput(`area_${node.outputs.length}`, node._type);
  }
}

app.registerExtension({
  name: 'fuwuffy.' + _ID,
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== _ID) {
      return;
    }
    
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function() {
      const me = onNodeCreated?.apply(this);
      this.index = 0;
      this.setProperty("area_values", [..._AREA_DEFAULTS]);

      ["image_width", "image_height"].forEach(name => {
        const widget = this.widgets.find(elt => elt.name == name);
        widget.callback = (value, _, node) => {
          node.properties[name] = value;
        };
      });

      addAreaGraphWidget(app, this, "area_conditioning_canvas");
      addNumberInput(this, "area_id", 0, (value, _, node) => {
        node.index = value;
        updateWidgetValues(node);
      }, { min: 0, max: 0, step: 10, precision: 0 });

      ["x", "y", "width", "height", "strength"].forEach((name, i) => {
        addNumberInput(this, name, [..._AREA_DEFAULTS][i], (value, _, node) => {
          node.properties["area_values"][node.index][i] = value;
        }, { min: 0, max: i === 4 ? 1 : 1, step: 0.1, precision: 2 });
      });

      this._type = "MASK";

      this.addWidget("button", "Update outputs", null, () => {
        updateOutputs(this);
        updateAreaIdAndInputs(this);
      });

      updateAreaIdAndInputs(this);
      updateOutputs(this);
      return me;
    };
    return nodeType;
  }
});
