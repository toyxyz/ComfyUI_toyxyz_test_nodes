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
    // mask_overlap_method가 추가되어 위젯 인덱스가 1 증가 (4 -> 5)
    node.widgets[i + 5].value = newValue;
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
  
  // canvas_image 출력이 첫 번째에 있어야 함
  // 출력이 없거나 첫 번째 출력이 canvas_image가 아닌 경우
  if (node.outputs.length === 0 || node.outputs[0].name !== "canvas_image") {
    // 기존 출력 제거
    while (node.outputs.length > 0) {
      node.removeOutput(node.outputs.length - 1);
    }
    // canvas_image 출력 추가
    node.addOutput("canvas_image", "IMAGE");
  }
  
  // targetNumber + 1 (canvas_image 포함)만큼 출력이 있어야 함
  const totalOutputs = targetNumber + 1;
  
  // 초과 출력 제거
  while (node.outputs.length > totalOutputs) {
    node.removeOutput(node.outputs.length - 1);
  }
  
  // 부족한 출력 추가 (canvas_image 다음부터)
  while (node.outputs.length < totalOutputs) {
    const areaIndex = node.outputs.length - 1; // canvas_image를 제외한 인덱스
    node.addOutput(`area_${areaIndex}`, "MASK");
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

      // mask_overlap_method는 Python에서 required로 정의되어 자동 생성됨
      // 따라서 여기서는 별도로 추가하지 않음

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
