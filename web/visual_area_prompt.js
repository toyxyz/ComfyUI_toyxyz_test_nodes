import { app } from "../../scripts/app.js";
import { addAreaGraphWidget, refreshAreaGraphWidget } from "./widgets/graph_widget.js";
import { addNumberInput } from "./util/util.js";

const _ID = "VisualAreaMask";
const _AREA_DEFAULTS = [0.0, 0.0, 1.0, 1.0, 1.0];
const _PREVIEW_PROPERTY_NAMES = ["image_width", "image_height", "area_number", "mask_overlap_method", "font_size"];


function normalizeAreaValue(areaValue) {
  if (!Array.isArray(areaValue)) {
    return [..._AREA_DEFAULTS];
  }

  return _AREA_DEFAULTS.map((defaultValue, i) => areaValue[i] ?? defaultValue);
}

function syncPreviewProperties(node) {
  if (!node.properties) {
    node.properties = {};
  }

  _PREVIEW_PROPERTY_NAMES.forEach((name) => {
    const widget = node.widgets?.find((entry) => entry.name === name);
    if (widget?.value !== undefined) {
      node.properties[name] = widget.value;
    }
  });
}


function updateWidgetValues(node) {
  syncPreviewProperties(node);

  if (!Array.isArray(node.properties["area_values"])) {
    node.properties["area_values"] = [];
  }
  node.properties["area_values"][node.index] = normalizeAreaValue(node.properties["area_values"][node.index]);
  const areaValues = node.properties["area_values"][node.index];

  // 위젯 이름으로 찾아서 업데이트 (인덱스 대신 이름 사용으로 안정성 향상)
  const widgetNames = ["x", "y", "width", "height", "strength"];
  widgetNames.forEach((name, i) => {
    const newValue = areaValues[i] ?? _AREA_DEFAULTS[i];
    node.properties["area_values"][node.index][i] = newValue;

    const widget = node.widgets.find(w => w.name === name);
    if (widget) {
      widget.value = newValue;
    }
  });

  refreshAreaGraphWidget(node);
}

function updateAreaIdAndInputs(node) {
  syncPreviewProperties(node);

  const countDynamicInputs = node.widgets.find(w => w.name === "area_number").value;
  const newMaxIdx = Math.max(countDynamicInputs - 1, 0);
  const areaIdWidget = node.widgets.find(w => w.name === "area_id");
  if(areaIdWidget) {
      areaIdWidget.options.max = newMaxIdx;
      
      if (areaIdWidget.value > newMaxIdx) {
        areaIdWidget.value = newMaxIdx;
        node.index = newMaxIdx;
      }
  }
  
  if (!node.properties["area_values"]) {
    node.properties["area_values"] = [];
  }

  node.properties["area_values"] = Array.from({ length: countDynamicInputs }, (_, index) =>
    normalizeAreaValue(node.properties["area_values"][index])
  );
  
  updateWidgetValues(node);
  node?.graph?.setDirtyCanvas(true);
  refreshAreaGraphWidget(node);
}

function updateOutputs(node) {
  syncPreviewProperties(node);

  const targetNumber = node.widgets.find(w => w.name === "area_number").value;
  if (!node.outputs) {
    node.outputs = [];
  }

  // canvas_image와 combined_mask 출력이 첫 번째, 두 번째에 있어야 함
  if (node.outputs.length === 0 ||
      node.outputs[0].name !== "canvas_image" ||
      node.outputs.length < 2 ||
      node.outputs[1].name !== "combined_mask") {
    // 기존 출력 제거
    while (node.outputs.length > 0) {
      node.removeOutput(node.outputs.length - 1);
    }
    // canvas_image 출력 추가
    node.addOutput("canvas_image", "IMAGE");
    // combined_mask 출력 추가
    node.addOutput("combined_mask", "MASK");
  }

  // targetNumber + 2 (canvas_image + combined_mask 포함)만큼 출력이 있어야 함
  const totalOutputs = targetNumber + 2;

  // 초과 출력 제거
  while (node.outputs.length > totalOutputs) {
    node.removeOutput(node.outputs.length - 1);
  }

  // 부족한 출력 추가 (canvas_image, combined_mask 다음부터)
  while (node.outputs.length < totalOutputs) {
    const areaIndex = node.outputs.length - 2; // canvas_image, combined_mask를 제외한 인덱스
    node.addOutput(`area_${areaIndex}`, "MASK");
  }
}

function updateInputs(node) {
  syncPreviewProperties(node);

  const targetNumber = node.widgets.find(w => w.name === "area_number").value;
  if (!node.inputs) {
    node.inputs = [];
  }

  // 현재 area_text 인풋 개수 계산
  let currentTextInputs = 0;
  for (let i = 0; i < node.inputs.length; i++) {
    if (node.inputs[i].name.startsWith("area_") && node.inputs[i].name.endsWith("_text")) {
      currentTextInputs++;
    }
  }

  // 초과 인풋 제거
  while (currentTextInputs > targetNumber) {
    currentTextInputs--;
    const inputName = `area_${currentTextInputs}_text`;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
      if (node.inputs[i].name === inputName) {
        node.removeInput(i);
        break;
      }
    }
  }

  // 부족한 인풋 추가
  for (let i = currentTextInputs; i < targetNumber; i++) {
    node.addInput(`area_${i}_text`, "STRING");
  }

  node.graph?.setDirtyCanvas(true);
  refreshAreaGraphWidget(node);
}

app.registerExtension({
  name: 'fuwuffy.' + _ID,
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== _ID) {
      return;
    }

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function(info) {
      if (onConfigure) {
        onConfigure.apply(this, arguments);
      }
      
      const self = this;
      requestAnimationFrame(() => {
        if (info && info.properties && info.properties.area_values) {
            self.setProperty("area_values", info.properties.area_values);
        }

        syncPreviewProperties(self);

        const areaWidget = self.widgets.find(w => w.name === "area_number");
        if(areaWidget) {
            const countDynamicInputs = areaWidget.value;
            const newMaxIdx = Math.max(countDynamicInputs - 1, 0);
            const areaIdWidget = self.widgets.find(w => w.name === "area_id");
            if(areaIdWidget) {
                areaIdWidget.options.max = newMaxIdx;
            }
        }

        updateOutputs(self);
        updateInputs(self);
        updateAreaIdAndInputs(self);
        refreshAreaGraphWidget(self);
      });
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function() {
      const me = onNodeCreated?.apply(this);
      this.index = 0;
      this.setProperty("area_values", [[..._AREA_DEFAULTS]]);
      syncPreviewProperties(this);

      ["image_width", "image_height"].forEach(name => {
        const widget = this.widgets.find(elt => elt.name == name);
        const origCallback = widget?.callback;
        widget.callback = (value, canvas, node, pos, event) => {
          const targetNode = node || this;
          origCallback?.call(widget, value, canvas, targetNode, pos, event);
          syncPreviewProperties(targetNode);
          targetNode.properties[name] = value;
          refreshAreaGraphWidget(targetNode);
        };
      });

      // mask_overlap_method는 Python에서 required로 정의되어 자동 생성됨
      // 따라서 여기서는 별도로 추가하지 않음

      addNumberInput(this, "area_id", 0, (value, _, node) => {
        node.index = value;
        updateWidgetValues(node);
      }, { min: 0, max: 0, step: 10, precision: 0 });

      ["x", "y", "width", "height", "strength"].forEach((name, i) => {
        addNumberInput(this, name, [..._AREA_DEFAULTS][i], (value, _, node) => {
          node.properties["area_values"][node.index][i] = value;
          refreshAreaGraphWidget(node);
        }, { min: 0, max: i === 4 ? 1 : 1, step: 0.1, precision: 2 });
      });

      const areaNumberWidget = this.widgets.find(w => w.name === "area_number");
      if (areaNumberWidget) {
        const origCallback = areaNumberWidget.callback;
        areaNumberWidget.callback = (value, canvas, _node, pos, event) => {
          const targetNode = _node || this;
          if (origCallback) origCallback.call(areaNumberWidget, value, canvas, targetNode, pos, event);
          syncPreviewProperties(targetNode);
          updateOutputs(targetNode);
          updateInputs(targetNode);
          updateAreaIdAndInputs(targetNode);
          refreshAreaGraphWidget(targetNode);
        };
      }

      this.addWidget("button", "Update outputs", null, () => {
        updateOutputs(this);
        updateInputs(this);
        updateAreaIdAndInputs(this);
      });

      addAreaGraphWidget(app, this, "area_conditioning_canvas");

      updateAreaIdAndInputs(this);
      updateOutputs(this);
      updateInputs(this);
      refreshAreaGraphWidget(this);
      return me;
    };
    return nodeType;
  }
});
