import { app } from "../../scripts/app.js";

const NODE_ID = "DrawAreaMask";
const BOXES_PROPERTY = "draw_area_boxes";
const BOXES_WIDGET_NAME = "boxes_state";
const WIDTH_PROPERTY = "draw_area_width";
const HEIGHT_PROPERTY = "draw_area_height";
const MAX_MASK_OUTPUTS = 32;
const HIDDEN_WIDGET_TYPE = "draw-area-mask-hidden";
const PREVIEW_MIN_HEIGHT = 220;
const PREVIEW_MAX_HEIGHT = 1120;
const PREVIEW_MARGIN = 10;
const BOX_MIN_PIXELS = 6;
const FALLBACK_ASPECT_RATIO = 1;

function ensureNodeProperties(node) {
  if (!node.properties) {
    node.properties = {};
  }

  if (!Array.isArray(node.properties[BOXES_PROPERTY])) {
    node.properties[BOXES_PROPERTY] = [];
  }

  return node.properties;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeColorString(color, fallback) {
  const normalized = typeof color === "string" ? color.trim() : "";
  return normalized ? normalized : fallback;
}

function readCssVariable(...names) {
  if (typeof window === "undefined" || !window.getComputedStyle) {
    return "";
  }

  const styles = [
    document.documentElement,
    document.body,
  ].filter(Boolean).map((element) => window.getComputedStyle(element));

  for (const name of names) {
    for (const style of styles) {
      const value = style.getPropertyValue(name)?.trim();
      if (value) {
        return value;
      }
    }
  }

  return "";
}

function getThemeColors() {
  const liteGraph = globalThis.LiteGraph;
  const cssBackground = readCssVariable("--comfy-input-bg", "--content-bg", "--bg-color");
  return {
    background: normalizeColorString(cssBackground || liteGraph?.WIDGET_BGCOLOR, "hsl(0, 0%, 15%)"),
  };
}

function normalizeBox(rawBox) {
  if (!rawBox || typeof rawBox !== "object") {
    return null;
  }

  const x = clamp(Number(rawBox.x ?? 0), 0, 1);
  const y = clamp(Number(rawBox.y ?? 0), 0, 1);
  const width = clamp(Number(rawBox.width ?? 0), 0, 1);
  const height = clamp(Number(rawBox.height ?? 0), 0, 1);
  const hue = ((Number(rawBox.hue ?? 0) % 360) + 360) % 360;

  const x2 = clamp(x + width, x, 1);
  const y2 = clamp(y + height, y, 1);
  const normalizedWidth = x2 - x;
  const normalizedHeight = y2 - y;

  if (normalizedWidth <= 0 || normalizedHeight <= 0) {
    return null;
  }

  return {
    x,
    y,
    width: normalizedWidth,
    height: normalizedHeight,
    hue,
  };
}

function normalizeBoxes(boxes) {
  if (!Array.isArray(boxes)) {
    return [];
  }

  return boxes.map(normalizeBox).filter(Boolean);
}

function parseBoxesState(value) {
  if (typeof value !== "string") {
    return [];
  }

  try {
    return normalizeBoxes(JSON.parse(value));
  } catch {
    return [];
  }
}

function getBoxes(node) {
  const properties = ensureNodeProperties(node);
  properties[BOXES_PROPERTY] = normalizeBoxes(properties[BOXES_PROPERTY]);
  return properties[BOXES_PROPERTY];
}

function getBoxesWidget(node) {
  return node.widgets?.find((widget) => widget.name === BOXES_WIDGET_NAME);
}

function syncBoxesWidget(node) {
  const boxesWidget = getBoxesWidget(node);
  if (!boxesWidget) {
    return;
  }

  boxesWidget.value = JSON.stringify(getBoxes(node));
}

function hideBoxesWidget(node) {
  const boxesWidget = getBoxesWidget(node);
  if (!boxesWidget || boxesWidget.__drawAreaMaskHidden) {
    return;
  }

  boxesWidget.__drawAreaMaskHidden = true;
  boxesWidget.hidden = true;
  boxesWidget.origType = boxesWidget.type;
  boxesWidget.origComputeSize = boxesWidget.computeSize;
  boxesWidget.type = HIDDEN_WIDGET_TYPE;
  boxesWidget.computeSize = () => [0, -4];
}

function setBoxes(node, boxes) {
  const properties = ensureNodeProperties(node);
  properties[BOXES_PROPERTY] = normalizeBoxes(boxes);
  node.setProperty?.(BOXES_PROPERTY, properties[BOXES_PROPERTY]);
  syncBoxesWidget(node);
  updateOutputs(node);
  node.graph?.setDirtyCanvas?.(true, true);
  refreshPreview(node);
}

function updateOutputs(node) {
  if (!node.outputs) {
    node.outputs = [];
  }

  const boxCount = getBoxes(node).length;
  const totalOutputs = boxCount + 1;

  while (node.outputs.length > totalOutputs) {
    node.removeOutput(node.outputs.length - 1);
  }

  while (node.outputs.length < totalOutputs) {
    if (node.outputs.length === 0) {
      node.addOutput("canvas_image", "IMAGE");
    } else {
      node.addOutput(`mask_${node.outputs.length - 1}`, "MASK");
    }
  }

  if (node.outputs[0]) {
    node.outputs[0].name = "canvas_image";
    node.outputs[0].type = "IMAGE";
  }

  for (let i = 1; i < node.outputs.length; i += 1) {
    node.outputs[i].name = `mask_${i - 1}`;
    node.outputs[i].type = "MASK";
  }
}

function getNumericWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function syncSizeProperties(node) {
  const properties = ensureNodeProperties(node);
  const widthWidget = getNumericWidget(node, "width");
  const heightWidget = getNumericWidget(node, "height");

  if (widthWidget?.value !== undefined) {
    properties[WIDTH_PROPERTY] = Number(widthWidget.value);
  }

  if (heightWidget?.value !== undefined) {
    properties[HEIGHT_PROPERTY] = Number(heightWidget.value);
  }
}

function getCanvasAspectRatio(node) {
  const properties = ensureNodeProperties(node);
  const width = Number(properties[WIDTH_PROPERTY] || 0);
  const height = Number(properties[HEIGHT_PROPERTY] || 0);

  if (width > 0 && height > 0) {
    return width / height;
  }

  return FALLBACK_ASPECT_RATIO;
}

function getPreviewHeight(node, width) {
  const safeWidth = Math.max(220, Math.round(width || node?.size?.[0] || 320));
  const innerWidth = Math.max(1, safeWidth - PREVIEW_MARGIN * 2);
  const aspectRatio = getCanvasAspectRatio(node);
  const rawHeight = Math.round(innerWidth / aspectRatio + PREVIEW_MARGIN * 2);
  return clamp(rawHeight, PREVIEW_MIN_HEIGHT, PREVIEW_MAX_HEIGHT);
}

function getPreviewState(node) {
  if (!node.__drawAreaMaskState) {
    node.__drawAreaMaskState = {
      drawScheduled: false,
      dragBox: null,
      dragStartPoint: null,
      canvasRect: null,
    };
  }

  return node.__drawAreaMaskState;
}

function getCanvasRect(canvas, node) {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const logicalWidth = canvas.width > 0 ? canvas.width / devicePixelRatio : (canvas.clientWidth || 320);
  const logicalHeight = canvas.height > 0 ? canvas.height / devicePixelRatio : (canvas.clientHeight || PREVIEW_MIN_HEIGHT);
  const canvasWidth = Math.max(1, Number(ensureNodeProperties(node)[WIDTH_PROPERTY] || 1));
  const canvasHeight = Math.max(1, Number(ensureNodeProperties(node)[HEIGHT_PROPERTY] || 1));
  const availableWidth = Math.max(1, logicalWidth - PREVIEW_MARGIN * 2);
  const availableHeight = Math.max(1, logicalHeight - PREVIEW_MARGIN * 2);
  const scale = Math.min(availableWidth / canvasWidth, availableHeight / canvasHeight);
  const drawWidth = canvasWidth * scale;
  const drawHeight = canvasHeight * scale;

  return {
    x: PREVIEW_MARGIN + (availableWidth - drawWidth) / 2,
    y: PREVIEW_MARGIN + (availableHeight - drawHeight) / 2,
    width: drawWidth,
    height: drawHeight,
  };
}

function toCanvasBox(box, canvasRect) {
  return {
    x: canvasRect.x + box.x * canvasRect.width,
    y: canvasRect.y + box.y * canvasRect.height,
    width: box.width * canvasRect.width,
    height: box.height * canvasRect.height,
  };
}

function hueBorderColor(hue) {
  return `hsla(${hue}, 90%, 65%, 0.95)`;
}

function hueFillColor(hue) {
  return `hsla(${hue}, 90%, 58%, 0.26)`;
}

function drawLabel(ctx, label, boxRect) {
  if (boxRect.width < 18 || boxRect.height < 18) {
    return;
  }

  ctx.save();
  ctx.font = "bold 18px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.9)";
  ctx.strokeText(label, boxRect.x + boxRect.width / 2, boxRect.y + boxRect.height / 2);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, boxRect.x + boxRect.width / 2, boxRect.y + boxRect.height / 2);
  ctx.restore();
}

function drawPreview(canvas, node) {
  const state = getPreviewState(node);
  const { background } = getThemeColors();
  const devicePixelRatio = window.devicePixelRatio || 1;
  const cssWidth = Math.max(220, canvas.parentElement?.clientWidth || node?.size?.[0] || 320);
  const cssHeight = getPreviewHeight(node, cssWidth);
  const pixelWidth = Math.max(1, Math.round(cssWidth * devicePixelRatio));
  const pixelHeight = Math.max(1, Math.round(cssHeight * devicePixelRatio));

  canvas.style.width = "100%";
  canvas.style.height = `${cssHeight}px`;
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.scale(devicePixelRatio, devicePixelRatio);

  ctx.fillStyle = background;
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const canvasRect = getCanvasRect(canvas, node);
  state.canvasRect = canvasRect;

  ctx.fillStyle = background;
  ctx.fillRect(canvasRect.x, canvasRect.y, canvasRect.width, canvasRect.height);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(canvasRect.x, canvasRect.y, canvasRect.width, canvasRect.height);

  const boxes = getBoxes(node);
  boxes.forEach((box, index) => {
    const boxRect = toCanvasBox(box, canvasRect);
    ctx.fillStyle = hueFillColor(box.hue);
    ctx.fillRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    ctx.strokeStyle = hueBorderColor(box.hue);
    ctx.lineWidth = 2;
    ctx.strokeRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    drawLabel(ctx, String(index), boxRect);
  });

  if (state.dragBox) {
    const boxRect = toCanvasBox(state.dragBox, canvasRect);
    ctx.fillStyle = hueFillColor(state.dragBox.hue);
    ctx.fillRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    ctx.strokeStyle = hueBorderColor(state.dragBox.hue);
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    ctx.setLineDash([]);
    drawLabel(ctx, String(boxes.length), boxRect);
  }

  ctx.fillStyle = "rgba(0, 0, 0, 0.58)";
  ctx.fillRect(canvasRect.x, canvasRect.y + canvasRect.height - 24, canvasRect.width, 24);
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.font = "12px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(
    `Ctrl+drag: add box   Alt+click: remove box   ${Math.round(ensureNodeProperties(node)[WIDTH_PROPERTY] || 0)} x ${Math.round(ensureNodeProperties(node)[HEIGHT_PROPERTY] || 0)}`,
    canvasRect.x + canvasRect.width / 2,
    canvasRect.y + canvasRect.height - 12
  );
}

function refreshPreview(node) {
  const state = getPreviewState(node);
  if (!node.__drawAreaMaskWidget?.drawPreview || state.drawScheduled) {
    return;
  }

  state.drawScheduled = true;
  requestAnimationFrame(() => {
    state.drawScheduled = false;
    node.__drawAreaMaskWidget.drawPreview();
  });
}

function getLocalPoint(canvas, event) {
  const bounds = canvas.getBoundingClientRect();
  const devicePixelRatio = window.devicePixelRatio || 1;
  const logicalWidth = canvas.width > 0 ? canvas.width / devicePixelRatio : (canvas.clientWidth || 1);
  const logicalHeight = canvas.height > 0 ? canvas.height / devicePixelRatio : (canvas.clientHeight || 1);
  const scaleX = bounds.width > 0 ? logicalWidth / bounds.width : 1;
  const scaleY = bounds.height > 0 ? logicalHeight / bounds.height : 1;
  return {
    x: (event.clientX - bounds.left) * scaleX,
    y: (event.clientY - bounds.top) * scaleY,
  };
}

function isPointInRect(point, rect) {
  return (
    rect &&
    point.x >= rect.x &&
    point.x <= rect.x + rect.width &&
    point.y >= rect.y &&
    point.y <= rect.y + rect.height
  );
}

function clampPointToRect(point, rect) {
  return {
    x: clamp(point.x, rect.x, rect.x + rect.width),
    y: clamp(point.y, rect.y, rect.y + rect.height),
  };
}

function getHitBoxIndex(node, point) {
  const state = getPreviewState(node);
  const boxes = getBoxes(node);
  const canvasRect = state.canvasRect;
  if (!canvasRect) {
    return -1;
  }

  for (let index = boxes.length - 1; index >= 0; index -= 1) {
    const boxRect = toCanvasBox(boxes[index], canvasRect);
    if (isPointInRect(point, boxRect)) {
      return index;
    }
  }

  return -1;
}

function randomHue() {
  return Math.floor(Math.random() * 360);
}

function finalizeDragBox(node) {
  const state = getPreviewState(node);
  const dragBox = normalizeBox(state.dragBox);
  state.dragBox = null;

  if (!dragBox) {
    refreshPreview(node);
    return;
  }

  const canvasRect = state.canvasRect;
  if (!canvasRect) {
    refreshPreview(node);
    return;
  }

  if (dragBox.width * canvasRect.width < BOX_MIN_PIXELS || dragBox.height * canvasRect.height < BOX_MIN_PIXELS) {
    refreshPreview(node);
    return;
  }

  if (getBoxes(node).length >= MAX_MASK_OUTPUTS) {
    refreshPreview(node);
    return;
  }

  setBoxes(node, [...getBoxes(node), dragBox]);
}

function createDragBox(startPoint, currentPoint, canvasRect, hue) {
  const left = Math.min(startPoint.x, currentPoint.x);
  const top = Math.min(startPoint.y, currentPoint.y);
  const right = Math.max(startPoint.x, currentPoint.x);
  const bottom = Math.max(startPoint.y, currentPoint.y);

  return {
    x: (left - canvasRect.x) / canvasRect.width,
    y: (top - canvasRect.y) / canvasRect.height,
    width: (right - left) / canvasRect.width,
    height: (bottom - top) / canvasRect.height,
    hue,
  };
}

function attachCanvasInteractions(node, canvas) {
  const handlePointerDown = (event) => {
    const state = getPreviewState(node);
    const canvasRect = state.canvasRect;
    const point = getLocalPoint(canvas, event);

    if (event.altKey) {
      const hitIndex = getHitBoxIndex(node, point);
      if (hitIndex >= 0) {
        const boxes = [...getBoxes(node)];
        boxes.splice(hitIndex, 1);
        setBoxes(node, boxes);
        event.preventDefault();
        event.stopPropagation();
      }
      return;
    }

    if (!event.ctrlKey || !isPointInRect(point, canvasRect)) {
      return;
    }

    const startPoint = clampPointToRect(point, canvasRect);
    state.dragStartPoint = startPoint;
    state.dragBox = createDragBox(startPoint, startPoint, canvasRect, randomHue());

    canvas.setPointerCapture?.(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
    refreshPreview(node);
  };

  const handlePointerMove = (event) => {
    const state = getPreviewState(node);
    const canvasRect = state.canvasRect;
    if (!state.dragBox || !canvasRect) {
      return;
    }

    const point = clampPointToRect(getLocalPoint(canvas, event), canvasRect);
    state.dragBox = createDragBox(state.dragStartPoint, point, canvasRect, state.dragBox.hue);
    event.preventDefault();
    event.stopPropagation();
    refreshPreview(node);
  };

  const handlePointerUp = (event) => {
    const state = getPreviewState(node);
    if (!state.dragBox) {
      return;
    }

    canvas.releasePointerCapture?.(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
    finalizeDragBox(node);
    state.dragStartPoint = null;
  };

  const handlePointerCancel = (event) => {
    const state = getPreviewState(node);
    if (!state.dragBox) {
      return;
    }

    state.dragBox = null;
    state.dragStartPoint = null;
    canvas.releasePointerCapture?.(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
    refreshPreview(node);
  };

  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", handlePointerUp);
  canvas.addEventListener("pointercancel", handlePointerCancel);

  return () => {
    canvas.removeEventListener("pointerdown", handlePointerDown);
    canvas.removeEventListener("pointermove", handlePointerMove);
    canvas.removeEventListener("pointerup", handlePointerUp);
    canvas.removeEventListener("pointercancel", handlePointerCancel);
  };
}

function addPreviewWidget(node) {
  const container = document.createElement("div");
  container.className = "toyxyz-draw-area-mask-preview";
  container.style.width = "100%";
  container.style.display = "block";
  container.style.boxSizing = "border-box";
  container.style.overflow = "hidden";
  container.style.background = "var(--comfy-input-bg, #222)";
  container.style.border = "1px solid var(--border-color, #555)";
  container.style.borderRadius = "8px";

  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  canvas.style.width = "100%";
  canvas.style.touchAction = "none";
  canvas.style.cursor = "crosshair";
  container.appendChild(canvas);

  const widget = node.addDOMWidget("draw_area_mask_preview", "draw_area_mask_preview", container, {
    hideOnZoom: false,
    serialize: false,
  });

  widget.serialize = false;
  widget.element = container;
  widget.canvas = canvas;
  widget.computeSize = (width) => [width, getPreviewHeight(node, width)];
  widget.computeLayoutSize = (currentNode) => {
    const width = currentNode?.size?.[0] || 320;
    const height = getPreviewHeight(currentNode || node, width);
    return {
      minHeight: height,
      maxHeight: height,
    };
  };
  widget.drawPreview = () => drawPreview(canvas, node);

  const resizeObserver = new ResizeObserver(() => {
    widget.drawPreview();
  });
  resizeObserver.observe(container);

  const detachInteractions = attachCanvasInteractions(node, canvas);

  node.__drawAreaMaskWidget = widget;

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function () {
    resizeObserver.disconnect();
    detachInteractions();
    originalOnRemoved?.apply(this, arguments);
  };

  return widget;
}

function hookSizeWidget(node, name) {
  const widget = getNumericWidget(node, name);
  if (!widget || widget.__drawAreaMaskHooked) {
    return;
  }

  widget.__drawAreaMaskHooked = true;
  const originalCallback = widget.callback;

  widget.callback = function (value, canvas, currentNode, pos, event) {
    const targetNode = currentNode || node;
    originalCallback?.call(this, value, canvas, targetNode, pos, event);
    syncSizeProperties(targetNode);
    refreshPreview(targetNode);
  };
}

app.registerExtension({
  name: "toyxyz.DrawAreaMask",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
      const result = await originalOnNodeCreated?.apply(this, arguments);

      ensureNodeProperties(this);
      syncSizeProperties(this);
      hideBoxesWidget(this);
      syncBoxesWidget(this);
      hookSizeWidget(this, "width");
      hookSizeWidget(this, "height");
      updateOutputs(this);
      this.setSize?.(this.computeSize());

      const clearBoxButton = this.addWidget("button", "clear box", null, () => {
        setBoxes(this, []);
      });
      clearBoxButton.serialize = false;

      addPreviewWidget(this);
      refreshPreview(this);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      originalOnConfigure?.apply(this, arguments);
      ensureNodeProperties(this);
      hideBoxesWidget(this);
      syncSizeProperties(this);

      const boxesWidget = getBoxesWidget(this);
      const widgetBoxes = boxesWidget ? parseBoxesState(boxesWidget.value || "[]") : [];
      this.properties[BOXES_PROPERTY] = widgetBoxes.length > 0 ? widgetBoxes : normalizeBoxes(this.properties[BOXES_PROPERTY]);
      syncBoxesWidget(this);

      hookSizeWidget(this, "width");
      hookSizeWidget(this, "height");
      updateOutputs(this);
      this.setSize?.(this.computeSize());
      refreshPreview(this);
    };

    const originalOnResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      originalOnResize?.apply(this, arguments);
      refreshPreview(this);
    };
  },
});
