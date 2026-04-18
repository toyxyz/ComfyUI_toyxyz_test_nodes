import { app } from "../../scripts/app.js";

const NODE_ID = "CropAreaMask";
const BOXES_PROPERTY = "crop_area_boxes";
const BOXES_WIDGET_NAME = "boxes_state";
const HIDDEN_WIDGET_TYPE = "crop-area-mask-hidden";
const IMAGE_WIDTH_PROPERTY = "crop_area_image_width";
const IMAGE_HEIGHT_PROPERTY = "crop_area_image_height";
const PREVIEW_MIN_HEIGHT = 220;
const PREVIEW_MAX_HEIGHT = 560;
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

function getBoxes(node) {
  const properties = ensureNodeProperties(node);
  properties[BOXES_PROPERTY] = normalizeBoxes(properties[BOXES_PROPERTY]);
  return properties[BOXES_PROPERTY];
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

  while (node.outputs.length > 3) {
    node.removeOutput(node.outputs.length - 1);
  }

  while (node.outputs.length < 3) {
    node.addOutput("crops", "IMAGE");
  }

  if (node.outputs[0]) {
    node.outputs[0].name = "crops";
    node.outputs[0].type = "IMAGE";
  }

  if (node.outputs[1]) {
    node.outputs[1].name = "masks";
    node.outputs[1].type = "MASK";
  }

  if (node.outputs[2]) {
    node.outputs[2].name = "source_image";
    node.outputs[2].type = "IMAGE";
  }
}

function getImageWidget(node) {
  return node.widgets?.find((widget) => widget.name === "image");
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
  if (!boxesWidget || boxesWidget.__cropAreaMaskHidden) {
    return;
  }

  boxesWidget.__cropAreaMaskHidden = true;
  boxesWidget.hidden = true;
  boxesWidget.origType = boxesWidget.type;
  boxesWidget.origComputeSize = boxesWidget.computeSize;
  boxesWidget.type = HIDDEN_WIDGET_TYPE;
  boxesWidget.computeSize = () => [0, -4];
}

function hideDefaultImagePreview(node) {
  if (!node) {
    return;
  }

  node.imgs = null;
  node.preview = null;
  node.imageIndex = null;
}

function getAspectRatio(node) {
  const properties = ensureNodeProperties(node);
  const width = Number(properties[IMAGE_WIDTH_PROPERTY] || 0);
  const height = Number(properties[IMAGE_HEIGHT_PROPERTY] || 0);

  if (width > 0 && height > 0) {
    return width / height;
  }

  return FALLBACK_ASPECT_RATIO;
}

function getPreviewHeight(node, width) {
  const safeWidth = Math.max(220, Math.round(width || node?.size?.[0] || 320));
  const innerWidth = Math.max(1, safeWidth - PREVIEW_MARGIN * 2);
  const aspectRatio = getAspectRatio(node);
  const rawHeight = Math.round(innerWidth / aspectRatio + PREVIEW_MARGIN * 2);
  return clamp(rawHeight, PREVIEW_MIN_HEIGHT, PREVIEW_MAX_HEIGHT);
}

function getPreviewState(node) {
  if (!node.__cropAreaMaskState) {
    node.__cropAreaMaskState = {
      image: null,
      imageValue: null,
      error: "",
      drawScheduled: false,
      dragBox: null,
      dragStartPoint: null,
      imageRect: null,
      reloadToken: 0,
    };
  }

  return node.__cropAreaMaskState;
}

function buildImageUrl(imageValue, reloadToken) {
  if (!imageValue) {
    return "";
  }

  const params = new URLSearchParams();
  params.set("filename", String(imageValue));
  params.set("preview", "webp;90");
  if (!String(imageValue).startsWith("blake3:")) {
    params.set("type", "input");
  }
  params.set("v", String(reloadToken));
  return `/view?${params.toString()}`;
}

function loadPreviewImage(node, forceReload = false) {
  const state = getPreviewState(node);
  const imageWidget = getImageWidget(node);
  const imageValue = imageWidget?.value;
  hideDefaultImagePreview(node);

  if (!imageValue) {
    state.image = null;
    state.imageValue = null;
    state.error = "";
    refreshPreview(node);
    return;
  }

  if (!forceReload && state.imageValue === imageValue && state.image) {
    return;
  }

  if (forceReload || state.imageValue !== imageValue) {
    state.reloadToken = Date.now();
  }

  state.imageValue = imageValue;
  state.error = "";

  const image = new Image();
  const currentValue = imageValue;
  const currentToken = state.reloadToken;

  image.onload = () => {
    const latestState = getPreviewState(node);
    if (latestState.imageValue !== currentValue || latestState.reloadToken !== currentToken) {
      return;
    }

    latestState.image = image;
    const properties = ensureNodeProperties(node);
    properties[IMAGE_WIDTH_PROPERTY] = image.naturalWidth;
    properties[IMAGE_HEIGHT_PROPERTY] = image.naturalHeight;
    hideDefaultImagePreview(node);
    refreshPreview(node);
  };

  image.onerror = () => {
    const latestState = getPreviewState(node);
    if (latestState.imageValue !== currentValue || latestState.reloadToken !== currentToken) {
      return;
    }

    latestState.image = null;
    latestState.error = "Failed to load preview image.";
    hideDefaultImagePreview(node);
    refreshPreview(node);
  };

  image.src = buildImageUrl(imageValue, currentToken);
}

function getCanvasRect(canvas, image) {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const logicalWidth = canvas.width > 0 ? canvas.width / devicePixelRatio : (canvas.clientWidth || 320);
  const logicalHeight = canvas.height > 0 ? canvas.height / devicePixelRatio : (canvas.clientHeight || PREVIEW_MIN_HEIGHT);
  const imageWidth = image?.naturalWidth || 1;
  const imageHeight = image?.naturalHeight || 1;
  const availableWidth = Math.max(1, logicalWidth - PREVIEW_MARGIN * 2);
  const availableHeight = Math.max(1, logicalHeight - PREVIEW_MARGIN * 2);
  const scale = Math.min(availableWidth / imageWidth, availableHeight / imageHeight);
  const drawWidth = imageWidth * scale;
  const drawHeight = imageHeight * scale;

  return {
    x: PREVIEW_MARGIN + (availableWidth - drawWidth) / 2,
    y: PREVIEW_MARGIN + (availableHeight - drawHeight) / 2,
    width: drawWidth,
    height: drawHeight,
  };
}

function toCanvasBox(box, imageRect) {
  return {
    x: imageRect.x + box.x * imageRect.width,
    y: imageRect.y + box.y * imageRect.height,
    width: box.width * imageRect.width,
    height: box.height * imageRect.height,
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

  hideDefaultImagePreview(node);

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.scale(devicePixelRatio, devicePixelRatio);

  ctx.fillStyle = "#222222";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const image = state.image;
  if (!image) {
    state.imageRect = null;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
    ctx.lineWidth = 1;
    ctx.strokeRect(PREVIEW_MARGIN, PREVIEW_MARGIN, cssWidth - PREVIEW_MARGIN * 2, cssHeight - PREVIEW_MARGIN * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
    ctx.font = "14px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(
      state.error || "Load image, then Ctrl+drag to add boxes. Alt+click removes a box.",
      cssWidth / 2,
      cssHeight / 2
    );
    return;
  }

  const imageRect = getCanvasRect(canvas, image);
  state.imageRect = imageRect;

  ctx.drawImage(image, imageRect.x, imageRect.y, imageRect.width, imageRect.height);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
  ctx.lineWidth = 1;
  ctx.strokeRect(imageRect.x, imageRect.y, imageRect.width, imageRect.height);

  const boxes = getBoxes(node);
  boxes.forEach((box, index) => {
    const boxRect = toCanvasBox(box, imageRect);
    ctx.fillStyle = hueFillColor(box.hue);
    ctx.fillRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    ctx.strokeStyle = hueBorderColor(box.hue);
    ctx.lineWidth = 2;
    ctx.strokeRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    drawLabel(ctx, String(index + 1), boxRect);
  });

  if (state.dragBox) {
    const boxRect = toCanvasBox(state.dragBox, imageRect);
    ctx.fillStyle = hueFillColor(state.dragBox.hue);
    ctx.fillRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    ctx.strokeStyle = hueBorderColor(state.dragBox.hue);
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(boxRect.x, boxRect.y, boxRect.width, boxRect.height);
    ctx.setLineDash([]);
    drawLabel(ctx, String(boxes.length + 1), boxRect);
  }

  ctx.fillStyle = "rgba(0, 0, 0, 0.58)";
  ctx.fillRect(imageRect.x, imageRect.y + imageRect.height - 24, imageRect.width, 24);
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.font = "12px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Ctrl+drag: add box   Alt+click: remove box", imageRect.x + imageRect.width / 2, imageRect.y + imageRect.height - 12);
}

function refreshPreview(node) {
  const state = getPreviewState(node);
  if (!node.__cropAreaMaskWidget?.drawPreview || state.drawScheduled) {
    return;
  }

  state.drawScheduled = true;
  requestAnimationFrame(() => {
    state.drawScheduled = false;
    node.__cropAreaMaskWidget.drawPreview();
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
  const imageRect = state.imageRect;
  if (!imageRect) {
    return -1;
  }

  for (let index = boxes.length - 1; index >= 0; index -= 1) {
    const boxRect = toCanvasBox(boxes[index], imageRect);
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

  const imageRect = state.imageRect;
  if (!imageRect) {
    refreshPreview(node);
    return;
  }

  if (dragBox.width * imageRect.width < BOX_MIN_PIXELS || dragBox.height * imageRect.height < BOX_MIN_PIXELS) {
    refreshPreview(node);
    return;
  }

  setBoxes(node, [...getBoxes(node), dragBox]);
}

function createDragBox(startPoint, currentPoint, imageRect, hue) {
  const left = Math.min(startPoint.x, currentPoint.x);
  const top = Math.min(startPoint.y, currentPoint.y);
  const right = Math.max(startPoint.x, currentPoint.x);
  const bottom = Math.max(startPoint.y, currentPoint.y);

  return {
    x: (left - imageRect.x) / imageRect.width,
    y: (top - imageRect.y) / imageRect.height,
    width: (right - left) / imageRect.width,
    height: (bottom - top) / imageRect.height,
    hue,
  };
}

function attachCanvasInteractions(node, canvas) {
  const handlePointerDown = (event) => {
    const state = getPreviewState(node);
    const imageRect = state.imageRect;
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

    if (!event.ctrlKey || !isPointInRect(point, imageRect)) {
      return;
    }

    const startPoint = clampPointToRect(point, imageRect);
    state.dragStartPoint = startPoint;
    state.dragBox = createDragBox(startPoint, startPoint, imageRect, randomHue());

    canvas.setPointerCapture?.(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
    refreshPreview(node);
  };

  const handlePointerMove = (event) => {
    const state = getPreviewState(node);
    const imageRect = state.imageRect;
    if (!state.dragBox || !imageRect) {
      return;
    }

    const point = clampPointToRect(getLocalPoint(canvas, event), imageRect);
    state.dragBox = createDragBox(state.dragStartPoint, point, imageRect, state.dragBox.hue);
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
  container.className = "toyxyz-crop-area-mask-preview";
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

  const widget = node.addDOMWidget("crop_area_mask_preview", "crop_area_mask_preview", container, {
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

  node.__cropAreaMaskWidget = widget;

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function () {
    resizeObserver.disconnect();
    detachInteractions();
    originalOnRemoved?.apply(this, arguments);
  };

  return widget;
}

function hookImageWidget(node) {
  const imageWidget = getImageWidget(node);
  if (!imageWidget || imageWidget.__cropAreaMaskHooked) {
    return;
  }

  imageWidget.__cropAreaMaskHooked = true;
  const originalCallback = imageWidget.callback;

  imageWidget.callback = function (value, canvas, currentNode, pos, event) {
    originalCallback?.call(this, value, canvas, currentNode, pos, event);
    const targetNode = currentNode || node;
    loadPreviewImage(targetNode, true);
    refreshPreview(targetNode);
  };
}

app.registerExtension({
  name: "toyxyz.CropAreaMask",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
      const result = await originalOnNodeCreated?.apply(this, arguments);

      ensureNodeProperties(this);
      hideDefaultImagePreview(this);
      hideBoxesWidget(this);
      syncBoxesWidget(this);
      hookImageWidget(this);
      updateOutputs(this);
      this.setSize?.(this.computeSize());

      const clearBoxButton = this.addWidget("button", "clear box", null, () => {
        setBoxes(this, []);
      });
      clearBoxButton.serialize = false;

      addPreviewWidget(this);
      loadPreviewImage(this, true);
      refreshPreview(this);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      originalOnConfigure?.apply(this, arguments);
      ensureNodeProperties(this);
      hideDefaultImagePreview(this);
      hideBoxesWidget(this);
      this.properties[BOXES_PROPERTY] = normalizeBoxes(this.properties[BOXES_PROPERTY]);
      syncBoxesWidget(this);
      hookImageWidget(this);
      updateOutputs(this);
      this.setSize?.(this.computeSize());
      loadPreviewImage(this, false);
      refreshPreview(this);
    };

    const originalOnResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      originalOnResize?.apply(this, arguments);
      hideDefaultImagePreview(this);
      refreshPreview(this);
    };

    const originalOnDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function () {
      hideDefaultImagePreview(this);
      return originalOnDrawBackground?.apply(this, arguments);
    };
  },
});
