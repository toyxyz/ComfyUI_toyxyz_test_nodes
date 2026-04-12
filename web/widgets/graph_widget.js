const LITEGRAPH_WIDGET_TYPE = "fuwuffyAreaCanvas";
const CANVAS_FALLBACK_WIDTH = 240;
const CANVAS_MAX_HEIGHT = 1200;
const CANVAS_MARGIN = 6;
const CANVAS_BORDER = 2;
const AREA_BORDER_SIZE = 3;

function getNodeNumericValue(node, name, fallback) {
    const widgetValue = node?.widgets?.find?.((widget) => widget.name === name)?.value;
    const propertyValue = node?.properties?.[name];
    const numericValue = Number(widgetValue ?? propertyValue ?? fallback);

    if (!Number.isFinite(numericValue)) {
        return fallback;
    }

    return numericValue;
}

function getPreviewAspectRatio(node) {
    const imageWidth = Math.max(1, getNodeNumericValue(node, "image_width", 512));
    const imageHeight = Math.max(1, getNodeNumericValue(node, "image_height", 512));
    return imageWidth / imageHeight;
}

function getDesiredCanvasHeight(node, widgetWidth) {
    const safeWidth = Math.max(Math.round(widgetWidth || 0), CANVAS_FALLBACK_WIDTH);
    const innerWidth = Math.max(safeWidth - CANVAS_MARGIN * 2, 1);
    const aspectRatio = getPreviewAspectRatio(node);
    const desiredHeight = Math.round(innerWidth / aspectRatio + CANVAS_MARGIN * 2);
    return Math.max(1, Math.min(CANVAS_MAX_HEIGHT, desiredHeight));
}

function isVueNodesEnabled(app) {
    return !!app?.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled", false);
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
        document.body
    ].filter(Boolean).map((element) => window.getComputedStyle(element));

    const resolveVariable = (value, depth = 0) => {
        if (!value || depth > 4) {
            return "";
        }

        const trimmed = value.trim();
        if (!trimmed.includes("var(")) {
            return trimmed;
        }

        return trimmed.replace(/var\((--[^),\s]+)(?:,[^)]+)?\)/g, (_, variableName) => {
            for (const style of styles) {
                const nestedValue = style.getPropertyValue(variableName);
                const resolvedValue = resolveVariable(nestedValue, depth + 1);
                if (resolvedValue) {
                    return resolvedValue;
                }
            }

            return "";
        }).trim();
    };

    for (const name of names) {
        for (const style of styles) {
            const rawValue = style.getPropertyValue(name);
            const resolvedValue = resolveVariable(rawValue);
            if (resolvedValue) {
                return resolvedValue;
            }
        }
    }

    return "";
}

function brightenHsl(hsla, amount) {
    const colorMatches = hsla.match(/[\d.]+/g);
    let [h, s, l, a = 1] = colorMatches ? colorMatches.map(Number) : [300, 100, 50, 1];
    l = Math.min(100, Math.max(0, l * amount));
    return `hsla(${h}, ${s}%, ${l}%, ${a})`;
}

function generateHslColor(value, max, alpha = 0.1) {
    if (max <= 0) {
        return `hsla(0, 0%, 0%, ${alpha})`;
    }
    const hue = Math.round(((value % max) / max) * 360);
    return `hsla(${hue}, 100%, 50%, ${alpha})`;
}

function getThemeColors() {
    const liteGraph = globalThis.LiteGraph;
    const cssBackground = readCssVariable("--comfy-input-bg", "--content-bg", "--bg-color");
    const cssBorder = readCssVariable("--comfy-border", "--border-color", "--content-fg", "--input-text");

    return {
        background: normalizeColorString(cssBackground || liteGraph?.WIDGET_BGCOLOR, "hsl(0, 0%, 15%)"),
        border: normalizeColorString(cssBorder || liteGraph?.WIDGET_OUTLINE_COLOR, "hsl(0, 0%, 40%)")
    };
}

function drawAreaLabel(ctx, label, centerX, centerY, width, height) {
    if (width <= 20 || height <= 20) {
        return;
    }

    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 16px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, centerX, centerY);
}

function drawPreviewContents(ctx, node, cssWidth, cssHeight, originX = 0, originY = 0) {
    const imageWidth = Math.max(1, getNodeNumericValue(node, "image_width", 512));
    const imageHeight = Math.max(1, getNodeNumericValue(node, "image_height", 512));
    const availableWidth = Math.max(cssWidth - CANVAS_MARGIN * 2, 1);
    const availableHeight = Math.max(cssHeight - CANVAS_MARGIN * 2, 1);
    const scale = Math.min(availableWidth / imageWidth, availableHeight / imageHeight);
    const backgroundWidth = imageWidth * scale;
    const backgroundHeight = imageHeight * scale;
    const xOffset = originX + CANVAS_MARGIN + Math.max(0, (availableWidth - backgroundWidth) / 2);
    const yOffset = originY + CANVAS_MARGIN + Math.max(0, (availableHeight - backgroundHeight) / 2);
    const values = Array.isArray(node?.properties?.["area_values"]) ? node.properties["area_values"] : [];
    const selectedIndex = Math.max(0, node?.index || 0);
    const { background, border } = getThemeColors();

    ctx.fillStyle = background;
    ctx.fillRect(originX, originY, cssWidth, cssHeight);

    ctx.fillStyle = border;
    ctx.fillRect(
        xOffset - CANVAS_BORDER,
        yOffset - CANVAS_BORDER,
        backgroundWidth + CANVAS_BORDER * 2,
        backgroundHeight + CANVAS_BORDER * 2
    );
    ctx.fillStyle = background;
    ctx.fillRect(xOffset, yOffset, backgroundWidth, backgroundHeight);

    const getDrawArea = (arr) => {
        if (!Array.isArray(arr) || arr.length < 4) {
            return [0, 0, 0, 0];
        }

        return [
            Math.min(arr[0] * backgroundWidth, backgroundWidth),
            Math.min(arr[1] * backgroundHeight, backgroundHeight),
            Math.max(0, Math.min(arr[2] * backgroundWidth, backgroundWidth - arr[0] * backgroundWidth)),
            Math.max(0, Math.min(arr[3] * backgroundHeight, backgroundHeight - arr[1] * backgroundHeight))
        ];
    };

    values.forEach((value, index) => {
        if (index === selectedIndex) {
            return;
        }

        const [x, y, width, height] = getDrawArea(value);
        if (width <= 0 || height <= 0) {
            return;
        }

        const fillColor = brightenHsl(generateHslColor(index + 1, values.length, 0.1), 0.7);
        const outlineColor = generateHslColor(index + 1, values.length, 0.1);
        ctx.fillStyle = fillColor;
        ctx.fillRect(xOffset + x, yOffset + y, width, height);
        ctx.fillStyle = outlineColor;
        ctx.fillRect(
            xOffset + x + AREA_BORDER_SIZE / 2,
            yOffset + y + AREA_BORDER_SIZE / 2,
            Math.max(0, width - AREA_BORDER_SIZE),
            Math.max(0, height - AREA_BORDER_SIZE)
        );
        drawAreaLabel(ctx, String(index), xOffset + x + width / 2, yOffset + y + height / 2, width, height);
    });

    const [selectedX, selectedY, selectedWidth, selectedHeight] = getDrawArea(values[selectedIndex]);
    if (selectedWidth > 0 && selectedHeight > 0) {
        const fillColor = brightenHsl(generateHslColor(selectedIndex + 1, values.length, 0.1), 0.7);
        const outlineColor = generateHslColor(selectedIndex + 1, values.length, 0.1);
        ctx.fillStyle = fillColor;
        ctx.fillRect(xOffset + selectedX, yOffset + selectedY, selectedWidth, selectedHeight);
        ctx.fillStyle = outlineColor;
        ctx.fillRect(
            xOffset + selectedX + AREA_BORDER_SIZE / 2,
            yOffset + selectedY + AREA_BORDER_SIZE / 2,
            Math.max(0, selectedWidth - AREA_BORDER_SIZE),
            Math.max(0, selectedHeight - AREA_BORDER_SIZE)
        );
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1;
        ctx.strokeRect(xOffset + selectedX, yOffset + selectedY, selectedWidth, selectedHeight);
        drawAreaLabel(
            ctx,
            String(selectedIndex),
            xOffset + selectedX + selectedWidth / 2,
            yOffset + selectedY + selectedHeight / 2,
            selectedWidth,
            selectedHeight
        );
    }
}

function drawDomPreview(canvas, node) {
    const container = canvas.parentElement;
    const fallbackWidth = Math.max(node?.size?.[0] || 0, CANVAS_FALLBACK_WIDTH);
    const containerWidth = Math.max(container?.clientWidth || 0, fallbackWidth);
    const previewHeight = getDesiredCanvasHeight(node, containerWidth);
    const devicePixelRatio = window.devicePixelRatio || 1;
    const cssWidth = containerWidth;
    const cssHeight = previewHeight;
    const pixelWidth = Math.max(1, Math.round(cssWidth * devicePixelRatio));
    const pixelHeight = Math.max(1, Math.round(cssHeight * devicePixelRatio));
    const imageWidth = Math.max(1, getNodeNumericValue(node, "image_width", 512));
    const imageHeight = Math.max(1, getNodeNumericValue(node, "image_height", 512));

    if (container) {
        container.style.minHeight = `${cssHeight}px`;
        container.style.maxHeight = `${cssHeight}px`;
        container.style.height = `${cssHeight}px`;
        container.style.overflow = "hidden";
        container.style.aspectRatio = `${imageWidth} / ${imageHeight}`;
    }

    canvas.style.width = "100%";
    canvas.style.height = "100%";
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
    drawPreviewContents(ctx, node, cssWidth, cssHeight);
}

function getLiteGraphPreviewHeight(node, widgetWidth, widgetY = 0) {
    const desiredHeight = getDesiredCanvasHeight(node, widgetWidth);
    const availableHeight = Math.max(1, Math.round((node?.size?.[1] || desiredHeight) - widgetY - 4));
    return Math.max(1, Math.min(desiredHeight, availableHeight));
}

export function refreshAreaGraphWidget(node) {
    if (!node?.__visualAreaPreviewWidget) {
        return;
    }

    if (node.__visualAreaPreviewMode === "litegraph") {
        node.graph?.setDirtyCanvas?.(true, true);
        return;
    }

    if (!node.__visualAreaPreviewWidget.drawPreview) {
        return;
    }

    if (node.__visualAreaPreviewScheduled) {
        return;
    }

    node.__visualAreaPreviewScheduled = true;
    requestAnimationFrame(() => {
        node.__visualAreaPreviewScheduled = false;
        node.__visualAreaPreviewWidget.drawPreview();
    });
}

function addLiteGraphAreaGraphWidget(node, name) {
    const widget = {
        name,
        type: LITEGRAPH_WIDGET_TYPE,
        serialize: false,
        computeSize(width) {
            return [width, getDesiredCanvasHeight(node, width)];
        },
        draw(ctx, currentNode, widgetWidth, widgetY) {
            const widgetHeight = getLiteGraphPreviewHeight(currentNode, widgetWidth, widgetY);
            ctx.save();
            drawPreviewContents(ctx, currentNode, widgetWidth, widgetHeight, 0, widgetY);
            ctx.restore();
        }
    };

    node.addCustomWidget(widget);

    const onResize = node.onResize;
    node.onResize = function () {
        onResize?.apply(this, arguments);
        this.graph?.setDirtyCanvas?.(true, true);
    };

    node.__visualAreaPreviewMode = "litegraph";
    node.__visualAreaPreviewWidget = widget;
    return { widget };
}

function addVueAreaGraphWidget(app, node, name) {
    const container = document.createElement("div");
    container.className = "fuwuffy-area-preview-container";
    container.style.width = "100%";
    container.style.minHeight = "1px";
    container.style.display = "block";
    container.style.boxSizing = "border-box";
    container.style.backgroundColor = "var(--comfy-input-bg)";
    container.style.pointerEvents = "none";
    container.style.overflow = "hidden";

    const canvas = document.createElement("canvas");
    canvas.className = "fuwuffy-area-canvas";
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.backgroundColor = "transparent";
    container.appendChild(canvas);

    const widget = node.addDOMWidget(name, "visual_area_mask_preview", container, {
        serialize: false,
        hideOnZoom: false
    });

    widget.serialize = false;
    widget.canvas = canvas;
    widget.element = container;
    widget.computeSize = function (width) {
        return [width, getDesiredCanvasHeight(node, width)];
    };
    widget.computeLayoutSize = function (currentNode) {
        const width = currentNode?.size?.[0] || CANVAS_FALLBACK_WIDTH;
        const height = getDesiredCanvasHeight(currentNode || node, width);
        return {
            minHeight: height,
            maxHeight: height
        };
    };
    widget.drawPreview = () => drawDomPreview(canvas, node);

    const resizeObserver = new ResizeObserver(() => {
        widget.drawPreview();
    });
    resizeObserver.observe(container);

    const onResize = node.onResize;
    node.onResize = function () {
        onResize?.apply(this, arguments);
        widget.drawPreview();
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        resizeObserver.disconnect();
        onRemoved?.apply(this, arguments);
    };

    node.__visualAreaPreviewMode = "vue";
    node.__visualAreaPreviewWidget = widget;
    refreshAreaGraphWidget(node);
    return { widget };
}

export function addAreaGraphWidget(app, node, name) {
    if (isVueNodesEnabled(app)) {
        return addVueAreaGraphWidget(app, node, name);
    }

    return addLiteGraphAreaGraphWidget(node, name);
}
