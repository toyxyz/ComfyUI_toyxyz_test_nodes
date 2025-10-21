// Type of the canvas widget
const WIDGET_CANVAS_TYPE = 'fuwuffyAreaCanvas';
// The canvas's className
const CANVAS_CLASS_NAME = 'fuwuffy-area-canvas';
// Default height for all widgets
const WIDGET_BASE_HEIGHT = LiteGraph.NODE_WIDGET_HEIGHT;
// Default size for canvas widget
const CANVAS_MIN_SIZE = 200;
// Margin of the canvas
const CANVAS_MARGIN = 3;
// Border size of the canvas
const CANVAS_BORDER = 2;
// Area border size
const AREA_BORDER_SIZE = 3;

// Convert an rgb hex color to hsl
function hexToHsl(hex) {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;
    if (max === min) {
        h = s = 0;
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    h = Math.round(h * 360);
    s = Math.round(s * 100);
    l = Math.round(l * 100);
    return `hsl(${h}, ${s}%, ${l}%)`;
}

// Adjust the brightness of an hsl color while maintaining alpha
function brightenHsl(hsla, amount) {
    const colorMatches = hsla.match(/[\d.]+/g);
    let [h, s, l, a = 1] = colorMatches ? colorMatches.map(Number) : [300, 100, 50, 1];
    l = Math.min(100, Math.max(0, l * amount));
    return `hsla(${h}, ${s}%, ${l}%, ${a})`;
}

// Generate an hsla color based on a value and a maximum range
function generateHslColor(value, max, alpha = 0.1) {
    if (max <= 0) {
        return `hsla(0, 0%, 0%, 0.1)`;
    }
    const hue = Math.round(((value % max) / max) * 360);
    return `hsla(${hue}, 100%, 50%, ${alpha})`;
}

// Compute the canvas size based on node dimensions
function computeCanvasSize(node, size) {
    node.widgets.sort((a, b) => {
        if (a.type === WIDGET_CANVAS_TYPE && b.type !== WIDGET_CANVAS_TYPE) {
            return 1;
        } else if (a.type !== WIDGET_CANVAS_TYPE && b.type === WIDGET_CANVAS_TYPE) {
            return -1;
        }
        return 0;
    });

    if (node.widgets[0].last_y == null) {
        return;
    }
    const yBase = WIDGET_BASE_HEIGHT * Math.max(node.inputs.length, node.outputs.length) + 5;
    let remainingHeight = size[1] - yBase;
    // Calculate total height of non-canvas widgets
    const widgetTotalHeight = node.widgets.reduce((totalHeight, widget) => {
        if (widget.type !== WIDGET_CANVAS_TYPE) {
            totalHeight += (widget.computeSize ? widget.computeSize()[1] : WIDGET_BASE_HEIGHT) + 5;
        }
        return totalHeight;
    }, 0);
    // Calculate canvas height
    remainingHeight = Math.max(remainingHeight - widgetTotalHeight, CANVAS_MIN_SIZE);
    node.size[1] = yBase + widgetTotalHeight + remainingHeight;
    node.graph.setDirtyCanvas(true);
    // Position each widget within the canvas
    let currentY = yBase;
    node.widgets.forEach(widget => {
        widget.y = currentY;
        currentY += (widget.type === WIDGET_CANVAS_TYPE ? remainingHeight : (widget.computeSize ? widget.computeSize()[1] : WIDGET_BASE_HEIGHT)) + 4;
    });
    node.canvasHeight = remainingHeight;
}

// Add area graph widget to the node
export function addAreaGraphWidget(app, node, name) {
    const widget = {
        type: WIDGET_CANVAS_TYPE,
        name: name,
        draw: function (ctx, node, widgetWidth, widgetY) {
            if (!node.canvasHeight) {
                computeCanvasSize(node, node.size);
            }
            // Canvas variables
            const visible = app.canvas.ds.scale > 0.5;
            const transform = ctx.getTransform();
            const widgetHeight = node.canvasHeight;
            const imageWidth = node.properties["image_width"] || 512;
            const imageHeight = node.properties["image_height"] || 512;
            const scale = Math.min((widgetWidth - CANVAS_MARGIN * 2) / imageWidth, (widgetHeight - CANVAS_MARGIN * 2) / imageHeight);
            // Get values from node
            const values = node.properties["area_values"];
            // Set canvas position and size in DOM
            Object.assign(this.canvas.style, {
                left: `${transform.e}px`,
                top: `${transform.f + (widgetY * transform.d)}px`,
                width: `${widgetWidth * transform.a}px`,
                height: `${widgetHeight * transform.d}px`,
                position: "absolute",
                zIndex: 1,
                pointerEvents: "none",
                display: visible ? "block" : "none"
            });
            // Calculate canvas draw dimensions
            const backgroundWidth = imageWidth * scale;
            const backgroundHeight = imageHeight * scale;
            const xOffset = CANVAS_MARGIN + (backgroundWidth < widgetWidth ? (widgetWidth - backgroundWidth) / 2 - CANVAS_MARGIN : 0);
            const yOffset = CANVAS_MARGIN + (backgroundHeight < widgetHeight ? (widgetHeight - backgroundHeight) / 2 - CANVAS_MARGIN : 0);
            // Transforms the node's area values to canvas pixel dimensions
            const getDrawArea = (arr) => {
                if (!arr || arr.length < 4) {
                    return [0, 0, 0, 0];
                }
                return [
                    Math.min(arr[0] * backgroundWidth, backgroundWidth),
                    Math.min(arr[1] * backgroundHeight, backgroundHeight),
                    Math.max(0, Math.min(arr[2] * backgroundWidth, backgroundWidth - arr[0] * backgroundWidth)),
                    Math.max(0, Math.min(arr[3] * backgroundHeight, backgroundHeight - arr[1] * backgroundHeight)),
                ];
            };
            // Draws a rectangle on the canvas
            const drawRect = (x, y, w, h, color) => {
                if (w <= 0 || h <= 0) {
                    return;
                }
                ctx.fillStyle = color;
                ctx.fillRect(x, y, w, h);
            };
            // Calculate widget positions
            const widgetX = xOffset;
            const widgetYOffset = widgetY + yOffset;
            // Color stuff
            const backgroundColor = hexToHsl(globalThis.LiteGraph.WIDGET_BGCOLOR);
            const borderColor = hexToHsl(globalThis.LiteGraph.WIDGET_OUTLINE_COLOR);
            // Draw the canvas's background and border
            drawRect(widgetX - CANVAS_BORDER, widgetYOffset - CANVAS_BORDER, backgroundWidth + CANVAS_BORDER * 2, backgroundHeight + CANVAS_BORDER * 2, borderColor);
            drawRect(widgetX, widgetYOffset, backgroundWidth, backgroundHeight, backgroundColor);
            if (!visible) {
                return;
            }
            // Draw all conditioning areas (non-selected)
            const halfBorder = AREA_BORDER_SIZE / 2;
            values.forEach((v, k) => {
                if (k === node.index) {
                    return; // Skip selected area to draw later
                }
                const [x, y, w, h] = getDrawArea(v);
                if (w <= 0 || h <= 0) {
                    return;
                }
                const areaColor = generateHslColor(k + 1, values.length, 0.1);
                const brightColor = brightenHsl(areaColor, 0.7);
                drawRect(widgetX + x, widgetYOffset + y, w, h, brightColor);
                drawRect(widgetX + x + halfBorder, widgetYOffset + y + halfBorder, w - AREA_BORDER_SIZE, h - AREA_BORDER_SIZE, areaColor);
                
                // Draw area_id number in the center
                if (w > 20 && h > 20) {
                    ctx.fillStyle = '#FFFFFF';
                    ctx.font = 'bold 16px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(k.toString(), widgetX + x + w / 2, widgetYOffset + y + h / 2);
                }
            });
            // Draw selected area with white outline
            const [x, y, w, h] = getDrawArea(values[node.index]);
            const areaColor = generateHslColor(node.index + 1, values.length, 0.1);
            const brightColor = brightenHsl(areaColor, 0.7);
            drawRect(widgetX + x, widgetYOffset + y, w, h, brightColor);
            drawRect(widgetX + x + halfBorder, widgetYOffset + y + halfBorder, w - AREA_BORDER_SIZE, h - AREA_BORDER_SIZE, areaColor);
            // Add white outline for selected area
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1;
            ctx.strokeRect(widgetX + x, widgetYOffset + y, w, h);
            
            // Draw area_id number for selected area
            if (w > 20 && h > 20) {
                ctx.fillStyle = '#FFFFFF';
                ctx.font = 'bold 16px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(node.index.toString(), widgetX + x + w / 2, widgetYOffset + y + h / 2);
            }
            
            if (node.is_selected) {
                node.inputs.filter(input => input.name.includes(node.index)).forEach(input => {
                    const link = input.link;
                    if (link) {
                        const nodeId = node.graph.links[link].origin_id;
                        const connectedNode = node.graph._nodes_by_id[nodeId];
                        const [x, y] = connectedNode.pos;
                        const [w, h] = connectedNode.size;
                        drawRect(x - node.pos[0], y - node.pos[1], w, h, generateHslColor(node.index + 1, values.length, 0.1));
                    }
                });
            }
        }
    };
    // Create a new canvas element
    widget.canvas = document.createElement("canvas");
    widget.canvas.className = CANVAS_CLASS_NAME;
    widget.canvas.style.display = "none";
    // Add the widget and canvas to the UI
    widget.parent = node;
    node.addCustomWidget(widget);
    node.onResize = size => computeCanvasSize(node, size);
    return { widget };
}
