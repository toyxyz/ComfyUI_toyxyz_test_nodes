const MIRROR_PROPERTIES = [
    "direction", "boxSizing", "width", "height", "overflowX", "overflowY",
    "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
    "borderStyle", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
    "fontStyle", "fontVariant", "fontWeight", "fontStretch", "fontSize", "fontSizeAdjust",
    "lineHeight", "fontFamily", "textAlign", "textTransform", "textIndent",
    "letterSpacing", "wordSpacing", "tabSize",
];

export function scaleCaretPosition(rect, offsetWidth, offsetHeight, caretLeft, caretTop,
        scrollLeft, scrollTop, caretHeight) {
    const scaleX = offsetWidth ? rect.width / offsetWidth : 1;
    const scaleY = offsetHeight ? rect.height / offsetHeight : scaleX;
    const localTop = caretTop - scrollTop;
    return {
        left: rect.left + (caretLeft - scrollLeft) * scaleX,
        top: rect.top + localTop * scaleY,
        bottom: rect.top + (localTop + caretHeight) * scaleY,
        scale: scaleY,
    };
}

// Measure the caret in a hidden textarea-shaped mirror, then account for scrolling.
export function textareaCaretRect(input, index = input.selectionStart) {
    index = Math.max(0, Math.min(index, input.value.length));
    const style = getComputedStyle(input);
    const mirror = document.createElement("div");
    mirror.style.position = "absolute";
    mirror.style.left = "-10000px";
    mirror.style.top = "0";
    mirror.style.visibility = "hidden";
    mirror.style.whiteSpace = "pre-wrap";
    mirror.style.overflowWrap = "break-word";
    for (const property of MIRROR_PROPERTIES) mirror.style[property] = style[property];
    mirror.style.overflowY = style.overflowY === "scroll" || input.scrollHeight > input.clientHeight
        ? "scroll" : "hidden";
    mirror.textContent = input.value.slice(0, index);
    const marker = document.createElement("span");
    // The remaining text belongs inside the marker so wrapping matches the
    // textarea at the measured position (as in comfyui-custom-scripts).
    marker.textContent = input.value.slice(index) || ".";
    mirror.append(marker);
    document.body.append(mirror);
    const rect = input.getBoundingClientRect();
    const caretHeight = Number.parseFloat(style.lineHeight) || marker.offsetHeight ||
        Number.parseFloat(style.fontSize) || 13;
    const result = scaleCaretPosition(rect, input.offsetWidth, input.offsetHeight,
        marker.offsetLeft + (Number.parseFloat(style.borderLeftWidth) || 0),
        marker.offsetTop + (Number.parseFloat(style.borderTopWidth) || 0),
        input.scrollLeft, input.scrollTop, caretHeight);
    mirror.remove();
    return result;
}
