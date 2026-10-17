// Consume only keys actually owned by a focused, visible editor. In particular,
// an old widget retained while switching workflows must not own graph shortcuts.
export function ownsEditorKey(event, root, node, app) {
    if(event.defaultPrevented ||
        !root?.isConnected || !root.contains(event.target) ||
        event.target!==root.ownerDocument.activeElement || !root.getClientRects().length) return false;
    if(root.checkVisibility && !root.checkVisibility()) return false;
    const canvas=app?.canvas;
    if(node?.graph && canvas) {
        if(canvas.graph!==node.graph) return false;
        if(!(canvas.selectedItems?.has(node) || canvas.selected_nodes?.[node.id]===node ||
            node.is_selected || node.selected)) return false;
    }
    return true;
}

export function plainEditorKey(event) {
    return !event.defaultPrevented && !event.ctrlKey && !event.metaKey &&
        !event.altKey && !event.isComposing && event.keyCode!==229;
}
