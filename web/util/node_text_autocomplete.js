// Resolve ownership from node DOM widgets, not from global textarea selectors:
// settings dialogs, search boxes and unrelated editors must remain untouched.
export class NodeTextAutocomplete {
    constructor(app, attach, dedicatedType) {
        this.app = app;
        this.attach = attach;
        this.dedicatedType = dedicatedType;
        this.enabled = false;
        this.nodes = new Set();
        this.tracked = new WeakSet();
        this.controllers = new Map();
        this.onFocus = event => this.activate(event.target);
        this.onInput = event => this.activate(event.target, false);
        this.observer = new MutationObserver(() => {
            for (const [input, entry] of this.controllers) {
                if (!input.isConnected) {
                    entry.controller.dispose();
                    this.controllers.delete(input);
                }
            }
        });
    }

    track(node) {
        if (this.tracked.has(node)) return;
        this.tracked.add(node);
        this.nodes.add(node);
        const manager = this, original = node.onRemoved;
        node.onRemoved = function (...args) {
            manager.nodes.delete(node);
            for (const [input, entry] of manager.controllers) {
                if (entry.node === node) {
                    entry.controller.dispose();
                    manager.controllers.delete(input);
                }
            }
            return original?.apply(this, args);
        };
    }

    owner(input) {
        const nodes = new Set([...(this.app.graph?._nodes || []), ...this.nodes]);
        for (const node of nodes) {
            if (node.comfyClass === this.dedicatedType || node.type === this.dedicatedType) continue;
            for (const widget of node.widgets || []) {
                if (widget.hidden || widget.options?.hidden) continue;
                const element = widget.element || widget.inputEl;
                if (element === input) return { node, widget };
                if (element?.contains?.(input)) return { node, widget: null };
            }
        }
        // Vue-rendered native widgets do not expose their live textarea through
        // inputEl. Their node/widget containers carry the ownership instead.
        const row = input.closest?.('.lg-node-widget');
        const nodeId = row?.closest('[data-node-id]')?.dataset.nodeId;
        if (nodeId !== undefined) {
            const node = [...nodes].find(item => String(item.id) === nodeId);
            if (node && node.comfyClass !== this.dedicatedType && node.type !== this.dedicatedType) {
                // Vue's bubbling input listener updates its widget model.
                return { node, widget: null };
            }
        }
        return null;
    }

    activate(input, update = true) {
        if (!this.enabled || !input?.isConnected || input.readOnly || input.disabled ||
                input.dataset?.toyxyzAutocomplete === 'off' || input.closest?.('[inert]') ||
                !(input instanceof HTMLTextAreaElement ||
                    (input instanceof HTMLInputElement && input.type === 'text'))) return;
        let entry = this.controllers.get(input);
        if (!entry) {
            const owner = this.owner(input);
            if (!owner) return;
            this.track(owner.node);
            const controller = this.attach(owner.node, owner.widget, input, { panels: false });
            if (!controller) return;
            entry = { ...owner, controller };
            this.controllers.set(input, entry);
            // Focus may occur after an already typed query, or the very first
            // input event may create the listeners too late for that event.
            entry.controller.update();
        } else if (update) entry.controller.update();
    }

    setEnabled(enabled) {
        enabled = Boolean(enabled);
        if (enabled === this.enabled) return;
        this.enabled = enabled;
        if (enabled) {
            document.addEventListener('focusin', this.onFocus, true);
            document.addEventListener('input', this.onInput, true);
            this.observer.observe(document.body, { childList: true, subtree: true });
            this.activate(document.activeElement);
        } else {
            document.removeEventListener('focusin', this.onFocus, true);
            document.removeEventListener('input', this.onInput, true);
            this.observer.disconnect();
            for (const entry of this.controllers.values()) entry.controller.dispose();
            this.controllers.clear();
        }
    }
}
