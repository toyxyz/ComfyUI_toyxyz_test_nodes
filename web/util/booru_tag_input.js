const SEPARATORS = /[,.;\n]/;

export function currentTagRange(value, selectionStart, selectionEnd = selectionStart) {
    if (selectionStart !== selectionEnd) return null;
    const cursor = Math.max(0, Math.min(selectionStart, value.length));
    let start = cursor;
    while (start > 0 && !SEPARATORS.test(value[start - 1])) start--;
    while (start < cursor && /\s/.test(value[start])) start++;
    const query = value.slice(start, cursor).trim();
    return query ? { start, end: cursor, query } : null;
}

export function insertTag(value, range, tag) {
    const before = value.slice(0, range.start);
    const after = value.slice(range.end);
    // Keep every character after the caret. Separate a remaining fragment
    // from the completed tag instead of treating it as text to replace.
    const nextIsDelimiter = /^\s*[,.;\n]/.test(after);
    const separator = nextIsDelimiter ? "" : /^\s/.test(after) ? "," : ", ";
    const insertion = tag + separator;
    return {
        value: before + insertion + after,
        caret: before.length + insertion.length,
    };
}

export function formatSelectedTag(tag) {
    return tag.replaceAll("_", " ").replaceAll("(", "\\(").replaceAll(")", "\\)");
}

export function matchRanges(label, query) {
    const needle = query.trim().toLowerCase()
        .replace(/\\([()])/g, "$1").replace(/\s+/g, "_");
    if (!needle) return [];
    let haystack = "";
    const starts = [];
    const ends = [];
    for (let index = 0; index < label.length; index++) {
        if (label[index] === "\\" && /[()]/.test(label[index + 1] || "")) continue;
        haystack += label[index] === " " ? "_" : label[index].toLowerCase();
        starts.push(index);
        ends.push(index + 1);
    }
    const ranges = [];
    for (let start = haystack.indexOf(needle); start >= 0;
            start = haystack.indexOf(needle, start + needle.length)) {
        ranges.push([starts[start], ends[start + needle.length - 1]]);
    }
    return ranges;
}
