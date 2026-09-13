"""Conservative source-selector bindings; target text stays verbatim user content.

No LLM call, translation, screen-side inference or target identity inference.
Only explicit color + proxy-shape/object assignment clauses are recognized.
"""
import colorsys
import re

COLORS = {
    'yellow': r'yellow|gold(?:en)?|orange|노란색?|노랑|노랗게|주황색?|황금색?',
    'blue': r'blue|cyan|파란색?|파랑|청록색?|하늘색',
    'red': r'red|빨간색?|빨강|붉은색?',
    'green': r'green|초록색?|녹색',
    'purple': r'purple|violet|보라색?',
    'pink': r'pink|magenta|분홍색?|핑크색?',
    'white': r'white|흰색?|하얀색?|하양',
    'black': r'black|검은색?|검정색?',
    'gray': r'gr[ae]y|회색',
}
NOUN = r'human(?:oid)?|figure|object|proxy|box|sphere|물체|오브젝트|휴먼|인물|박스|상자|스피어|구체'


def color_family(value):
    try:
        r, g, b = (int(value[i:i+2], 16)/255 for i in (1, 3, 5))
    except (ValueError, TypeError):
        return ''
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < .18: return 'black'
    if s < .15: return 'white' if v > .8 else 'gray'
    degrees = h*360
    if degrees < 20 or degrees >= 345: return 'red'
    if degrees < 75: return 'yellow'  # Gold/orange family; ambiguity is never guessed.
    if degrees < 170: return 'green'
    if degrees < 265: return 'blue'
    if degrees < 300: return 'purple'
    return 'pink'


def explicit_bindings(texts, objects):
    """Return unique explicit assignments only; contradictions remain with the user/Qwen."""
    found = {}
    ambiguous = False
    colors = '|'.join(f'(?:{p})' for p in COLORS.values())
    pattern = re.compile(
        rf'(?<![A-Za-z])(?P<color>{colors}|#[0-9a-f]{{6}})\s+(?P<shape>{NOUN})\s*'
        r'(?:은|는|을|를|위치에|자리에|is\b|becomes\b|as\b|->|→|=|:)\s*'
        r'(?P<target>[^,;\n.!?。]+)', re.I)
    for text in texts:
        # Quoted dialogue/visible text is not an instruction to assign a target.
        text = re.sub(r'"[^"\n]*"|“[^”\n]*”', '', str(text))
        text = re.sub(rf'\s+(?:and|그리고)\s+(?=(?:the\s+|its\s+)?(?:{colors})\s+(?:{NOUN}))', '; ', str(text), flags=re.I)
        for match in pattern.finditer(text):
            color, shape, target = match['color'], match['shape'].lower(), match['target'].strip()
            # Negations/exclusions are not positive object-to-target assignments.
            if not target or re.search(r'제외|사용하지|빼|무시|\b(?:exclude|ignore|omit|not|remove)\b', target, re.I):
                continue
            family = next((name for name, p in COLORS.items() if re.fullmatch(p, color, re.I)), '')
            candidates = [o for o in objects if
                          (o.get('color', '').lower() == color.lower() if color.startswith('#')
                           else color_family(o.get('color', '')) == family)]
            explicit_shape = ('box' if shape in ('box', '박스', '상자') else
                              'sphere' if shape in ('sphere', '스피어', '구체') else
                              'human' if shape in ('human', 'humanoid', 'figure', '휴먼', '인물') else None)
            if explicit_shape:
                candidates = [o for o in candidates if o.get('shape') == explicit_shape]
            if len(candidates) != 1:
                ambiguous = True
                continue
            obj = candidates[0]
            ident = obj.get('id', obj.get('selector'))
            assignment = {'object': obj, 'target_text': target, 'source_phrase': match[0].strip()}
            found.setdefault(ident, []).append(assignment)
    bindings = []
    for rows in found.values():
        targets = {r['target_text'] for r in rows}
        if len(targets) == 1:
            bindings.append(rows[0])
        else:
            ambiguous = True
    return bindings, ambiguous


def identity_bindings(texts, objects, camera_source, identity_labels):
    """Bind explicit, already-resolved Subject labels; never infer an identity.

    Camera scope carries across adjacent assignment sentences, but another source
    or an unresolved alias ends it. Conflicting assignments remain semantic work.
    """
    scoped = []
    for text in texts:
        text = re.sub(r'"[^"\n]*"|“[^”\n]*”|<d>.*?</d>', '', str(text), flags=re.S)
        active = False
        for clause in re.split(r'[.;\n!?。]', text):
            sources = re.findall(r'<(?:Video|Picture|Audio) \d+>|@[\w-]+', clause)
            if sources:
                active = all(source == camera_source for source in sources)
            if active:
                scoped.append(clause)
    rows, ambiguous = explicit_bindings(scoped, objects)
    resolved = []
    for row in rows:
        target = re.fullmatch(r'(<Subject \d+>)\s*(?:이다|다|입니다)?', row['target_text'])
        if target and target[1] in identity_labels:
            resolved.append({**row, 'target_label': target[1], 'target_text': target[1]})
    counts = {}
    for row in resolved:
        counts[row['target_label']] = counts.get(row['target_label'], 0) + 1
    if any(count > 1 for count in counts.values()):
        ambiguous = True
    return [row for row in resolved if counts[row['target_label']] == 1], ambiguous
