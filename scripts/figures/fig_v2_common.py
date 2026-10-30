"""Shared by fig1_front_v2.py and fig2_humility_v2.py.

1. The paper's nomenclature: config codes (mn10, w4g128, n4@512, w8a8 ...) never
   appear on a figure; `system_name` turns a static system code into the paper's
   name and `CODE_MAP` is written into every JSON and the README.
2. The overlap check: after rendering, every text artist's bounding box is tested
   against every other text box, every marker it does not label, every data line
   (hull, stems, series lines, leader lines) and every other panel's box; the
   legend is tested against every panel. `place_label` tries candidate offsets
   for a direct label and keeps the first that passes, with a short leader line
   when the point itself has no free side.
"""
import re

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

MN = {"mn04": "MN04", "mn05": "MN05", "mn10": "MN10", "mn20": "MN20", "dymn04": "DyMN04"}
HEAD = {"esc": "FP32 head", "8": "Q8 head", "4": "Q4 head", "bitnet": "BitNet head"}
CANON_GROUP = {8: 64, 4: 128, 3: 64, 2: 128}     # the group size the paper's AST-Qb names imply
BUDGET = {"b05": "5%", "b10": "10%", "b20": "20%", "b30": "30%", "b50": "50%"}
MID = "#858480"      # leader lines and reference lines: 3.7:1 on white (figstyle AXIS is 2.4:1, kept for spines only)


def ast_name(body):
    """'w4g128@512' -> 'AST-Q4 (512 frames)', 'n4' -> 'AST-Q4n', 'n3g64' -> 'AST-Q3n'."""
    m = re.fullmatch(r"(fp32|([wn])(\d)(?:g(\d+))?(_p8)?)(?:@(\d+)(x2)?)?", body)
    assert m, body
    whole, kind, bits, group, p8, frames, x2 = m.groups()
    if whole == "fp32":
        name = "AST-FP32"
    else:
        bits = int(bits)
        if kind == "w" and group is None:
            name = f"AST-Q{bits} (per-channel, direct)"
        elif kind == "w":
            name = f"AST-Q{bits}" if int(group) == CANON_GROUP.get(bits) else f"AST-Q{bits} (group {group})"
        elif group is None:
            name = f"AST-Q{bits}n"
        else:
            name = f"AST-Q{bits}n" if int(group) == CANON_GROUP.get(bits) else f"AST-Q{bits}n (group {group})"
        if p8:
            name += " (8-bit parent)"
    if frames:
        name += f" ({frames} frames, mean of two windows)" if x2 else f" ({frames} frames)"
    return name


def cheap_name(src, bb):
    """('mn10', 'w8a8') -> 'MN10-Q8'; ('mn10unpr', 'fp32') -> 'MN10-FP32 (exact-length clips)';
    ('mn10int8a', 'int8') -> 'MN10-INT8'; ('eff', 'w8a8') -> 'EffNet-B3-Q8'."""
    base = re.sub(r"(unpr|int8a|int8b)$", "", src)
    model = "EffNet-B3" if base == "eff" else MN[base]
    prec = {"w8a8": "Q8", "fp32": "FP32", "int8": "INT8"}[bb]
    name = f"{model}-{prec}"
    if src.endswith("unpr"):
        name += " (exact-length clips)"
    if src.endswith("int8b"):
        name += " (variant b)"
    return name


def system_name(system):
    """Full name of a static system code, with its head."""
    if system.startswith("ast:"):
        body, head = system[4:].split("+")
        return f"{ast_name(body)}, {HEAD[head]}"
    fam, *rest = system.split(":")
    if fam == "eff":
        bb, head = rest
        return f"{cheap_name('eff', bb)}, {HEAD[head]}"
    src, bb, head = rest
    return f"{cheap_name(src, bb)}, {HEAD[head]}"


def cascade_name(cheap_src, cheap_bb, head, expensive, budget):
    return f"{cheap_name(cheap_src, cheap_bb)}, {HEAD[str(head)]}, to {ast_name(expensive)}, {BUDGET[budget]}"


CODE_MAP = {
    "mn04 / mn05 / mn10 / mn20 / dymn04": "MN04 / MN05 / MN10 / MN20 / DyMN04 (EfficientAT MobileNetV3)",
    "w8a8 (MobileNet, EfficientNet)": "-Q8 suffix, simulated 8-bit weights and activations",
    "fp32": "-FP32 suffix",
    "int8 (mnXXint8a, mnXXint8b)": "-INT8 suffix, real integer kernels",
    "mnXXunpr": "exact-length clips, head trained at exact length",
    "eff": "EffNet-B3",
    "ast:w8g64": "AST-Q8", "ast:w4g128": "AST-Q4", "ast:w3g64": "AST-Q3",
    "ast:n4": "AST-Q4n (nested, per-channel)", "ast:n3g64": "AST-Q3n (nested, group 64)",
    "ast:wB (no group)": "AST-QB (per-channel, direct)", "ast:wBgG (other G)": "AST-QB (group G)",
    "ast:nBgG": "AST-QBn", "@F": "(F frames); omitted at the full 1024", "@256x2": "(256 frames, mean of two windows)",
    "head esc / 8 / 4 / bitnet": "FP32 head / Q8 head / Q4 head / BitNet head",
    "b05 / b20 / b30 / b50": "5% / 20% / 30% / 50% escalation budget",
    "family P3 (stage 3c pick)": "cascade, pre-registered pick",
    "family S4 (stage 3b pick re-tested)": "cascade, re-tested pick",
    "family P5c (unpadded run)": "same, exact-length clips",
}


# ---------------------------------------------------------------- overlap check

def _pad(bb, p):
    return Bbox.from_extents(bb.x0 - p, bb.y0 - p, bb.x1 + p, bb.y1 + p)


def _marker_boxes(fig):
    """One box per scatter point: side = sqrt(s) points plus the edge width, in pixels."""
    px = fig.dpi / 72
    out = []
    for ax in fig.axes:
        for coll in ax.collections:
            if not isinstance(coll, PathCollection) or not coll.get_visible():
                continue
            offs = coll.get_offset_transform().transform(coll.get_offsets())
            sizes = coll.get_sizes()
            lws = coll.get_linewidths()
            for i, (x, y) in enumerate(offs):
                if not ax.bbox.contains(x, y):
                    continue
                side = (np.sqrt(sizes[i % len(sizes)]) + lws[i % len(lws)]) * px
                out.append((Bbox.from_bounds(x - side / 2, y - side / 2, side, side), (x, y), ax, coll.get_label()))
    return out


def _line_paths(fig):
    out = []
    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_visible() or getattr(ln, "_fc_skip", False):
                continue
            xy = ln.get_transform().transform(ln.get_xydata())
            xy = xy[np.isfinite(xy).all(1)]
            if len(xy) < 2:
                continue
            out.append((MPath(xy), ax, ln.get_label() or "line"))
        for a in ax.texts:
            if isinstance(a, Annotation) and a.arrow_patch is not None and a.arrow_patch.get_visible():
                out.append((a.arrow_patch.get_path().transformed(a.arrow_patch.get_transform()), ax, f"leader of '{a.get_text()}'"))
    return out


def _texts(fig, renderer):
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if isinstance(t, Annotation):
            if not t._check_xy(renderer):
                continue
            t.update_positions(renderer)
        bb = Text.get_window_extent(t, renderer)
        if bb.width == 0 or bb.height == 0:
            continue
        out.append((t, bb))
    return out


def _owner_xy(t):
    """Display coords of the point this text labels, or None."""
    if hasattr(t, "_fc_owner"):
        return t._fc_owner
    if isinstance(t, Annotation) and t.axes is not None:
        return tuple(t.axes.transData.transform(t.xy))
    return None


def _crosses(bb, ax, path):
    """Does the line path cross the part of bb that lies inside ax (lines are clipped to their axes)?"""
    inter = Bbox.intersection(bb, ax.bbox)
    if inter is None or inter.width <= 0 or inter.height <= 0:
        return False
    return path.intersects_bbox(inter, filled=False)


def collisions(fig, pad=1.0):
    """List of (kind, a, b) collisions on the rendered figure. Empty when the figure is clean."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = _texts(fig, renderer)
    markers = _marker_boxes(fig)
    lines = _line_paths(fig)
    bad = []
    for i, (t, bb) in enumerate(texts):
        tb = _pad(bb, pad)
        for u, cb in texts[i + 1:]:
            if tb.overlaps(cb):
                bad.append(("text-text", t.get_text(), u.get_text()))
        own = _owner_xy(t)
        for mb, xy, ax, lab in markers:
            if own is not None and abs(xy[0] - own[0]) < 0.5 and abs(xy[1] - own[1]) < 0.5:
                continue
            if tb.overlaps(mb):
                bad.append(("text-marker", t.get_text(), f"{lab} at {ax.get_title(loc='left')!r} {_data(ax, xy)}"))
        for path, ax, lab in lines:
            if lab.startswith("leader of") and lab == f"leader of '{t.get_text()}'":
                continue
            if _crosses(tb, ax, path):
                bad.append(("text-line", t.get_text(), f"{lab} in {ax.get_title(loc='left')!r}"))
        for ax in fig.axes:
            if ax is not t.axes and tb.overlaps(ax.bbox):
                bad.append(("text-panel", t.get_text(), ax.get_title(loc='left')))
    for leg in fig.legends + [l for ax in fig.axes for l in ax.get_children() if isinstance(l, Legend)]:
        lb = _pad(leg.get_window_extent(renderer), pad)
        for ax in fig.axes:
            if lb.overlaps(ax.bbox):
                bad.append(("legend-panel", "legend", ax.get_title(loc='left')))
        for mb, xy, ax, lab in markers:
            if lb.overlaps(mb):
                bad.append(("legend-marker", "legend", lab))
    return bad, len(texts), len(markers), len(lines)


def _data(ax, xy):
    x, y = ax.transData.inverted().transform(xy)
    return f"({x:.3g}, {y:.4g})"


def report(fig, name, strict=True):
    bad, nt, nm, nl = collisions(fig)
    head = f"OVERLAP CHECK {name}: {len(bad)} collisions among {nt} text boxes, {nm} markers, {nl} lines"
    lines = [head] + [f"  {k}: {a!r} vs {b!r}" for k, a, b in bad]
    text = "\n".join(lines)
    print(text)
    if strict:
        assert not bad, text
    return text


# ---------------------------------------------------------------- label placement

# Candidate offsets in points: (dx, dy, ha, va). The near ring sits beside the point without a
# leader; the far rings get a thin leader line. Direction keys: r l u d ur ul dr dl (and finer
# angles on the far rings).
DIRS = {"r": 0, "ur": 45, "u": 90, "ul": 135, "l": 180, "dl": 225, "d": 270, "dr": 315}
ORDER = ["r", "l", "u", "d", "ur", "ul", "dr", "dl"]
# stacked variants: beside the point but one text line up or down, for two labels on one side
STACK = {"ul+": (-4, 5, "right", "bottom"), "dl+": (-4, -5, "right", "top"),
         "ur+": (4, 5, "left", "bottom"), "dr+": (4, -5, "left", "top")}


def _offset(angle, dist):
    rad = np.deg2rad(angle)
    dx, dy = dist * np.cos(rad), dist * np.sin(rad)
    ha = "center" if abs(np.cos(rad)) < 0.3 else ("left" if dx > 0 else "right")
    va = "center" if abs(np.sin(rad)) < 0.3 else ("bottom" if dy > 0 else "top")
    return (round(float(dx), 2), round(float(dy), 2), ha, va)


def _candidates(prefer, with_leader):
    order = list(prefer or []) + [k for k in ORDER if k not in (prefer or [])]
    near = [(k, STACK[k] if k in STACK else _offset(DIRS[k], 5.0 if k in ("r", "l") else 4.5), False) for k in order]
    order = [k for k in order if k in DIRS]
    far = []
    if with_leader:
        for dist, step in ((13, 45), (22, 30), (32, 30), (42, 15), (52, 15), (62, 15)):
            angles = sorted({DIRS[k] for k in order} | set(np.arange(0, 360, step).tolist()),
                            key=lambda a: min(abs(a - DIRS[k]) % 360 for k in order[:1]) if order else a)
            pref_angles = [DIRS[k] for k in order]
            angles = pref_angles + [a for a in angles if a not in pref_angles]
            for a in angles:
                dx, dy, ha, va = _offset(a, dist)
                far.append((f"{dist}pt@{a:g}", (dx, dy, ha, va), True))
                for ha2 in ("right", "center", "left"):       # the box may also hug a panel edge
                    if ha2 != ha:
                        far.append((f"{dist}pt@{a:g}/{ha2}", (dx, dy, ha2, va), True))
    return near + far


def place_label(fig, ax, xy, text, prefer=None, leader_color=MID, pad=1.0, forbid_leader=False,
                alt_text=None, quiet=False, inside=True, **kw):
    """Annotate `xy` (data coords) with `text`, trying nearby offsets first (in `prefer` order,
    then the rest), then farther offsets with a leader line; then the same with `alt_text` (a
    one- or two-line variant of the same name). Returns the Annotation or None."""
    renderer = fig.canvas.get_renderer()
    cands = [(k, o, l, text) for k, o, l in _candidates(prefer, not forbid_leader)]
    if alt_text:
        cands += [(k, o, l, alt_text) for k, o, l in _candidates(prefer, not forbid_leader)]
    scene = (_marker_boxes(fig), _line_paths(fig), _texts(fig, renderer))   # fixed while this label moves
    why = []
    for key, (dx, dy, ha, va), leader, txt in cands:
        props = dict(arrowstyle="-", color=leader_color, lw=0.5, shrinkA=1.5, shrinkB=2.5) if leader else None
        ann = ax.annotate(txt, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                          arrowprops=props, **kw)
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_zorder(ann.get_zorder() - 0.5)
        reason = _blocked(fig, ax, ann, renderer, pad, scene, inside)
        if reason is None:
            ann._fc_where = key + (" (leader)" if leader else "")
            return ann
        why.append(f"{'far ' if leader else ''}{key}: {reason}")
        ann.remove()
    if not quiet:
        print(f"  WARNING: no free spot for label {text!r} at {xy} in {ax.get_title(loc='left')!r}; dropped")
        for w in why[:(60 if __import__('os').environ.get('FIGCHECK_VERBOSE') else 12)]:
            print(f"      {w}")
    return None


def _segments_cross(p, q):
    """Do two polylines (display coords) intersect? Vectorised segment test."""
    a = p.vertices
    b = q.vertices
    a0, a1 = a[:-1], a[1:]
    b0, b1 = b[:-1], b[1:]

    def orient(o, p1, p2):
        return (p1[..., 0] - o[..., 0]) * (p2[..., 1] - o[..., 1]) - (p1[..., 1] - o[..., 1]) * (p2[..., 0] - o[..., 0])

    A0, A1 = a0[:, None, :], a1[:, None, :]
    B0, B1 = b0[None, :, :], b1[None, :, :]
    d1 = orient(A0, A1, B0) * orient(A0, A1, B1)
    d2 = orient(B0, B1, A0) * orient(B0, B1, A1)
    return bool(((d1 < 0) & (d2 < 0)).any())


def _blocked(fig, ax, ann, renderer, pad, scene, inside=True):
    """None when the annotation (text box and leader) is clear of every other text, marker, line
    and panel; otherwise a short reason. `inside` keeps the box inside its own panel; otherwise
    it may spill into empty figure space but never into another panel."""
    markers, lines, texts = scene
    ann.update_positions(renderer)
    bb = _pad(Text.get_window_extent(ann, renderer), pad)
    box = ax.bbox.padded(2) if inside else fig.bbox
    if not (box.contains(bb.x0, bb.y0) and box.contains(bb.x1, bb.y1)):
        return "outside panel" if inside else "outside figure"
    if not inside and any(bb.overlaps(o.bbox) for o in fig.axes if o is not ax):
        return "inside another panel"
    own = _owner_xy(ann)
    leader = None
    if ann.arrow_patch is not None:
        leader = ann.arrow_patch.get_path().transformed(ann.arrow_patch.get_transform())
    for mb, xy, max_, lab in markers:
        if abs(xy[0] - own[0]) < 0.5 and abs(xy[1] - own[1]) < 0.5:
            continue
        if bb.overlaps(mb):
            return f"marker {lab} {_data(max_, xy)}"
        if leader is not None and leader.intersects_bbox(_pad(mb, 1.0), filled=False):
            return f"leader hits marker {lab}"
    for path, pax, lab in lines:
        if _crosses(bb, pax, path):
            return f"line {lab}"
        if leader is not None and pax is ax and not lab.startswith("leader of") and _segments_cross(leader, path):
            return f"leader crosses {lab}"
    for t, tb in texts:
        if bb.overlaps(tb):
            return f"text {t.get_text()!r}"
        if leader is not None and leader.intersects_bbox(_pad(tb, pad + 0.5), filled=False):
            return f"leader hits text {t.get_text()!r}"
    return None
