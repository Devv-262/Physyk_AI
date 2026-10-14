"""Non-overlapping label placement for the perception overlay.

Boxes for a stacked column sit almost on top of each other, so a label above each box collides.
Instead each label goes in a column beside the cluster, ordered by vertical position and pushed
apart to a minimum spacing, with a leader line back to its box. Pure geometry — no cv2 — so it is
unit-testable.
"""


def layout_labels(boxes, text_sizes, img_w, img_h, gap=10, min_spacing=4, margin=6):
    """boxes: list of (x0, y0, x1, y1); text_sizes: list of (w, h) in px.
    Returns a list of dicts {"text_xy": (x, y_baseline), "anchor": (x, y), "side": "right"|"left",
    "label_x", "label_mid_y"} in the same order as `boxes`. `anchor` is the box-edge point the
    leader line starts from."""
    n = len(boxes)
    if n == 0:
        return []
    # Put the label column on whichever side of the cluster has room (right preferred).
    cluster_x1 = max(b[2] for b in boxes)
    cluster_x0 = min(b[0] for b in boxes)
    max_w = max(w for w, _ in text_sizes)
    side = "right" if cluster_x1 + gap + max_w + margin <= img_w else "left"

    centers = [((b[1] + b[3]) / 2.0) for b in boxes]
    order = sorted(range(n), key=lambda i: centers[i])
    ys = {}
    prev_bottom = margin
    for i in order:
        h = text_sizes[i][1]
        top = max(centers[i] - h / 2.0, prev_bottom + min_spacing)
        ys[i] = top
        prev_bottom = top + h
    # If the column ran off the bottom, shift everything up (keeping order and spacing).
    overflow = prev_bottom - (img_h - margin)
    if overflow > 0:
        for i in ys:
            ys[i] -= overflow

    out = [None] * n
    for i in range(n):
        w, h = text_sizes[i]
        if side == "right":
            x = cluster_x1 + gap
            anchor = (boxes[i][2], (boxes[i][1] + boxes[i][3]) // 2)
        else:
            x = max(margin, cluster_x0 - gap - w)
            anchor = (boxes[i][0], (boxes[i][1] + boxes[i][3]) // 2)
        out[i] = {"text_xy": (int(x), int(ys[i] + h)), "anchor": anchor, "side": side,
                  "label_mid_y": int(ys[i] + h / 2.0), "label_x": int(x)}
    return out
