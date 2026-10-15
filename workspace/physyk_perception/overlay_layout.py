"""Label placement for the perception overlay: each label sits right beside its own box, and only moves out to a column with a
short leader line when nothing beside the box is free. Pure geometry (no cv2), so it is unit-testable.

Boxes for a stacked column nearly coincide, so the usual spot (above the box) collides; the placer tries above, below, right and
left of each box in turn and takes the first spot that touches no box, no earlier label and no reserved area (the HUD text).
"""


def _hit(a, b, pad=0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def place_labels(boxes, text_sizes, img_w, img_h, reserved=(), gap=4, margin=4, priority=None):
    """boxes: [(x0, y0, x1, y1)]; text_sizes: [(w, h)]; reserved: rects labels must not cover.
    Returns, in the order of `boxes`, dicts {"rect": (x0, y0, x1, y1), "leader": None | ((x, y) on the box, (x, y) at the label)}.
    Lower `priority` is placed first (cubes before the tray, so the tray's wide label never takes a cube's spot); within a priority the
    boxes with the most neighbours go first so the crowded ones get the free sides."""
    n = len(boxes)
    placed = [None] * n
    taken = []                                      # label rects already placed
    obstacles = [tuple(b) for b in boxes] + [tuple(r) for r in reserved]

    def free(rect, own):
        if rect[0] < margin or rect[1] < margin or rect[2] > img_w - margin or rect[3] > img_h - margin:
            return False
        for j, o in enumerate(obstacles):
            if j == own and own < n:
                continue
            if _hit(rect, o, 1):
                return False
        return not any(_hit(rect, t, 2) for t in taken)

    crowd = [sum(1 for j in range(n) if j != i and _hit(boxes[i], boxes[j], gap + 14)) for i in range(n)]
    prio = list(priority) if priority is not None else [0] * n
    for i in sorted(range(n), key=lambda i: (prio[i], -crowd[i], boxes[i][1])):
        x0, y0, x1, y1 = boxes[i]
        w, h = text_sizes[i]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cands = [
            (x0, y0 - gap - h),                     # above, left-aligned
            (x0, y1 + gap),                         # below
            (x1 + gap, cy - h / 2.0),               # right
            (x0 - gap - w, cy - h / 2.0),           # left
            (cx - w / 2.0, y0 - gap - h),           # above, centred
            (cx - w / 2.0, y1 + gap),               # below, centred
            (x1 + gap, y0 - gap - h),               # diagonal corners
            (x1 + gap, y1 + gap),
            (x0 - gap - w, y0 - gap - h),
            (x0 - gap - w, y1 + gap),
            (x0, y0 - 3 * gap - h),                 # a little further out
            (x0, y1 + 3 * gap),
            (x1 + 3 * gap, cy - h / 2.0),
            (x0 - 3 * gap - w, cy - h / 2.0),
        ]
        for (lx, ly) in cands:
            rect = (int(lx), int(ly), int(lx + w), int(ly + h))
            if free(rect, i):
                placed[i] = {"rect": rect, "leader": None}
                taken.append(rect)
                break

    # whatever is still unplaced goes to a column beside the cluster, in vertical order, with a leader line to its box
    rest = [i for i in range(n) if placed[i] is None]
    if rest:
        right = max(b[2] for b in boxes) + 14 + max(text_sizes[i][0] for i in rest) + margin <= img_w
        col_x = (max(b[2] for b in boxes) + 14) if right else None
        y = margin
        for i in sorted(rest, key=lambda i: boxes[i][1] + boxes[i][3]):
            w, h = text_sizes[i]
            x0, y0, x1, y1 = boxes[i]
            lx = col_x if right else max(margin, min(b[0] for b in boxes) - 14 - w)
            ly = max((y0 + y1) / 2.0 - h / 2.0, y)
            for _ in range(40):                      # slide down until it clears everything already placed or reserved
                rect = (int(lx), int(ly), int(lx + w), int(ly + h))
                if not any(_hit(rect, t, 2) for t in taken) and not any(_hit(rect, tuple(r)) for r in reserved):
                    break
                ly += h + 3
            ly = min(ly, img_h - margin - h)
            rect = (int(lx), int(ly), int(lx + w), int(ly + h))
            taken.append(rect)
            anchor = (x1, (y0 + y1) // 2) if right else (x0, (y0 + y1) // 2)
            near = (rect[0] - 2, (rect[1] + rect[3]) // 2) if right else (rect[2] + 2, (rect[1] + rect[3]) // 2)
            placed[i] = {"rect": rect, "leader": (anchor, near)}
            y = ly + h + 3
    return placed
