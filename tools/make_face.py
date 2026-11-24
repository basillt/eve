"""Turn a reference image into a character "face": a mosaic of glowing light-dots, like the brain.

    python tools/make_face.py path/to/portrait.png alexander

It cuts the figure out of its background (GrabCut), crops to the subject, and turns each kept cell
into one coloured dot. The app draws them as additive glows, so only the character — not the photo's
background — shows, as a clear, multi-colour cloud of light. Point a character card at it with
"face": "<id>". Needs Pillow, opencv-python and numpy (dev-only; not part of the app).
A clear, high-contrast reference of a single subject gives the best result.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageEnhance

GRID_W = 86
COLORS = 24
DARK_FLOOR = 0.10
PAD = 0.04


def cutout(bgr):
    """A foreground mask of the main figure, via GrabCut + keep-largest-blob."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.06), int(h * 0.08), int(w * 0.88), int(h * 0.92))
    cv2.grabCut(bgr, mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 6, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return fg


def make(src, out_id):
    rgb = Image.open(src).convert("RGB")
    fg = cutout(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
    w = GRID_W
    h = int(w * rgb.height / rgb.width)
    small = ImageEnhance.Color(ImageEnhance.Contrast(rgb.resize((w, h), Image.LANCZOS)).enhance(1.2)).enhance(1.4)
    fgs = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)
    q = small.convert("P", palette=Image.ADAPTIVE, colors=COLORS)
    pal = q.getpalette()[:COLORS * 3]
    palette = [[pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]] for i in range(COLORS)]
    qp, gp = q.load(), small.convert("L").load()
    raw = []
    for y in range(h):
        for x in range(w):
            if fgs[y, x] < 0.5:
                continue
            L = gp[x, y] / 255.0
            if L < DARK_FLOOR:
                continue
            raw.append([x / w, y / h, int(qp[x, y]), round(0.4 + 0.6 * L, 3)])
    if not raw:
        sys.exit("no foreground found — try a clearer reference")
    xs = [p[0] for p in raw]; ys = [p[1] for p in raw]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = (x1 - x0) or 1, (y1 - y0) or 1
    pts = [[round(PAD + (1 - 2 * PAD) * (p[0] - x0) / bw, 4),
            round(PAD + (1 - 2 * PAD) * (p[1] - y0) / bh, 4), p[2], p[3]] for p in raw]
    grid = [max(30, round(w * bw)), max(30, round(w * bh))]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.join(root, "web", "faces", f"{out_id}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump({"aspect": round(bh / bw, 3), "grid": grid, "palette": palette, "points": pts}, f)
    print(f"{dest}: {len(pts)} dots, aspect {round(bh / bw, 3)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/make_face.py <image> <id>")
    make(sys.argv[1], sys.argv[2])
