"""Turn a reference image into a character "face": a mosaic of glowing light-dots, like the brain.

    python tools/make_face.py path/to/portrait.png alexander

Writes web/faces/<id>.json: { aspect, grid:[w,h], palette:[[r,g,b],...], points:[[x,y,ci,b],...] }.
Each kept cell of the image becomes one coloured dot; the app draws them as additive glows, so the
portrait reads clearly and glows in many colours. Point a character card at it with "face": "<id>".
Needs Pillow. A clear, high-contrast reference gives the best result.
"""
import json
import os
import sys

from PIL import Image, ImageEnhance

GRID_W = 86          # dots across; height follows the image aspect
COLORS = 24          # palette size
DARK_FLOOR = 0.12    # drop near-black cells (empty background on a dark stage)


def make(src, out_id):
    im = Image.open(src).convert("RGB")
    w = GRID_W
    h = int(w * im.height / im.width)
    small = im.resize((w, h), Image.LANCZOS)
    small = ImageEnhance.Contrast(small).enhance(1.25)
    small = ImageEnhance.Color(small).enhance(1.35)
    q = small.convert("P", palette=Image.ADAPTIVE, colors=COLORS)
    pal = q.getpalette()[:COLORS * 3]
    palette = [[pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]] for i in range(COLORS)]
    qp = q.load()
    gp = small.convert("L").load()
    pts = []
    for y in range(h):
        for x in range(w):
            L = gp[x, y] / 255.0
            if L < DARK_FLOOR:
                continue
            pts.append([round(x / w, 4), round(y / h, 4), int(qp[x, y]), round(0.35 + 0.65 * L, 3)])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.join(root, "web", "faces", f"{out_id}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump({"aspect": round(h / w, 3), "grid": [w, h], "palette": palette, "points": pts}, f)
    print(f"{dest}: {len(pts)} dots, {COLORS} colours, aspect {round(h / w, 3)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/make_face.py <image> <id>")
    make(sys.argv[1], sys.argv[2])
