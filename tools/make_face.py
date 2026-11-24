"""Turn a reference image into a character "face": a cloud of glowing points, like the brain.

    python tools/make_face.py path/to/portrait.png alexander

Writes web/faces/<id>.json ({aspect, points:[[x,y,b],...]}). The portrait is abstracted into
points of light (its shadows and edges), not a copy of the image — so a public-domain figure
reads as a glowing constellation that reacts to mood in the app. Point a character card at it with
"face": "<id>". Needs Pillow.
"""
import json
import os
import random
import sys

from PIL import Image, ImageFilter, ImageOps

MAX_POINTS = 2400
WIDTH = 200


def make(src, out_id, seed=7):
    random.seed(seed)
    g = Image.open(src).convert("L")
    w = WIDTH
    h = int(w * g.height / g.width)
    g = g.resize((w, h)).filter(ImageFilter.GaussianBlur(1.4))      # drop texture/mosaic tiles
    g = ImageOps.autocontrast(g, cutoff=2)
    lum = g.load()
    edge = g.filter(ImageFilter.GaussianBlur(0.6)).filter(ImageFilter.FIND_EDGES).load()
    pts = []
    for y in range(h):
        for x in range(w):
            L = lum[x, y] / 255.0
            dark = max(0.0, (0.62 - L)) / 0.62                       # subject shadows: hair, eyes, outline
            e = edge[x, y] / 255.0
            weight = 1.25 * (dark ** 1.3) + 0.5 * e
            if weight <= 0.03:
                continue
            if random.random() < min(0.9, weight * 0.8):
                b = round(0.5 + 0.5 * min(1.0, weight), 3)
                pts.append((round(x / w, 4), round(y / h, 4), b))
    random.shuffle(pts)
    pts = pts[:MAX_POINTS]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.join(root, "web", "faces", f"{out_id}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump({"aspect": round(h / w, 3), "points": pts}, f)
    print(f"{dest}: {len(pts)} points, aspect {round(h / w, 3)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/make_face.py <image> <id>")
    make(sys.argv[1], sys.argv[2])
