"""Render the 15 s launch video (1080x1920, 30 fps) from tools/promo.html.

    python -m http.server 8170          # from the repo root
    python tools/promo_capture.py ar full   (or: en, and: preview for 4 test frames)

Needs playwright + Chrome and ffmpeg. Every frame is drawn by brain.js with advance(), so the video is the same every time."""
import asyncio, os, subprocess, sys
from playwright.async_api import async_playwright
ARGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
lang, mode = sys.argv[1], sys.argv[2]            # mode: preview | full
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
SCENES = [(0, 120), (120, 240), (240, 360), (360, 450)]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=ARGS)
        pg = await b.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=1)
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(f"http://127.0.0.1:8170/tools/promo.html?lang={lang}")
        await pg.wait_for_function("window.ready === true"); await pg.wait_for_timeout(1500)
        if mode == "preview":
            marks = {60: "a", 200: "b", 330: "c", 420: "d"}
            for f in range(0, max(marks) + 1):
                await pg.evaluate(f"F({f})")
                if f in marks:
                    await pg.screenshot(path=f"{OUT}/prev_{lang}_{marks[f]}.jpg", type="jpeg", quality=85)
        else:
            files = []
            for i, (a, z) in enumerate(SCENES):
                tmp, final = f"{OUT}/{lang}_s{i}.tmp.mp4", f"{OUT}/{lang}_s{i}.mp4"
                ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", "30", "-c:v", "mjpeg", "-i", "-",
                                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", tmp], stdin=subprocess.PIPE)
                for f in range(a, z):
                    await pg.evaluate(f"F({f})")
                    ff.stdin.write(await pg.screenshot(type="jpeg", quality=92))
                ff.stdin.close()
                assert ff.wait() == 0
                os.replace(tmp, final); files.append(final); print("scene", i, "done", flush=True)
            lst = f"{OUT}/{lang}_list.txt"
            open(lst, "w").write("".join(f"file '{f}'\n" for f in files))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                            "-movflags", "+faststart", f"{OUT}/eve_launch_{lang}.mp4"], check=True)
        print("errors:", errs)
        await b.close()
asyncio.run(main())
