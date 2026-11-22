"""Film a walkthrough of the app: real UI, real clicks, scripted demo replies.

    python tools/walkthrough.py http://127.0.0.1:8162/ path/to/some.docx

Start the app first (a demo backend with scripted replies works best for a repeatable film).
Time inside the page runs K times slower and the screencast is played back K times faster,
so the result is smooth even though the browser renders WebGL in software."""
import asyncio, base64, os, subprocess, sys, time
from playwright.async_api import async_playwright

K = 2.0
FPS = 30
HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/"
DOC = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "project-plan.docx")
OUT = os.path.join(HERE, "out", "eve_walkthrough.mp4")
TMP = OUT.replace(".mp4", ".tmp.mp4")
ARGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
BANNER = base64.b64encode(open(os.path.join(os.path.dirname(HERE), "docs", "banner.jpg"), "rb").read()).decode()

SLOW_TIME = f"""
(() => {{
  const K = {K};
  const realNow = performance.now.bind(performance), t0 = realNow();
  performance.now = () => t0 + (realNow() - t0) / K;
  const rAF = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => rAF(() => cb(performance.now()));
  const sT = window.setTimeout.bind(window), sI = window.setInterval.bind(window);
  window.setTimeout = (f, d, ...a) => sT(f, (d || 0) * K, ...a);
  window.setInterval = (f, d, ...a) => sI(f, (d || 0) * K, ...a);
  localStorage.setItem('lang', 'en');
  localStorage.removeItem('attached');
}})();
"""

OVERLAY = """
(banner) => {
  const css = document.createElement('style');
  css.textContent = `
  #__cur{position:fixed;left:0;top:0;z-index:99999;pointer-events:none;transition:transform .6s cubic-bezier(.45,0,.2,1);transform:translate(900px,700px)}
  #__cur svg{filter:drop-shadow(0 2px 4px rgba(0,0,0,.6))}
  .__rip{position:fixed;z-index:99998;width:44px;height:44px;margin:-22px 0 0 -22px;border:3px solid #19e3ff;border-radius:50%;pointer-events:none;animation:__r .55s ease-out forwards}
  @keyframes __r{from{transform:scale(.2);opacity:1}to{transform:scale(1.3);opacity:0}}
  #__cap{position:fixed;z-index:99990;pointer-events:none;top:64px;padding:12px 20px;font:600 21px/1.35 "Inter","Segoe UI",system-ui,sans-serif;color:#fff;
    background:rgba(8,10,26,.82);border-inline-start:3px solid #19e3ff;box-shadow:0 0 30px rgba(25,227,255,.18);opacity:0;transition:opacity .35s;text-align:start;
    clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%)}
  #__end{position:fixed;inset:0;z-index:99995;pointer-events:none;background:#05060d url(data:image/jpeg;base64,${banner}) center/contain no-repeat;opacity:0;transition:opacity .8s}
  #__end p{position:absolute;left:0;right:0;bottom:70px;text-align:center;font:600 26px "Inter","Segoe UI",sans-serif;color:#19e3ff;letter-spacing:.02em;margin:0}`;
  document.head.append(css);
  const cur = document.createElement('div'); cur.id = '__cur';
  cur.innerHTML = '<svg width="26" height="34" viewBox="0 0 26 34"><path d="M2 2 L2 27 L8.5 21 L13 31 L17.5 29 L13 19.5 L22 19.5 Z" fill="#fff" stroke="#05060d" stroke-width="2" stroke-linejoin="round"/></svg>';
  const cap = document.createElement('div'); cap.id = '__cap';
  const end = document.createElement('div'); end.id = '__end'; end.innerHTML = '<p>github.com/basillt/eve</p>';
  document.body.append(cap, end, cur);
  let cx = 900, cy = 700;
  window.__move = (x, y) => { cx = x; cy = y; cur.style.transform = `translate(${x - 3}px,${y - 2}px)`; };
  window.__tap = () => { const r = document.createElement('div'); r.className = '__rip'; r.style.left = cx + 'px'; r.style.top = cy + 'px'; document.body.append(r); setTimeout(() => r.remove(), 700); };
  window.__cap = (t, mode) => {
    if (!t) { cap.style.opacity = 0; return; }
    const box = document.querySelector(mode === 'side' ? '.side' : '.stage').getBoundingClientRect();
    cap.textContent = t; cap.style.maxWidth = (box.width - 80) + 'px';
    if (mode === 'side') {                       // middle of the empty chat area
      cap.style.left = (box.left + (box.width - cap.offsetWidth) / 2) + 'px'; cap.style.top = Math.round(box.top + box.height * 0.6) + 'px';
    } else { cap.style.left = (box.left + 40) + 'px'; cap.style.top = '64px'; }
    cap.style.opacity = 1;
  };
  window.__end = () => { end.style.opacity = 1; cap.style.opacity = 0; cur.style.opacity = 0; };
}
"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=ARGS)
        pg = await b.new_page(viewport={"width": 1440, "height": 810}, device_scale_factor=4 / 3)
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.add_init_script(SLOW_TIME)
        await pg.goto(URL); await pg.wait_for_timeout(int(3000 * K))
        cdp = await pg.context.new_cdp_session(pg)
        await cdp.send("Animation.enable"); await cdp.send("Animation.setPlaybackRate", {"playbackRate": 1 / K})
        await pg.evaluate(OVERLAY, BANNER)

        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-",
                               "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                               "-r", str(FPS), "-movflags", "+faststart", TMP], stdin=subprocess.PIPE)
        st = {"t0": None, "prev": None, "next": 0.0, "n": 0, "on": True}
        def put(ts, jpg):
            if st["t0"] is None:
                st["t0"] = ts
            vt = (ts - st["t0"]) / K
            if st["prev"] is not None:
                while st["next"] <= vt:
                    ff.stdin.write(st["prev"]); st["next"] += 1 / FPS; st["n"] += 1
            st["prev"] = jpg
        async def on_frame(f):
            try:
                await cdp.send("Page.screencastFrameAck", {"sessionId": f["sessionId"]})
            except Exception:
                pass
            if st["on"]:
                put(f["metadata"]["timestamp"], base64.b64decode(f["data"]))
        cdp.on("Page.screencastFrame", lambda f: asyncio.ensure_future(on_frame(f)))
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 1920, "maxHeight": 1080})

        V = lambda s: pg.wait_for_timeout(int(s * K * 1000))     # wait in video seconds
        cap = lambda t, mode="stage": pg.evaluate("([t, m]) => window.__cap(t, m)", [t, mode])
        async def point(sel, nth=0):
            box = await pg.locator(sel).nth(nth).bounding_box()
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            await pg.evaluate("([x, y]) => window.__move(x, y)", [x, y]); await V(0.7)
        async def click(sel, nth=0, real=True):
            await point(sel, nth); await pg.evaluate("window.__tap()")
            if real:
                await pg.locator(sel).nth(nth).click()
            await V(0.35)
        async def ask(text):
            await click("#input")
            await pg.locator("#input").press_sequentially(text, delay=int(32 * K))
            await V(0.3); await pg.press("#input", "Enter")

        await V(0.6)
        await cap("Eve: an AI assistant that runs on your computer"); await V(3.0)
        await cap("Ask anything. It thinks right here, offline.")
        await ask("Can you plan my week? I have an electronics exam on Tuesday.")
        await pg.wait_for_function("!document.getElementById('send').disabled", timeout=int(60000 * K)); await V(1.4)
        await cap("Mark good answers. Fix the wrong ones.")
        await click(".msg.bot .acts button", 0); await V(1.6)
        await cap("Give it your own files")
        await click("#attach", real=False)
        await pg.set_input_files("#filepick", [DOC]); await V(1.6)
        await cap("Answers show which file they came from")
        await ask("When is the first deadline?")
        await pg.wait_for_function("!document.getElementById('send').disabled", timeout=int(60000 * K)); await V(2.4)
        await cap("Everything it remembers. Delete any line.")
        await click('.tabs button[data-v="memory"]'); await V(2.4)
        await cap("Your files stay on your computer")
        await click('.tabs button[data-v="files"]'); await V(2.0)
        await cap("See what runs on your machine")
        await click('.tabs button[data-v="brain"]'); await V(2.2)
        await cap("At night it learns about you")
        await click("#sleep"); await V(1.2)
        await click('.tabs button[data-v="chat"]')
        await pg.wait_for_selector("#morning.on", timeout=int(90000 * K)); await V(0.6)
        await cap("Every morning: what it learned", "side"); await V(2.4)
        await cap("Share it. Only the lines you pick.", "side")
        await click("#m-share"); await V(0.6)
        await click("#m-pick input", 0); await click("#m-pick input", 2); await V(2.2)
        await click("#m-back"); await click("#m-ok"); await V(0.5)
        await cap(None)
        await click("#langbtn"); await V(0.3)
        await cap("Arabic and English"); await V(2.4)
        await pg.evaluate("window.__end()"); await V(3.2)

        st["on"] = False
        await cdp.send("Page.stopScreencast")
        if st["prev"] is not None:
            for _ in range(FPS // 2):
                ff.stdin.write(st["prev"])
        ff.stdin.close(); code = ff.wait()
        print("frames", st["n"], "seconds", st["n"] / FPS, "ffmpeg", code, "errors", errs)
        if code == 0:
            os.replace(TMP, OUT)
        await b.close()
asyncio.run(main())
