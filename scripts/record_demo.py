"""Re-record docs/demo.gif from docs/app/: the projection chart, sizing as the stop moves, then the memo checks catching two errors.

    python scripts/record_demo.py
"""
import io
import pathlib

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = 960, 600


def main():
    frames, durations = [], []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="dark")
        pg.goto((ROOT / "docs/app/index.html").as_uri())

        def press(sel):   # a click through the page, so the browser doesn't scroll the button into view
            pg.evaluate("(s) => document.querySelector(s).click()", sel)

        def snap(ms):
            frames.append(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
            durations.append(ms)

        # the projection: the chart at rest, the crosshair walking forward, then a riskier stock
        pg.evaluate("window.scrollTo(0, document.getElementById('pj-tiles').getBoundingClientRect().top + scrollY - 84)")   # under the sticky status bar
        pg.wait_for_timeout(400)
        snap(1800)
        box = pg.locator("#fan svg").bounding_box()
        for f in (0.25, 0.45, 0.65):
            pg.mouse.move(box["x"] + 54 + (box["width"] - 166) * f, box["y"] + box["height"] / 2)
            pg.wait_for_timeout(120)
            snap(700)
        pg.mouse.move(0, 0)
        press('[data-preset="ipo"]')
        pg.wait_for_timeout(300)
        snap(1900)
        press('[data-preset="memo"]')
        press("#t-size")
        pg.evaluate("window.scrollTo(0, document.getElementById('sym').closest('.card').getBoundingClientRect().top + scrollY - 84)")
        pg.wait_for_timeout(200)
        snap(1400)
        for stop in ("47", "48", "49"):
            pg.fill("#stop", stop)
            pg.wait_for_timeout(150)
            snap(1100)
        pg.select_option("#conv", "High")
        pg.wait_for_timeout(150)
        snap(1600)
        pg.evaluate("window.scrollTo(0, 0)")
        press("#t-check")
        # the score and the failing checks (listed first) at the top of the frame
        pg.evaluate("window.scrollTo(0, document.getElementById('score').closest('.card').getBoundingClientRect().top + scrollY - 84)")
        for which, ms in (("good", 1500), ("ev", 1900), ("rating", 2200)):
            press(f'[data-load="{which}"]')
            pg.wait_for_timeout(150)
            snap(ms)
        b.close()
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    out = ROOT / "docs/demo.gif"
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"{out} ({out.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
