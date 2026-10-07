"""Re-record docs/demo.gif from docs/app/: sizing as the stop moves, then the memo checks catching two errors.

    python tools_dev/record_demo.py
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
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="light")
        pg.goto((ROOT / "docs/app/index.html").as_uri())

        def snap(ms):
            frames.append(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
            durations.append(ms)

        pg.evaluate("window.scrollTo(0, 260)")
        snap(1600)
        for stop in ("47", "48", "49"):
            pg.fill("#stop", stop)
            pg.wait_for_timeout(150)
            snap(1100)
        pg.select_option("#conv", "High")
        pg.wait_for_timeout(150)
        snap(1600)
        pg.evaluate("window.scrollTo(0, 0)")
        pg.click("#t-check")
        # the score and the failing checks (listed first) at the top of the frame
        pg.evaluate("window.scrollTo(0, document.getElementById('p-check').getBoundingClientRect().top + scrollY - 12)")
        for which, ms in (("good", 1500), ("ev", 1900), ("rating", 2200)):
            pg.click(f'[data-load="{which}"]')
            pg.wait_for_timeout(150)
            snap(ms)
        b.close()
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    out = ROOT / "docs/demo.gif"
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"{out} ({out.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
