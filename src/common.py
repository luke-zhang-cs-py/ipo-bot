"""What the scripts in src/ share: where the repo is, and a console that can print a memo."""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]   # the repo root: .env, prompts/, memos/ and forecasts/ live there


def utf8_console(streams=None):
    """Print in UTF-8: memos use characters a Windows console's default code page lacks (≥, —, ☒ from filings).
    A stream that can't be reconfigured (a test's capture, an old file object) is left as it is."""
    for stream in streams or (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
