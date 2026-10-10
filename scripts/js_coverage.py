"""Line, function and branch coverage of the browser demo's JavaScript (docs/app/*.js), with no npm.

    python scripts/js_coverage.py                  # runs the Node tests with NODE_V8_COVERAGE set, then reports
    python scripts/js_coverage.py DIR              # reports on raw V8 coverage already written to DIR
    python scripts/js_coverage.py --fail-under 95  # exits 1 if any file's lines, functions or branches are under 95%

Node writes V8's raw block coverage to NODE_V8_COVERAGE for every process it starts (the tests spawn Node, and the
variable reaches it through the inherited environment). This reads those files the way c8 does:
- each process's ranges are nested, so the innermost one gives a character's count; processes are summed;
- a function is covered if it was called in any process (the CommonJS wrapper around the whole file is not counted);
- a branch is one of V8's block ranges (an if/else arm, a ?: arm, the right side of && || ??, a loop body, a catch,
  the code after a return or throw), covered if any character in it ran;
- a line is a line with code (not blank, not only a comment), covered if any of its code ran.
"""
import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "docs" / "app"
TRAILING_COMMENT = re.compile(r"\s+//\s.*$")


def script_path(url):
    """The local file a coverage entry is for: a file:// URL, or the plain path vm.Script was given."""
    if url.startswith("file:"):
        return pathlib.Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path)).resolve()
    return pathlib.Path(url).resolve() if url and os.path.isabs(url) else None


def utf16_text(path):
    """The source and, for each UTF-16 offset V8 reports, the index of its character."""
    with open(path, encoding="utf-8", newline="") as fh:     # as V8 read it: a CRLF file keeps its CRs
        text = fh.read()
    index = []
    for i, ch in enumerate(text):
        index.extend([i] * (2 if ord(ch) > 0xFFFF else 1))
    return text, index


def process_counts(functions, length):
    """One process's count for every UTF-16 offset: the innermost range that holds it."""
    counts = [0] * length
    ranges = sorted((r for f in functions for r in f["ranges"]), key=lambda r: (r["startOffset"], -r["endOffset"]))
    for r in ranges:
        counts[r["startOffset"]:r["endOffset"]] = [r["count"]] * (min(r["endOffset"], length) - r["startOffset"])
    return counts


def collect(cov_dir):
    """{path: (counts summed over processes, {function extent: name}, set of block ranges)} for docs/app/*.js."""
    wanted = {p.resolve() for p in APP.glob("*.js")}
    files = {p: [None, {}, set()] for p in wanted}
    for name in sorted(glob.glob(os.path.join(cov_dir, "*.json"))):
        with open(name, encoding="utf-8") as fh:
            for entry in json.load(fh)["result"]:
                path = script_path(entry["url"])
                if path not in wanted:
                    continue
                length = len(utf16_text(path)[1])
                counts, funcs, blocks = files[path]
                one = process_counts(entry["functions"], length)
                files[path][0] = one if counts is None else [a + b for a, b in zip(counts, one)]
                for f in entry["functions"]:
                    first = f["ranges"][0]
                    if not (first["startOffset"] == 0 and first["endOffset"] >= length):   # the module wrapper
                        funcs[(first["startOffset"], first["endOffset"])] = f["functionName"] or "(anonymous)"
                    if f["isBlockCoverage"]:
                        blocks.update((r["startOffset"], r["endOffset"]) for r in f["ranges"][1:])
    return files


def report(path, counts, funcs, blocks):
    """Coverage figures for one file, and where it misses."""
    text, index = utf16_text(path)
    if counts is None:                                  # never loaded
        counts = [0] * len(index)
    line_of = [text.count("\n", 0, index[o]) + 1 for o in range(len(index))] if index else []
    by_char = {}
    for o, c in enumerate(counts):
        by_char[index[o]] = max(by_char.get(index[o], 0), c)

    lines, missed_lines, start = 0, [], 0
    for n, raw in enumerate(text.split("\n"), 1):
        code = TRAILING_COMMENT.sub("", raw)
        stripped = code.strip()
        if stripped and not stripped.startswith("//"):
            lines += 1
            if not any(by_char.get(start + i, 0) > 0 for i, ch in enumerate(code) if not ch.isspace()):
                missed_lines.append(n)
        start += len(raw) + 1

    def ran(s, e):
        return any(c > 0 for c in counts[s:e])

    missed_funcs = [f"{name} (line {line_of[s]})" for (s, e), name in sorted(funcs.items()) if counts[s] <= 0]
    missed_blocks = sorted({line_of[s] for s, e in blocks if not ran(s, e)})
    hit_blocks = sum(1 for s, e in blocks if ran(s, e))
    return {"lines": (lines - len(missed_lines), lines), "functions": (len(funcs) - len(missed_funcs), len(funcs)),
            "branches": (hit_blocks, len(blocks)), "missed_lines": missed_lines, "missed_functions": missed_funcs,
            "missed_branch_lines": missed_blocks}


def pct(hit_total):
    hit, total = hit_total
    return 100.0 * hit / total if total else 100.0


def cell(hit_total):
    return f"{pct(hit_total):5.1f}% {hit_total[0]:>3}/{hit_total[1]:<3}"


def ranges_text(nums):
    """[3, 4, 5, 9] -> '3-5, 9'"""
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def run_tests(cov_dir):
    tests = sorted(str(p) for p in (ROOT / "tests" / "src").glob("test_js_*.py"))
    env = {**os.environ, "NODE_V8_COVERAGE": cov_dir}
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=ROOT, env=env)
    if r.returncode:
        sys.exit(r.returncode)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cov_dir", nargs="?", help="raw V8 coverage (NODE_V8_COVERAGE); omitted: run the Node tests first")
    ap.add_argument("--fail-under", type=float, default=0, help="exit 1 if any file's lines, functions or branches are below this %%")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        cov_dir = args.cov_dir or tmp
        if not args.cov_dir:
            run_tests(cov_dir)
        files = collect(cov_dir)
        if not any(c is not None for c, _, _ in files.values()):
            sys.exit(f"no coverage for docs/app/*.js in {cov_dir}: is Node installed (IPO_BOT_NODE) and NODE_V8_COVERAGE set?")
        ok = True
        print(f"{'file':<16} {'lines':>16} {'functions':>14} {'branches':>16}")
        for path in sorted(files):
            r = report(path, *files[path])
            print(f"{path.name:<16} {cell(r['lines']):>16} {cell(r['functions']):>14} {cell(r['branches']):>16}")
            for label, key in (("lines", "missed_lines"), ("branches on lines", "missed_branch_lines")):
                if r[key]:
                    print(f"  missed {label}: {ranges_text(r[key])}")
            if r["missed_functions"]:
                print("  missed functions: " + ", ".join(r["missed_functions"]))
            ok = ok and all(pct(r[key]) >= args.fail_under for key in ("lines", "functions", "branches"))
    if not ok:
        sys.exit(f"JavaScript coverage under {args.fail_under}%")


if __name__ == "__main__":
    main()
