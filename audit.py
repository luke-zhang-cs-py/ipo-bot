"""Second-pass auditor: a separate model call checks a memo against the accuracy rules and its sources.

    python audit.py memos/<memo>.md                      audit a saved memo (no sources: rules A1-A11 on the text)
    python audit.py memos/<memo>.md --sources src.json   with the tool results it was written from

From code: audit(memo, bot.sources) after bot.ask(question). Returns
    {"verdict": "pass" | "fail", "checks": [{"id": "A1", "result": "pass|fail|not_applicable", "reason": "..."}],
     "summary": "...", "mechanical": [...verify.py results...]}
The verdict is recomputed here from the checks, so a "pass" with a failed check cannot slip through.
Costs API credit: one call per memo.
"""
import json
import pathlib
import sys

import anthropic

import ipo_bot
import verify

HERE = pathlib.Path(__file__).resolve().parent
AUDITOR = (HERE / "prompts" / "auditor_prompt.md").read_text(encoding="utf-8")
RULES = [f"A{n}" for n in range(1, 12)]
SOURCES_CHARS = 120_000

SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string", "enum": RULES},
                           "result": {"type": "string", "enum": ["pass", "fail", "not_applicable"]},
                           "reason": {"type": "string"}},
            "required": ["id", "result", "reason"], "additionalProperties": False}},
        "summary": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
    },
    "required": ["checks", "summary", "verdict"], "additionalProperties": False,
}


def build_request(memo, sources):
    mech = verify.check_memo(memo)
    src = json.dumps(sources or [], default=str)
    if len(src) > SOURCES_CHARS:
        src = src[:SOURCES_CHARS] + " ...[sources truncated]"
    content = (f"MEMO:\n<<<\n{memo}\n>>>\n\nSOURCES ({len(sources or [])} tool results):\n<<<\n{src}\n>>>\n\n"
               "MECHANICAL CHECKS:\n" + "\n".join(f"- {'PASS' if ok else 'FAIL'} {n}" + (f" ({d})" if d and not ok else "")
                                                 for n, ok, d in mech))
    return content, mech


def parse(text, mech):
    data = json.loads(text)
    seen = {c["id"]: c for c in data.get("checks", []) if c.get("id") in RULES}
    checks = [seen.get(r, {"id": r, "result": "fail", "reason": "the auditor gave no result for this rule"})
              for r in RULES]
    failed = [c for c in checks if c["result"] == "fail"] or [m for m in mech if not m[1]]
    return {"verdict": "fail" if failed else "pass", "checks": checks, "summary": data.get("summary", ""),
            "mechanical": [{"check": n, "pass": ok, "detail": d} for n, ok, d in mech]}


def audit(memo, sources=None, client=None):
    client = client or anthropic.Anthropic()
    content, mech = build_request(memo, sources)
    with client.messages.stream(
        model=ipo_bot.MODEL,
        max_tokens=16_000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=AUDITOR,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        return {"verdict": "fail", "checks": [], "summary": "the auditor declined to review this memo",
                "mechanical": [{"check": n, "pass": ok, "detail": d} for n, ok, d in mech]}
    text = "".join(b.text for b in msg.content if b.type == "text")
    return parse(text, mech)


def report(result):
    lines = [f"Verdict: {result['verdict'].upper()}", result.get("summary", "")]
    for c in result["checks"]:
        lines.append(f"  {c['id']:<4}{c['result']:<15}{c['reason']}")
    bad = [m for m in result["mechanical"] if not m["pass"]]
    lines.append(f"  mechanical: {len(result['mechanical']) - len(bad)} of {len(result['mechanical'])} pass"
                 + "".join(f"\n    FAIL {m['check']} ({m['detail']})" for m in bad))
    return "\n".join(lines)


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if not argv:
        sys.exit(__doc__)
    memo = pathlib.Path(argv[0]).read_text(encoding="utf-8")
    sources = None
    if "--sources" in argv:
        sources = json.loads(pathlib.Path(argv[argv.index("--sources") + 1]).read_text(encoding="utf-8"))
    try:
        result = audit(memo, sources)
    except anthropic.AuthenticationError:
        sys.exit("Claude rejected the API key: set ANTHROPIC_API_KEY or run `ant auth login`.")
    print(report(result))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
