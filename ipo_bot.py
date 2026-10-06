"""IPO and market research bot: Claude with web search and data tools, answering with a one-page memo.

    python ipo_bot.py "Rate the <company> IPO"      one question, memo printed and saved to memos/
    python ipo_bot.py                                interactive: ask follow-ups in one conversation

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). Data tools use SEC_USER_AGENT, FRED_API_KEY and
FMP_API_KEY when set; without them the bot says which data is missing instead of guessing.
"""
import datetime as dt
import json
import pathlib
import sys
import zoneinfo

import anthropic

try:
    # Verify HTTPS with the operating system's certificate store: needed behind a network that inspects
    # traffic, harmless elsewhere.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from tools import TOOL_DEFS, run_tool

HERE = pathlib.Path(__file__).resolve().parent
MODEL = "claude-opus-5-5"
# Opus 5.5 can decline a request; "default" re-runs a declined request on the fallback Anthropic
# recommends for that category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORT = "high"            # thinking depth; Opus 5.5 defaults to medium
MAX_TOKENS = 32_000
MAX_TURNS = 40             # model turns per question, tool rounds included
WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 12}
BOSTON = zoneinfo.ZoneInfo("America/New_York")

SYSTEM = (HERE / "system_prompt.md").read_text(encoding="utf-8")


def tools():
    # Our own tools stream their input as it is written; run_tool validates it before running anything.
    own = [{**t, "eager_input_streaming": True} for t in TOOL_DEFS]
    return [WEB_SEARCH] + own


def now_line():
    return "Now: " + dt.datetime.now(BOSTON).strftime("%A %d %B %Y, %H:%M %Z")


class Bot:
    def __init__(self, client=None, log=print):
        self.client = client or anthropic.Anthropic()
        self.messages = []
        self.log = log

    def _request(self):
        with self.client.beta.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
            # The system prompt and tools never change, so they are cached across turns and questions.
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=tools(),
            messages=self.messages,
        ) as stream:
            return stream.get_final_message()

    def ask(self, question):
        """Answer one user message; the conversation carries on across calls."""
        # The date goes in the user turn, not the system prompt, so the cached prefix stays the same.
        self.messages.append({"role": "user", "content": f"{now_line()}\n\n{question}"})
        for _ in range(MAX_TURNS):
            msg = self._request()
            # Append the whole content unchanged (thinking and server-tool blocks included).
            self.messages.append({"role": "assistant", "content": msg.content})
            if msg.stop_reason == "refusal":
                return "The request was declined.\n\n" + _disclaimer()
            if msg.stop_reason == "pause_turn":      # a long server-tool turn: send it back to continue
                continue
            if msg.stop_reason == "max_tokens":
                return _text(msg) + "\n\n[cut off at the length limit]\n\n" + _disclaimer()
            calls = [b for b in msg.content if b.type == "tool_use"]
            if not calls:
                return _text(msg)
            results = []
            for call in calls:
                args = call.input if isinstance(call.input, dict) else {}
                self.log(f"  [tool] {call.name} {json.dumps(args)[:120]}")
                text, is_error = run_tool(call.name, args)
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": is_error})
            self.messages.append({"role": "user", "content": results})   # all results in one message
        return "Stopped after too many tool rounds without an answer.\n\n" + _disclaimer()


def _text(msg):
    return "\n".join(b.text for b in msg.content if b.type == "text").strip()


def _disclaimer():
    from checks import DISCLAIMER
    return DISCLAIMER


def save(question, memo):
    out = HERE / "memos"
    out.mkdir(exist_ok=True)
    stamp = dt.datetime.now(BOSTON).strftime("%Y-%m-%d_%H%M%S")
    path = out / f"{stamp}.md"
    path.write_text(f"> {question}\n\n{memo}\n", encoding="utf-8")
    return path


def main(argv):
    # Memos use characters a Windows console's default code page lacks (≥, —, ☒ from filings).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        bot = Bot()
    except anthropic.AnthropicError as e:
        sys.exit(f"Could not start: {e}")
    questions = [" ".join(argv)] if argv else None
    while True:
        q = questions.pop() if questions else input("\nAsk (blank to quit): ").strip()
        if not q:
            return
        try:
            memo = bot.ask(q)
        except anthropic.AuthenticationError:
            sys.exit("Claude rejected the API key: set ANTHROPIC_API_KEY or run `ant auth login`.")
        except anthropic.RateLimitError:
            print("Rate limited; wait a minute and ask again.")
            continue
        except anthropic.APIStatusError as e:
            print(f"Claude API error {e.status_code}: {e.message}")
            continue
        except anthropic.APIConnectionError:
            print("Could not reach the Claude API.")
            continue
        print("\n" + memo)
        print(f"\n(saved to {save(q, memo)})")
        if argv:
            return


if __name__ == "__main__":
    main(sys.argv[1:])
