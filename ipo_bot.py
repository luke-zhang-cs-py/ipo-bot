"""IPO and market research bot: Claude with web search and data tools, answering with a one-page memo.

    python ipo_bot.py "Rate the <company> IPO"      one question, memo printed and saved to memos/
    python ipo_bot.py                                interactive: ask follow-ups in one conversation
    python ipo_bot.py --portfolio portfolio.json     review your holdings: buy/add/hold/trim/sell, sized by your rules
    python ipo_bot.py --portfolio portfolio.json "Should I buy <ticker>?"
    python ipo_bot.py --audit "Rate <ticker>"        also run the second-pass auditor on the answer (one more call)

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

HERE = pathlib.Path(__file__).resolve().parent


def load_env(path=HERE / ".env"):
    """KEY=value lines from .env into the environment; a variable already set wins. No other syntax."""
    import os
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env()

import tools as tools_module  # noqa: E402
from tools import TOOL_DEFS, run_tool  # noqa: E402

REVIEW = ("Review my portfolio. For each holding say BUY/ADD/HOLD/TRIM/SELL with reasons, and suggest up to "
          "3 new ideas to buy that fit my rules, each sized with portfolio_size.")
MODEL = "claude-opus-5-5"
# Opus 5.5 can decline a request; "default" re-runs a declined request on the fallback Anthropic
# recommends for that category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORT = "high"            # thinking depth; Opus 5.5 defaults to medium
MAX_TOKENS = 32_000
MAX_TURNS = 40             # model turns per question, tool rounds included
WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 12}
BOSTON = zoneinfo.ZoneInfo("America/New_York")
SOURCE_CHARS = 6000        # how much of each tool result is kept for the auditor

SYSTEM = (HERE / "system_prompt.md").read_text(encoding="utf-8")


def tools(web=True):
    # Our own tools stream their input as it is written; run_tool validates it before running anything.
    own = [{**t, "eager_input_streaming": True} for t in TOOL_DEFS]
    return ([WEB_SEARCH] if web else []) + own


def now_line(as_of=None):
    if as_of:
        return (f"Now: {dt.date.fromisoformat(as_of):%A %d %B %Y} (backtest: use only information available on "
                f"or before this date; the data tools are limited to it and web search is off)")
    return "Now: " + dt.datetime.now(BOSTON).strftime("%A %d %B %Y, %H:%M %Z")


class Bot:
    """use_tools=False: no tools at all (the stale-data test). as_of="YYYY-MM-DD": a backtest, with the data
    tools limited to that date and no web search. Every tool result is kept in self.sources for the auditor."""

    def __init__(self, client=None, log=print, use_tools=True, as_of=None):
        self.client = client or anthropic.Anthropic()
        self.messages = []
        self.log = log
        self.use_tools = use_tools
        self.as_of = as_of
        self.sources = []

    def _request(self):
        extra = {"tools": tools(web=not self.as_of)} if self.use_tools else {}
        with self.client.beta.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
            # The system prompt and tools never change, so they are cached across turns and questions.
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
            **extra,
        ) as stream:
            return stream.get_final_message()

    def ask(self, question):
        """Answer one user message; the conversation carries on across calls."""
        # The date (and whether a portfolio is loaded) goes in the user turn, not the system prompt, so the
        # cached prefix stays the same.
        mode = "\nPortfolio mode: on. The user's portfolio file is loaded (portfolio_view, portfolio_size)." \
            if tools_module.PORTFOLIO["path"] else ""
        if not self.use_tools:
            mode += "\nTools: none are available in this session."
        self.messages.append({"role": "user", "content": f"{now_line(self.as_of)}{mode}\n\n{question}"})
        for _ in range(MAX_TURNS):
            msg = self._request()
            # Append the whole content unchanged (thinking and server-tool blocks included).
            self.messages.append({"role": "assistant", "content": msg.content})
            self._keep_search_results(msg)
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
                self.sources.append({"tool": call.name, "input": args, "result": text[:SOURCE_CHARS], "error": is_error})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": is_error})
            self.messages.append({"role": "user", "content": results})   # all results in one message
        return "Stopped after too many tool rounds without an answer.\n\n" + _disclaimer()

    def _keep_search_results(self, msg):
        for block in msg.content:
            if getattr(block, "type", "") == "web_search_tool_result" and isinstance(getattr(block, "content", None), list):
                hits = [{"title": getattr(r, "title", ""), "url": getattr(r, "url", ""),
                         "page_age": getattr(r, "page_age", None)} for r in block.content]
                self.sources.append({"tool": "web_search", "input": {}, "result": json.dumps(hits)[:SOURCE_CHARS],
                                     "error": False})


def _text(msg):
    return "\n".join(b.text for b in msg.content if b.type == "text").strip()


def _disclaimer():
    from checks import DISCLAIMER
    return DISCLAIMER


def save(question, memo, sources=None):
    """The memo, and beside it the tool results it was written from (for audit.py --sources)."""
    out = HERE / "memos"
    out.mkdir(exist_ok=True)
    stamp = dt.datetime.now(BOSTON).strftime("%Y-%m-%d_%H%M%S")
    path = out / f"{stamp}.md"
    path.write_text(f"> {question}\n\n{memo}\n", encoding="utf-8")
    if sources is not None:
        (out / f"{stamp}.sources.json").write_text(json.dumps(sources, indent=1, default=str), encoding="utf-8")
    return path


def main(argv):
    # Memos use characters a Windows console's default code page lacks (≥, —, ☒ from filings).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = list(argv)
    do_audit = "--audit" in argv
    if do_audit:
        argv.remove("--audit")
    if "--portfolio" in argv:
        i = argv.index("--portfolio")
        if i + 1 >= len(argv):
            sys.exit("--portfolio needs a file, e.g. --portfolio portfolio.json")
        path = pathlib.Path(argv[i + 1]).resolve()
        del argv[i:i + 2]
        import portfolio
        try:
            portfolio.load(path)            # check it now, not halfway through a research run
        except portfolio.PortfolioError as e:
            sys.exit(f"Portfolio file: {e}")
        tools_module.PORTFOLIO["path"] = str(path)
        if not argv:
            argv = [REVIEW]
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
        print(f"\n(saved to {save(q, memo, bot.sources)})")
        if do_audit:
            import audit
            try:
                print("\n" + audit.report(audit.audit(memo, bot.sources)))
            except anthropic.APIError as e:
                print(f"Audit failed: {e}")
        if argv:
            return


if __name__ == "__main__":
    main(sys.argv[1:])
