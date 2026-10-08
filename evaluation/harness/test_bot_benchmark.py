"""Accuracy and integrity tests for a market / IPO prediction bot against rival bots.

Run with either:
    python -m pytest -q test_bot_benchmark.py
    python -m unittest -v test_bot_benchmark
Or print a full scoreboard first:
    python test_bot_benchmark.py

To test your own bot, edit get_setup() below. Everything else stays the same.

In ipo-bot: HARNESS_SETUP picks the data and bots (see real_setup.py):
    demo     the synthetic demo below (the default)
    ipo      the IPO model in evaluation/ipo_eval.py on the real IPO dataset, 2018-2023 (the holdout stays sealed)
    market   the benchmark's stock algorithms on 30 US large caps, monthly
"""
import os
import unittest

import numpy as np
import pandas as pd

import bot_benchmark as bb

# Thresholds. Tighten them as your bot improves; loosen them only with a written reason.
ALPHA = 0.05               # significance level for every "beats X" claim (Holm-adjusted)
MAX_ECE = 0.05             # largest acceptable calibration error
DM_LAG = 0                 # set to (horizon in prediction steps - 1) if horizons overlap
MIN_IPO_EVENTS = 100       # below this, IPO-only claims are skipped as statistically underpowered
MASKING_TOLERANCE = 0.005  # largest Brier change allowed when company identities are hidden
SLICE_WIN_SHARE = 0.75     # share of years / kinds the candidate must win against each opponent
SLICE_MAX_LOSS = 0.01      # and the most it may lose any single slice by (Brier points)
LEAK_CHECK_POINTS = 3      # cut dates used by the scrambled-future test
COST_BPS = 10.0            # trading cost per leg per period in the long-short test


def get_setup():
    """Return your data and bots. The demo uses synthetic data from demo_setup.py.

    store      bb.PointInTimeData with columns available_at, entity, field, value, kind.
               Outcomes live in the same store (field="outcome", entity=event_id,
               available_at = when the outcome became public).
    events     list of bb.Event, the things every bot must predict.
    candidate  factory(store) -> your bot.
    rivals     {label: factory} for the other bots you want to beat. Only bots that can be run
               point-in-time belong here; compare live-only bots in a forward test instead.
    baselines  {label: factory} for the naive models any real edge must clear.
    """
    which = os.environ.get("HARNESS_SETUP", "demo")
    if which in ("ipo", "market"):
        import real_setup
        return real_setup.ipo_setup() if which == "ipo" else real_setup.market_setup()

    from demo_setup import LearningBot, build_demo

    store, events = build_demo()
    return {
        "store": store,
        "events": events,
        "candidate": lambda s: LearningBot(name="candidate"),
        "rivals": {"rival_bot": lambda s: LearningBot(name="rival_bot", signal_noise=1.5, use_revision=False)},
        "baselines": {"base_rate": lambda s: bb.BaseRateBot(), "coin_flip": lambda s: bb.CoinFlipBot()},
    }


def _fmt(df: pd.DataFrame) -> str:
    df = df.copy()
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    return "\n" + df.to_string(index=False)


class BotBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = get_setup()
        cls.cfg = cfg
        cls.store, cls.events = cfg["store"], cfg["events"]
        factories = {"candidate": cfg["candidate"], **cfg["rivals"], **cfg["baselines"]}
        runs = {label: bb.run_walk_forward(f, cls.store, cls.events) for label, f in factories.items()}
        cls.names = {label: r["bot"].iloc[0] for label, r in runs.items()}
        cls.candidate = cls.names["candidate"]
        cls.rivals = [cls.names[k] for k in cfg["rivals"]]
        cls.baselines = [cls.names[k] for k in cfg["baselines"]]
        cls.runs = runs
        cls.joined = bb.attach_outcomes(pd.concat(runs.values(), ignore_index=True), cls.store, cls.events)
        cls.mine = cls.joined[cls.joined["bot"] == cls.candidate]
        cls.cut_dates = bb.pick_cut_dates(cls.events, LEAK_CHECK_POINTS)

    # ---- 1. The test itself must be fair -------------------------------------------------
    def test_01_events_and_outcomes_are_well_formed(self):
        ids = [e.event_id for e in self.events]
        self.assertEqual(len(ids), len(set(ids)), "duplicate event ids")
        o = self.store.frame
        o = o[o["field"] == "outcome"]
        self.assertFalse(o["entity"].duplicated().any(), "some events have more than one outcome row")
        published = o.set_index("entity")["available_at"]
        missing = [i for i in ids if i not in published.index]
        self.assertFalse(missing, f"events without a stored outcome, e.g. {missing[:3]}")
        early = [e.event_id for e in self.events if published[e.event_id] <= e.as_of]
        self.assertFalse(early, f"outcomes visible at prediction time (a built-in leak), e.g. {early[:3]}")

    def test_02_every_bot_has_a_unique_name(self):
        self.assertEqual(len(set(self.names.values())), len(self.names),
                         f"bots share a name, so their predictions would merge: {self.names}")

    def test_03_candidate_answers_every_event(self):
        self.assertEqual(len(self.mine), len(self.events))

    # ---- 2. No cheating ---------------------------------------------------------------
    def test_04_predictions_are_reproducible(self):
        mid = self.cut_dates[len(self.cut_dates) // 2]
        again = bb.run_walk_forward(self.cfg["candidate"], self.store, self.events, until=mid)
        first = self.runs["candidate"]
        first = first[first["as_of"] <= mid]
        m = first.merge(again, on="event_id", suffixes=("", "_again"))
        self.assertEqual(len(m), len(first))
        np.testing.assert_allclose(m["prob_up"], m["prob_up_again"], rtol=0, atol=1e-12,
                                   err_msg="same inputs gave different outputs: pin seeds, model "
                                           "snapshots and sampling temperature")

    def test_05_no_future_information_leaks(self):
        changed = bb.future_invariance(self.cfg["candidate"], self.store, self.events, self.cut_dates)
        self.assertTrue(changed.empty, "predictions changed when only FUTURE data was scrambled, so the "
                                       "bot is reading data it should not have yet:" + _fmt(changed.head()))

    def test_06_leak_detector_catches_known_leaks(self):
        """Tests the test: deliberately leaky bots must be flagged."""
        from demo_setup import CheaterBot, LeakyScalerBot, build_demo

        store, events = build_demo()
        for factory in (CheaterBot, LeakyScalerBot):
            flagged = bb.future_invariance(factory, store, events, bb.pick_cut_dates(events, 1))
            self.assertFalse(flagged.empty, f"leak detector missed {factory.__name__}")

    def test_07_hiding_company_identities_does_not_change_skill(self):
        """For LLM-based bots: a big drop means it remembers outcomes instead of predicting them."""
        m_store, m_events, back = bb.mask_identities(self.store, self.events)
        preds = bb.run_walk_forward(self.cfg["candidate"], m_store, m_events)
        preds["event_id"] = preds["event_id"].map(back)
        masked = bb.attach_outcomes(preds, self.store, self.events)
        change = bb.brier(masked["prob_up"], masked["y"]) - bb.brier(self.mine["prob_up"], self.mine["y"])
        self.assertLessEqual(abs(change), MASKING_TOLERANCE,
                             f"Brier moved by {change:+.4f} when names were hidden; the bot may be "
                             f"recalling these companies rather than forecasting them")

    def test_08_events_postdate_the_models_knowledge_cutoff(self):
        bot = self.cfg["candidate"](self.store)
        cutoff = getattr(bot, "knowledge_cutoff", None)
        if cutoff is None:
            self.skipTest("candidate declares no knowledge_cutoff (only needed for LLM-based bots)")
        stale = [e.event_id for e in self.events if pd.Timestamp(e.as_of) <= pd.Timestamp(cutoff)]
        self.assertFalse(stale, f"{len(stale)} events are older than the model's training data, so it "
                                f"may already know how they turned out, e.g. {stale[:3]}")

    # ---- 3. Is it actually more accurate? ----------------------------------------------
    def test_09_beats_every_baseline(self):
        h2h = bb.head_to_head(self.joined, self.candidate, self.baselines, lag=DM_LAG, alpha=ALPHA)
        self.assertTrue(h2h["wins"].all(), "not significantly better than:" + _fmt(h2h[~h2h["wins"]]))

    def test_10_beats_every_rival(self):
        h2h = bb.head_to_head(self.joined, self.candidate, self.rivals, lag=DM_LAG, alpha=ALPHA)
        self.assertTrue(h2h["wins"].all(), "not significantly better than:" + _fmt(h2h[~h2h["wins"]]))

    def test_11_beats_everyone_on_ipos_alone(self):
        ipo = self.joined[self.joined["kind"] == "ipo"]
        n = int((ipo["bot"] == self.candidate).sum())
        if n < MIN_IPO_EVENTS:
            self.skipTest(f"only {n} IPO events; too few for a reliable IPO-only claim")
        h2h = bb.head_to_head(ipo, self.candidate, self.rivals + self.baselines, lag=DM_LAG, alpha=ALPHA)
        self.assertTrue(h2h["wins"].all(), "on IPOs, not significantly better than:" + _fmt(h2h[~h2h["wins"]]))

    def test_12_probabilities_are_calibrated(self):
        groups = [("all events", self.mine)] + [(k, g) for k, g in self.mine.groupby("kind") if len(g) >= 200]
        for label, g in groups:
            ece = bb.expected_calibration_error(g["prob_up"], g["y"])
            self.assertLessEqual(ece, MAX_ECE, f"{label}: when the bot says X%, it happens {ece:.1%} "
                                               f"more or less often on average")

    def test_13_advantage_holds_across_years_and_kinds(self):
        for by in ("year", "kind"):
            for opp in self.rivals + self.baselines:
                s = bb.slice_comparison(self.joined, self.candidate, opp, by)
                self.assertGreaterEqual((s["diff"] < 0).mean(), SLICE_WIN_SHARE,
                                        f"vs {opp}, wins too few {by} slices:" + _fmt(s))
                self.assertLessEqual(s["diff"].max(), SLICE_MAX_LOSS,
                                     f"vs {opp}, loses one {by} slice badly:" + _fmt(s))

    def test_14_ranks_stocks_better_than_chance(self):
        market = self.mine[self.mine["kind"] == "market"]
        ic, t, n = bb.information_coefficient(market)
        if n < 20:
            self.skipTest("fewer than 20 dates with enough names to measure ranking skill")
        self.assertGreater(ic, 0, f"mean information coefficient is {ic:.3f}")
        self.assertGreater(t, 2.0, f"information coefficient {ic:.3f} is not significant (t = {t:.2f})")

    def test_15_edge_survives_trading_costs(self):
        mine = bb.long_short_returns(self.joined, self.candidate, cost_bps=COST_BPS)
        if len(mine) < 20:
            self.skipTest("not enough market dates for a long-short test")
        self.assertGreater(mine.mean(), 0, f"long-short loses {mine.mean():.4%} per period after costs")
        for opp in self.rivals:
            theirs = bb.long_short_returns(self.joined, opp, cost_bps=COST_BPS)
            if len(theirs) < 20:
                continue
            ci = bb.block_bootstrap_diff(mine, theirs)
            self.assertGreater(ci["ci_low"], 0, f"after costs, the edge over {opp} could be zero: "
                                                f"95% CI [{ci['ci_low']:.4%}, {ci['ci_high']:.4%}] per period")


if __name__ == "__main__":
    cfg = get_setup()
    factories = [cfg["candidate"], *cfg["rivals"].values(), *cfg["baselines"].values()]
    preds = pd.concat([bb.run_walk_forward(f, cfg["store"], cfg["events"]) for f in factories], ignore_index=True)
    joined = bb.attach_outcomes(preds, cfg["store"], cfg["events"])
    names = list(dict.fromkeys(preds["bot"]))
    print(bb.format_report(joined, names[0], names[1:]))
    print()
    unittest.main(verbosity=2)
