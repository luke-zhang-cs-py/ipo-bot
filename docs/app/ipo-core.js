// The browser demo's logic: portfolio.py's sizing rules and verify.py's KEY NUMBERS checks, line for line.
// tests/test_js_parity.py runs both on the same inputs and fails if they disagree, so this file and the
// Python cannot drift apart. The projection charts at the end are JS-only; the same test checks their properties.
// Loaded by docs/app/index.html, and by Node for that test.
(function (root) {
  "use strict";
  const DEFAULT_RULES = { max_position_pct: 10.0, max_sector_pct: 30.0, risk_per_trade_pct: 1.0, min_cash_pct: 5.0,
    conviction_scale: { Low: 0.5, Medium: 1.0, High: 1.5 } };

  const GUARDRAILS = { max_daily_loss_pct: 100, earnings_blackout_hours: 24 * 30, max_volume_pct: 100,   // each one's ceiling
    max_order_value: 1e12, max_gross_exposure_pct: 100, max_drawdown_pct: 100 };

  class PortfolioError extends Error {}

  function num(v, what, minimum = 0, allowZero = true) {
    if (typeof v !== "number" || !isFinite(v)) throw new PortfolioError(what + " must be a number");
    if (v < minimum || (!allowZero && v === 0)) throw new PortfolioError(what + " must be " + (allowZero ? "at least" : "above") + " " + minimum);
    return v;
  }

  function validate(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new PortfolioError("the portfolio file must be a JSON object");
    const rules = Object.assign({}, DEFAULT_RULES, raw.rules || {});
    for (const k of ["max_position_pct", "max_sector_pct", "risk_per_trade_pct", "min_cash_pct"]) {
      rules[k] = num(rules[k], "rules." + k);
      if (rules[k] > 100) throw new PortfolioError("rules." + k + " is a percentage: at most 100");
    }
    for (const [k, ceiling] of Object.entries(GUARDRAILS)) {
      if (rules[k] != null) {
        rules[k] = num(rules[k], "rules." + k);
        if (rules[k] > ceiling) throw new PortfolioError("rules." + k + " is at most " + ceiling);
      }
    }
    const equityHigh = raw.equity_high == null ? null : num(raw.equity_high, "equity_high");
    const halted = raw.halted === undefined ? false : raw.halted;
    if (typeof halted !== "boolean") throw new PortfolioError("halted must be true or false");
    const dayPnl = raw.day_pnl === undefined ? 0 : raw.day_pnl;
    if (typeof dayPnl !== "number" || !isFinite(dayPnl)) throw new PortfolioError("day_pnl must be a number (negative for a loss)");
    const scale = {};
    for (const [k, v] of Object.entries(rules.conviction_scale || {})) scale[String(k)] = num(v, "conviction_scale." + k);
    rules.conviction_scale = scale;
    const holdings = [], seen = new Set();
    (raw.holdings || []).forEach((h, i) => {
      if (!h || typeof h !== "object" || !String(h.symbol || "").trim()) throw new PortfolioError("holding " + (i + 1) + " needs a symbol");
      const sym = String(h.symbol).trim().toUpperCase();
      if (seen.has(sym)) throw new PortfolioError(sym + " is listed twice");
      seen.add(sym);
      holdings.push({ symbol: sym, shares: num(h.shares === undefined ? 0 : h.shares, sym + " shares"),
        cost_basis: h.cost_basis == null ? null : num(h.cost_basis, sym + " cost_basis"), sector: String(h.sector || "Unknown"),
        price: h.price == null ? null : num(h.price, sym + " price", 0, false), price_date: String(h.price_date || raw.as_of || "") });
    });
    return { as_of: String(raw.as_of || ""), cash: num(raw.cash === undefined ? 0 : raw.cash, "cash"), holdings, rules, day_pnl: dayPnl,
      equity_high: equityHigh, halted };
  }

  function valued(pf) {
    const rows = [], unpriced = [];
    for (const h of pf.holdings) {
      if (h.price == null) { unpriced.push(h.symbol); continue; }
      rows.push(Object.assign({}, h, { price_source: "portfolio file, " + (h.price_date || "undated"), value: h.shares * h.price }));
    }
    const invested = rows.reduce((a, r) => a + r.value, 0), equity = invested + pf.cash, sectors = Object.create(null);
    for (const r of rows) {
      r.weight_pct = equity ? 100 * r.value / equity : 0;
      if (r.cost_basis) r.gain_pct = Math.round(100 * 100 * (r.price / r.cost_basis - 1)) / 100;
      sectors[r.sector] = (sectors[r.sector] || 0) + r.value;
    }
    const rules = pf.rules, breaches = [];
    for (const r of rows) if (r.weight_pct > rules.max_position_pct + 1e-9) {
      const excess = r.value - equity * rules.max_position_pct / 100;
      breaches.push({ rule: "max_position_pct", symbol: r.symbol, weight_pct: round2(r.weight_pct), shares_over: Math.ceil(excess / r.price) });
    }
    for (const [s, v] of Object.entries(sectors)) {
      const pct = equity ? 100 * v / equity : 0;
      if (pct > rules.max_sector_pct + 1e-9) breaches.push({ rule: "max_sector_pct", sector: s, weight_pct: round2(pct) });
    }
    const cashPct = equity ? 100 * pf.cash / equity : 0;
    if (cashPct < rules.min_cash_pct - 1e-9) breaches.push({ rule: "min_cash_pct", cash_pct: round2(cashPct) });
    const sectorPct = {};
    for (const [s, v] of Object.entries(sectors)) sectorPct[s] = equity ? 100 * v / equity : 0;
    return { as_of: pf.as_of, equity, cash: pf.cash, cash_pct: cashPct, holdings: rows, sector_pct: sectorPct, unpriced, rule_breaches: breaches, rules,
      day_pnl: pf.day_pnl || 0, equity_high: pf.equity_high == null ? null : pf.equity_high, halted: !!pf.halted, invested };
  }

  const round2 = (x) => Math.round(x * 100) / 100;

  // An ISO date or date-time in ms since the epoch; a bare date (or one with no zone) is read as UTC, as portfolio.py does.
  function when(v, what) {
    const s = String(v);
    const t = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(s) || !s.includes("T") ? s : s + "Z");
    if (!isFinite(t)) throw new PortfolioError(what + " must be an ISO date or date-time");
    return t;
  }

  function sizePosition(view, symbol, entryPrice, stopPrice, conviction = "Medium", sector = null, opts = {}) {
    const entry = num(entryPrice, "entry_price", 0, false), stop = num(stopPrice, "stop_price");
    if (stop >= entry) throw new PortfolioError("the stop-working price must be below the entry price for a purchase");
    const rules = view.rules, equity = view.equity, scale = Object.hasOwn(rules.conviction_scale, String(conviction)) ? rules.conviction_scale[String(conviction)] : undefined;
    if (scale === undefined) throw new PortfolioError("conviction must be one of " + Object.keys(rules.conviction_scale).sort().join(", "));
    const sym = String(symbol).trim().toUpperCase();
    if ((view.unpriced || []).includes(sym)) throw new PortfolioError(sym + " is held but has no price: its position and sector limits cannot be checked");
    const held = view.holdings.find((h) => h.symbol === sym) || null;
    sector = sector || (held ? held.sector : "Unknown");
    const heldValue = held ? held.value : 0, sectorValue = (Object.hasOwn(view.sector_pct, sector) ? view.sector_pct[sector] : 0) * equity / 100;
    const riskBudget = equity * rules.risk_per_trade_pct / 100 * scale;
    const limits = {
      risk: Math.floor(riskBudget / (entry - stop)),
      position: Math.floor(Math.max(0, equity * rules.max_position_pct / 100 - heldValue) / entry),
      cash: Math.floor(Math.max(0, view.cash - equity * rules.min_cash_pct / 100) / entry),
    };
    if (sector !== "Unknown") limits.sector = Math.floor(Math.max(0, equity * rules.max_sector_pct / 100 - sectorValue) / entry);
    const warnings = [];
    if (rules.max_daily_loss_pct != null) {
      const lost = -Math.min(0, view.day_pnl || 0);
      if (equity && lost >= equity * rules.max_daily_loss_pct / 100 - 1e-9) limits.daily_loss = 0;
    }
    if (rules.earnings_blackout_hours != null) {
      if (opts.next_earnings == null) warnings.push("earnings_blackout_hours is set but no earnings date was given: the blackout was not checked");
      else {
        const hours = (when(opts.next_earnings, "next_earnings") - (opts.now != null ? when(opts.now, "now") : Date.now())) / 3.6e6;
        if (hours >= 0 && hours <= rules.earnings_blackout_hours) limits.earnings_blackout = 0;
      }
    }
    if (rules.max_volume_pct != null) {
      if (opts.avg_volume == null) warnings.push("max_volume_pct is set but no average volume was given: the order size was not checked against it");
      else limits.liquidity = Math.floor(num(opts.avg_volume, "avg_volume") * rules.max_volume_pct / 100);
    }
    if (rules.max_order_value != null) limits.order_value = Math.floor(rules.max_order_value / entry);
    if (rules.max_gross_exposure_pct != null) {
      const invested = view.invested != null ? view.invested : view.holdings.reduce((a, h) => a + h.value, 0);
      limits.exposure = Math.floor(Math.max(0, equity * rules.max_gross_exposure_pct / 100 - invested) / entry);
    }
    if (rules.max_drawdown_pct != null) {
      if (view.equity_high == null) warnings.push("max_drawdown_pct is set but equity_high is not: the drawdown was not checked");
      else if (equity <= view.equity_high * (1 - rules.max_drawdown_pct / 100) + 1e-9) limits.drawdown = 0;
    }
    if (view.halted) limits.kill_switch = 0;
    const shares = Math.max(0, Math.min(...Object.values(limits)));
    let binding = null;
    for (const k of Object.keys(limits)) if (binding === null || limits[k] < limits[binding]) binding = k;
    const cost = shares * entry, newValue = heldValue + cost;
    return { symbol: sym, shares, cost: round2(cost), entry_price: entry, stop_price: stop, conviction, binding_rule: binding,
      limits_in_shares: limits, risk_budget: round2(riskBudget), loss_at_stop: round2(shares * (entry - stop)),
      weight_after_pct: equity ? round2(100 * newValue / equity) : 0, cash_after: round2(view.cash - cost), warnings };
  }

  // ---------------------------------------------------------------- verify.py
  const REL_TOL = 0.005, OVERWEIGHT = 15, UNDERWEIGHT = -10, CONVICTION_OK = new Set(["Medium", "High"]);
  const RATINGS = new Set(["Overweight", "Equal-weight", "Underweight", "NOT RATED"]);

  function extract(memo) {
    const blocks = [...String(memo).matchAll(/```json\s*([\s\S]*?)```/gi)].map((m) => m[1]);
    let bad = null;
    for (const raw of blocks.reverse()) {           // an invalid block (an example, a stray snippet) does not hide a valid one
      let data;
      try { data = JSON.parse(raw); } catch (e) { bad = bad || e; continue; }
      if (isDict(data) && isDict(data.key_numbers)) return data.key_numbers;
    }
    if (bad) throw new Error("the KEY NUMBERS block is not valid JSON (" + bad.message + ")");
    throw new Error('no ```json block with a "key_numbers" object');
  }
  const isNum = (x) => typeof x === "number" && isFinite(x);
  // The block is written by a model, so any shape can arrive: a wrong shape is a failed check, read the way the Python reads it.
  const isDict = (x) => !!x && typeof x === "object" && !Array.isArray(x);
  const truthy = (x) => Array.isArray(x) || typeof x === "string" ? x.length > 0 : isDict(x) ? Object.keys(x).length > 0 : !!x;
  const list = (x) => Array.isArray(x) ? x : typeof x === "string" ? [...x] : isDict(x) ? Object.keys(x) : [];    // what Python iterates
  const pyStr = (x) => typeof x === "string" ? x : x == null ? "None" : typeof x === "boolean" ? (x ? "True" : "False") : typeof x === "object" ? JSON.stringify(x) : String(x);
  const pyType = (x) => Array.isArray(x) ? "list" : typeof x === "string" ? "str" : typeof x === "number" ? (Number.isInteger(x) ? "int" : "float") : typeof x === "boolean" ? "bool" : "dict";
  const DIGITS = "\\d(?:_?\\d)*", FLOAT = new RegExp("^[+-]?(?:" + DIGITS + "(?:\\.(?:" + DIGITS + ")?)?|\\." + DIGITS + ")(?:[eE][+-]?" + DIGITS + ")?$");
  function pyFloat(x) {                         // Python's float(): a number or a numeric string; null, "", lists and objects raise
    if (typeof x === "number") return x;
    if (typeof x === "boolean") return +x;      // float(True) is 1.0
    if (typeof x === "string") {
      const s = x.trim();
      if (/^[+-]?(inf|infinity)$/i.test(s)) return s[0] === "-" ? -Infinity : Infinity;
      if (/^[+-]?nan$/i.test(s)) return NaN;
      if (FLOAT.test(s)) return Number(s.replace(/_/g, ""));
    }
    throw new TypeError("not a number: " + pyStr(x));
  }
  const close = (a, b, rel = REL_TOL, abs = 1e-9) => a != null && b != null && Math.abs(a - b) <= Math.max(rel * Math.max(Math.abs(a), Math.abs(b)), abs);
  const date = (s) => { const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s || "")); if (!m) return null; const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])); return d.getUTCMonth() === +m[2] - 1 ? d : null; };
  function tradingDays(a, b) { let n = 0; for (let d = new Date(a.getTime() + 864e5); d <= b; d = new Date(d.getTime() + 864e5)) if (d.getUTCDay() % 6) n++; return n; }

  function check(k) {
    const out = [], add = (name, ok, detail = "") => out.push([name, !!ok, detail]);
    const rawIn = truthy(k.inputs) ? k.inputs : {}, rawOut = truthy(k.outputs) ? k.outputs : {};
    add("inputs is an object", isDict(rawIn), pyType(rawIn));
    add("outputs is an object", isDict(rawOut), pyType(rawOut));
    const inputs = isDict(rawIn) ? rawIn : {}, outputs = isDict(rawOut) ? rawOut : {}, unknown = new Set(list(k.unknown));
    const asOf = date(k.as_of);
    add("as_of date present", asOf !== null);
    const subj = k.subject || {};
    add("identity: company, ticker, exchange and share class given", ["company", "ticker", "exchange", "share_class"].every((f) => String(subj[f] || "").trim()));
    const input = (n) => (Object.hasOwn(inputs, n) && isDict(inputs[n]) && isNum(inputs[n].value)) ? inputs[n].value : null;
    for (const [name, v] of Object.entries(inputs)) {
      if (!v || typeof v !== "object" || Array.isArray(v)) { add("input " + name + " is an object", false); continue; }
      if (v.value == null) { add(name + ": missing value is listed in unknown", unknown.has(name)); continue; }
      add(name + ": value is a plain number", isNum(v.value));
      add(name + ": has a source", !!String(v.source || "").trim());
      add(name + ": has an as-of date", date(v.as_of) !== null);
    }
    add("sources are listed", truthy(k.sources));
    const flagged = (...words) => { const t = list(k.flags).map(pyStr).join(" ").toLowerCase(); return words.every((w) => t.includes(w.toLowerCase())); };
    const p = isDict(inputs.price) ? inputs.price : null;
    if (p && date(p.as_of) && asOf && (p.basis || "last close") !== "offer midpoint") {
      const lag = tradingDays(date(p.as_of), asOf);
      if (lag > 1) add("stale price is flagged", flagged("stale"), "price is " + lag + " trading days old");
    }
    const price = input("price"), shares = input("diluted_shares"), mc = isNum(outputs.market_cap) ? outputs.market_cap : null;
    if (price !== null && shares !== null) add("market cap = price x fully diluted shares", close(mc, price * shares));
    else if (mc !== null) add("market cap has both inputs", false);
    const ev = isNum(outputs.enterprise_value) ? outputs.enterprise_value : null, debt = input("debt"), cash = input("cash");
    if (mc !== null && debt !== null && cash !== null) add("EV = market cap + debt + preferred + minority interest - cash", close(ev, mc + debt + (input("preferred") || 0) + (input("minority_interest") || 0) - cash));
    else if (ev !== null) add("EV has its inputs", false);
    const values = Object.assign({}, ...Object.keys(inputs).map((n) => ({ [n]: input(n) })), { market_cap: mc, enterprise_value: ev });
    const look = (x) => typeof x === "string" && Object.hasOwn(values, x) ? values[x] : null;
    const allMultiples = list(outputs.multiples), multiples = allMultiples.filter(isDict);
    add("every multiple is an object", multiples.length === allMultiples.length);
    for (const m of multiples) {
      const n = look(m.numerator), d = look(m.denominator), val = isNum(m.value) ? m.value : null;
      if (n == null || d == null || d === 0) add("multiple " + pyStr(m.name) + ": inputs present", false);
      else add("multiple " + pyStr(m.name) + " recomputes", close(val, n / d, 0.01));
    }
    for (const seg of list(k.segments).filter(isDict)) {
      const total = look(seg.total), parts = list(seg.parts).map((x) => isDict(x) && isNum(x.value) ? x.value : null);
      if (total != null && parts.length && !parts.includes(null)) add("segments add up to " + pyStr(seg.total), close(parts.reduce((a, b) => a + b, 0), total, 0.01));
    }
    for (const m of multiples) {
      const periods = [m.numerator, m.denominator].filter((x) => typeof x === "string" && Object.hasOwn(inputs, x) && isDict(inputs[x])).map((x) => String(inputs[x].period || ""));
      const kinds = new Set();
      for (const per of periods.concat([String(m.name || "")])) for (const w of ["LTM", "NTM", "FY"]) if (new RegExp("\\b" + w).test(per.toUpperCase())) kinds.add(w);
      add("multiple " + pyStr(m.name) + ": one kind of period", kinds.size <= 1);
    }
    const rev = input("revenue"), rev0 = input("revenue_prior");
    if (ev !== null && rev && ev / rev > 50) add("EV/revenue above 50x is flagged", flagged("outlier"));
    if (rev && rev0 && rev0 > 0 && rev / rev0 - 1 > 3) add("revenue growth above 300% is flagged", flagged("outlier"));
    for (const name of ["net_income", "operating_income", "gross_profit"]) {
      const x = input(name);
      if (x !== null && rev && !(x / rev >= -1 && x / rev <= 1)) add(name + " margin outside -100%..100% is flagged", flagged("outlier"));
    }
    const scen = k.scenarios, rating = k.rating;
    add("rating is one of the four", RATINGS.has(rating));
    if (rating === "NOT RATED") add("NOT RATED has no scenarios", !truthy(scen));
    else if (isDict(scen)) {
      const cases = ["bull", "base", "bear"];
      const field = (c, f) => { if (!isDict(scen[c]) || !Object.hasOwn(scen[c], f)) throw new TypeError(c + "." + f + " missing"); return pyFloat(scen[c][f]); };
      let vals = null, probs = null;
      try { vals = Object.fromEntries(cases.map((c) => [c, field(c, "value")])); probs = Object.fromEntries(cases.map((c) => [c, field(c, "prob")])); }
      catch (e) { vals = null; }
      if (vals === null) add("scenarios have bull, base and bear values and probabilities", false, JSON.stringify(scen).slice(0, 200));
      else {
        const sum = probs.bull + probs.base + probs.bear;
        add("probabilities add up to 100%", Math.abs(sum - 100) < 0.01);
        add("probabilities in 5% steps", cases.every((c) => Math.abs(probs[c] / 5 - Math.round(probs[c] / 5)) < 1e-9));
        add("bear < base < bull", vals.bear < vals.base && vals.base < vals.bull);
        const want = cases.reduce((a, c) => a + vals[c] * probs[c] / 100, 0);
        add("PWV matches the probabilities and values", close(isNum(scen.pwv) ? scen.pwv : null, want));   // from the checked numbers, not the raw text
        const ref = isNum(scen.reference_price) ? scen.reference_price : null, er = isNum(scen.expected_return_pct) ? scen.expected_return_pct : null;
        if (ref) {
          add("expected return = PWV / reference price - 1", er !== null && Math.abs(er - 100 * (want / ref - 1)) < 0.5);
          for (const c of cases) if (vals[c] > 5 * ref || vals[c] < ref / 5) add(c + " value beyond 5x / one-fifth of the price is flagged", flagged("outlier"));
        }
        if (er !== null) {
          const wantR = expectedRating(er, k.conviction);
          add("rating follows the thresholds", rating === wantR, "expected return " + er + "%, conviction " + k.conviction + " -> " + wantR + "; block says " + rating);
          add("no positive rating with an expected loss", !(er < 0 && rating === "Overweight"));
        }
        const ipo = k.ipo_ratings;
        if (isDict(ipo)) {
          add("Participate only with an Overweight rating", (ipo.at_offer === "Participate") === (rating === "Overweight"));
          if (ipo.aftermarket === "Buy below") add("buy-below = PWV / 1.15", close(isNum(ipo.buy_below) ? ipo.buy_below : null, want / 1.15, 0.01));
        }
        const stop = isNum(k.stop_working_price) ? k.stop_working_price : null;
        add("a rating comes with a stop-working price", stop !== null);
        if (stop !== null && ref) add("stop-working price below the reference price", stop < ref);
      }
    } else add("a rated stock has scenarios", false);
    return out;
  }

  function checkMemo(memo) {
    let k;
    try { k = extract(memo); } catch (e) { return [["KEY NUMBERS block present and valid", false, e.message]]; }
    const head = [["KEY NUMBERS block present and valid", true, ""]];
    try { return head.concat(check(k)); }
    catch (e) { return head.concat([["the block's shape can be checked", false, e.message]]); }   // never freeze the page on an odd block
  }

  function expectedRating(er, conviction) {     // verify.expected_rating
    if (er >= OVERWEIGHT && CONVICTION_OK.has(conviction)) return "Overweight";
    if (er <= UNDERWEIGHT) return "Underweight";
    return "Equal-weight";
  }

  // ---------- projections (the demo's charts; not in the Python, so tests/test_js_parity.py checks their properties)
  function rng(seed) {                          // mulberry32: the same seed draws the same paths
    let a = seed >>> 0;
    return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }

  const quantile = (sorted, q) => { const i = (sorted.length - 1) * q, lo = Math.floor(i); return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (i - lo); };

  // Each simulated path picks bull, base or bear by its probability, ends near that target (lognormal, spread by
  // half the volatility, mean exactly on the target) and gets there as a Brownian bridge with the full volatility.
  // So the paths' average ending is the PWV, and the bands show how wide the road is, not just where it ends.
  function project(o) {
    const cases = ["bull", "base", "bear"], price = o.price, months = o.months ?? 12, vol = (o.vol ?? 35) / 100;
    if (!(isFinite(price) && price > 0)) throw new Error("price must be above 0");
    for (const c of cases) if (!(o[c] && isFinite(o[c].value) && o[c].value > 0 && isFinite(o[c].prob) && o[c].prob >= 0)) throw new Error(c + " needs a value above 0 and a probability");
    const sum = cases.reduce((a, c) => a + o[c].prob, 0);
    if (Math.abs(sum - 100) > 0.01) throw new Error("probabilities add up to " + +sum.toFixed(2) + "%, not 100%");
    if (!(o.bear.value < o.base.value && o.base.value < o.bull.value)) throw new Error("needs bear < base < bull");
    if (!(Number.isInteger(months) && months >= 1 && months <= 36)) throw new Error("horizon must be 1 to 36 months");
    if (!(vol >= 0 && vol <= 2)) throw new Error("volatility must be 0% to 200%");
    const stop = isFinite(o.stop) && o.stop > 0 ? o.stop : null;
    const nPaths = o.paths ?? 2000, steps = o.steps ?? months * 4, h = months / 12, dt = h / steps, sw = vol / 2;
    const t = Array.from({ length: steps + 1 }, (_, i) => i * months / steps);
    const pwv = cases.reduce((a, c) => a + o[c].value * o[c].prob / 100, 0), er = 100 * (pwv / price - 1);
    const scenario = Object.fromEntries(cases.map((c) => [c, t.map((m) => price * Math.pow(o[c].value / price, m / months))]));
    const weighted = t.map((_, i) => cases.reduce((a, c) => a + scenario[c][i] * o[c].prob / 100, 0));
    const r = rng(o.seed ?? 1), normal = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    const cols = t.map(() => new Float64Array(nPaths)), ends = new Float64Array(nPaths), picks = { bull: 0, base: 0, bear: 0 };
    let touched = 0;
    const l0 = Math.log(price), w = new Float64Array(steps + 1);
    for (let p = 0; p < nPaths; p++) {
      const u = r() * 100, c = u < o.bull.prob ? "bull" : u < o.bull.prob + o.base.prob ? "base" : "bear";
      picks[c]++;
      const lEnd = Math.log(o[c].value) + sw * Math.sqrt(h) * normal() - sw * sw * h / 2;
      for (let i = 1; i <= steps; i++) w[i] = w[i - 1] + vol * Math.sqrt(dt) * normal();
      let low = Infinity;
      for (let i = 0; i <= steps; i++) {
        const f = i / steps, x = Math.exp(l0 + f * (lEnd - l0) + w[i] - f * w[steps]);
        cols[i][p] = x; if (x < low) low = x;
      }
      ends[p] = cols[steps][p];
      if (stop !== null && low <= stop) touched++;
    }
    const qs = [0.1, 0.25, 0.5, 0.75, 0.9], bands = Object.fromEntries(qs.map((q) => ["p" + Math.round(q * 100), []]));
    for (const col of cols) { const s = Float64Array.from(col).sort(); for (const q of qs) bands["p" + Math.round(q * 100)].push(quantile(s, q)); }
    const sorted = Float64Array.from(ends).sort(), lo = quantile(sorted, 0.01), hi = quantile(sorted, 0.99), nb = o.bins ?? 32;
    const counts = new Array(nb).fill(0), width = (hi - lo) / nb || 1;
    for (const x of sorted) if (x >= lo && x <= hi) counts[Math.min(nb - 1, Math.floor((x - lo) / width))]++;
    const share = (f) => { let k = 0; for (const x of ends) if (f(x)) k++; return k / nPaths; };
    return {
      price, months, vol: vol * 100, stop, t, pwv, expected_return_pct: er, rating: expectedRating(er, o.conviction ?? "Medium"),
      scenario, weighted, bands, picks,
      histogram: { lo, width, counts, edges: counts.map((_, i) => lo + i * width) },
      mean_end: ends.reduce((a, b) => a + b, 0) / nPaths,
      p_end_above_price: share((x) => x > price), p_end_below_stop: stop === null ? null : share((x) => x < stop),
      p_touch_stop: stop === null ? null : touched / nPaths,
      range80: [quantile(sorted, 0.1), quantile(sorted, 0.9)],
    };
  }

  // ---------- comparing trade ideas: each projected (and sized, given a portfolio), then sorted by one measure
  function tradeMetrics(idea, view = null) {
    const r = project(idea), stop = r.stop, price = r.price;
    const risk = stop !== null && stop < price ? price - stop : null;
    let shares = null, loss = null;
    if (view && stop !== null) {
      try { const s = sizePosition(view, idea.symbol || "NEW", price, stop, idea.conviction ?? "Medium", idea.sector ?? null); shares = s.shares; loss = s.loss_at_stop; } catch (e) { /* no size for this idea */ }
    }
    return { symbol: idea.symbol || "NEW", price, stop, pwv: r.pwv, expected_return_pct: r.expected_return_pct, rating: r.rating,
      reward_risk: risk === null ? null : (r.pwv - price) / risk, p_touch_stop: r.p_touch_stop, p_end_above_price: r.p_end_above_price,
      range80: r.range80, shares, loss_at_stop: loss, idea };
  }

  // Each measure sorts best first by default; "dir" flips it. Ideas without a value (no stop, no size) always go last,
  // and ties keep the symbol order, so the same list always sorts the same way.
  const TRADE_SORTS = {
    expected_return_pct: { label: "Expected return", best: "high" },
    reward_risk: { label: "Reward : risk", best: "high" },
    p_end_above_price: { label: "Chance it ends up", best: "high" },
    p_touch_stop: { label: "Chance it hits the stop", best: "low" },
    shares: { label: "Shares the rules allow", best: "high" },
    rating: { label: "Rating", best: "high" },
    symbol: { label: "Symbol", best: "low" },
  };
  const RATING_RANK = { Overweight: 2, "Equal-weight": 1, Underweight: 0 };
  function sortTrades(rows, key, dir = "best") {
    if (!TRADE_SORTS[key]) throw new Error("unknown sort: " + key);
    const value = (r) => key === "rating" ? RATING_RANK[r.rating] ?? null : r[key];
    let sign = TRADE_SORTS[key].best === "high" ? -1 : 1;
    if (dir === "worst") sign = -sign;
    return rows.slice().sort((a, b) => {
      const x = value(a), y = value(b);
      if (x === null || x === undefined) return (y === null || y === undefined) ? a.symbol.localeCompare(b.symbol) : 1;
      if (y === null || y === undefined) return -1;
      if (x !== y) return typeof x === "string" ? sign * x.localeCompare(y) : sign * (x - y);
      return key === "rating" ? b.expected_return_pct - a.expected_return_pct || a.symbol.localeCompare(b.symbol) : a.symbol.localeCompare(b.symbol);
    });
  }

  // The rating rule's numbers, for the page's gauge and notes: one source, the same as verify.py.
  const THRESHOLDS = Object.freeze({ overweight: OVERWEIGHT, underweight: UNDERWEIGHT, conviction_ok: Object.freeze([...CONVICTION_OK]) });

  const api = { PortfolioError, validate, valued, sizePosition, extract, check, checkMemo, expectedRating, project, tradeMetrics, sortTrades, TRADE_SORTS, THRESHOLDS, round2 };
  if (typeof module === "object" && module.exports) module.exports = api; else root.IpoCore = api;
})(typeof self !== "undefined" ? self : this);
