// The browser demo's logic: portfolio.py's sizing rules and verify.py's KEY NUMBERS checks, line for line.
// tests/test_js_parity.py runs both on the same inputs and fails if they disagree, so this file and the
// Python cannot drift apart. Loaded by docs/app/index.html, and by Node for that test.
(function (root) {
  "use strict";
  const DEFAULT_RULES = { max_position_pct: 10.0, max_sector_pct: 30.0, risk_per_trade_pct: 1.0, min_cash_pct: 5.0,
    conviction_scale: { Low: 0.5, Medium: 1.0, High: 1.5 } };

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
    return { as_of: String(raw.as_of || ""), cash: num(raw.cash === undefined ? 0 : raw.cash, "cash"), holdings, rules };
  }

  function valued(pf) {
    const rows = [], unpriced = [];
    for (const h of pf.holdings) {
      if (h.price == null) { unpriced.push(h.symbol); continue; }
      rows.push(Object.assign({}, h, { price_source: "portfolio file, " + (h.price_date || "undated"), value: h.shares * h.price }));
    }
    const invested = rows.reduce((a, r) => a + r.value, 0), equity = invested + pf.cash, sectors = {};
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
    return { as_of: pf.as_of, equity, cash: pf.cash, cash_pct: cashPct, holdings: rows, sector_pct: sectorPct, unpriced, rule_breaches: breaches, rules };
  }

  const round2 = (x) => Math.round(x * 100) / 100;

  function sizePosition(view, symbol, entryPrice, stopPrice, conviction = "Medium", sector = null) {
    const entry = num(entryPrice, "entry_price", 0, false), stop = num(stopPrice, "stop_price");
    if (stop >= entry) throw new PortfolioError("the stop-working price must be below the entry price for a purchase");
    const rules = view.rules, equity = view.equity, scale = rules.conviction_scale[String(conviction)];
    if (scale === undefined) throw new PortfolioError("conviction must be one of " + Object.keys(rules.conviction_scale).sort().join(", "));
    const sym = String(symbol).trim().toUpperCase(), held = view.holdings.find((h) => h.symbol === sym) || null;
    sector = sector || (held ? held.sector : "Unknown");
    const heldValue = held ? held.value : 0, sectorValue = (view.sector_pct[sector] || 0) * equity / 100;
    const riskBudget = equity * rules.risk_per_trade_pct / 100 * scale;
    const limits = {
      risk: Math.floor(riskBudget / (entry - stop)),
      position: Math.floor(Math.max(0, equity * rules.max_position_pct / 100 - heldValue) / entry),
      cash: Math.floor(Math.max(0, view.cash - equity * rules.min_cash_pct / 100) / entry),
    };
    if (sector !== "Unknown") limits.sector = Math.floor(Math.max(0, equity * rules.max_sector_pct / 100 - sectorValue) / entry);
    const shares = Math.max(0, Math.min(...Object.values(limits)));
    let binding = null;
    for (const k of Object.keys(limits)) if (binding === null || limits[k] < limits[binding]) binding = k;
    const cost = shares * entry, newValue = heldValue + cost;
    return { symbol: sym, shares, cost: round2(cost), entry_price: entry, stop_price: stop, conviction, binding_rule: binding,
      limits_in_shares: limits, risk_budget: round2(riskBudget), loss_at_stop: round2(shares * (entry - stop)),
      weight_after_pct: equity ? round2(100 * newValue / equity) : 0, cash_after: round2(view.cash - cost) };
  }

  // ---------------------------------------------------------------- verify.py
  const REL_TOL = 0.005, OVERWEIGHT = 15, UNDERWEIGHT = -10, CONVICTION_OK = new Set(["Medium", "High"]);
  const RATINGS = new Set(["Overweight", "Equal-weight", "Underweight", "NOT RATED"]);

  function extract(memo) {
    const blocks = [...String(memo).matchAll(/```json\s*([\s\S]*?)```/gi)].map((m) => m[1]);
    for (const raw of blocks.reverse()) {
      let data;
      try { data = JSON.parse(raw); } catch (e) { throw new Error("the KEY NUMBERS block is not valid JSON"); }
      if (data && typeof data === "object" && data.key_numbers && typeof data.key_numbers === "object") return data.key_numbers;
    }
    throw new Error('no ```json block with a "key_numbers" object');
  }
  const isNum = (x) => typeof x === "number" && isFinite(x);
  const close = (a, b, rel = REL_TOL, abs = 1e-9) => a != null && b != null && Math.abs(a - b) <= Math.max(rel * Math.max(Math.abs(a), Math.abs(b)), abs);
  const date = (s) => { const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s || "")); if (!m) return null; const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])); return d.getUTCMonth() === +m[2] - 1 ? d : null; };
  function tradingDays(a, b) { let n = 0; for (let d = new Date(a.getTime() + 864e5); d <= b; d = new Date(d.getTime() + 864e5)) if (d.getUTCDay() % 6) n++; return n; }

  function check(k) {
    const out = [], add = (name, ok, detail = "") => out.push([name, !!ok, detail]);
    const inputs = k.inputs || {}, outputs = k.outputs || {}, unknown = new Set(k.unknown || []), sources = k.sources || [];
    const asOf = date(k.as_of);
    add("as_of date present", asOf !== null);
    const subj = k.subject || {};
    add("identity: company, ticker, exchange and share class given", ["company", "ticker", "exchange", "share_class"].every((f) => String(subj[f] || "").trim()));
    const input = (n) => (inputs[n] && typeof inputs[n] === "object" && isNum(inputs[n].value)) ? inputs[n].value : null;
    for (const [name, v] of Object.entries(inputs)) {
      if (!v || typeof v !== "object" || Array.isArray(v)) { add("input " + name + " is an object", false); continue; }
      if (v.value == null) { add(name + ": missing value is listed in unknown", unknown.has(name)); continue; }
      add(name + ": value is a plain number", isNum(v.value));
      add(name + ": has a source", !!String(v.source || "").trim());
      add(name + ": has an as-of date", date(v.as_of) !== null);
    }
    add("sources are listed", sources.length > 0);
    const flagged = (...words) => { const t = (k.flags || []).join(" ").toLowerCase(); return words.every((w) => t.includes(w.toLowerCase())); };
    const p = inputs.price && typeof inputs.price === "object" ? inputs.price : null;
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
    for (const m of outputs.multiples || []) {
      const n = values[m.numerator], d = values[m.denominator], val = isNum(m.value) ? m.value : null;
      if (n == null || d == null || d === 0) add("multiple " + m.name + ": inputs present", false);
      else add("multiple " + m.name + " recomputes", close(val, n / d, 0.01));
    }
    for (const seg of k.segments || []) {
      const total = values[seg.total], parts = (seg.parts || []).map((x) => isNum(x.value) ? x.value : null);
      if (total != null && parts.length && !parts.includes(null)) add("segments add up to " + seg.total, close(parts.reduce((a, b) => a + b, 0), total, 0.01));
    }
    for (const m of outputs.multiples || []) {
      const periods = [m.numerator, m.denominator].filter((x) => inputs[x] && typeof inputs[x] === "object").map((x) => String(inputs[x].period || ""));
      const kinds = new Set();
      for (const per of periods.concat([String(m.name || "")])) for (const w of ["LTM", "NTM", "FY"]) if (new RegExp("\\b" + w).test(per.toUpperCase())) kinds.add(w);
      add("multiple " + m.name + ": one kind of period", kinds.size <= 1);
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
    if (rating === "NOT RATED") add("NOT RATED has no scenarios", !scen);
    else if (scen && typeof scen === "object") {
      const cases = ["bull", "base", "bear"];
      if (!cases.every((c) => scen[c] && isNum(+scen[c].value) && isNum(+scen[c].prob))) add("scenarios have bull, base and bear values and probabilities", false);
      else {
        const vals = Object.fromEntries(cases.map((c) => [c, +scen[c].value])), probs = Object.fromEntries(cases.map((c) => [c, +scen[c].prob]));
        const sum = probs.bull + probs.base + probs.bear;
        add("probabilities add up to 100%", Math.abs(sum - 100) < 0.01);
        add("probabilities in 5% steps", cases.every((c) => Math.abs(probs[c] / 5 - Math.round(probs[c] / 5)) < 1e-9));
        add("bear < base < bull", vals.bear < vals.base && vals.base < vals.bull);
        const want = cases.reduce((a, c) => a + vals[c] * probs[c] / 100, 0);
        add("PWV matches the probabilities and values", close(isNum(scen.pwv) ? scen.pwv : null, want));
        const ref = isNum(scen.reference_price) ? scen.reference_price : null, er = isNum(scen.expected_return_pct) ? scen.expected_return_pct : null;
        if (ref) {
          add("expected return = PWV / reference price - 1", er !== null && Math.abs(er - 100 * (want / ref - 1)) < 0.5);
          for (const c of cases) if (vals[c] > 5 * ref || vals[c] < ref / 5) add(c + " value beyond 5x / one-fifth of the price is flagged", flagged("outlier"));
        }
        if (er !== null) {
          const wantR = er >= OVERWEIGHT && CONVICTION_OK.has(k.conviction) ? "Overweight" : er <= UNDERWEIGHT ? "Underweight" : "Equal-weight";
          add("rating follows the thresholds", rating === wantR, "expected return " + er + "%, conviction " + k.conviction + " -> " + wantR + "; block says " + rating);
          add("no positive rating with an expected loss", !(er < 0 && rating === "Overweight"));
        }
        const ipo = k.ipo_ratings;
        if (ipo && typeof ipo === "object") {
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
    return [["KEY NUMBERS block present and valid", true, ""]].concat(check(k));
  }

  const api = { PortfolioError, validate, valued, sizePosition, extract, check, checkMemo };
  if (typeof module === "object" && module.exports) module.exports = api; else root.IpoCore = api;
})(typeof self !== "undefined" ? self : this);
