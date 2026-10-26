(() => {
  const R = window.RESULTS;
  const H = Charts.html;
  const $ = (id) => document.getElementById(id);
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const MINUS = "−";

  // --- formatting -------------------------------------------------------------

  const fmt = {
    pct: (v, d = 1) => (v == null ? "–" : `${v < 0 ? MINUS : ""}${Math.abs(v * 100).toFixed(d)}%`),
    spct: (v, d = 1) => (v == null ? "–" : `${v >= 0 ? "+" : MINUS}${Math.abs(v * 100).toFixed(d)}%`),
    usd: (v, d = 0) => (v == null ? "–" : `${v < 0 ? MINUS : ""}$${Math.abs(v).toLocaleString("en-US",
      { minimumFractionDigits: d, maximumFractionDigits: d })}`),
    usdShort: (v) => {
      const a = Math.abs(v), s = v < 0 ? MINUS : "";
      return a >= 1000 ? `${s}$${(a / 1000).toFixed(a % 1000 === 0 ? 0 : 1)}k` : `${s}$${a.toFixed(0)}`;
    },
    num: (v, d = 4) => (v == null ? "–" : `${v < 0 ? MINUS : ""}${Math.abs(v).toFixed(d)}`),
    snum: (v, d = 4) => {
      if (v == null) return "–";
      const digits = Math.abs(v).toFixed(d);
      return Number(digits) === 0 ? digits : `${v > 0 ? "+" : MINUS}${digits}`;
    },
    p: (v) => (v == null ? "n/a" : v < 0.001 ? "<0.001" : v.toFixed(3)),
    int: (v) => (v == null ? "–" : Number(v).toLocaleString("en-US")),
    date: (s) => {
      const d = new Date(`${s}T00:00:00Z`);
      return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    },
    model: (m) => m.replaceAll("_", " "),
    cap: (s) => s.charAt(0).toUpperCase() + s.slice(1),
    regime: (r) => ({ bull: "Bull", bear: "Bear", high_vol: "High vol", low_vol: "Low vol" }[r] || r),
  };

  const ICONS = {
    bad: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>',
    ok: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M4.8 8.3l2.2 2.2 4.2-4.6" stroke="#fff" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    neutral: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M5 8h6" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>',
  };

  function status(kind, text) {
    const span = H("span", { class: `status ${kind}` });
    span.innerHTML = ICONS[kind];
    span.appendChild(document.createTextNode(text));
    return span;
  }

  function tile(container, { label, value, detail, statusNode }) {
    const card = H("div", { class: "card tile" });
    card.appendChild(H("span", { class: "label" }, label));
    card.appendChild(H("span", { class: "value" }, value));
    if (statusNode) card.appendChild(statusNode);
    if (detail) card.appendChild(H("span", { class: "detail" }, detail));
    container.appendChild(card);
  }

  function segmented(container, label, options, current, onChange) {
    const wrap = H("div", { class: "control" });
    wrap.appendChild(H("span", { id: `seg-${label.replace(/\W+/g, "-")}` }, label));
    const group = H("div", { class: "segmented", role: "group", "aria-labelledby": `seg-${label.replace(/\W+/g, "-")}` });
    for (const opt of options) {
      const btn = H("button", { type: "button", "aria-pressed": String(opt.value === current) }, opt.label);
      btn.addEventListener("click", () => {
        group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
        btn.setAttribute("aria-pressed", "true");
        onChange(opt.value);
      });
      group.appendChild(btn);
    }
    wrap.appendChild(group);
    container.appendChild(wrap);
  }

  const pCls = (v) => (v != null && v < 0.05 ? "pos" : "");
  const signCls = (v) => (v == null ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "");

  // --- pages ------------------------------------------------------------------

  function renderOverview() {
    const tiles = $("overview-tiles");
    const bt = R.risk_backtest;
    if (bt) {
      const pass99 = bt.rows.filter((r) => !r["rejected_99%"]).length;
      tile(tiles, {
        label: "Risk models passing 99% VaR", value: `${pass99} of ${bt.rows.length}`,
        statusNode: pass99 ? status("ok", "some pass") : status("bad", "none pass"),
        detail: `1-day VaR, out of sample, ${fmt.date(bt.start)} to ${fmt.date(bt.end)}`,
      });
      const best = bt.rows.slice().sort((a, b) => Math.abs(a["rate_95%"] - 0.05) - Math.abs(b["rate_95%"] - 0.05))[0];
      tile(tiles, { label: "Best calibrated at 95%", value: fmt.pct(best["rate_95%"], 2),
                    detail: `${best.model}: share of days breached vs 5% expected` });
    }
    if (R.robustness) {
      const cells = R.robustness.cells;
      const sig = cells.filter((c) => (c.cw_pvalue_holm ?? 1) < 0.05 || (c.pt_pvalue_holm ?? 1) < 0.05).length;
      tile(tiles, { label: "Forecast signal", value: `${sig} of ${cells.length}`,
                    statusNode: sig ? status("ok", "significant cells") : status("bad", "no signal"),
                    detail: "model × horizon cells significant after Holm adjustment" });
    }
    if (R.forecast && R.forecast["10y"]) {
      const e = R.forecast["10y"].economics;
      tile(tiles, { label: "Best model vs buy & hold", value: `${fmt.spct(e.strategy_total_return, 0)} vs ${fmt.spct(e.buy_hold_total_return, 0)}`,
                    detail: `${fmt.model(e.model_name)} strategy over 10 years, after costs` });
    }
    if (R.options) {
      const o = R.options;
      tile(tiles, { label: `ATM call, ${o.inputs.maturity_days} trading days`, value: fmt.usd(o.black76, 2),
                    detail: `Black-76. Black-Scholes on the futures price says ${fmt.usd(o.bs_as_spot, 2)} (${fmt.spct(o.bs_as_spot / o.black76 - 1)})` });
    }

    if (R.prices) {
      const { dates, close } = R.prices;
      const markers = [];
      if (R.risk) {
        const i = dates.indexOf(R.risk.worst_day.date);
        if (i >= 0) markers.push({ index: i, label: `${fmt.spct(R.risk.worst_day.return)} on ${fmt.date(R.risk.worst_day.date)}` });
      }
      Charts.line($("price-chart"), {
        x: dates, xType: "date", height: 300, ariaLabel: "Silver futures daily close over 10 years",
        series: [{ name: "Close", values: close, color: "--series-1", width: 1.5 }],
        yFormat: (v, tip) => (tip ? fmt.usd(v, 2) : `$${v.toFixed(0)}`), xFormat: fmt.date, markers,
      });
      const monthly = [];
      dates.forEach((d, i) => { if (i === dates.length - 1 || dates[i + 1].slice(0, 7) !== d.slice(0, 7)) monthly.push({ date: d, close: close[i] }); });
      Charts.table($("price-table"), {
        columns: [{ key: "date", label: "Month end", format: fmt.date, sortValue: (r) => r.date },
                  { key: "close", label: "Close", num: true, format: (v) => fmt.usd(v, 2) }],
        rows: monthly.reverse(),
      });
    }

    const cards = $("overview-cards");
    const nav = [
      ["risk", "Risk simulation", "Monte Carlo VaR and CVaR for a 1,000 oz position, across 16 model variants."],
      ["backtest", "Risk backtest", "Every variant's VaR checked day by day against what actually happened."],
      ["forecast", "Forecasting", "Ten models predicting 21-day returns, and whether trading them would pay."],
      ["robustness", "Robustness", "The same question at four horizons and in four market regimes."],
      ["options", "Options", "Black-76 pricing of silver futures options, checked by Monte Carlo."],
      ["method", "Method", "Safeguards against lookahead and false positives, and known limitations."],
    ];
    for (const [id, title, text] of nav) {
      const a = H("a", { class: "card", href: `#${id}` });
      a.appendChild(H("h3", {}, title));
      a.appendChild(H("p", {}, text));
      cards.appendChild(a);
    }
  }

  function renderRisk() {
    const risk = R.risk;
    const s = risk.settings;
    const state = { calibration: "point", process: "gbm", dist: "normal" };
    const controls = $("risk-controls");
    segmented(controls, "Calibration", [
      { value: "point", label: "Flat" }, { value: "ewma", label: "EWMA" },
      { value: "garch", label: "GARCH" }, { value: "bootstrap", label: "Bootstrap" },
    ], state.calibration, (v) => { state.calibration = v; draw(); });
    segmented(controls, "Process", [{ value: "gbm", label: "GBM" }, { value: "gbm-jump", label: "GBM + jumps" }],
              state.process, (v) => { state.process = v; draw(); });
    segmented(controls, "Shocks", [{ value: "normal", label: "Normal" }, { value: "t", label: "Student-t" }],
              state.dist, (v) => { state.dist = v; draw(); });

    const calibrationText = {
      point: `flat volatility over the last ${s.period}`,
      ewma: "EWMA volatility (recent days weighted most)",
      garch: `GARCH(1,1) volatility averaged over the ${s.horizon_days}-day horizon`,
      bootstrap: "block bootstrap: each path draws its own drift and volatility",
    };

    function draw() {
      const v = risk.variants.find((x) => x.id === `${state.calibration}|${state.process}|${state.dist}`);
      const tiles = $("risk-tiles");
      tiles.innerHTML = "";
      tile(tiles, { label: "Silver price", value: fmt.usd(risk.s0, 2), detail: `front-month futures, ${fmt.date(risk.end)}` });
      tile(tiles, { label: v.jumps ? "Diffusion volatility" : "Volatility", value: fmt.pct(v.sigma),
                    detail: state.calibration === "point" ? `annualized, ${s.period} history` : `annualized; flat ${s.period}: ${fmt.pct(v.sigma_flat)}` });
      tile(tiles, { label: "Expected return", value: fmt.spct(v.mu),
                    detail: v.mu_range ? `5–95% range across paths: ${fmt.spct(v.mu_range[0], 0)} to ${fmt.spct(v.mu_range[1], 0)}` : "annualized, from history" });
      for (const m of v.metrics) {
        tile(tiles, { label: `${Math.round(m.confidence * 100)}% VaR, ${s.horizon_days} days`, value: fmt.usd(m.var),
                      detail: `${fmt.pct(m.var_pct)} of position · CVaR ${fmt.usd(m.cvar)}` });
      }
      if (v.jumps) {
        tile(tiles, { label: "Jumps detected", value: String(v.jumps.n),
                      detail: `${v.jumps.per_year.toFixed(1)} a year, mean ${fmt.spct(v.jumps.mean)}, std ${fmt.pct(v.jumps.std)}` });
      }

      $("fan-sub").textContent = `${v.label}: ${calibrationText[state.calibration]}. ${fmt.int(s.n_sims)} paths.`;
      const days = v.fan.p50.map((_, i) => i);
      $("fan-legend").innerHTML = "";
      Charts.legend($("fan-legend"), [
        { name: "5–95% of paths", color: "--band-outer" }, { name: "25–75%", color: "--band-inner" },
        { name: "Median", color: "--band-line", line: true },
      ]);
      Charts.line($("fan-chart"), {
        x: days, height: 290, ariaLabel: "Fan chart of simulated silver prices",
        bands: [{ lower: v.fan.p05, upper: v.fan.p95, color: "--band-outer" },
                { lower: v.fan.p25, upper: v.fan.p75, color: "--band-inner" }],
        series: [{ name: "Median", values: v.fan.p50, color: "--band-line", width: 2.5 }],
        yFormat: (y, tip) => (tip ? fmt.usd(y, 2) : `$${y.toFixed(0)}`),
        xTickFormat: (d) => String(d),
        tooltipTitle: (i) => (i === 0 ? "Today" : `Day ${i}`),
        tooltipRows: (i) => [["95th pct", v.fan.p95], ["75th", v.fan.p75], ["Median", v.fan.p50], ["25th", v.fan.p25], ["5th pct", v.fan.p05]]
          .map(([name, arr]) => ({ name, value: fmt.usd(arr[i], 2) })),
      });
      Charts.table($("fan-table"), {
        columns: [{ key: "day", label: "Day", num: true }, ...["p05", "p25", "p50", "p75", "p95"].map((k) => (
          { key: k, label: k === "p50" ? "Median" : `${k.slice(1)}th pct`, num: true, format: (x) => fmt.usd(x, 2) }))],
        rows: days.map((d) => ({ day: d, p05: v.fan.p05[d], p25: v.fan.p25[d], p50: v.fan.p50[d], p75: v.fan.p75[d], p95: v.fan.p95[d] })),
      });

      $("pnl-sub").textContent = `${fmt.int(s.position_oz)} oz long over ${s.horizon_days} trading days, ${fmt.int(s.n_sims)} simulated outcomes.`;
      const [m95, m99] = v.metrics;
      $("pnl-legend").innerHTML = "";
      Charts.legend($("pnl-legend"), [
        { name: `95% VaR ${fmt.usd(m95.var)}`, color: "--series-2", line: true },
        { name: `99% VaR ${fmt.usd(m99.var)}`, color: "--ink", line: true },
      ]);
      Charts.histogram($("pnl-chart"), {
        edges: v.hist.edges, counts: v.hist.counts, height: 290, xFormat: fmt.usdShort,
        ariaLabel: "Histogram of simulated profit and loss",
        vlines: [{ x: -m95.var, color: "--series-2" }, { x: -m99.var, color: "--ink" }],
      });
      Charts.table($("pnl-table"), {
        columns: [{ key: "lo", label: "From", num: true, format: (x) => fmt.usd(x) },
                  { key: "hi", label: "To", num: true, format: (x) => fmt.usd(x) },
                  { key: "n", label: "Paths", num: true, format: fmt.int }],
        rows: v.hist.counts.map((n, i) => ({ lo: v.hist.edges[i], hi: v.hist.edges[i + 1], n })),
      });

      $("risk-bt-sub").textContent = `Over the ${s.period} history (${fmt.date(risk.start)} to ${fmt.date(risk.end)}), after the first ${s.backtest_window} trading days, the model is recalibrated each day on the previous ${s.backtest_window} days only, and its VaR compared with the next day's (or the next ${s.horizon_days} days') actual loss.`;
      const rows = [];
      for (const [h, byConf] of Object.entries(v.backtest)) {
        for (const [c, bt] of Object.entries(byConf)) rows.push({ h: Number(h), c: Number(c), ...bt });
      }
      Charts.table($("risk-bt-table"), {
        columns: [
          { key: "h", label: "Horizon", format: (x) => `${x} day${x > 1 ? "s" : ""}` },
          { key: "c", label: "VaR", format: (x) => `${Math.round(x * 100)}%` },
          { key: "n_obs", label: "Windows", num: true, format: fmt.int },
          { key: "n_exceptions", label: "Breaches", num: true },
          { key: "rate", label: "Rate", num: true, format: (x) => fmt.pct(x, 2) },
          { key: "expected", label: "Expected", num: true, format: (x) => fmt.pct(x, 0) },
          { key: "kupiec_p", label: "Kupiec p", num: true, format: fmt.p },
          { key: "cluster_p", label: "Clustering p", num: true, format: fmt.p },
          { key: "verdict", label: "Verdict", format: (x, r) => status(r.rejected ? "bad" : "ok", r.rejected ? x.replace("REJECTED", "Rejected") : "Not rejected") },
        ],
        rows,
      });
    }
    draw();
  }

  function renderBacktest() {
    const bt = R.risk_backtest;
    $("backtest-intro").textContent = `All 16 combinations of calibration, price process and shock distribution, backtested on ${fmt.int(bt.n_tests)} trading days from ${fmt.date(bt.start)} to ${fmt.date(bt.end)}. Each day's 1-day VaR is calibrated on the previous ${bt.window} trading days only.`;
    const pass95 = bt.rows.filter((r) => !r["rejected_95%"]).length;
    const pass99 = bt.rows.filter((r) => !r["rejected_99%"]).length;
    const rates99 = bt.rows.map((r) => r["rate_99%"]);
    const avg = (dist, key) => {
      const rows = bt.rows.filter((r) => r.dist === dist);
      return rows.reduce((a, r) => a + r[key], 0) / rows.length;
    };
    const best95 = bt.rows.slice().sort((a, b) => Math.abs(a["rate_95%"] - 0.05) - Math.abs(b["rate_95%"] - 0.05))[0];
    const understate = Math.min(...rates99) > 0.01 ? ": every variant understates silver's daily tail risk" : "";
    $("backtest-callout").innerHTML = "";
    $("backtest-callout").append(
      H("strong", {}, `${pass99} of ${bt.rows.length} variants pass at 99%; ${pass95} pass at 95%. `),
      document.createTextNode(`At 99%, breaches run from ${fmt.pct(Math.min(...rates99), 2)} to ${fmt.pct(Math.max(...rates99), 2)} of days against 1% expected${understate}. Student-t shocks average ${fmt.pct(avg("t", "rate_99%"), 2)} at 99% vs ${fmt.pct(avg("normal", "rate_99%"), 2)} for normal shocks, and ${fmt.pct(avg("t", "rate_95%"), 2)} vs ${fmt.pct(avg("normal", "rate_95%"), 2)} at 95%. Closest to target at 95%: ${best95.model}, at ${fmt.pct(best95["rate_95%"], 2)}.`),
    );
    for (const c of [95, 99]) {
      Charts.legend($(`dots-legend-${c}`), [{ name: "Not rejected", color: "--series-1" }, { name: "Rejected (Kupiec or clustering, 5% level)", color: "--critical" }]);
      Charts.dots($(`dots-${c}`), {
        categories: bt.rows.map((r) => r.model), values: bt.rows.map((r) => r[`rate_${c}%`]),
        flagged: bt.rows.map((r) => r[`rejected_${c}%`]), ref: 1 - c / 100, refLabel: `expected ${100 - c}%`,
        valueFormat: (v) => fmt.pct(v, v < 0.1 && c === 99 ? 1 : 0), ariaLabel: `${c}% VaR breach rate by model variant`,
        tooltip: (i) => {
          const r = bt.rows[i];
          return { title: r.model, rows: [
            { name: "Breaches", value: `${r[`exceptions_${c}%`]} of ${fmt.int(bt.n_tests)}` },
            { name: "Rate", value: fmt.pct(r[`rate_${c}%`], 2) },
            { name: "Kupiec p", value: fmt.p(r[`kupiec_p_${c}%`]) },
            { name: "Clustering p", value: fmt.p(r[`cluster_p_${c}%`]) },
            { name: "Verdict", value: r[`rejected_${c}%`] ? "Rejected" : "Not rejected" },
          ] };
        },
      });
    }
    const cols = [{ key: "model", label: "Variant" }];
    for (const c of [95, 99]) {
      cols.push(
        { key: `rate_${c}%`, label: `${c}% rate`, num: true, format: (x) => fmt.pct(x, 2) },
        { key: `kupiec_p_${c}%`, label: `${c}% Kupiec p`, num: true, format: fmt.p },
        { key: `cluster_p_${c}%`, label: `${c}% clustering p`, num: true, format: fmt.p },
        { key: `verdict_${c}%`, label: `${c}% verdict`, sortValue: (r) => (r[`rejected_${c}%`] ? 1 : 0),
          format: (x, r) => status(r[`rejected_${c}%`] ? "bad" : "ok", r[`rejected_${c}%`] ? "Rejected" : "Not rejected") },
      );
    }
    Charts.table($("backtest-table"), { columns: cols, rows: bt.rows });
  }

  function renderForecast() {
    let period = R.forecast["10y"] ? "10y" : Object.keys(R.forecast)[0];
    segmented($("forecast-controls"), "History", Object.keys(R.forecast).map((p) => ({ value: p, label: p.replace("y", " years") })),
              period, (p) => { period = p; draw(); });

    function draw() {
      const f = R.forecast[period];
      const e = f.economics;
      const summary = f.summary;
      const positive = summary.filter((r) => r.oos_r2_vs_zero > 0 && r.model !== "zero_return");
      const sigHolm = summary.filter((r) => (r.cw_holm ?? 1) < 0.05 || (r.pt_holm ?? 1) < 0.05);
      const rawSig = summary.filter((r) => (r.cw_pvalue ?? 1) < 0.05 || (r.pt_pvalue ?? 1) < 0.05);

      $("forecast-callout").innerHTML = "";
      $("forecast-callout").append(
        H("strong", {}, positive.length ? `Only ${positive.map((r) => fmt.model(r.model)).join(", ")} beat${positive.length === 1 ? "s" : ""} the random walk` : "Nothing beats the random walk"),
        document.createTextNode(` over ${f.n_predictions} non-overlapping test periods (${fmt.date(f.start)} to ${fmt.date(f.end)}). ${sigHolm.length ? `${sigHolm.length} model(s) are significant after Holm adjustment.` : `No model is significant after Holm adjustment${rawSig.length ? "" : ", or even before it"}.`}`),
      );

      const tiles = $("forecast-tiles");
      tiles.innerHTML = "";
      const best = summary[0];
      tile(tiles, { label: "Test periods", value: String(f.n_predictions), detail: `non-overlapping ${f.horizon_days}-day forecasts` });
      tile(tiles, { label: "Best OOS R²", value: fmt.snum(best.oos_r2_vs_zero, 4),
                    detail: best.kind === "ml" ? fmt.model(best.model) : `${fmt.model(best.model)}, a baseline (no ML)` });
      tile(tiles, { label: "Significant after Holm", value: `${sigHolm.length} of ${summary.length}`,
                    statusNode: sigHolm.length ? status("ok", "some signal") : status("bad", "no signal") });
      tile(tiles, { label: "Strategy vs buy & hold", value: `${fmt.spct(e.strategy_total_return, 0)} vs ${fmt.spct(e.buy_hold_total_return, 0)}`,
                    detail: `${e.n_trades} trades, ${e.cost_bps} bps per unit traded` });

      Charts.hbar($("r2-chart"), {
        categories: summary.map((r) => fmt.model(r.model)), values: summary.map((r) => r.oos_r2_vs_zero),
        colorFn: (v) => (v >= 0 ? "--div-pos" : "--div-neg"), valueFormat: (v) => fmt.num(v, 2),
        ariaLabel: "Out-of-sample R squared by model",
        tooltip: (i) => {
          const r = summary[i];
          return { title: fmt.model(r.model), rows: [
            { name: "R² vs random walk", value: fmt.snum(r.oos_r2_vs_zero) },
            { name: "R² vs historical mean", value: fmt.snum(r.oos_r2_vs_mean) },
            { name: "Directional accuracy", value: fmt.pct(r.directional_accuracy) },
            { name: "Clark-West p", value: fmt.p(r.cw_pvalue) },
            { name: "Pesaran-Timmermann p", value: fmt.p(r.pt_pvalue) },
          ] };
        },
      });

      $("equity-title").textContent = `Trading the best model: ${fmt.model(f.best_model)}`;
      $("equity-sub").textContent = `Long when it predicts a rise, short when it predicts a fall, ${f.cost_bps} bps per unit traded. Cumulative return.`;
      $("equity-legend").innerHTML = "";
      Charts.legend($("equity-legend"), [{ name: "Strategy", color: "--series-1", line: true }, { name: "Buy & hold", color: "--series-2", line: true }]);
      Charts.line($("equity-chart"), {
        x: f.equity.dates, xType: "date", height: 280, includeZero: true, ariaLabel: "Cumulative return of strategy and buy and hold",
        series: [{ name: "Strategy", values: f.equity.strategy, color: "--series-1", endLabel: true },
                 { name: "Buy & hold", values: f.equity.buy_hold, color: "--series-2", endLabel: true }],
        yFormat: (v, tip) => fmt.spct(v, tip ? 1 : 0), xFormat: fmt.date,
        hlines: [{ value: 0, color: "--axis" }],
      });

      Charts.table($("forecast-table"), {
        columns: [
          { key: "model", label: "Model", format: fmt.model },
          { key: "kind", label: "Type", format: (x) => (x === "ml" ? "ML" : "Baseline") },
          { key: "oos_r2_vs_zero", label: "R² vs random walk", num: true, format: (x) => fmt.snum(x), cls: signCls },
          { key: "oos_r2_vs_mean", label: "R² vs hist. mean", num: true, format: (x) => fmt.snum(x), cls: signCls },
          { key: "directional_accuracy", label: "Direction right", num: true, format: (x) => fmt.pct(x) },
          { key: "cw_pvalue", label: "CW p", num: true, format: fmt.p, cls: pCls },
          { key: "cw_holm", label: "CW Holm", num: true, format: fmt.p, cls: pCls },
          { key: "pt_pvalue", label: "PT p", num: true, format: fmt.p, cls: pCls },
          { key: "pt_holm", label: "PT Holm", num: true, format: fmt.p, cls: pCls },
        ],
        rows: summary, highlight: (r) => r.model === f.best_model,
      });

      const ab = f.ablation;
      Charts.hbar($("ablation-chart"), {
        categories: ab.map((r) => fmt.model(r.model)), values: ab.map((r) => r.oos_r2_delta),
        colorFn: (v) => (v >= 0 ? "--div-pos" : "--div-neg"), valueFormat: (v) => fmt.snum(v, 2),
        ariaLabel: "Change in R squared from adding COT features",
        tooltip: (i) => {
          const r = ab[i];
          return { title: fmt.model(r.model), rows: [
            { name: "With COT", value: fmt.snum(r.oos_r2_vs_zero_with) },
            { name: "Without COT", value: fmt.snum(r.oos_r2_vs_zero_without) },
            { name: "Change", value: fmt.snum(r.oos_r2_delta) },
          ] };
        },
      });
      Charts.table($("ablation-table"), {
        columns: [{ key: "model", label: "Model", format: fmt.model },
                  { key: "oos_r2_vs_zero_with", label: "R² with COT", num: true, format: (x) => fmt.snum(x) },
                  { key: "oos_r2_vs_zero_without", label: "R² without", num: true, format: (x) => fmt.snum(x) },
                  { key: "oos_r2_delta", label: "Change", num: true, format: (x) => fmt.snum(x), cls: signCls }],
        rows: ab,
      });

      const cot = new Set(["cot_mm_net_pct_oi", "cot_mm_net_pct_oi_chg_21d", "cot_comm_net_pct_oi"]);
      $("stability-sub").textContent = f.stability_model
        ? `${fmt.model(f.stability_model)} refit ${f.n_refits} times. Sign consistency: share of refits where the coefficient is nonzero and points the same way as its average. COT features highlighted.`
        : "No linear ML model to track.";
      if (f.stability) {
        Charts.table($("stability-table"), {
          columns: [
            { key: "feature", label: "Feature", format: (x) => x.replaceAll("_", " ") },
            { key: "mean", label: "Mean coef.", num: true, format: (x) => fmt.snum(x, 5) },
            { key: "sign_consistency", label: "Sign consistency", num: true, format: (x) => {
              if (x == null) return "never used";
              const wrap = H("span");
              wrap.appendChild(H("span", { class: "mini-bar", style: `width:${Math.round(x * 48)}px` }));
              wrap.appendChild(document.createTextNode(fmt.pct(x, 0)));
              return wrap;
            } },
          ],
          rows: f.stability, highlight: (r) => cot.has(r.feature),
        });
      }
    }
    draw();
  }

  function renderRobustness() {
    const rb = R.robustness;
    const cells = rb.cells;
    const cell = (m, h) => cells.find((c) => c.model === m && c.horizon_days === h);
    const testable = rb.counts.horizon.cw_pvalue.testable;
    const hc = rb.counts.horizon;
    const rawHits = hc.cw_pvalue.raw + hc.pt_pvalue.raw, holmHits = hc.cw_pvalue.Holm + hc.pt_pvalue.Holm;
    $("robustness-intro").textContent = `${rb.models.length} models at ${rb.horizons.length} horizons (${rb.horizons.map((h) => `${h}d`).join(", ")}) over ${rb.period}, plus the ${rb.regime_horizon}-day forecasts sliced by market regime. ${rawHits} of ${testable * 2} model × horizon tests reach a raw p < 0.05 and ${holmHits} survive Holm adjustment.`;

    const metrics = {
      oos_r2_vs_zero: { label: "R² vs random walk", center: 0, format: (v) => fmt.snum(v, 3) },
      oos_r2_vs_mean: { label: "R² vs historical mean", center: 0, format: (v) => fmt.snum(v, 3) },
      directional_accuracy: { label: "Directional accuracy", center: 0.5, format: (v) => fmt.pct(v, 0) },
    };
    let metric = "oos_r2_vs_zero";
    segmented($("robustness-controls"), "Metric", Object.entries(metrics).map(([value, m]) => ({ value, label: m.label })),
              metric, (v) => { metric = v; draw(); });

    function draw() {
      const m = metrics[metric];
      const values = cells.map((c) => c[metric]).filter((v) => v != null);
      const limit = Math.max(...values.map((v) => Math.abs(v - m.center)));
      $("heatmap-title").textContent = `${m.label}, model × horizon`;
      $("heatmap-sub").textContent = metric === "directional_accuracy"
        ? "Share of test periods where the predicted direction was right. Blue above 50%, red below. Hover for test results."
        : "Blue: smaller errors than the benchmark. Red: larger. Hover a cell for its test results.";
      Charts.heatmap($("heatmap"), {
        rows: rb.models, cols: rb.horizons, rowLabel: fmt.model, colLabel: (h) => `${h} day${h > 1 ? "s" : ""}`,
        value: (r, c) => cell(r, c)?.[metric] ?? null, format: m.format,
        fill: (v) => Charts.diverging(v, limit, m.center),
        tooltip: (r, c) => {
          const x = cell(r, c);
          return { title: `${fmt.model(r)}, ${c}-day horizon`, rows: [
            { name: "R² vs random walk", value: fmt.snum(x.oos_r2_vs_zero) },
            { name: "R² vs historical mean", value: fmt.snum(x.oos_r2_vs_mean) },
            { name: "Directional accuracy", value: fmt.pct(x.directional_accuracy) },
            { name: "Clark-West p (Holm)", value: `${fmt.p(x.cw_pvalue)} (${fmt.p(x.cw_pvalue_holm)})` },
            { name: "Pesaran-Timmermann p (Holm)", value: `${fmt.p(x.pt_pvalue)} (${fmt.p(x.pt_pvalue_holm)})` },
            { name: "Test periods", value: String(x.n_predictions) },
          ] };
        },
      });
      const scale = $("heatmap-scale");
      scale.innerHTML = "";
      scale.append(H("span", {}, m.format(m.center - limit)),
                   H("span", { class: "bar", style: `background:linear-gradient(90deg, ${Charts.diverging(-1, 1)}, ${Charts.diverging(0, 1)}, ${Charts.diverging(1, 1)})` }),
                   H("span", {}, m.format(m.center + limit)));
    }
    draw();

    const contrib = rb.contributions.filter((c) => c.horizon_days === 1);
    if (contrib.length) {
      const top = contrib.reduce((a, b) => (a.oos_r2_vs_zero > b.oos_r2_vs_zero ? a : b));
      $("contrib-callout").append(
        H("strong", {}, "The 1-day R² of the tree models is one day. "),
        document.createTextNode(`${fmt.cap(fmt.model(top.model))}'s R² of ${fmt.snum(top.oos_r2_vs_zero, 3)} drops to ${fmt.snum(top.oos_r2_without, 3)} without a single test day, ${fmt.date(top.date)}, when silver then moved ${fmt.spct(top.actual_return)} during the January 2026 crash. Squared errors let one extreme day dominate.`),
      );
    } else {
      $("contrib-callout").hidden = true;
    }

    const countRows = [];
    for (const [family, label] of [["horizon", "Horizon"], ["regime", "Regime"]]) {
      for (const [test, name] of [["cw_pvalue", "Clark-West"], ["pt_pvalue", "Pesaran-Timmermann"]]) {
        countRows.push({ family: label, test: name, ...rb.counts[family][test] });
      }
    }
    Charts.table($("counts-table"), {
      columns: [{ key: "family", label: "Model ×", nosort: true }, { key: "test", label: "Test", nosort: true },
                { key: "testable", label: "Tested", num: true, nosort: true }, { key: "raw", label: "Raw", num: true, nosort: true },
                { key: "Bonferroni", label: "Bonf", num: true, nosort: true }, { key: "Holm", label: "Holm", num: true, nosort: true },
                { key: "BH", label: "BH", num: true, nosort: true }],
      rows: countRows,
    });

    Charts.table($("contrib-table"), {
      columns: [
        { key: "model", label: "Model", format: fmt.model },
        { key: "horizon_days", label: "Horizon", num: true, format: (x) => `${x}d` },
        { key: "oos_r2_vs_zero", label: "R²", num: true, format: (x) => fmt.snum(x, 3) },
        { key: "share_of_gain", label: "From best day", num: true, format: (x) => fmt.pct(x, 0) },
        { key: "oos_r2_without", label: "R² without it", num: true, format: (x) => fmt.snum(x, 3), cls: signCls },
        { key: "date", label: "Best day", format: fmt.date },
      ],
      rows: rb.contributions, sort: { key: "oos_r2_vs_zero", dir: "desc" },
    });

    let regime = "all";
    const regimeNames = ["all", ...new Set(rb.regimes.map((r) => r.regime))];
    segmented($("regime-controls"), "Regime", regimeNames.map((r) => ({ value: r, label: r === "all" ? "All" : fmt.regime(r) })),
              regime, (r) => { regime = r; drawRegimes(); });
    function drawRegimes() {
      Charts.table($("regime-table"), {
        columns: [
          { key: "model", label: "Model", format: fmt.model },
          { key: "regime", label: "Regime", format: fmt.regime },
          { key: "n_predictions", label: "Periods", num: true },
          { key: "oos_r2_vs_zero", label: "R² vs random walk", num: true, format: (x) => fmt.snum(x), cls: signCls },
          { key: "directional_accuracy", label: "Direction right", num: true, format: (x) => fmt.pct(x) },
          { key: "cw_pvalue", label: "CW p", num: true, format: fmt.p, cls: pCls },
          { key: "pt_pvalue", label: "PT p", num: true, format: fmt.p, cls: pCls },
          { key: "pt_pvalue_bh", label: "PT p (BH)", num: true, format: fmt.p, cls: pCls },
        ],
        rows: regime === "all" ? rb.regimes : rb.regimes.filter((r) => r.regime === regime),
      });
    }
    drawRegimes();
  }

  function renderOptions() {
    const o = R.options, inp = o.inputs;
    $("options-intro").textContent = `A ${inp.maturity_days}-trading-day European ${inp.option_type} on COMEX silver futures, struck at the money ($${inp.strike.toFixed(2)}). SI=F is a futures price, so it is priced with Black-76, where the futures price has no drift under the pricing measure. Volatility is a GARCH(1,1) forecast averaged to expiry (${fmt.pct(inp.sigma)}); the rate is the 3-month T-bill (${fmt.pct(inp.rate, 2)}).`;

    const tiles = $("options-tiles");
    tile(tiles, { label: "Black-76 price", value: fmt.usd(o.black76, 2), detail: "closed form, the correct model for futures" });
    tile(tiles, { label: "Black-Scholes on futures", value: fmt.usd(o.bs_as_spot, 2),
                  statusNode: status("bad", `${fmt.spct(o.bs_as_spot / o.black76 - 1)} too high`),
                  detail: "treats the futures price as spot" });
    tile(tiles, { label: "Monte Carlo", value: fmt.usd(o.mc.price, 2),
                  detail: `95% CI ${fmt.usd(o.mc.low, 3)} to ${fmt.usd(o.mc.high, 3)}, ${fmt.int(o.mc.n_paths)} paths` });
    const both = o.variance_reduction.find((v) => v.setting === "both");
    tile(tiles, { label: "Variance reduction", value: `${both.paths_saved.toFixed(0)}× fewer paths`,
                  detail: "antithetic + control variate vs plain" });

    const c = o.curve;
    Charts.legend($("curve-legend"), [
      { name: "Black-76 call", color: "--series-1", line: true },
      { name: "Black-Scholes call (wrong for futures)", color: "--series-2", line: true },
      { name: "Black-76 put", color: "--series-3", line: true },
    ]);
    Charts.line($("curve-chart"), {
      x: c.strikes, height: 280, ariaLabel: "Option price by strike",
      series: [{ name: "Black-76 call", values: c.black76_call, color: "--series-1" },
               { name: "Black-Scholes call", values: c.bs_spot_call, color: "--series-2" },
               { name: "Black-76 put", values: c.black76_put, color: "--series-3" }],
      yFormat: (v, tip) => (tip ? fmt.usd(v, 3) : `$${v.toFixed(0)}`), xTickFormat: (v) => `$${v.toFixed(0)}`,
      tooltipTitle: (i) => `Strike ${fmt.usd(c.strikes[i], 2)}`,
    });
    Charts.table($("curve-table"), {
      columns: [{ key: "k", label: "Strike", num: true, format: (x) => fmt.usd(x, 2) },
                { key: "c76", label: "Black-76 call", num: true, format: (x) => fmt.usd(x, 4) },
                { key: "cbs", label: "Black-Scholes call", num: true, format: (x) => fmt.usd(x, 4) },
                { key: "p76", label: "Black-76 put", num: true, format: (x) => fmt.usd(x, 4) }],
      rows: c.strikes.map((k, i) => ({ k, c76: c.black76_call[i], cbs: c.bs_spot_call[i], p76: c.black76_put[i] })),
    });

    const conv = (setting) => o.convergence.filter((r) => r.setting === setting);
    const plain = conv("plain"), red = conv("both");
    Charts.legend($("conv-legend"), [
      { name: "Plain Monte Carlo", color: "--series-2", line: true },
      { name: "Antithetic + control variate", color: "--series-1", line: true },
      { name: "Black-76", color: "--ink-2", line: true },
    ]);
    Charts.line($("conv-chart"), {
      x: plain.map((r) => r.n_paths), xType: "log", height: 280, ariaLabel: "Monte Carlo price convergence",
      bands: [{ lower: plain.map((r) => r.low), upper: plain.map((r) => r.high), color: "--series-2", opacity: 0.18 },
              { lower: red.map((r) => r.low), upper: red.map((r) => r.high), color: "--series-1", opacity: 0.28 }],
      series: [{ name: "Plain", values: plain.map((r) => r.price), color: "--series-2" },
               { name: "Antithetic + control variate", values: red.map((r) => r.price), color: "--series-1" }],
      hlines: [{ value: o.black76, color: "--ink-2" }],
      yFormat: (v, tip) => (tip ? fmt.usd(v, 4) : `$${v.toFixed(2)}`),
      xTickFormat: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)),
      tooltipTitle: (i) => `${fmt.int(plain[i].n_paths)} paths`,
      tooltipRows: (i) => [
        { name: "Plain", value: `${fmt.usd(plain[i].price, 3)} ± ${(1.96 * plain[i].std_error).toFixed(3)}`, color: "--series-2" },
        { name: "Reduced", value: `${fmt.usd(red[i].price, 3)} ± ${(1.96 * red[i].std_error).toFixed(3)}`, color: "--series-1" },
        { name: "Black-76", value: fmt.usd(o.black76, 3), color: "--ink-2" },
      ],
    });
    Charts.table($("conv-table"), {
      columns: [{ key: "n_paths", label: "Paths", num: true, format: fmt.int }, { key: "setting", label: "Setting" },
                { key: "price", label: "Price", num: true, format: (x) => fmt.usd(x, 4) },
                { key: "std_error", label: "Std error", num: true, format: (x) => x.toFixed(5) }],
      rows: o.convergence,
    });

    const vr = o.variance_reduction;
    const vrLabel = { plain: "Plain", antithetic: "Antithetic", "control variate": "Control variate", both: "Both" };
    Charts.hbar($("vr-chart"), {
      categories: vr.map((v) => vrLabel[v.setting] || v.setting), values: vr.map((v) => v.paths_saved),
      colorFn: () => "--series-1", valueFormat: (v) => `${v.toFixed(0)}×`, ariaLabel: "Paths saved by variance reduction",
      rowHeight: 62,
      tooltip: (i) => ({ title: vrLabel[vr[i].setting], rows: [
        { name: "Std error", value: vr[i].std_error.toFixed(5) },
        { name: "Fewer paths needed", value: `${vr[i].paths_saved.toFixed(1)}×` },
      ] }),
    });

    const g = o.greeks;
    Charts.table($("greeks-table"), {
      columns: [{ key: "name", label: "", nosort: true }, { key: "value", label: "Value", num: true, nosort: true }],
      rows: [
        { name: "Futures price", value: fmt.usd(inp.futures, 2) },
        { name: "Strike", value: fmt.usd(inp.strike, 2) },
        { name: "Volatility (GARCH, to expiry)", value: fmt.pct(inp.sigma, 2) },
        { name: "Risk-free rate (3M T-bill)", value: fmt.pct(inp.rate, 2) },
        { name: "Maturity", value: `${inp.maturity_days} trading days` },
        { name: "Delta", value: fmt.snum(g.delta, 4) },
        { name: "Gamma", value: fmt.num(g.gamma, 4) },
        { name: "Vega, per vol point", value: fmt.usd(g.vega_per_vol_pt, 4) },
        { name: "Theta, per trading day", value: fmt.usd(g.theta_per_trading_day, 4) },
        { name: "Rho, per 1% rate move", value: fmt.usd(g.rho_per_pct, 4) },
      ],
    });

    const b = o.bias_check;
    $("bias-callout").append(
      H("strong", {}, "The Monte Carlo pricer is unbiased and its error bars are honest. "),
      document.createTextNode(`Across ${b.runs} independent seeds of ${fmt.int(b.n_sims)} paths, the average pricing error is ${fmt.snum(b.mean_error, 6)} (z = ${b.z.toFixed(2)}), the reported standard error (${b.mean_reported_se.toFixed(5)}) matches the actual spread of prices (${b.empirical_sd.toFixed(5)}), and ${fmt.pct(b.outside_ci, 1)} of runs land outside their own 95% interval, as about 5% should.`),
    );
  }

  // --- shell ------------------------------------------------------------------

  const renderers = {
    overview: renderOverview, risk: renderRisk, backtest: renderBacktest, forecast: renderForecast,
    robustness: renderRobustness, options: renderOptions, method: () => {},
  };
  const sectionData = { risk: "risk", backtest: "risk_backtest", forecast: "forecast", robustness: "robustness", options: "options" };
  const rendered = new Set();

  function route() {
    const id = location.hash.slice(1) in renderers ? location.hash.slice(1) : "overview";
    for (const key of Object.keys(renderers)) $(`page-${key}`).classList.toggle("active", key === id);
    document.querySelectorAll(".nav-inner a").forEach((a) => {
      if (a.getAttribute("href") === `#${id}`) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (!rendered.has(id)) {
      rendered.add(id);
      if (sectionData[id] && !R[sectionData[id]]) {
        $(`page-${id}`).appendChild(H("div", { class: "empty" }, `This section hasn't been exported yet. Run: python export_results.py --only ${sectionData[id]}`));
      } else {
        renderers[id]();
      }
    }
  }

  $("theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable: theme lasts for this visit */ }
    Charts.redrawAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => Charts.redrawAll());

  if (!R) {
    $("as-of-badge").textContent = "No results";
    document.querySelectorAll(".page").forEach((p) => p.remove());
    $("no-data").hidden = false;
    return;
  }

  $("as-of-badge").textContent = `Data as of ${fmt.date(R.meta.as_of)}`;
  $("method-as-of").textContent = R.meta.as_of;
  const footer = $("footer");
  footer.append(
    document.createTextNode(`Bohdan Bilokon · Results as of ${fmt.date(R.meta.as_of)}, generated ${R.meta.generated} · `),
    H("a", { href: "https://github.com/BohdanBil/silver-quant-finance" }, "Source on GitHub"),
    document.createTextNode(" · "),
    H("a", { href: "data/results.json", download: "silver-results.json" }, "Download all results (JSON)"),
  );

  window.addEventListener("hashchange", route);
  route();
})();
