// Small dependency-free SVG chart toolkit for the results site.
// Every chart reads its colors from CSS custom properties at draw time and is
// re-drawn on resize and on theme change, so light/dark both stay correct.

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const color = (c) => (c && c.startsWith("--") ? css(c) : c);

  function svgEl(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function html(tag, attrs = {}, text = null) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "style") node.style.cssText = v;
      else node.setAttribute(k, v);
    }
    if (text !== null) node.textContent = text;
    return node;
  }

  // --- scales & ticks -------------------------------------------------------

  function linear(d0, d1, r0, r1) {
    const span = d1 - d0 || 1;
    const f = (v) => r0 + ((v - d0) / span) * (r1 - r0);
    f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * span;
    return f;
  }

  function log10(d0, d1, r0, r1) {
    const l = linear(Math.log10(d0), Math.log10(d1), r0, r1);
    const f = (v) => l(Math.log10(v));
    f.invert = (p) => 10 ** l.invert(p);
    return f;
  }

  function niceStep(span, count) {
    const raw = span / Math.max(count, 1);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  }

  function niceTicks(min, max, count = 5) {
    if (min === max) { min -= 1; max += 1; }
    const step = niceStep(max - min, count);
    const start = Math.ceil(min / step - 1e-9) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return ticks;
  }

  function niceDomain(min, max, count = 5) {
    if (min === max) { min -= 1; max += 1; }
    const step = niceStep(max - min, count);
    return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
  }

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function dateTicks(t0, t1, maxTicks = 8) {
    const years = (t1 - t0) / (365.25 * 864e5);
    const start = new Date(t0);
    const ticks = [];
    if (years > 2.5) {
      const stride = [1, 2, 5, 10].find((s) => years / s <= maxTicks) || 10;
      for (let y = start.getUTCFullYear() + 1; ; y += stride) {
        const t = Date.UTC(y, 0, 1);
        if (t > t1) break;
        ticks.push({ t, label: String(y) });
      }
    } else {
      const stride = [1, 3, 6, 12].find((s) => (years * 12) / s <= maxTicks) || 12;
      let y = start.getUTCFullYear(), m = Math.ceil((start.getUTCMonth() + 1) / stride) * stride;
      for (;;) {
        if (m >= 12) { y += Math.floor(m / 12); m %= 12; }
        const t = Date.UTC(y, m, 1);
        if (t > t1) break;
        if (t >= t0) ticks.push({ t, label: `${MONTHS[m]} ${y}` });
        m += stride;
      }
    }
    return ticks;
  }

  function textWidth(s, px = 11) { return String(s).length * px * 0.6; }

  // --- lifecycle --------------------------------------------------------------

  const mounted = new Set();

  function mount(container, draw) {
    container.innerHTML = "";
    const entry = { container, draw, width: 0 };
    container.__chart = entry;
    mounted.add(entry);
    render(entry, true);
    if (!container.__observer) {
      container.__observer = new ResizeObserver(() => render(container.__chart));
      container.__observer.observe(container);
    }
  }

  function render(entry, force = false) {
    if (!entry) return;
    const width = Math.floor(entry.container.clientWidth);
    if (!width || (!force && width === entry.width)) return;
    entry.width = width;
    entry.container.innerHTML = "";
    entry.draw(entry.container, width);
  }

  function redrawAll() {
    for (const entry of mounted) {
      if (!entry.container.isConnected) { mounted.delete(entry); continue; }
      entry.width = 0;  // charts on hidden pages redraw when they next get a width
      render(entry, true);
    }
  }

  // --- tooltip ----------------------------------------------------------------

  function tooltip(container) {
    const tip = html("div", { class: "tooltip", role: "status" });
    container.appendChild(tip);
    return {
      show(content, x, y) {
        tip.innerHTML = "";
        if (content.title) tip.appendChild(html("div", { class: "tt-title" }, content.title));
        for (const row of content.rows || []) {
          const line = html("div", { class: "tt-row" });
          const name = html("span");
          if (row.color) name.appendChild(html("i", { style: `background:${color(row.color)}` }));
          name.appendChild(document.createTextNode(row.name));
          line.appendChild(name);
          line.appendChild(html("span", {}, row.value));
          tip.appendChild(line);
        }
        const cw = container.clientWidth;
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        let left = x + 14;
        if (left + tw > cw) left = x - tw - 14;
        tip.style.left = `${Math.max(0, left)}px`;
        tip.style.top = `${Math.max(0, y - th / 2)}px`;
        tip.classList.add("visible");
      },
      hide() { tip.classList.remove("visible"); },
    };
  }

  function legend(container, items) {
    const box = html("div", { class: "legend" });
    for (const item of items) {
      const span = html("span");
      const swatch = item.color.startsWith("--") ? `var(${item.color})` : item.color;
      span.appendChild(html("i", { class: item.line ? "line" : "", style: `background:${swatch}` }));
      span.appendChild(document.createTextNode(item.name));
      box.appendChild(span);
    }
    container.appendChild(box);
  }

  // --- shared axis drawing ------------------------------------------------------

  function yAxis(svg, y, ticks, left, right, format) {
    for (const v of ticks) {
      const py = y(v);
      svgEl("line", { x1: left, x2: right, y1: py, y2: py, stroke: css("--grid"), "stroke-width": 1 }, svg);
      svgEl("text", { x: left - 8, y: py, "text-anchor": "end", "dominant-baseline": "middle" }, svg)
        .textContent = format(v);
    }
  }

  function xAxisLine(svg, left, right, py) {
    svgEl("line", { x1: left, x2: right, y1: py, y2: py, stroke: css("--axis"), "stroke-width": 1 }, svg);
  }

  // --- line / band chart --------------------------------------------------------

  // opts: { x, xType: "number"|"date"|"log", series: [{name, values, color, width, endLabel}],
  //         bands: [{name, lower, upper, color}], height, yFormat, xFormat, xTickFormat,
  //         hlines: [{value, label, color}], markers: [{index, label}], yLabel, tooltipTitle(i) }
  function line(container, opts) {
    mount(container, (root, width) => {
      const height = opts.height || 280;
      const isDate = opts.xType === "date";
      const xs = opts.x.map((v) => (isDate ? Date.parse(v) : v));
      const all = [];
      for (const s of opts.series || []) all.push(...s.values.filter((v) => v !== null));
      for (const b of opts.bands || []) all.push(...b.lower, ...b.upper);
      for (const h of opts.hlines || []) all.push(h.value);
      let [ymin, ymax] = [Math.min(...all), Math.max(...all)];
      if (opts.includeZero) { ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0); }
      [ymin, ymax] = niceDomain(ymin, ymax, 5);

      const endLabels = (opts.series || []).filter((s) => s.endLabel);
      const labelSpace = endLabels.length ? Math.max(...endLabels.map((s) => textWidth(s.name, 12))) + 12 : 0;
      const yTicks = niceTicks(ymin, ymax, 5);
      const left = Math.max(...yTicks.map((t) => textWidth(opts.yFormat(t)))) + 14;
      const m = { top: 10, right: 12 + labelSpace, bottom: 26, left };
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img",
                                 "aria-label": opts.ariaLabel || "" }, root);

      const x = opts.xType === "log" ? log10(xs[0], xs[xs.length - 1], m.left, width - m.right)
                                     : linear(xs[0], xs[xs.length - 1], m.left, width - m.right);
      const y = linear(ymin, ymax, height - m.bottom, m.top);
      yAxis(svg, y, yTicks, m.left, width - m.right, opts.yFormat);
      xAxisLine(svg, m.left, width - m.right, height - m.bottom);

      let xTicks;
      if (isDate) xTicks = dateTicks(xs[0], xs[xs.length - 1], Math.max(2, Math.floor((width - m.left - m.right) / 64)))
        .map((d) => ({ v: d.t, label: d.label }));
      else if (opts.xType === "log") xTicks = xs.filter((v) => [1, 2, 5].includes(Math.round(v / 10 ** Math.floor(Math.log10(v)))))
        .map((v) => ({ v, label: opts.xTickFormat ? opts.xTickFormat(v) : String(v) }));
      else xTicks = niceTicks(xs[0], xs[xs.length - 1], Math.max(2, Math.floor(width / 90)))
        .map((v) => ({ v, label: opts.xTickFormat ? opts.xTickFormat(v) : String(v) }));
      for (const t of xTicks) {
        svgEl("text", { x: x(t.v), y: height - m.bottom + 17, "text-anchor": "middle" }, svg).textContent = t.label;
      }

      for (const b of opts.bands || []) {
        const top = xs.map((v, i) => `${x(v)},${y(b.upper[i])}`);
        const bottom = xs.map((v, i) => `${x(v)},${y(b.lower[i])}`).reverse();
        svgEl("polygon", { points: top.concat(bottom).join(" "), fill: color(b.color), opacity: b.opacity ?? 1 }, svg);
      }
      for (const h of opts.hlines || []) {
        const py = y(h.value);
        svgEl("line", { x1: m.left, x2: width - m.right, y1: py, y2: py, stroke: color(h.color || "--ink-2"),
                        "stroke-width": 1.25 }, svg);
        if (h.label) {
          svgEl("text", { x: m.left + 6, y: py - 6, class: "direct-label" }, svg).textContent = h.label;
        }
      }
      for (const s of opts.series || []) {
        let d = "", pen = false;
        s.values.forEach((v, i) => {
          if (v === null) { pen = false; return; }
          d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`;
          pen = true;
        });
        svgEl("path", { d, fill: "none", stroke: color(s.color), "stroke-width": s.width || 2,
                        "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
        if (s.endLabel) {
          const last = s.values.length - 1;
          svgEl("text", { x: x(xs[last]) + 8, y: y(s.values[last]), "dominant-baseline": "middle",
                          class: "direct-label" }, svg).textContent = s.name;
        }
      }
      for (const mk of opts.markers || []) {
        const s = opts.series[0];
        const px = x(xs[mk.index]), py = y(s.values[mk.index]);
        svgEl("circle", { cx: px, cy: py, r: 5, fill: color(s.color), stroke: css("--surface"), "stroke-width": 2 }, svg);
        const anchor = px > width * 0.7 ? "end" : "start";
        svgEl("text", { x: px + (anchor === "end" ? -10 : 10), y: py + 4, "text-anchor": anchor,
                        class: "direct-label" }, svg).textContent = mk.label;
      }

      // crosshair + tooltip
      const tip = tooltip(root);
      const cross = svgEl("line", { y1: m.top, y2: height - m.bottom, stroke: css("--ink-2"), "stroke-width": 1,
                                    visibility: "hidden" }, svg);
      const dots = (opts.series || []).map((s) => svgEl("circle", { r: 4, fill: color(s.color),
        stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }, svg));
      const hit = svgEl("rect", { x: m.left, y: m.top, width: width - m.left - m.right,
                                  height: height - m.top - m.bottom, fill: "transparent" }, svg);
      const nearest = (px) => {
        const target = x.invert(px);
        let lo = 0, hi = xs.length - 1;
        while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] < target) lo = mid; else hi = mid; }
        return Math.abs(xs[lo] - target) <= Math.abs(xs[hi] - target) ? lo : hi;
      };
      hit.addEventListener("pointermove", (ev) => {
        const rect = svg.getBoundingClientRect();
        const px = (ev.clientX - rect.left) * (width / rect.width);
        const i = nearest(px);
        const cx = x(xs[i]);
        cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
        (opts.series || []).forEach((s, k) => {
          if (s.values[i] === null) { dots[k].setAttribute("visibility", "hidden"); return; }
          dots[k].setAttribute("cx", cx); dots[k].setAttribute("cy", y(s.values[i]));
          dots[k].setAttribute("visibility", "visible");
        });
        const rows = opts.tooltipRows ? opts.tooltipRows(i) : (opts.series || []).map((s) => ({
          name: s.name, value: s.values[i] === null ? "–" : opts.yFormat(s.values[i], true), color: s.color }));
        const title = opts.tooltipTitle ? opts.tooltipTitle(i) : (opts.xFormat ? opts.xFormat(opts.x[i]) : String(opts.x[i]));
        tip.show({ title, rows }, cx * (rect.width / width), ev.clientY - rect.top);
      });
      hit.addEventListener("pointerleave", () => {
        cross.setAttribute("visibility", "hidden");
        dots.forEach((d) => d.setAttribute("visibility", "hidden"));
        tip.hide();
      });
    });
  }

  // --- histogram ------------------------------------------------------------------

  // opts: { edges, counts, color, vlines: [{x, label, color}], xFormat, height }
  function histogram(container, opts) {
    mount(container, (root, width) => {
      const height = opts.height || 280;
      const yTicks = niceTicks(0, Math.max(...opts.counts), 4);
      const count = (v) => v.toLocaleString("en-US");
      const m = { top: 10, right: 12, bottom: 26, left: Math.max(...yTicks.map((t) => textWidth(count(t)))) + 14 };
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img",
                                 "aria-label": opts.ariaLabel || "" }, root);
      const lo = opts.edges[0], hi = opts.edges[opts.edges.length - 1];
      const x = linear(lo, hi, m.left, width - m.right);
      const y = linear(0, yTicks[yTicks.length - 1], height - m.bottom, m.top);
      yAxis(svg, y, yTicks, m.left, width - m.right, count);
      for (const v of niceTicks(lo, hi, Math.max(3, Math.floor(width / 110)))) {
        svgEl("text", { x: x(v), y: height - m.bottom + 17, "text-anchor": "middle" }, svg).textContent = opts.xFormat(v);
      }
      const tip = tooltip(root);
      const fill = color(opts.color || "--series-1");
      opts.counts.forEach((c, i) => {
        const x0 = x(opts.edges[i]), x1 = x(opts.edges[i + 1]);
        const bar = svgEl("rect", { x: x0 + 0.5, y: y(c), width: Math.max(0.5, x1 - x0 - 1), height: y(0) - y(c),
                                    fill, rx: 1 }, svg);
        const column = svgEl("rect", { x: x0, y: m.top, width: x1 - x0, height: height - m.top - m.bottom,
                                       fill: "transparent" }, svg);
        column.addEventListener("pointermove", (ev) => {
          const rect = svg.getBoundingClientRect();
          tip.show({ title: `${opts.xFormat(opts.edges[i])} to ${opts.xFormat(opts.edges[i + 1])}`,
                     rows: [{ name: "simulated paths", value: count(c) }] },
                   ev.clientX - rect.left, ev.clientY - rect.top);
          bar.setAttribute("opacity", 0.75);
        });
        column.addEventListener("pointerleave", () => { tip.hide(); bar.setAttribute("opacity", 1); });
      });
      xAxisLine(svg, m.left, width - m.right, height - m.bottom);
      for (const v of opts.vlines || []) {
        const px = x(v.x);
        svgEl("line", { x1: px, x2: px, y1: m.top, y2: height - m.bottom, stroke: color(v.color), "stroke-width": 2 }, svg);
      }
    });
  }

  // --- horizontal bars ------------------------------------------------------------

  // opts: { categories, values, colorFn(v, i), valueFormat, tooltip(i), rowHeight, xLabel, domain }
  function hbar(container, opts) {
    mount(container, (root, width) => {
      const rowH = opts.rowHeight || 28;
      const height = opts.categories.length * rowH + 34;
      const left = Math.max(...opts.categories.map((c) => textWidth(c, 12))) + 16;
      const m = { top: 4, right: 16, bottom: 26, left };
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img",
                                 "aria-label": opts.ariaLabel || "" }, root);
      let [lo, hi] = opts.domain || [Math.min(0, ...opts.values), Math.max(0, ...opts.values)];
      [lo, hi] = niceDomain(lo, hi, Math.max(3, Math.floor(width / 120)));
      const x = linear(lo, hi, m.left, width - m.right);
      const ticks = niceTicks(lo, hi, Math.max(3, Math.floor(width / 120)));
      for (const v of ticks) {
        svgEl("line", { x1: x(v), x2: x(v), y1: m.top, y2: height - m.bottom, stroke: css("--grid") }, svg);
        svgEl("text", { x: x(v), y: height - m.bottom + 17, "text-anchor": "middle" }, svg).textContent = opts.valueFormat(v);
      }
      const tip = tooltip(root);
      opts.categories.forEach((cat, i) => {
        const cy = m.top + i * rowH + rowH / 2;
        svgEl("text", { x: m.left - 10, y: cy, "text-anchor": "end", "dominant-baseline": "middle",
                        style: `fill:${css("--ink-2")};font-size:12px` }, svg).textContent = cat;
        const v = opts.values[i];
        const x0 = x(Math.min(0, v)), x1 = x(Math.max(0, v));
        const bar = svgEl("rect", { x: x0, y: cy - rowH * 0.3, width: Math.max(1, x1 - x0), height: rowH * 0.6,
                                    rx: 3, fill: color(opts.colorFn(v, i)) }, svg);
        const hitRow = svgEl("rect", { x: 0, y: cy - rowH / 2, width, height: rowH, fill: "transparent" }, svg);
        hitRow.addEventListener("pointermove", (ev) => {
          const rect = svg.getBoundingClientRect();
          tip.show(opts.tooltip ? opts.tooltip(i) : { title: cat, rows: [{ name: "value", value: opts.valueFormat(v) }] },
                   ev.clientX - rect.left, ev.clientY - rect.top);
          bar.setAttribute("opacity", 0.75);
        });
        hitRow.addEventListener("pointerleave", () => { tip.hide(); bar.setAttribute("opacity", 1); });
      });
      if (lo < 0 && hi > 0) {
        svgEl("line", { x1: x(0), x2: x(0), y1: m.top, y2: height - m.bottom, stroke: css("--ink-2"), "stroke-width": 1.25 }, svg);
      }
    });
  }

  // --- dot plot -------------------------------------------------------------------

  // opts: { categories, values, flagged: [bool], ref, refLabel, valueFormat, tooltip(i), domain }
  function dots(container, opts) {
    mount(container, (root, width) => {
      const rowH = 26;
      const height = opts.categories.length * rowH + 44;
      const left = Math.max(...opts.categories.map((c) => textWidth(c, 12))) + 16;
      const m = { top: 20, right: 16, bottom: 26, left };
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img",
                                 "aria-label": opts.ariaLabel || "" }, root);
      const [lo, hi] = niceDomain(0, Math.max(opts.ref, ...opts.values) * 1.1, 4);
      const x = linear(lo, hi, m.left, width - m.right);
      for (const v of niceTicks(lo, hi, Math.max(3, Math.floor(width / 110)))) {
        svgEl("line", { x1: x(v), x2: x(v), y1: m.top, y2: height - m.bottom, stroke: css("--grid") }, svg);
        svgEl("text", { x: x(v), y: height - m.bottom + 17, "text-anchor": "middle" }, svg).textContent = opts.valueFormat(v);
      }
      const rx = x(opts.ref);
      svgEl("line", { x1: rx, x2: rx, y1: m.top - 6, y2: height - m.bottom, stroke: css("--ink-2"), "stroke-width": 1.5 }, svg);
      svgEl("text", { x: rx + 6, y: m.top - 8, class: "direct-label" }, svg).textContent = opts.refLabel;
      const tip = tooltip(root);
      const bad = css("--critical"), ok = color("--series-1"), surface = css("--surface");
      opts.categories.forEach((cat, i) => {
        const cy = m.top + i * rowH + rowH / 2;
        svgEl("line", { x1: m.left, x2: width - m.right, y1: cy, y2: cy, stroke: css("--grid") }, svg);
        svgEl("text", { x: m.left - 10, y: cy, "text-anchor": "end", "dominant-baseline": "middle",
                        style: `fill:${css("--ink-2")};font-size:12px` }, svg).textContent = cat;
        const cx = x(opts.values[i]);
        let mark;
        if (opts.flagged[i]) {
          mark = svgEl("g", {}, svg);
          const s = 5;
          for (const [a, b] of [[[-s, -s], [s, s]], [[-s, s], [s, -s]]]) {
            svgEl("line", { x1: cx + a[0], y1: cy + a[1], x2: cx + b[0], y2: cy + b[1], stroke: surface, "stroke-width": 5,
                            "stroke-linecap": "round" }, mark);
            svgEl("line", { x1: cx + a[0], y1: cy + a[1], x2: cx + b[0], y2: cy + b[1], stroke: bad, "stroke-width": 2.5,
                            "stroke-linecap": "round" }, mark);
          }
        } else {
          mark = svgEl("circle", { cx, cy, r: 5.5, fill: ok, stroke: surface, "stroke-width": 2 }, svg);
        }
        const hitRow = svgEl("rect", { x: 0, y: cy - rowH / 2, width, height: rowH, fill: "transparent" }, svg);
        hitRow.addEventListener("pointermove", (ev) => {
          const rect = svg.getBoundingClientRect();
          tip.show(opts.tooltip(i), ev.clientX - rect.left, ev.clientY - rect.top);
        });
        hitRow.addEventListener("pointerleave", () => tip.hide());
      });
    });
  }

  // --- heatmap (HTML grid) ----------------------------------------------------------

  function hexToRgb(hex) {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }

  function mix(a, b, t) {
    const [ra, ga, ba] = hexToRgb(a), [rb, gb, bb] = hexToRgb(b);
    return `rgb(${Math.round(ra + (rb - ra) * t)},${Math.round(ga + (gb - ga) * t)},${Math.round(ba + (bb - ba) * t)})`;
  }

  function diverging(value, limit, center = 0) {
    const t = Math.max(-1, Math.min(1, (value - center) / (limit || 1)));
    const mid = css("--div-mid");
    return t >= 0 ? mix(mid, css("--div-pos"), t) : mix(mid, css("--div-neg"), -t);
  }

  function readableOn(rgb) {
    const [r, g, b] = rgb.match(/\d+/g).map(Number);
    const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    return lum > 0.55 ? "#0b0b0b" : "#ffffff";
  }

  // opts: { rows, cols, rowLabel(r), colLabel(c), value(r, c), format(v), fill(v), tooltip(r, c) }
  function heatmap(container, opts) {
    mount(container, (root) => {
      const grid = html("div", { class: "heatmap", role: "table",
                                 style: `grid-template-columns: max-content repeat(${opts.cols.length}, minmax(56px, 1fr))` });
      grid.appendChild(html("div"));
      for (const c of opts.cols) grid.appendChild(html("div", { class: "hm-head", role: "columnheader" }, opts.colLabel(c)));
      const tip = tooltip(root);
      for (const r of opts.rows) {
        grid.appendChild(html("div", { class: "hm-row", role: "rowheader" }, opts.rowLabel ? opts.rowLabel(r) : r));
        for (const c of opts.cols) {
          const v = opts.value(r, c);
          const bg = v === null ? css("--surface-2") : opts.fill(v);
          const cell = html("div", { class: "hm-cell", role: "cell",
                                     style: `background:${bg};color:${v === null ? css("--muted") : readableOn(bg)}` },
                            v === null ? "–" : opts.format(v));
          cell.addEventListener("pointermove", (ev) => {
            const rect = root.getBoundingClientRect();
            tip.show(opts.tooltip(r, c), ev.clientX - rect.left, ev.clientY - rect.top);
          });
          cell.addEventListener("pointerleave", () => tip.hide());
          grid.appendChild(cell);
        }
      }
      root.appendChild(grid);
    });
  }

  // --- sortable table ---------------------------------------------------------------

  // columns: [{key, label, format(v, row), num, sortValue(row), cls(v, row)}]
  function table(container, { columns, rows, sort, highlight, caption }) {
    container.innerHTML = "";
    const wrap = html("div", { class: "table-wrap" });
    const tbl = html("table");
    if (caption) tbl.appendChild(html("caption", { style: "position:absolute;left:-9999px" }, caption));
    const thead = html("thead"), tbody = html("tbody");
    const headRow = html("tr");
    let state = sort ? { ...sort } : null;
    const value = (col, row) => (col.sortValue ? col.sortValue(row) : row[col.key]);

    function draw() {
      tbody.innerHTML = "";
      let sorted = rows.slice();
      if (state) {
        const col = columns.find((c) => c.key === state.key);
        sorted.sort((a, b) => {
          const va = value(col, a), vb = value(col, b);
          if (va === null || va === undefined) return 1;
          if (vb === null || vb === undefined) return -1;
          const cmp = typeof va === "string" ? va.localeCompare(vb) : va - vb;
          return state.dir === "asc" ? cmp : -cmp;
        });
      }
      for (const row of sorted) {
        const tr = html("tr", highlight && highlight(row) ? { class: "highlight" } : {});
        for (const col of columns) {
          const v = row[col.key];
          const td = html("td", { class: [col.num ? "num" : "", col.cls ? col.cls(v, row) : ""].join(" ").trim() });
          const content = col.format ? col.format(v, row) : (v ?? "–");
          if (content instanceof Node) td.appendChild(content); else td.textContent = content;
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      headRow.querySelectorAll("th").forEach((th, k) => {
        th.setAttribute("aria-sort", state && columns[k].key === state.key ? (state.dir === "asc" ? "ascending" : "descending") : "none");
      });
    }

    columns.forEach((col) => {
      const th = html("th", { class: [col.num ? "num" : "", col.nosort ? "" : "sortable"].join(" ").trim(), scope: "col" }, col.label);
      if (!col.nosort) {
        th.tabIndex = 0;
        const toggle = () => {
          state = state && state.key === col.key ? { key: col.key, dir: state.dir === "asc" ? "desc" : "asc" }
                                                 : { key: col.key, dir: col.num ? "desc" : "asc" };
          draw();
        };
        th.addEventListener("click", toggle);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      }
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    tbl.appendChild(thead);
    tbl.appendChild(tbody);
    wrap.appendChild(tbl);
    container.appendChild(wrap);
    draw();
  }

  return { line, histogram, hbar, dots, heatmap, table, legend, diverging, redrawAll, css, html };
})();
