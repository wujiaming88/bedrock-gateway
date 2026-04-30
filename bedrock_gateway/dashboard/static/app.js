/* ------------------------------------------------------------------
 * Bedrock Gateway Cockpit — front-end
 *
 * Responsibilities:
 *   - Poll the /api/metrics/* endpoints every 5s.
 *   - Render 4 radial gauges (QPS, P95 latency, success rate, tokens/h).
 *   - Render a traffic+latency dual-axis line chart (Chart.js).
 *   - Render model distribution doughnut + tokens bar (Chart.js).
 *   - Render the recent-requests table and the error panel.
 *
 * Kept deliberately to ES5-ish syntax: no `?.`, no `??`, no `let`/`const`
 * block scoping gymnastics, no arrow-function class fields. Only top-level
 * `var` declarations and plain `function` expressions so this runs in
 * anything with Promise + fetch.
 * ------------------------------------------------------------------ */

(function () {
  "use strict";

  // ----- Configuration ---------------------------------------------
  var POLL_MS = 5000;
  var MONO_STACK =
    "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Cascadia Code', " +
    "SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace";
  var MODEL_COLORS = [
    "#00ff88",
    "#00aaff",
    "#ffaa00",
    "#aa66ff",
    "#ff4488",
    "#44ddff",
    "#ffdd44",
    "#88ff44",
    "#ff8844",
    "#ff4444"
  ];

  // ----- State -----------------------------------------------------
  var currentWindow = "1h";
  var currentFilter = "all";
  // Rendered window options — (re)built from /api/metrics/system retain_days.
  // Always starts with the 1h/6h/24h trio; longer windows appear as retention
  // allows.
  var windowOptions = ["1h", "6h", "24h"];
  var charts = {
    traffic: null,
    modelsPie: null,
    modelsTokens: null,
    memory: null
  };
  var lastSuccessfulTick = 0;

  // ----- Utility helpers -------------------------------------------
  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function $$(sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  }

  function formatNumber(n) {
    if (n === null || n === undefined || isNaN(n)) {
      return "—";
    }
    if (n >= 1e9) {
      return (n / 1e9).toFixed(1) + "B";
    }
    if (n >= 1e6) {
      return (n / 1e6).toFixed(1) + "M";
    }
    if (n >= 1e3) {
      return (n / 1e3).toFixed(1) + "k";
    }
    if (n >= 10) {
      return String(Math.round(n));
    }
    return n.toFixed(2);
  }

  function formatUptime(seconds) {
    seconds = Math.max(0, Math.floor(seconds || 0));
    var d = Math.floor(seconds / 86400);
    var h = Math.floor((seconds % 86400) / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = seconds % 60;
    if (d > 0) {
      return d + "d " + h + "h";
    }
    if (h > 0) {
      return h + "h " + m + "m";
    }
    if (m > 0) {
      return m + "m " + s + "s";
    }
    return s + "s";
  }

  function formatTime(ts) {
    if (!ts) {
      return "—";
    }
    var d = new Date(ts * 1000);
    var hh = String(d.getHours()).padStart(2, "0");
    var mm = String(d.getMinutes()).padStart(2, "0");
    var ss = String(d.getSeconds()).padStart(2, "0");
    return hh + ":" + mm + ":" + ss;
  }

  function formatClock(date) {
    var hh = String(date.getHours()).padStart(2, "0");
    var mm = String(date.getMinutes()).padStart(2, "0");
    var ss = String(date.getSeconds()).padStart(2, "0");
    return hh + ":" + mm + ":" + ss;
  }

  // Turn a Bedrock model id like "us.anthropic.claude-sonnet-4-6-20250514-v1:0"
  // into the shorter "claude-sonnet-4-6" that fits in a legend/table cell.
  function friendlyModel(name) {
    if (!name || typeof name !== "string") {
      return "-";
    }
    var m = name;
    // Strip common regional prefixes.
    m = m.replace(/^(us|eu|apac|ap|ca|sa)\./, "");
    // Strip vendor namespace.
    m = m.replace(/^anthropic\./, "");
    // Drop a trailing ":0" inference-profile suffix.
    m = m.replace(/:\d+$/, "");
    // Drop a trailing "-vN" or "-YYYYMMDD" version tag (but keep X-Y-Z dotted
    // variant numbers like "claude-sonnet-4-6").
    m = m.replace(/-v\d+(?:[-.]\d+)*$/, "");
    m = m.replace(/-\d{8}$/, "");
    return m || name;
  }

  function statusClass(status) {
    if (!status) {
      return "";
    }
    if (status >= 500) {
      return "status-5xx";
    }
    if (status >= 400) {
      return "status-4xx";
    }
    if (status >= 300) {
      return "status-3xx";
    }
    return "status-2xx";
  }

  function latencyClass(ms) {
    if (!ms && ms !== 0) {
      return "";
    }
    if (ms < 500) {
      return "latency-fast";
    }
    if (ms < 2000) {
      return "latency-mid";
    }
    return "latency-slow";
  }

  function fetchJSON(url) {
    return fetch(url, {
      headers: { Accept: "application/json" },
      cache: "no-store"
    }).then(function (resp) {
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status + " for " + url);
      }
      return resp.json();
    });
  }

  function safeGet(obj, key, fallback) {
    if (obj && Object.prototype.hasOwnProperty.call(obj, key) && obj[key] !== null && obj[key] !== undefined) {
      return obj[key];
    }
    return fallback;
  }

  // ----- Gauges (Canvas) -------------------------------------------
  /**
   * Draw a radial gauge on a canvas.
   *
   * @param {HTMLCanvasElement} canvas
   * @param {object} opts
   *   - value  (number)   current reading
   *   - max    (number)   value that corresponds to 100%
   *   - label  (string)   bottom title e.g. "QPS"
   *   - unit   (string)   suffix e.g. " ms"
   *   - format (function) optional value→string
   *   - color  (string)   primary stroke colour
   */
  function drawGauge(canvas, opts) {
    var ctx = canvas.getContext("2d");
    var dpr = window.devicePixelRatio || 1;

    // Ensure canvas has a CSS size (fall back to attribute size when layout
    // hasn't happened yet).
    var cssW = canvas.clientWidth || canvas.width || 200;
    var cssH = canvas.clientHeight || canvas.height || 200;

    // Set the CSS size explicitly so the backing-store scaling we apply below
    // never inflates the displayed element.
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";

    var targetW = Math.max(1, Math.round(cssW * dpr));
    var targetH = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    // Draw in CSS-pixel coordinates, let scale() handle DPR.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(dpr, dpr);

    var w = cssW;
    var h = cssH;
    var cx = w / 2;
    var cy = h / 2 + h * 0.06;
    var radius = Math.min(w, h) * 0.38;
    var thickness = Math.max(10, radius * 0.18);

    var value = typeof opts.value === "number" && isFinite(opts.value) ? opts.value : 0;
    var max = opts.max > 0 ? opts.max : 1;
    var ratio = Math.max(0, Math.min(1, value / max));

    var primary = opts.color || "#00ff88";
    var secondary = opts.color2 || "#00aaff";
    var mono = opts.mono || MONO_STACK;

    // Arc spans 225° → 315° (bottom-left to bottom-right going over the top)
    var startAngle = Math.PI * 0.75;
    var endAngle = Math.PI * 2.25;

    // Soft glow halo under the track
    var halo = ctx.createRadialGradient(cx, cy, radius * 0.4, cx, cy, radius * 1.25);
    halo.addColorStop(0, "rgba(0, 0, 0, 0)");
    halo.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = halo;

    // Track (background arc — very subtle)
    ctx.lineWidth = thickness;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle, false);
    ctx.stroke();

    // Gradient fill arc with glow
    var grad = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
    grad.addColorStop(0, primary);
    grad.addColorStop(1, secondary);
    ctx.strokeStyle = grad;
    ctx.shadowColor = primary;
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.arc(
      cx,
      cy,
      radius,
      startAngle,
      startAngle + (endAngle - startAngle) * ratio,
      false
    );
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Tick marks
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    var ticks = 10;
    for (var i = 0; i <= ticks; i++) {
      var t = i / ticks;
      var ang = startAngle + (endAngle - startAngle) * t;
      var inner = radius + thickness / 2 + 3;
      var outer = inner + (i % 5 === 0 ? 9 : 4);
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(ang) * inner, cy + Math.sin(ang) * inner);
      ctx.lineTo(cx + Math.cos(ang) * outer, cy + Math.sin(ang) * outer);
      ctx.stroke();
    }

    // Reflection below the gauge — a subtle highlight streak
    var reflect = ctx.createLinearGradient(
      cx - radius,
      cy + radius * 0.9,
      cx + radius,
      cy + radius * 0.9
    );
    reflect.addColorStop(0, "rgba(255,255,255,0)");
    reflect.addColorStop(0.5, "rgba(255,255,255,0.04)");
    reflect.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = reflect;
    ctx.fillRect(cx - radius, cy + radius * 0.82, radius * 2, 2);

    // Center value
    ctx.fillStyle = primary;
    ctx.shadowColor = primary;
    ctx.shadowBlur = 8;
    ctx.font = "600 " + Math.floor(radius * 0.52) + "px " + mono;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    var display = opts.format ? opts.format(value) : formatNumber(value);
    ctx.fillText(display, cx, cy - radius * 0.1);
    ctx.shadowBlur = 0;

    // Unit suffix
    if (opts.unit) {
      ctx.fillStyle = "rgba(138, 147, 166, 0.9)";
      ctx.font = "500 " + Math.floor(radius * 0.2) + "px " + mono;
      ctx.fillText(opts.unit, cx, cy + radius * 0.24);
    }

    // Label
    ctx.fillStyle = "rgba(138, 147, 166, 0.7)";
    ctx.font = "600 " + Math.floor(radius * 0.16) + "px " + mono;
    ctx.fillText((opts.label || "").toUpperCase(), cx, cy + radius * 0.62);
  }

  function renderGauges(overview, traffic) {
    // QPS: prefer the server-side rolling value, which compensates for the
    // current minute being partially elapsed. Fall back to the last sample.
    var qps = safeGet(overview, "qps", null);
    if (qps === null) {
      qps = 0;
      if (traffic && traffic.qps && traffic.qps.length) {
        qps = traffic.qps[traffic.qps.length - 1] || 0;
      }
    }
    // P95: prefer the pooled 5-minute value from the server; fall back to the
    // max of the last 5 per-minute p95s.
    var p95 = safeGet(overview, "p95_ms", null);
    if (p95 === null) {
      p95 = 0;
      if (traffic && traffic.p95 && traffic.p95.length) {
        var look = traffic.p95.slice(-5);
        for (var i = 0; i < look.length; i++) {
          if (look[i] > p95) {
            p95 = look[i];
          }
        }
      }
    }
    var successRate = safeGet(overview, "success_rate", 0);
    // Tokens/h is a rolling 60-minute sum on the server. For legacy payloads
    // fall back to lifetime total / uptime.
    var tokensPerHour = safeGet(overview, "tokens_per_hour", null);
    if (tokensPerHour === null) {
      var totalTokens =
        safeGet(overview, "prompt_tokens", 0) + safeGet(overview, "completion_tokens", 0);
      var uptimeSec = safeGet(overview, "uptime_seconds", 1) || 1;
      tokensPerHour = totalTokens / (uptimeSec / 3600);
    }

    var canvasQps = $('canvas[data-key="qps"]');
    var canvasSuccess = $('canvas[data-key="success"]');
    var canvasLatency = $('canvas[data-key="latency"]');
    var canvasTokens = $('canvas[data-key="tokens"]');

    if (canvasQps) {
      // QPS: 0 is healthy (no traffic). Ramp to amber at 50, red at 100.
      var qpsColor = qps < 50 ? "#00ff88" : qps < 100 ? "#ffaa00" : "#ff4444";
      var qpsColor2 = qps < 50 ? "#00aaff" : qps < 100 ? "#ffcc44" : "#ff8844";
      drawGauge(canvasQps, {
        value: qps,
        max: Math.max(5, qps * 1.4),
        label: "QPS",
        color: qpsColor,
        color2: qpsColor2,
        format: function (v) {
          return v.toFixed(2);
        }
      });
    }

    if (canvasSuccess) {
      // Success %: green >99, amber >95, red ≤95. With no traffic successRate
      // is 0 → treat as healthy/green so an empty dashboard isn't all red.
      var hasTraffic = safeGet(overview, "total_requests", 0) > 0;
      var sColor, sColor2;
      if (!hasTraffic || successRate > 99) {
        sColor = "#00ff88";
        sColor2 = "#00aaff";
      } else if (successRate > 95) {
        sColor = "#ffaa00";
        sColor2 = "#ffcc44";
      } else {
        sColor = "#ff4444";
        sColor2 = "#ff8844";
      }
      drawGauge(canvasSuccess, {
        value: successRate,
        max: 100,
        label: "SUCCESS %",
        color: sColor,
        color2: sColor2,
        format: function (v) {
          return v.toFixed(1);
        },
        unit: "%"
      });
    }

    if (canvasLatency) {
      // P95: <1000ms green, <3000ms amber, ≥3000ms red.
      var lColor = p95 < 1000 ? "#00ff88" : p95 < 3000 ? "#ffaa00" : "#ff4444";
      var lColor2 = p95 < 1000 ? "#00aaff" : p95 < 3000 ? "#ffcc44" : "#ff8844";
      drawGauge(canvasLatency, {
        value: p95,
        max: Math.max(2000, p95 * 1.3),
        label: "P95 LATENCY",
        color: lColor,
        color2: lColor2,
        format: function (v) {
          return Math.round(v);
        },
        unit: "ms"
      });
    }

    if (canvasTokens) {
      // Pure information — no threshold colouring.
      drawGauge(canvasTokens, {
        value: tokensPerHour,
        max: Math.max(1000, tokensPerHour * 1.3),
        label: "TOKENS / H",
        color: "#00aaff",
        color2: "#44ddff",
        format: function (v) {
          return formatNumber(v);
        }
      });
    }

    // TTFT — first-token latency gauge (streaming only).
    var canvasTtft = $('canvas[data-key="ttft"]');
    if (canvasTtft) {
      var ttft = safeGet(overview, "ttft_p50_ms", 0) || 0;
      var tColor = ttft < 500 ? "#00ff88" : ttft < 1500 ? "#ffaa00" : "#ff4444";
      var tColor2 = ttft < 500 ? "#00aaff" : ttft < 1500 ? "#ffcc44" : "#ff8844";
      drawGauge(canvasTtft, {
        value: ttft,
        max: Math.max(1500, ttft * 1.3),
        label: "TTFT",
        color: tColor,
        color2: tColor2,
        format: function (v) {
          return Math.round(v);
        },
        unit: "ms"
      });
    }

    // Tokens / second — output throughput. Informational blue.
    var canvasTps = $('canvas[data-key="tps"]');
    if (canvasTps) {
      var tps = safeGet(overview, "tokens_per_sec_avg", 0) || 0;
      drawGauge(canvasTps, {
        value: tps,
        max: Math.max(50, tps * 1.3),
        label: "TOKENS / S",
        color: "#00aaff",
        color2: "#44ddff",
        format: function (v) {
          return v.toFixed(1);
        }
      });
    }
  }

  function renderMiniTiles(overview) {
    var retry = safeGet(overview, "retry_rate", null);
    var timeout = safeGet(overview, "timeout_rate", null);
    var p50 = safeGet(overview, "p50_ms", null);
    var p99 = safeGet(overview, "p99_ms", null);

    var retryEl = $("#tile-retry");
    var timeoutEl = $("#tile-timeout");
    var p50El = $("#tile-p50");
    var p99El = $("#tile-p99");

    if (retryEl) {
      retryEl.textContent = retry === null ? "—" : retry.toFixed(2) + "%";
      retryEl.classList.remove("warn", "bad");
      if (retry !== null) {
        if (retry >= 5) retryEl.classList.add("bad");
        else if (retry >= 1) retryEl.classList.add("warn");
      }
    }
    if (timeoutEl) {
      timeoutEl.textContent = timeout === null ? "—" : timeout.toFixed(2) + "%";
      timeoutEl.classList.remove("warn", "bad");
      if (timeout !== null) {
        if (timeout >= 2) timeoutEl.classList.add("bad");
        else if (timeout >= 0.5) timeoutEl.classList.add("warn");
      }
    }
    if (p50El) {
      p50El.textContent = p50 === null ? "—" : Math.round(p50) + " ms";
    }
    if (p99El) {
      p99El.textContent = p99 === null ? "—" : Math.round(p99) + " ms";
      p99El.classList.remove("warn", "bad");
      if (p99 !== null) {
        if (p99 >= 3000) p99El.classList.add("bad");
        else if (p99 >= 1000) p99El.classList.add("warn");
      }
    }
  }

  // ----- Traffic chart ---------------------------------------------
  function labelsToHHMM(labels) {
    var out = [];
    for (var i = 0; i < labels.length; i++) {
      var d = new Date(labels[i] * 1000);
      out.push(String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0"));
    }
    return out;
  }

  // Format x-axis labels to suit the active window. Short windows stay as
  // "HH:MM"; multi-day windows prefix "MM-DD" so the day boundary is visible.
  function labelsForWindow(labels, win) {
    var out = [];
    for (var i = 0; i < labels.length; i++) {
      var d = new Date(labels[i] * 1000);
      var mo = String(d.getMonth() + 1).padStart(2, "0");
      var dy = String(d.getDate()).padStart(2, "0");
      var hh = String(d.getHours()).padStart(2, "0");
      var mm = String(d.getMinutes()).padStart(2, "0");
      if (win === "7d") {
        out.push(mo + "-" + dy);
      } else if (win === "3d") {
        out.push(mo + "-" + dy + " " + hh + ":" + mm);
      } else {
        out.push(hh + ":" + mm);
      }
    }
    return out;
  }

  function downsample(arr, maxPoints) {
    if (!arr || arr.length <= maxPoints) {
      return arr || [];
    }
    var step = Math.ceil(arr.length / maxPoints);
    var out = [];
    for (var i = 0; i < arr.length; i += step) {
      // bucket mean
      var slice = arr.slice(i, i + step);
      var sum = 0;
      for (var j = 0; j < slice.length; j++) {
        sum += slice[j];
      }
      out.push(sum / slice.length);
    }
    return out;
  }

  function downsampleLabels(arr, maxPoints) {
    if (!arr || arr.length <= maxPoints) {
      return arr || [];
    }
    var step = Math.ceil(arr.length / maxPoints);
    var out = [];
    for (var i = 0; i < arr.length; i += step) {
      out.push(arr[i]);
    }
    return out;
  }

  function makeLineGradient(ctx, chartArea, hex, alphaTop, alphaBottom) {
    if (!chartArea) {
      return "rgba(0,0,0,0)";
    }
    var top = chartArea.top;
    var bottom = chartArea.bottom;
    var g = ctx.createLinearGradient(0, top, 0, bottom);
    g.addColorStop(0, hexToRgba(hex, alphaTop));
    g.addColorStop(1, hexToRgba(hex, alphaBottom));
    return g;
  }

  function hexToRgba(hex, alpha) {
    // Accept #rgb / #rrggbb; fall back to the raw input if it's already rgba().
    if (!hex || hex[0] !== "#") {
      return hex;
    }
    var h = hex.slice(1);
    if (h.length === 3) {
      h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    }
    var r = parseInt(h.slice(0, 2), 16);
    var g = parseInt(h.slice(2, 4), 16);
    var b = parseInt(h.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function renderTraffic(traffic) {
    var el = $("#chart-traffic");
    if (!el || !window.Chart) {
      return;
    }

    // Down-sample big windows so labels stay legible
    var maxPoints = 90;
    var labelsTs = downsampleLabels(traffic.labels || [], maxPoints);
    var qps = downsample(traffic.qps || [], maxPoints);
    var p50 = downsample(traffic.p50 || [], maxPoints);
    var p95 = downsample(traffic.p95 || [], maxPoints);
    var p99 = downsample(traffic.p99 || [], maxPoints);
    var labels = labelsForWindow(labelsTs, traffic.window || currentWindow);

    var datasets = [
      {
        type: "bar",
        label: "QPS",
        data: qps,
        backgroundColor: function (ctx) {
          return makeLineGradient(ctx.chart.ctx, ctx.chart.chartArea, "#00ff88", 0.55, 0.05);
        },
        borderColor: "rgba(0, 255, 136, 0.8)",
        borderWidth: 0,
        borderRadius: 3,
        yAxisID: "yQps",
        order: 4
      },
      {
        type: "line",
        label: "p50",
        data: p50,
        borderColor: "#00aaff",
        backgroundColor: function (ctx) {
          return makeLineGradient(ctx.chart.ctx, ctx.chart.chartArea, "#00aaff", 0.25, 0);
        },
        fill: true,
        borderWidth: 2.5,
        tension: 0.35,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: "#00aaff",
        pointHoverBorderColor: "#0e131c",
        pointHoverBorderWidth: 2,
        yAxisID: "yMs",
        order: 3
      },
      {
        type: "line",
        label: "p95",
        data: p95,
        borderColor: "#ffaa00",
        backgroundColor: function (ctx) {
          return makeLineGradient(ctx.chart.ctx, ctx.chart.chartArea, "#ffaa00", 0.18, 0);
        },
        fill: true,
        borderWidth: 2.5,
        tension: 0.35,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: "#ffaa00",
        pointHoverBorderColor: "#0e131c",
        pointHoverBorderWidth: 2,
        yAxisID: "yMs",
        order: 2
      },
      {
        type: "line",
        label: "p99",
        data: p99,
        borderColor: "#ff4488",
        backgroundColor: "transparent",
        borderWidth: 2,
        borderDash: [5, 4],
        tension: 0.35,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: "#ff4488",
        pointHoverBorderColor: "#0e131c",
        pointHoverBorderWidth: 2,
        yAxisID: "yMs",
        order: 1
      }
    ];

    if (charts.traffic) {
      charts.traffic.data.labels = labels;
      charts.traffic.data.datasets = datasets;
      charts.traffic.update("none");
      return;
    }

    charts.traffic = new Chart(el.getContext("2d"), {
      type: "bar",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        devicePixelRatio: window.devicePixelRatio || 1,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: {
              color: "rgba(255,255,255,0.6)",
              font: { family: MONO_STACK, size: 11, weight: "500" },
              usePointStyle: true,
              pointStyle: "rectRounded",
              padding: 14
            }
          },
          tooltip: {
            backgroundColor: "rgba(10, 14, 20, 0.92)",
            titleColor: "#d6dbe3",
            bodyColor: "#d6dbe3",
            borderColor: "rgba(0, 255, 136, 0.2)",
            borderWidth: 1,
            padding: 10,
            titleFont: { family: MONO_STACK, size: 11, weight: "600" },
            bodyFont: { family: MONO_STACK, size: 11 }
          }
        },
        scales: {
          x: {
            ticks: {
              color: "rgba(255,255,255,0.45)",
              font: { family: MONO_STACK, size: 10 },
              maxRotation: 0,
              autoSkip: true,
              maxTicksLimit: 10
            },
            grid: { color: "rgba(255,255,255,0.03)", drawTicks: false },
            border: { color: "rgba(255,255,255,0.06)" }
          },
          yQps: {
            type: "linear",
            position: "left",
            beginAtZero: true,
            ticks: {
              color: "rgba(0, 255, 136, 0.75)",
              font: { family: MONO_STACK, size: 10 }
            },
            grid: { color: "rgba(255,255,255,0.03)", drawTicks: false },
            border: { display: false },
            title: {
              display: true,
              text: "QPS",
              color: "rgba(0, 255, 136, 0.75)",
              font: { family: MONO_STACK, size: 10, weight: "600" }
            }
          },
          yMs: {
            type: "linear",
            position: "right",
            beginAtZero: true,
            ticks: {
              color: "rgba(255, 170, 0, 0.75)",
              font: { family: MONO_STACK, size: 10 }
            },
            grid: { display: false },
            border: { display: false },
            title: {
              display: true,
              text: "ms",
              color: "rgba(255, 170, 0, 0.75)",
              font: { family: MONO_STACK, size: 10, weight: "600" }
            }
          }
        }
      }
    });
  }

  // ----- Model charts ----------------------------------------------
  function renderModels(modelsResp) {
    var list = (modelsResp && modelsResp.models) || [];
    list.sort(function (a, b) {
      return b.requests - a.requests;
    });
    var topN = list.slice(0, 10);

    var labels = [];
    var reqs = [];
    var tokens = [];
    var colors = [];
    var totalReqs = 0;

    for (var i = 0; i < topN.length; i++) {
      labels.push(friendlyModel(topN[i].model));
      reqs.push(topN[i].requests);
      tokens.push(topN[i].tokens);
      colors.push(MODEL_COLORS[i % MODEL_COLORS.length]);
      totalReqs += topN[i].requests;
    }

    // Legend
    var legendEl = $("#models-legend");
    if (legendEl) {
      legendEl.innerHTML = "";
      if (!topN.length) {
        var empty = document.createElement("div");
        empty.className = "muted";
        empty.textContent = "no traffic yet";
        legendEl.appendChild(empty);
      }
      for (var j = 0; j < topN.length; j++) {
        var row = document.createElement("div");
        row.className = "row";
        var sw = document.createElement("span");
        sw.className = "swatch";
        sw.style.background = colors[j];
        var nm = document.createElement("span");
        nm.className = "name";
        nm.textContent = friendlyModel(topN[j].model);
        nm.title = topN[j].model;
        var pct = document.createElement("span");
        pct.className = "pct";
        pct.textContent =
          totalReqs > 0 ? ((topN[j].requests / totalReqs) * 100).toFixed(1) + "%" : "—";
        var cnt = document.createElement("span");
        cnt.className = "count";
        cnt.textContent = formatNumber(topN[j].requests);
        row.appendChild(sw);
        row.appendChild(nm);
        row.appendChild(pct);
        row.appendChild(cnt);
        legendEl.appendChild(row);
      }
    }

    var pieEl = $("#chart-models-pie");
    var tokenEl = $("#chart-models-tokens");

    if (pieEl && window.Chart) {
      var pieCfg = {
        data: {
          labels: labels,
          datasets: [{
            data: reqs,
            backgroundColor: colors,
            borderColor: "rgba(10, 14, 20, 0.95)",
            borderWidth: 3,
            hoverOffset: 6,
            hoverBorderColor: "rgba(10, 14, 20, 1)"
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          devicePixelRatio: window.devicePixelRatio || 1,
          cutout: "68%",
          plugins: {
            legend: { display: false },
            tooltip: {
              enabled: true,
              backgroundColor: "rgba(10, 14, 20, 0.92)",
              borderColor: "rgba(0, 255, 136, 0.2)",
              borderWidth: 1,
              titleFont: { family: MONO_STACK, size: 11, weight: "600" },
              bodyFont: { family: MONO_STACK, size: 11 }
            }
          }
        }
      };
      if (charts.modelsPie) {
        charts.modelsPie.data = pieCfg.data;
        charts.modelsPie.update("none");
      } else {
        charts.modelsPie = new Chart(pieEl.getContext("2d"), Object.assign({ type: "doughnut" }, pieCfg));
      }
    }

    if (tokenEl && window.Chart) {
      var barCfg = {
        data: {
          labels: labels,
          datasets: [
            {
              label: "tokens",
              data: tokens,
              backgroundColor: colors,
              borderWidth: 0,
              borderRadius: 3,
              barThickness: "flex",
              maxBarThickness: 14
            }
          ]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          devicePixelRatio: window.devicePixelRatio || 1,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "rgba(10, 14, 20, 0.92)",
              borderColor: "rgba(0, 255, 136, 0.2)",
              borderWidth: 1,
              titleFont: { family: MONO_STACK, size: 11, weight: "600" },
              bodyFont: { family: MONO_STACK, size: 11 }
            }
          },
          scales: {
            x: {
              beginAtZero: true,
              ticks: {
                color: "rgba(255,255,255,0.45)",
                font: { family: MONO_STACK, size: 10 }
              },
              grid: { color: "rgba(255,255,255,0.03)", drawTicks: false },
              border: { display: false }
            },
            y: {
              ticks: {
                color: "rgba(255,255,255,0.6)",
                font: { family: MONO_STACK, size: 10 }
              },
              grid: { display: false },
              border: { display: false }
            }
          }
        }
      };
      if (charts.modelsTokens) {
        charts.modelsTokens.data = barCfg.data;
        charts.modelsTokens.update("none");
      } else {
        charts.modelsTokens = new Chart(tokenEl.getContext("2d"), Object.assign({ type: "bar" }, barCfg));
      }
    }
  }

  // ----- Requests table --------------------------------------------
  function renderRequests(data) {
    var tbody = $("#req-body");
    if (!tbody) {
      return;
    }
    var items = (data && data.requests) || [];
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="muted">no recent requests</td></tr>';
      return;
    }

    var rows = [];
    for (var i = 0; i < items.length; i++) {
      var r = items[i];
      var tokens = (r.prompt_tokens || 0) + (r.completion_tokens || 0);
      var statusTxt = r.status ? String(r.status) : "—";
      var statusBadge = r.status
        ? '<span class="status-badge ' + statusClass(r.status) + '">' + escapeHtml(statusTxt) + "</span>"
        : '<span class="muted">—</span>';
      rows.push(
        "<tr>" +
          "<td>" + escapeHtml(formatTime(r.ts)) + "</td>" +
          '<td><span class="method-badge">' + escapeHtml(r.method || "-") + "</span></td>" +
          "<td title=\"" + escapeHtml(r.path || "") + "\">" + escapeHtml(r.path || "-") + "</td>" +
          "<td title=\"" + escapeHtml(r.model || "") + "\">" + escapeHtml(friendlyModel(r.model || "-")) + "</td>" +
          '<td class="num">' + statusBadge + "</td>" +
          "<td class=\"num " + latencyClass(r.latency_ms) + "\">" + escapeHtml(formatNumber(r.latency_ms)) + " ms</td>" +
          "<td class=\"num\">" + escapeHtml(formatNumber(tokens)) + "</td>" +
          "</tr>"
      );
    }
    tbody.innerHTML = rows.join("");
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) {
      return "";
    }
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ----- Errors panel ----------------------------------------------
  function renderErrors(data) {
    var byStatus = (data && data.by_status) || {};
    var byType = (data && data.by_type) || {};
    var recent = (data && data.recent) || [];

    var statusEl = $("#err-by-status");
    var typeEl = $("#err-by-type");
    var listEl = $("#err-list");

    if (statusEl) {
      statusEl.innerHTML = "";
      var statusKeys = Object.keys(byStatus);
      if (!statusKeys.length) {
        statusEl.innerHTML = '<span class="muted">none</span>';
      }
      for (var i = 0; i < statusKeys.length; i++) {
        var k = statusKeys[i];
        var chip = document.createElement("span");
        var code = parseInt(k, 10);
        chip.className = "err-chip" + (code >= 500 ? "" : " amber");
        chip.textContent = k + " × " + byStatus[k];
        statusEl.appendChild(chip);
      }
    }

    if (typeEl) {
      typeEl.innerHTML = "";
      var typeKeys = Object.keys(byType);
      if (!typeKeys.length) {
        typeEl.innerHTML = '<span class="muted">none</span>';
      }
      for (var j = 0; j < typeKeys.length; j++) {
        var tk = typeKeys[j];
        var tchip = document.createElement("span");
        tchip.className = "err-chip";
        tchip.textContent = tk + " × " + byType[tk];
        typeEl.appendChild(tchip);
      }
    }

    if (listEl) {
      listEl.innerHTML = "";
      if (!recent.length) {
        var li = document.createElement("li");
        li.textContent = "no recent errors";
        li.style.borderLeftColor = "#222a3b";
        li.style.background = "transparent";
        li.style.color = "#5a6278";
        li.style.gridTemplateColumns = "1fr";
        listEl.appendChild(li);
      }
      var max = Math.min(10, recent.length);
      for (var k2 = 0; k2 < max; k2++) {
        var e = recent[k2];
        var eli = document.createElement("li");
        eli.innerHTML =
          '<span class="err-ts">' + escapeHtml(formatTime(e.ts)) + "</span>" +
          '<span class="err-type">' + escapeHtml(e.error_type || "-") + "</span>" +
          '<span class="err-status">' + escapeHtml(String(e.status || "—")) + "</span>" +
          '<span class="err-msg">' + escapeHtml(e.error_message || e.path || "") + "</span>";
        listEl.appendChild(eli);
      }
    }
  }

  // ----- Request sources -------------------------------------------
  function renderSources(data) {
    var el = $("#sources-list");
    if (!el) {
      return;
    }
    var items = (data && data.sources) || [];
    el.innerHTML = "";
    if (!items.length) {
      var empty = document.createElement("li");
      empty.className = "muted";
      empty.textContent = "no traffic yet";
      el.appendChild(empty);
      return;
    }
    var maxCount = 0;
    for (var i = 0; i < items.length; i++) {
      if (items[i].count > maxCount) {
        maxCount = items[i].count;
      }
    }
    for (var j = 0; j < items.length; j++) {
      var it = items[j];
      var li = document.createElement("li");
      var ip = document.createElement("span");
      ip.className = "src-ip";
      ip.textContent = it.ip || "—";
      var cnt = document.createElement("span");
      cnt.className = "src-count";
      cnt.textContent = formatNumber(it.count);
      var pct = document.createElement("span");
      pct.className = "src-pct";
      pct.textContent = (it.percentage || 0).toFixed(1) + "%";
      var bar = document.createElement("span");
      bar.className = "src-bar";
      var pctWidth = maxCount > 0 ? (it.count / maxCount) * 100 : 0;
      bar.style.width = pctWidth.toFixed(1) + "%";
      li.appendChild(ip);
      li.appendChild(cnt);
      li.appendChild(pct);
      li.appendChild(bar);
      el.appendChild(li);
    }
  }

  // ----- System health --------------------------------------------
  function classForThresholds(value, warnAt, badAt) {
    if (value === null || value === undefined || isNaN(value)) {
      return "";
    }
    if (value >= badAt) return "h-bad";
    if (value >= warnAt) return "h-warn";
    return "h-ok";
  }

  function formatAgeSeconds(iso) {
    if (!iso) return "never";
    var ts = Date.parse(iso);
    if (!ts) return "—";
    var age = Math.max(0, Math.round((Date.now() - ts) / 1000));
    if (age < 90) return age + "s ago";
    if (age < 3600) return Math.floor(age / 60) + "m ago";
    return Math.floor(age / 3600) + "h ago";
  }

  function addHealthRow(list, label, value, tone, sub) {
    var li = document.createElement("li");
    if (tone) li.className = tone;
    var lbl = document.createElement("span");
    lbl.className = "h-label";
    lbl.textContent = label;
    var val = document.createElement("span");
    val.className = "h-value";
    val.textContent = value;
    if (sub) {
      var s = document.createElement("span");
      s.className = "h-sub";
      s.textContent = sub;
      val.appendChild(s);
    }
    li.appendChild(lbl);
    li.appendChild(val);
    list.appendChild(li);
  }

  function renderHealth(data) {
    var el = $("#health-list");
    if (!el) return;
    el.innerHTML = "";
    if (!data) {
      var li = document.createElement("li");
      li.className = "muted";
      li.textContent = "no health data";
      el.appendChild(li);
      return;
    }

    // Active connections
    var active = safeGet(data, "active_connections", 0);
    addHealthRow(
      el, "Active Connections", String(active),
      classForThresholds(active, 50, 100)
    );

    // Upstream pool
    var up = data.upstream_pool || {};
    var upVal =
      safeGet(up, "active", 0) + " active / " +
      safeGet(up, "idle", 0) + " idle / " +
      safeGet(up, "total", 0) + " total";
    addHealthRow(el, "Upstream Pool", upVal, "h-ok");

    // Open FDs
    var fd = data.open_fds || {};
    var cur = safeGet(fd, "current", null);
    var lim = safeGet(fd, "limit", null);
    var fdVal = (cur === null ? "—" : cur) + " / " + (lim === null ? "∞" : lim);
    var fdTone = "h-ok";
    if (cur !== null && lim !== null && lim > 0) {
      var pct = cur / lim;
      if (pct >= 0.9) fdTone = "h-bad";
      else if (pct >= 0.75) fdTone = "h-warn";
    }
    addHealthRow(el, "Open FDs", fdVal, fdTone);

    // Auth
    var auth = data.auth || {};
    var mode = safeGet(auth, "mode", "-");
    var status = safeGet(auth, "status", "unknown");
    var expires = safeGet(auth, "expires_at", null);
    var authTone = "h-ok";
    if (status === "expired") authTone = "h-bad";
    else if (status === "expiring_soon") authTone = "h-warn";
    else if (status === "unknown") authTone = "";
    var authVal = mode + " · " + status;
    var authSub = expires ? "expires " + expires : null;
    addHealthRow(el, "Auth", authVal, authTone, authSub);

    // Consecutive errors
    var ce = safeGet(data, "consecutive_errors", 0);
    var ceTone = ce === 0 ? "h-ok" : ce >= 5 ? "h-bad" : "h-warn";
    addHealthRow(el, "Consecutive Errors", String(ce), ceTone);

    // Event loop lag
    var lag = safeGet(data, "event_loop_lag_ms", 0);
    var lagTone = lag < 10 ? "h-ok" : lag < 50 ? "h-warn" : "h-bad";
    addHealthRow(el, "Event Loop Lag", lag.toFixed(1) + " ms", lagTone);

    // Upstream reachability
    var ups = data.upstream || {};
    var reach = safeGet(ups, "reachable", null);
    var lat = safeGet(ups, "latency_ms", null);
    var lastCheck = safeGet(ups, "last_check", null);
    var lastSuccess = safeGet(ups, "last_success", null);
    var upsTone, upsVal, upsSub;
    if (reach === null) {
      upsTone = "";
      upsVal = "probing…";
      upsSub = null;
    } else if (reach) {
      upsTone = "h-ok";
      upsVal = "reachable (" + (lat !== null ? Math.round(lat) + " ms" : "—") + ")";
      upsSub = lastCheck ? "last check " + formatAgeSeconds(lastCheck) : null;
    } else {
      upsTone = "h-bad";
      upsVal = "unreachable";
      upsSub = lastSuccess
        ? "last seen " + formatAgeSeconds(lastSuccess)
        : "never seen";
    }
    addHealthRow(el, "Upstream", upsVal, upsTone, upsSub);
  }

  // ----- Memory time series ----------------------------------------
  function renderMemory(data) {
    var el = $("#chart-memory");
    if (!el || !window.Chart) {
      return;
    }
    var labelsTs = (data && data.labels) || [];
    var values = (data && data.memory_mb) || [];
    var labels = labelsForWindow(labelsTs, (data && data.window) || currentWindow);

    var dataset = {
      type: "line",
      label: "RSS (MB)",
      data: values,
      borderColor: "#aa66ff",
      backgroundColor: function (ctx) {
        return makeLineGradient(
          ctx.chart.ctx, ctx.chart.chartArea, "#aa66ff", 0.22, 0
        );
      },
      fill: true,
      borderWidth: 2.5,
      tension: 0.3,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: "#aa66ff",
      pointHoverBorderColor: "#0e131c",
      pointHoverBorderWidth: 2,
      spanGaps: true
    };

    if (charts.memory) {
      charts.memory.data.labels = labels;
      charts.memory.data.datasets = [dataset];
      charts.memory.update("none");
      return;
    }

    charts.memory = new Chart(el.getContext("2d"), {
      type: "line",
      data: { labels: labels, datasets: [dataset] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        devicePixelRatio: window.devicePixelRatio || 1,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(10, 14, 20, 0.92)",
            borderColor: "rgba(170, 102, 255, 0.3)",
            borderWidth: 1,
            titleFont: { family: MONO_STACK, size: 11, weight: "600" },
            bodyFont: { family: MONO_STACK, size: 11 },
            callbacks: {
              label: function (ctx) {
                var v = ctx.parsed.y;
                return v === null || v === undefined ? "—" : v.toFixed(1) + " MB";
              }
            }
          }
        },
        scales: {
          x: {
            ticks: {
              color: "rgba(255,255,255,0.45)",
              font: { family: MONO_STACK, size: 10 },
              maxRotation: 0,
              autoSkip: true,
              maxTicksLimit: 8
            },
            grid: { color: "rgba(255,255,255,0.03)", drawTicks: false },
            border: { color: "rgba(255,255,255,0.06)" }
          },
          y: {
            beginAtZero: false,
            ticks: {
              color: "rgba(170, 102, 255, 0.75)",
              font: { family: MONO_STACK, size: 10 },
              callback: function (v) {
                return v + " MB";
              }
            },
            grid: { color: "rgba(255,255,255,0.03)", drawTicks: false },
            border: { display: false }
          }
        }
      }
    });
  }

  // ----- System / status bar ---------------------------------------
  function renderSystem(system) {
    if (!system) {
      return;
    }
    // If the server's retain_days reshapes the window set, rebuild buttons.
    var retain = safeGet(system, "retain_days", null);
    var nextOptions = buildWindowOptions(retain);
    if (nextOptions.join("|") !== windowOptions.join("|")) {
      windowOptions = nextOptions;
      renderWindowButtons(windowOptions);
    }
    var v = $("#pill-version");
    var r = $("#pill-region");
    var a = $("#pill-auth");
    var u = $("#pill-uptime");
    var m = $("#pill-memory");

    if (v) {
      v.textContent = "v" + safeGet(system, "version", "—");
    }
    if (r) {
      r.textContent = "region: " + safeGet(system, "region", "—");
    }
    if (a) {
      a.textContent = "auth: " + safeGet(system, "auth_mode", "—");
    }
    if (u) {
      u.textContent = "uptime " + formatUptime(safeGet(system, "uptime_seconds", 0));
    }
    if (m) {
      var mem = safeGet(system, "memory_rss_mb", null);
      m.textContent = "RSS " + (mem !== null ? mem + " MB" : "—");
    }
  }

  function updateClock() {
    var el = $("#last-updated");
    if (!el) {
      return;
    }
    var now = new Date();
    var stale = lastSuccessfulTick && now.getTime() - lastSuccessfulTick > POLL_MS * 3;
    var livePill = $("#pill-live");
    if (livePill) {
      if (stale) {
        livePill.classList.add("stale");
        livePill.textContent = "STALE";
      } else {
        livePill.classList.remove("stale");
        livePill.textContent = "LIVE";
      }
    }
    var stamp =
      lastSuccessfulTick > 0 ? new Date(lastSuccessfulTick) : now;
    el.textContent = "clock: " + formatClock(now) + "  —  last update: " + formatClock(stamp);
  }

  // ----- Polling loop ----------------------------------------------
  function tick() {
    var urls = [
      fetchJSON("/api/metrics/overview"),
      fetchJSON("/api/metrics/traffic?window=" + encodeURIComponent(currentWindow)),
      fetchJSON("/api/metrics/models"),
      fetchJSON("/api/metrics/requests?limit=50&filter=" + encodeURIComponent(currentFilter)),
      fetchJSON("/api/metrics/errors"),
      fetchJSON("/api/metrics/system"),
      fetchJSON("/api/metrics/sources?limit=10"),
      fetchJSON("/api/metrics/memory?window=" + encodeURIComponent(currentWindow)),
      fetchJSON("/api/metrics/health")
    ];

    Promise.all(urls)
      .then(function (results) {
        var overview = results[0];
        var traffic = results[1];
        var models = results[2];
        var requests = results[3];
        var errors = results[4];
        var system = results[5];
        var sources = results[6];
        var memory = results[7];
        var health = results[8];

        renderGauges(overview, traffic);
        renderMiniTiles(overview);
        renderTraffic(traffic);
        renderModels(models);
        renderRequests(requests);
        renderErrors(errors);
        renderSystem(system);
        renderSources(sources);
        renderMemory(memory);
        renderHealth(health);

        lastSuccessfulTick = Date.now();
      })
      .catch(function (err) {
        // Non-fatal: just leave whatever was last drawn, mark UI stale.
        if (window.console && console.warn) {
          console.warn("poll failed:", err);
        }
      });
  }

  // ----- Window switch ---------------------------------------------
  // Generate the set of window options the backend can serve given the
  // configured retention horizon. 1h/6h/24h are always available;
  // multi-day windows are additive as retention allows.
  function buildWindowOptions(retainDays) {
    var opts = ["1h", "6h", "24h"];
    var n = parseInt(retainDays, 10);
    if (!isFinite(n) || n <= 0) {
      return opts;
    }
    var candidates = [3, 7, 14, 30];
    for (var i = 0; i < candidates.length; i++) {
      if (n >= candidates[i]) {
        opts.push(candidates[i] + "d");
      }
    }
    return opts;
  }

  function renderWindowButtons(options) {
    var host = $("#window-switch");
    if (!host) {
      return;
    }
    if (options.indexOf(currentWindow) === -1) {
      // The previously selected window is no longer offered — fall back.
      currentWindow = options[0];
    }
    host.innerHTML = "";
    for (var i = 0; i < options.length; i++) {
      var w = options[i];
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "win-btn" + (w === currentWindow ? " active" : "");
      btn.setAttribute("data-window", w);
      btn.textContent = w.toUpperCase();
      host.appendChild(btn);
    }
    wireWindowButtons();
  }

  function wireWindowButtons() {
    var winButtons = $$("#window-switch .win-btn");
    for (var i = 0; i < winButtons.length; i++) {
      (function (btn) {
        btn.addEventListener("click", function () {
          var w = btn.getAttribute("data-window");
          if (!w || w === currentWindow) {
            return;
          }
          currentWindow = w;
          for (var k = 0; k < winButtons.length; k++) {
            winButtons[k].classList.toggle("active", winButtons[k] === btn);
          }
          tick();
        });
      })(winButtons[i]);
    }
  }

  // ----- Event wiring ----------------------------------------------
  function bindControls() {
    var filButtons = $$("#filter-switch .fil-btn");
    for (var j = 0; j < filButtons.length; j++) {
      (function (btn) {
        btn.addEventListener("click", function () {
          var f = btn.getAttribute("data-filter");
          if (!f || f === currentFilter) {
            return;
          }
          currentFilter = f;
          for (var k = 0; k < filButtons.length; k++) {
            filButtons[k].classList.toggle("active", filButtons[k] === btn);
          }
          tick();
        });
      })(filButtons[j]);
    }

    // Redraw gauges on resize so they stay crisp on DPR changes.
    // Debounce so dragging a window edge doesn't hammer the API.
    var resizeTimer = null;
    window.addEventListener("resize", function () {
      if (resizeTimer) {
        clearTimeout(resizeTimer);
      }
      resizeTimer = setTimeout(tick, 150);
    });
  }

  // ----- Bootstrap -------------------------------------------------
  function boot() {
    // Render the safe default 1h/6h/24h set immediately so the header
    // isn't empty while we wait for the first poll. renderSystem() will
    // refresh the set once the real retain_days is known.
    renderWindowButtons(windowOptions);
    bindControls();
    tick();
    setInterval(tick, POLL_MS);
    setInterval(updateClock, 1000);
    updateClock();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
