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
  var charts = {
    traffic: null,
    modelsPie: null,
    modelsTokens: null
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

    // Handle HiDPI once per resize
    if (canvas.width !== canvas.clientWidth * dpr || canvas.height !== canvas.clientHeight * dpr) {
      canvas.width = Math.max(1, canvas.clientWidth) * dpr;
      canvas.height = Math.max(1, canvas.clientHeight) * dpr;
    }

    var w = canvas.width;
    var h = canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var cx = w / 2;
    var cy = h / 2 + h * 0.08;
    var radius = Math.min(w, h) * 0.38;
    var thickness = Math.max(6, radius * 0.14);

    var value = typeof opts.value === "number" && isFinite(opts.value) ? opts.value : 0;
    var max = opts.max > 0 ? opts.max : 1;
    var ratio = Math.max(0, Math.min(1, value / max));

    // Arc spans 225° → 315° (bottom-left to bottom-right going over the top)
    var startAngle = Math.PI * 0.75;
    var endAngle = Math.PI * 2.25;

    // Track
    ctx.lineWidth = thickness;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle, false);
    ctx.stroke();

    // Gradient fill arc
    var grad = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
    grad.addColorStop(0, opts.color || "#00ff88");
    grad.addColorStop(1, opts.color2 || "#00aaff");
    ctx.strokeStyle = grad;
    ctx.shadowColor = opts.color || "#00ff88";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + (endAngle - startAngle) * ratio, false);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Tick marks
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    var ticks = 10;
    for (var i = 0; i <= ticks; i++) {
      var t = i / ticks;
      var ang = startAngle + (endAngle - startAngle) * t;
      var inner = radius + thickness / 2 + 2;
      var outer = inner + (i % 5 === 0 ? 8 : 4);
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(ang) * inner, cy + Math.sin(ang) * inner);
      ctx.lineTo(cx + Math.cos(ang) * outer, cy + Math.sin(ang) * outer);
      ctx.stroke();
    }

    // Center value
    ctx.fillStyle = opts.color || "#00ff88";
    ctx.font = "bold " + Math.floor(radius * 0.55) + "px " + (opts.mono || "SFMono-Regular, Consolas, monospace");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    var display = opts.format ? opts.format(value) : formatNumber(value);
    ctx.fillText(display, cx, cy - radius * 0.1);

    // Unit suffix
    if (opts.unit) {
      ctx.fillStyle = "#8a93a6";
      ctx.font = Math.floor(radius * 0.22) + "px " + (opts.mono || "SFMono-Regular, Consolas, monospace");
      ctx.fillText(opts.unit, cx, cy + radius * 0.25);
    }

    // Label
    ctx.fillStyle = "#8a93a6";
    ctx.font = "bold " + Math.floor(radius * 0.18) + "px " + (opts.mono || "SFMono-Regular, Consolas, monospace");
    ctx.fillText((opts.label || "").toUpperCase(), cx, cy + radius * 0.62);
  }

  function renderGauges(overview, traffic) {
    // QPS: average over the last 60 seconds (or visible window's last point)
    var qps = 0;
    if (traffic && traffic.qps && traffic.qps.length) {
      qps = traffic.qps[traffic.qps.length - 1] || 0;
    }
    var p95 = 0;
    if (traffic && traffic.p95 && traffic.p95.length) {
      // Take the max p95 over the last 5 minutes so the gauge feels reactive
      var look = traffic.p95.slice(-5);
      for (var i = 0; i < look.length; i++) {
        if (look[i] > p95) {
          p95 = look[i];
        }
      }
    }
    var successRate = safeGet(overview, "success_rate", 0);
    var totalTokens =
      safeGet(overview, "prompt_tokens", 0) + safeGet(overview, "completion_tokens", 0);
    var uptimeSec = safeGet(overview, "uptime_seconds", 1) || 1;
    var tokensPerHour = totalTokens / (uptimeSec / 3600);

    var canvasQps = $('canvas[data-key="qps"]');
    var canvasSuccess = $('canvas[data-key="success"]');
    var canvasLatency = $('canvas[data-key="latency"]');
    var canvasTokens = $('canvas[data-key="tokens"]');

    if (canvasQps) {
      drawGauge(canvasQps, {
        value: qps,
        max: Math.max(5, qps * 1.4),
        label: "QPS",
        color: "#00ff88",
        color2: "#00aaff",
        format: function (v) {
          return v.toFixed(2);
        }
      });
    }

    if (canvasSuccess) {
      drawGauge(canvasSuccess, {
        value: successRate,
        max: 100,
        label: "SUCCESS %",
        color: successRate >= 99 ? "#00ff88" : successRate >= 90 ? "#ffaa00" : "#ff4444",
        color2: successRate >= 99 ? "#00aaff" : successRate >= 90 ? "#ffcc44" : "#ff8844",
        format: function (v) {
          return v.toFixed(1);
        },
        unit: "%"
      });
    }

    if (canvasLatency) {
      var lColor = p95 < 500 ? "#00ff88" : p95 < 2000 ? "#ffaa00" : "#ff4444";
      drawGauge(canvasLatency, {
        value: p95,
        max: Math.max(2000, p95 * 1.3),
        label: "P95 LATENCY",
        color: lColor,
        color2: "#00aaff",
        format: function (v) {
          return Math.round(v);
        },
        unit: "ms"
      });
    }

    if (canvasTokens) {
      drawGauge(canvasTokens, {
        value: tokensPerHour,
        max: Math.max(1000, tokensPerHour * 1.3),
        label: "TOKENS / H",
        color: "#aa66ff",
        color2: "#00aaff",
        format: function (v) {
          return formatNumber(v);
        }
      });
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
    var labels = labelsToHHMM(labelsTs);

    var datasets = [
      {
        type: "bar",
        label: "QPS",
        data: qps,
        backgroundColor: "rgba(0, 255, 136, 0.35)",
        borderColor: "#00ff88",
        borderWidth: 1,
        yAxisID: "yQps",
        order: 3
      },
      {
        type: "line",
        label: "p50",
        data: p50,
        borderColor: "#00aaff",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        tension: 0.3,
        pointRadius: 0,
        yAxisID: "yMs",
        order: 2
      },
      {
        type: "line",
        label: "p95",
        data: p95,
        borderColor: "#ffaa00",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        tension: 0.3,
        pointRadius: 0,
        yAxisID: "yMs",
        order: 1
      },
      {
        type: "line",
        label: "p99",
        data: p99,
        borderColor: "#ff4444",
        backgroundColor: "transparent",
        borderWidth: 1.2,
        borderDash: [4, 3],
        tension: 0.3,
        pointRadius: 0,
        yAxisID: "yMs",
        order: 0
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
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: {
              color: "#8a93a6",
              font: { family: "SFMono-Regular, Consolas, monospace", size: 11 }
            }
          },
          tooltip: {
            backgroundColor: "#0e131c",
            titleColor: "#d6dbe3",
            bodyColor: "#d6dbe3",
            borderColor: "#222a3b",
            borderWidth: 1
          }
        },
        scales: {
          x: {
            ticks: {
              color: "#5a6278",
              maxRotation: 0,
              autoSkip: true,
              maxTicksLimit: 10
            },
            grid: { color: "rgba(255,255,255,0.04)" }
          },
          yQps: {
            type: "linear",
            position: "left",
            beginAtZero: true,
            ticks: { color: "#00ff88" },
            grid: { color: "rgba(255,255,255,0.04)" },
            title: {
              display: true,
              text: "QPS",
              color: "#00ff88",
              font: { family: "SFMono-Regular, Consolas, monospace", size: 10 }
            }
          },
          yMs: {
            type: "linear",
            position: "right",
            beginAtZero: true,
            ticks: { color: "#ffaa00" },
            grid: { display: false },
            title: {
              display: true,
              text: "ms",
              color: "#ffaa00",
              font: { family: "SFMono-Regular, Consolas, monospace", size: 10 }
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
      labels.push(topN[i].model);
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
        nm.textContent = topN[j].model;
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
        data: { labels: labels, datasets: [{ data: reqs, backgroundColor: colors, borderColor: "#0e131c", borderWidth: 2 }] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          cutout: "60%",
          plugins: { legend: { display: false }, tooltip: { enabled: true } }
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
              borderWidth: 0
            }
          ]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              beginAtZero: true,
              ticks: { color: "#5a6278" },
              grid: { color: "rgba(255,255,255,0.04)" }
            },
            y: {
              ticks: { color: "#8a93a6", font: { size: 10 } },
              grid: { display: false }
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
      rows.push(
        "<tr>" +
          "<td>" + escapeHtml(formatTime(r.ts)) + "</td>" +
          "<td>" + escapeHtml(r.method || "-") + "</td>" +
          "<td title=\"" + escapeHtml(r.path || "") + "\">" + escapeHtml(r.path || "-") + "</td>" +
          "<td title=\"" + escapeHtml(r.model || "") + "\">" + escapeHtml(r.model || "-") + "</td>" +
          "<td class=\"num " + statusClass(r.status) + "\">" + escapeHtml(String(r.status || "—")) + "</td>" +
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

  // ----- System / status bar ---------------------------------------
  function renderSystem(system) {
    if (!system) {
      return;
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
      fetchJSON("/api/metrics/system")
    ];

    Promise.all(urls)
      .then(function (results) {
        var overview = results[0];
        var traffic = results[1];
        var models = results[2];
        var requests = results[3];
        var errors = results[4];
        var system = results[5];

        renderGauges(overview, traffic);
        renderTraffic(traffic);
        renderModels(models);
        renderRequests(requests);
        renderErrors(errors);
        renderSystem(system);

        lastSuccessfulTick = Date.now();
      })
      .catch(function (err) {
        // Non-fatal: just leave whatever was last drawn, mark UI stale.
        if (window.console && console.warn) {
          console.warn("poll failed:", err);
        }
      });
  }

  // ----- Event wiring ----------------------------------------------
  function bindControls() {
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
    window.addEventListener("resize", function () {
      // Trigger a cheap redraw with last overview/traffic — easiest: poll now.
      tick();
    });
  }

  // ----- Bootstrap -------------------------------------------------
  function boot() {
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
