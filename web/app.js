(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let sessionId = null;
  let info = null;
  let currentStep = 1;
  let adviceLoaded = false;

  const STEP_TITLES = {
    1: "Step 1 of 5 · Select or Upload a Specimen",
    2: "Step 2 of 5 · Visual Verdict & Grad-CAM Attribution",
    3: "Step 3 of 5 · Refine Diagnosis with Field Observations",
    4: "Step 4 of 5 · Sourced Extension Treatment Protocol",
    5: "Step 5 of 5 · Interactive AI Plant Pathologist Chat",
  };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");

  async function api(path, options) {
    const res = await fetch(path, options);
    let data = {};
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    return data;
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  function showError(msg) { const e = $("start-error"); if (e) { e.textContent = msg; e.hidden = !msg; } }
  function setLoading(on) { $("loading").hidden = !on; $("sample-go").disabled = on || !$("sample").value; }

  function goToStep(stepNum) {
    currentStep = Math.max(1, Math.min(5, Number(stepNum)));
    document.querySelectorAll(".step-view").forEach((el, idx) => {
      el.classList.toggle("active", idx + 1 === currentStep);
    });
    document.querySelectorAll(".step-tab").forEach((tab) => {
      tab.classList.toggle("active", Number(tab.getAttribute("data-target")) === currentStep);
    });
    $("step-indicator").textContent = STEP_TITLES[currentStep] || "";
    $("prev-step-btn").disabled = currentStep <= 1;
    $("next-step-btn").disabled = !sessionId || currentStep >= 5;

    if (currentStep === 4 && sessionId && !adviceLoaded) {
      loadAdvice();
    }
  }

  function renderRanking(ranking) {
    const html = ranking.map((r) => {
      const pct = Math.round(r.prob * 100);
      return `<li><div class="bar-label"><span>${esc(r.name)}</span><strong>${pct}%</strong></div>
        <div class="bar" role="img" aria-label="${esc(r.name)} ${pct} percent"><span style="width:${Math.max(pct, 1)}%"></span></div></li>`;
    }).join("");
    $("ranking").innerHTML = html;
    if ($("ranking-step3")) $("ranking-step3").innerHTML = html;
  }

  function renderQuestions() {
    $("questions-form").innerHTML = info.questions.map((q) => `
      <fieldset><legend>${esc(q.text)}</legend><div class="chips">
        ${q.options.map((o) => `<label class="chip"><input type="radio" name="${esc(q.id)}" value="${esc(o.value)}"><span>${esc(o.label)}</span></label>`).join("")}
      </div></fieldset>`).join("");
  }

  function renderResult(d) {
    sessionId = d.session_id;
    adviceLoaded = false;
    $("img-original").src = d.original;
    const hasOverlay = !!d.overlay;
    $("img-overlay").hidden = !hasOverlay;
    $("no-overlay").hidden = hasOverlay;
    if (hasOverlay) $("img-overlay").src = d.overlay;
    const labels = { confident: "Confident Diagnosis", uncertain: "Uncertain · Needs Review", refuse: "Cannot Diagnose" };
    const badge = $("tier-badge");
    badge.textContent = labels[d.tier] || d.tier;
    badge.className = `badge ${d.tier}`;
    $("message").innerHTML = esc(d.message).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const sb = $("sample-banner");
    if (d.sample) {
      sb.hidden = false;
      sb.className = "sample-banner " + (d.sample.matches ? "ok" : "miss");
      sb.innerHTML = d.sample.matches
        ? `✓ Verified Dataset Specimen · Ground Truth: <strong>${esc(d.sample.true_label)}</strong> · Top-1 prediction matches.`
        : `Dataset Specimen · Ground Truth: <strong>${esc(d.sample.true_label)}</strong> · Top-1 prediction does <strong>not</strong> match.`;
    } else { sb.hidden = true; }
    $("notes").innerHTML = (d.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join("");
    renderRanking(d.ranking);
    $("evidence").hidden = true;
    $("advice").innerHTML = `<p class="hint">Click <strong>Refresh Sourced Protocol</strong> or open this step to load extension management guidelines.</p>`;
    $("chat-log").innerHTML = `<div class="bubble bot"><p>Specimen analyzed! Ask me any follow-up questions about symptoms, spread, or treatment below.</p></div>`;
    renderQuestions();

    const usable = d.can_ask;
    ["tab-2", "tab-3", "tab-4", "tab-5"].forEach((tid) => {
      const t = $(tid);
      if (t) t.disabled = false;
    });
    $("reset-scan-btn").hidden = false;
    $("chat-form").querySelector("button").disabled = !usable;
    $("chat-input").disabled = !usable;
    $("advice-btn").disabled = !usable;
    if (!usable) {
      $("advice").innerHTML = `<p class="disclaimer">I won't give treatment advice without a reliable diagnosis. Try a clearer, well-lit close-up of the affected leaf or fruit.</p>`;
      adviceLoaded = true;
    }

    // Automatically flip to Step Page 2 (Diagnosis & Grad-CAM) with no scrolling
    goToStep(2);
  }

  async function diagnose(formData) {
    showError("");
    setLoading(true);
    try { renderResult(await api("/api/diagnose", { method: "POST", body: formData })); }
    catch (e) { showError(e.message); }
    finally { setLoading(false); }
  }

  function runSampleById(sampleId) {
    if (!sampleId) return;
    const sel = $("sample");
    sel.value = sampleId;
    $("sample-go").disabled = !sel.value;
    const fd = new FormData();
    fd.append("sample", sampleId);
    diagnose(fd);
  }

  function bindDropzone() {
    const dz = $("dropzone"), input = $("file");
    const getFormData = (file) => {
      const fd = new FormData();
      fd.append("file", file);
      const c = $("user-crop"), o = $("user-organ");
      if (c && c.value) fd.append("crop", c.value);
      if (o && o.value) fd.append("organ", o.value);
      return fd;
    };
    input.addEventListener("change", () => {
      if (!input.files[0]) return;
      diagnose(getFormData(input.files[0]));
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files[0];
      if (f) { diagnose(getFormData(f)); }
    });
  }

  function addBubble(kind, html) {
    const b = document.createElement("div");
    b.className = `bubble ${kind}`;
    b.innerHTML = html;
    $("chat-log").appendChild(b);
    $("chat-log").scrollTop = $("chat-log").scrollHeight;
  }

  const sourcesHtml = (sources) => !sources || !sources.length ? "" :
    `<div class="sources"><strong>Verified Academic &amp; Extension Sources</strong><ol>${sources.map((s) =>
      `<li><a href="${esc(s.source_url)}" target="_blank" rel="noopener noreferrer">${esc(s.source_title)}</a> — ${esc(s.publisher_type)}</li>`).join("")}</ol></div>`;

  async function loadAdvice() {
    if (!sessionId) return;
    try {
      $("advice").innerHTML = `<p class="hint">Loading sourced agronomic protocol…</p>`;
      const r = await post("/api/advice", { session_id: sessionId });
      $("advice").innerHTML = md(r.text) + sourcesHtml(r.sources) + `<p class="hint" style="margin-top:8px;">${esc(r.kb_note)}</p>`;
      adviceLoaded = true;
    } catch (e) {
      $("advice").innerHTML = `<p class="error">${esc(e.message)}</p>`;
    }
  }

  async function submitQuestion(qText) {
    const q = qText.trim();
    if (!q || !sessionId) return;
    $("chat-input").value = "";
    addBubble("user", esc(q));
    const typing = document.createElement("div");
    typing.className = "bubble bot typing";
    typing.innerHTML = 'Consulting knowledge base…';
    $("chat-log").appendChild(typing);
    $("chat-log").scrollTop = $("chat-log").scrollHeight;
    $("chat-input").disabled = true;
    try {
      const r = await post("/api/ask", { session_id: sessionId, question: q });
      typing.remove();
      addBubble("bot", md(r.text) + sourcesHtml(r.sources));
    } catch (e) { typing.remove(); addBubble("bot", `<p>${esc(e.message)}</p>`); }
    finally { $("chat-input").disabled = false; $("chat-input").focus(); }
  }

  async function init() {
    bindDropzone();
    try { info = await api("/api/info"); }
    catch (e) { showError("Could not reach the local server. Is `python server.py` running?"); return; }
    $("engine").textContent = info.engine_label;
    const sel = $("sample");
    info.samples.forEach((s) => sel.add(new Option(s.name, s.id)));
    sel.addEventListener("change", () => { $("sample-go").disabled = !sel.value; });
    $("sample-go").addEventListener("click", () => { runSampleById(sel.value); });

    document.querySelectorAll("[data-sample]").forEach((btn) => {
      btn.addEventListener("click", () => { runSampleById(btn.getAttribute("data-sample")); });
    });

    document.querySelectorAll(".step-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        if (!tab.disabled) goToStep(Number(tab.getAttribute("data-target")));
      });
    });

    $("prev-step-btn").addEventListener("click", () => { if (currentStep > 1) goToStep(currentStep - 1); });
    $("next-step-btn").addEventListener("click", () => { if (sessionId && currentStep < 5) goToStep(currentStep + 1); });
    $("reset-scan-btn").addEventListener("click", () => { goToStep(1); });

    document.querySelectorAll(".prompt-pill").forEach((pill) => {
      pill.addEventListener("click", () => { submitQuestion(pill.getAttribute("data-q") || ""); });
    });

    $("update").addEventListener("click", async () => {
      const answers = {};
      info.questions.forEach((q) => { const c = document.querySelector(`input[name="${q.id}"]:checked`); if (c) answers[q.id] = c.value; });
      if (!Object.keys(answers).length) return;
      try {
        const r = await post("/api/answers", { session_id: sessionId, answers });
        renderRanking(r.ranking);
        adviceLoaded = false;
        $("evidence").hidden = !r.evidence_log.length;
        $("evidence-list").innerHTML = r.evidence_log.map((e) => `<li><strong>${esc(e.question)}</strong> → ${esc(e.answer)}: ${esc(e.effect)}</li>`).join("");
      } catch (e) { alert(e.message); }
    });

    $("advice-btn").addEventListener("click", loadAdvice);

    $("chat-form").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      await submitQuestion($("chat-input").value);
    });

    const params = new URLSearchParams(window.location.search);
    const initialSample = params.get("sample");
    if (initialSample) {
      runSampleById(initialSample);
    }
  }
  init();
})();
