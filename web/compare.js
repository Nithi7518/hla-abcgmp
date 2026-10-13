(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v == null || isNaN(v)) ? "—" : `${(Number(v) * 100).toFixed(2)}%`;
  const fmtM = (n) => n ? `${(n / 1e6).toFixed(2)}M` : "—";

  let benchData = null;
  let currentCStep = 1;

  const COMPARE_TITLES = {
    1: "Comparison Page 1 of 5 · Model Leaderboard & KPIs",
    2: "Comparison Page 2 of 5 · Architectural & Methodological Contrast Matrix",
    3: "Comparison Page 3 of 5 · Head-to-Head 23-Class Comparator (metrics.json)",
    4: "Comparison Page 4 of 5 · Authentic Confusion Matrix & Convergence Curves",
    5: "Comparison Page 5 of 5 · Epoch Training Logs, Cross-Crop & Calibration",
  };

  function goToCStep(num) {
    currentCStep = Math.max(1, Math.min(5, Number(num)));
    document.querySelectorAll(".step-view").forEach((el, idx) => {
      el.classList.toggle("active", idx + 1 === currentCStep);
    });
    document.querySelectorAll("[data-cstep]").forEach((tab) => {
      tab.classList.toggle("active", Number(tab.getAttribute("data-cstep")) === currentCStep);
    });
    $("compare-indicator").textContent = COMPARE_TITLES[currentCStep] || "";
    $("c-prev-btn").disabled = currentCStep <= 1;
    $("c-next-btn").disabled = currentCStep >= 5;
  }

  function renderLeaderboard(runs, baselines) {
    const all = [...runs, ...(baselines || [])];
    all.sort((a, b) => (b.test_metrics?.macro_f1 || 0) - (a.test_metrics?.macro_f1 || 0));

    $("leaderboard-body").innerHTML = all.map((r, idx) => {
      const tm = r.test_metrics || {};
      const vm = r.val_metrics || {};
      const ciAcc = r.bootstrap_ci?.accuracy;
      const ciStr = ciAcc ? `<br><small style="color:var(--text-muted);">[${pct(ciAcc.ci_lower)}, ${pct(ciAcc.ci_upper)}]</small>` : (r.multi_seed_summary ? `<br><small style="color:var(--text-muted);">${esc(r.multi_seed_summary)}</small>` : "");
      const isTop = idx === 0;
      return `<tr class="${isTop ? "row-highlight" : ""}">
        <td>
          <strong>${esc(r.title)}</strong>
          <br><code class="tiny-code">${esc(r.run_id)}</code>
        </td>
        <td>
          <span class="table-pill">${esc(r.modality)}</span>
          <br><small style="color:var(--text-muted);">${esc(r.pretraining)}</small>
        </td>
        <td><strong>${fmtM(r.params_total)}</strong></td>
        <td>${r.epochs_run ?? "—"} <small style="color:var(--text-muted);">(best: ${r.best_epoch ?? "—"})</small></td>
        <td>${pct(vm.macro_f1)}</td>
        <td><strong>${pct(tm.accuracy)}</strong>${ciStr}</td>
        <td><strong>${pct(tm.balanced_accuracy)}</strong></td>
        <td><span class="score-badge ${isTop ? "best" : ""}">${pct(tm.macro_f1)}</span></td>
        <td>${pct(tm.weighted_f1)}</td>
      </tr>`;
    }).join("");
  }

  function renderPerClassComparison() {
    if (!benchData) return;
    const runs = benchData.evaluated_runs.filter((r) => r.test_metrics?.per_class);
    const idA = $("comp-model-a").value;
    const idB = $("comp-model-b").value;
    const filter = $("comp-organ-filter").value;

    const runA = runs.find((r) => r.run_id === idA) || runs[0];
    const runB = runs.find((r) => r.run_id === idB) || runs[1] || runs[0];
    if (!runA || !runB) return;

    $("th-model-a").textContent = `${runA.title.split("(")[0].trim()} (P / R / F1)`;
    $("th-model-b").textContent = `${runB.title.split("(")[0].trim()} (P / R / F1)`;

    const pcA = runA.test_metrics.per_class || {};
    const pcB = runB.test_metrics.per_class || {};
    const labels = Object.keys(pcA).sort();

    const rows = [];
    for (const lbl of labels) {
      const [crop, cond] = lbl.split("__");
      const isFruit = cond.includes("fruit");
      const organ = isFruit ? "Fruit" : (cond.includes("leaf") || cond === "black_rot" || cond === "cordana" || cond === "red_rust" ? "Leaf" : "Leaf / Fruit");
      if (filter === "fruit" && !isFruit) continue;
      if (filter === "leaf" && isFruit) continue;

      const a = pcA[lbl] || { precision: 0, recall: 0, f1: 0, support: 0 };
      const b = pcB[lbl] || { precision: 0, recall: 0, f1: 0, support: 0 };
      const diff = (a.f1 - b.f1) * 100;
      const diffSign = diff > 0.05 ? `+${diff.toFixed(2)}%` : (diff < -0.05 ? `${diff.toFixed(2)}%` : "0.00%");
      const diffCls = diff > 0.5 ? "delta-pos" : (diff < -0.5 ? "delta-neg" : "delta-neutral");

      rows.push(`<tr>
        <td><strong>${esc(lbl.replace("__", " / ").replace(/_/g, " "))}</strong><br><code class="tiny-code">${esc(lbl)}</code></td>
        <td style="text-transform:capitalize;">${esc(crop)}</td>
        <td><span class="organ-tag ${isFruit ? "fruit" : "leaf"}">${organ}</span></td>
        <td><strong>${a.support}</strong></td>
        <td>
          <div class="prf-cell">
            <span>P: ${pct(a.precision)}</span> · <span>R: ${pct(a.recall)}</span> · <strong>F1: ${pct(a.f1)}</strong>
          </div>
          <div class="mini-bar"><span style="width:${Math.max(a.f1 * 100, 1)}%"></span></div>
        </td>
        <td>
          <div class="prf-cell">
            <span>P: ${pct(b.precision)}</span> · <span>R: ${pct(b.recall)}</span> · <strong>F1: ${pct(b.f1)}</strong>
          </div>
          <div class="mini-bar secondary"><span style="width:${Math.max(b.f1 * 100, 1)}%"></span></div>
        </td>
        <td><span class="delta-badge ${diffCls}">${diffSign}</span></td>
      </tr>`);
    }
    $("per-class-body").innerHTML = rows.join("");
  }

  function renderRunArtifacts() {
    if (!benchData) return;
    const rid = $("artifact-run-select").value;
    const run = benchData.evaluated_runs.find((r) => r.run_id === rid) || benchData.evaluated_runs[0];
    if (!run) return;

    if ($("hist-title")) $("hist-title").textContent = `Epoch Log · ${run.title}`;

    const cmImg = $("cm-img");
    const curveImg = $("curve-img");
    if (run.confusion_matrix_url) {
      cmImg.src = run.confusion_matrix_url;
      cmImg.hidden = false;
    } else {
      cmImg.hidden = true;
    }
    if (run.training_curve_url) {
      curveImg.src = run.training_curve_url;
      curveImg.hidden = false;
    } else {
      curveImg.hidden = true;
    }

    const hist = run.history || [];
    if (!hist.length) {
      $("history-body").innerHTML = `<tr><td colspan="7">No epoch history CSV recorded for this run.</td></tr>`;
      return;
    }
    $("history-body").innerHTML = hist.map((h) => {
      const isBest = Number(h.epoch) === Number(run.best_epoch);
      return `<tr class="${isBest ? "row-highlight" : ""}">
        <td><strong>Ep ${h.epoch}</strong> ${isBest ? '<span class="organ-tag leaf">★ Best</span>' : ""}</td>
        <td>${Number(h.train_loss).toFixed(4)}</td>
        <td>${Number(h.val_loss).toFixed(4)}</td>
        <td>${pct(h.val_accuracy)}</td>
        <td>${pct(h.val_balanced_accuracy)}</td>
        <td><strong>${pct(h.val_macro_f1)}</strong></td>
        <td>${Number(h.epoch_seconds).toFixed(1)}s</td>
      </tr>`;
    }).join("");
  }

  function renderCrossCropAndCalibration(cc, cal) {
    const byCrop = (cc?.by_crop || []).slice().sort((a, b) => b.mean_macro_f1 - a.mean_macro_f1);
    $("crosscrop-body").innerHTML = byCrop.map((c) => `<tr>
      <td style="text-transform:capitalize;"><strong>${esc(c.crop)}</strong></td>
      <td>${c.n_evaluations}</td>
      <td><strong>${pct(c.mean_macro_f1)}</strong></td>
      <td>${pct(c.mean_roc_auc)}</td>
    </tr>`).join("");

    const cov = cal?.test_coverage_table || [];
    $("calibration-body").innerHTML = cov.map((row) => `<tr>
      <td><strong>p ≥ ${row.threshold.toFixed(2)}</strong></td>
      <td>${pct(row.coverage)}</td>
      <td><strong>${pct(row.accuracy_on_accepted)}</strong></td>
    </tr>`).join("");
  }

  async function init() {
    document.querySelectorAll("[data-cstep]").forEach((tab) => {
      tab.addEventListener("click", () => { goToCStep(Number(tab.getAttribute("data-cstep"))); });
    });
    $("c-prev-btn").addEventListener("click", () => { if (currentCStep > 1) goToCStep(currentCStep - 1); });
    $("c-next-btn").addEventListener("click", () => { if (currentCStep < 5) goToCStep(currentCStep + 1); });

    try {
      const res = await fetch("/api/benchmarks");
      benchData = await res.json();
    } catch (e) {
      $("leaderboard-body").innerHTML = `<tr><td colspan="9">Failed to load benchmark data: ${esc(e.message)}</td></tr>`;
      return;
    }

    const runs = benchData.evaluated_runs || [];
    renderLeaderboard(runs, benchData.reference_baselines || []);

    const selA = $("comp-model-a");
    const selB = $("comp-model-b");
    const selArt = $("artifact-run-select");

    runs.forEach((r) => {
      selA.add(new Option(r.title, r.run_id));
      selB.add(new Option(r.title, r.run_id));
      selArt.add(new Option(`${r.title} (${r.run_id})`, r.run_id));
    });

    if (runs.length > 1) selB.selectedIndex = 1;

    selA.addEventListener("change", renderPerClassComparison);
    selB.addEventListener("change", renderPerClassComparison);
    $("comp-organ-filter").addEventListener("change", renderPerClassComparison);
    selArt.addEventListener("change", renderRunArtifacts);

    renderPerClassComparison();
    renderRunArtifacts();
    renderCrossCropAndCalibration(benchData.cross_crop_summary, benchData.calibration);
  }

  init();
})();
