const state = {
      meta: null,
      scriptView: null,
      windowDetail: null,
      focusRows: [],
      featureLookupRow: null,
      selectedBundleId: null,
      selectedScriptId: null,
      selectedLayer: null,
      selectedWindowId: null,
      selectedSentence: null,
      selectedToken: null,
      selectedFeatureFilter: "",
      featureLookupInput: "",
      mode: "token",
      lens: "strongest",
      search: "",
      topFeatureSearch: "",
      topFeatureLimit: 10,
      spanStart: null,
      spanEnd: null,
      pendingSpanStart: null,
      probeStatus: null,
    };

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        let detail = "";
        try {
          detail = await response.text();
        } catch (error) {
          detail = "";
        }
        throw new Error("Request failed: " + response.status + " " + url + (detail ? " | " + detail : ""));
      }
      return response.json();
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        let detail = "";
        try {
          detail = await response.text();
        } catch (error) {
          detail = "";
        }
        throw new Error("Request failed: " + response.status + " " + url + (detail ? " | " + detail : ""));
      }
      return response.json();
    }

    function sleep(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function formatFloat(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
      }
      return Number(value).toFixed(2);
    }

    function compactText(value, width = 100) {
      const text = String(value ?? "").replace(/\s+/g, " ").trim();
      if (text.length <= width) {
        return text;
      }
      return text.slice(0, width - 3) + "...";
    }

    function ensureList(value) {
      if (Array.isArray(value)) return value;
      if (value === null || value === undefined || value === "") return [];
      return [value];
    }

    function currentFeatureRow() {
      if (state.featureLookupRow) return state.featureLookupRow;
      if (state.selectedFeatureFilter) {
        const exact = (state.focusRows || []).find((row) => String(row.feature_id) === String(state.selectedFeatureFilter));
        if (exact) return exact;
      }
      return (state.focusRows || [])[0] || null;
    }

    function filteredTopFeatures() {
      const rows = state.scriptView?.top_script_features || [];
      const needle = String(state.topFeatureSearch || "").trim().toLowerCase();
      if (!needle) return rows;
      return rows.filter((row) => {
        const label = String(row.label || "");
        return (
          label.toLowerCase().includes(needle) ||
          String(row.feature_id).includes(needle) ||
          String(row.transcript_relevance_rank || "").includes(needle)
        );
      });
    }

    function currentWindow() {
      if (state.windowDetail && String(state.windowDetail.sample_id) === String(state.selectedWindowId)) {
        return state.windowDetail;
      }
      const windows = state.scriptView?.windows || [];
      return windows.find((window) => String(window.sample_id) === String(state.selectedWindowId)) || windows[0] || null;
    }

    function currentBundleMeta() {
      return (state.meta?.bundles || []).find((bundle) => bundle.bundle_id === state.selectedBundleId) || state.meta?.bundles?.[0] || null;
    }

    function currentSentence() {
      const window = currentWindow();
      if (!window) return null;
      return (window.sentences || []).find((sentence) => Number(sentence.sentence_id) === Number(state.selectedSentence)) || null;
    }

    async function initialize() {
      document.getElementById("status").textContent = "Loading model bundles...";
      while (true) {
        const health = await fetchJson("/api/health");
        if (health.ready) break;
        if (health.error) {
          throw new Error(health.error);
        }
        await sleep(2000);
      }
      state.meta = await fetchJson("/api/meta");
      state.selectedBundleId = state.meta.default_bundle_id;
      state.selectedScriptId = state.meta.default_script_id;
      state.selectedLayer = state.meta.default_layer;
      await loadScriptView();
    }

    async function loadScriptView() {
      const params = new URLSearchParams({
        bundle_id: state.selectedBundleId,
        script_id: state.selectedScriptId,
        layer: String(state.selectedLayer),
      });
      state.scriptView = await fetchJson("/api/script-view?" + params.toString());
      const windows = filteredWindows();
      if (!windows.length) {
        state.selectedWindowId = null;
        state.windowDetail = null;
        state.focusRows = [];
        renderAll();
        return;
      }
      if (!windows.some((window) => String(window.sample_id) === String(state.selectedWindowId))) {
        state.selectedWindowId = String(windows[0].sample_id);
      }
      await loadWindowDetail();
      const window = currentWindow();
      if (window && state.selectedToken === null && window.token_details.length) {
        state.selectedToken = Number(window.token_details[0].token_position);
      }
      if (window && state.selectedSentence === null && window.sentences.length) {
        state.selectedSentence = Number(window.sentences[0].sentence_id);
      }
      await loadFocus();
      renderAll();
    }

    async function loadWindowDetail() {
      const summary = (state.scriptView?.windows || []).find((window) => String(window.sample_id) === String(state.selectedWindowId));
      if (!summary) {
        state.windowDetail = null;
        return;
      }
      const params = new URLSearchParams({
        bundle_id: state.selectedBundleId,
        script_id: state.selectedScriptId,
        layer: String(state.selectedLayer),
        sample_id: String(summary.sample_id),
      });
      const payload = await fetchJson("/api/window?" + params.toString());
      state.windowDetail = payload.window || null;
    }

    function filteredWindows() {
      const windows = state.scriptView?.windows || [];
      const needle = state.search.trim().toLowerCase();
      if (!needle) return windows;
      return windows.filter((window) => String(window.text || "").toLowerCase().includes(needle));
    }

    async function loadFocus() {
      const window = currentWindow();
      if (!window) {
        state.focusRows = [];
        state.probeStatus = null;
        renderRight();
        return;
      }
      const params = new URLSearchParams({
        bundle_id: state.selectedBundleId,
        script_id: state.selectedScriptId,
        layer: String(state.selectedLayer),
        sample_id: String(window.sample_id),
        mode: state.mode,
        lens: state.lens,
      });
      if (state.mode === "token" && state.selectedToken !== null) {
        params.set("token_position", String(state.selectedToken));
      }
      if (state.mode === "sentence" && state.selectedSentence !== null) {
        params.set("sentence_id", String(state.selectedSentence));
      }
      if (state.mode === "span" && state.spanStart !== null && state.spanEnd !== null) {
        params.set("span_start", String(state.spanStart));
        params.set("span_end", String(state.spanEnd));
      }
      if (state.selectedFeatureFilter) {
        params.set("feature_filter", String(state.selectedFeatureFilter));
      }
      const payload = await fetchJson("/api/focus?" + params.toString());
      state.focusRows = payload.rows || [];
      await loadProbeStatus();
      renderRight();
      renderSurface();
      renderTokenTable();
      renderTopFeatures();
    }

    async function runFeatureLookup() {
      const value = String(state.featureLookupInput || "").trim();
      if (!value) {
        state.featureLookupRow = null;
        renderRight();
        return;
      }
      const params = new URLSearchParams({
        bundle_id: state.selectedBundleId,
        script_id: state.selectedScriptId,
        layer: String(state.selectedLayer),
        feature_id: value,
      });
      const payload = await fetchJson("/api/feature?" + params.toString());
      state.featureLookupRow = payload.row || null;
      if (state.featureLookupRow) {
        state.selectedFeatureFilter = String(state.featureLookupRow.feature_id);
      }
      await loadProbeStatus();
      renderAll();
    }

    async function loadProbeStatus() {
      const row = currentFeatureRow();
      if (!row) {
        state.probeStatus = null;
        return;
      }
      const params = new URLSearchParams({
        bundle_id: state.selectedBundleId,
        script_id: state.selectedScriptId,
        layer: String(row.layer || state.selectedLayer),
        feature_id: String(row.feature_id),
      });
      try {
        state.probeStatus = await fetchJson("/api/probe-status?" + params.toString());
      } catch (error) {
        state.probeStatus = { error: error.message };
      }
    }

    async function startProbeForCurrentFeature() {
      const row = currentFeatureRow();
      if (!row) return;
      state.probeStatus = { running: true, pending: true };
      renderProbeStatus();
      try {
        state.probeStatus = await postJson("/api/probe-start", {
          bundle_id: state.selectedBundleId,
          script_id: state.selectedScriptId,
          layer: Number(row.layer || state.selectedLayer),
          feature_id: Number(row.feature_id),
        });
      } catch (error) {
        state.probeStatus = { error: error.message };
      }
      renderProbeStatus();
    }

    function renderAll() {
      renderToolbar();
      renderStats();
      renderSurface();
      renderSentenceButtons();
      renderInteractiveTokens();
      renderTokenTable();
      renderTopFeatures();
      renderRight();
    }

    function renderToolbar() {
      const container = document.getElementById("toolbar");
      const bundleOptions = (state.meta?.bundles || []).map((bundle) =>
        `<option value="${escapeHtml(bundle.bundle_id)}" ${bundle.bundle_id === state.selectedBundleId ? "selected" : ""}>${escapeHtml(bundle.label || bundle.bundle_id)}</option>`
      ).join("");
      const bundleMeta = currentBundleMeta();
      const scriptOptions = (bundleMeta?.scripts || []).map((script) =>
        `<option value="${escapeHtml(script.script_id)}" ${script.script_id === state.selectedScriptId ? "selected" : ""}>${escapeHtml(script.stimulus_id || script.filename || script.script_id)}</option>`
      ).join("");
      const selectedScript = (bundleMeta?.scripts || []).find((script) => script.script_id === state.selectedScriptId) || bundleMeta?.scripts?.[0];
      const layerOptions = (selectedScript?.layers || []).map((layer) =>
        `<option value="${layer}" ${Number(layer) === Number(state.selectedLayer) ? "selected" : ""}>Layer ${layer}</option>`
      ).join("");
      const windowOptions = filteredWindows().map((window) =>
        `<option value="${escapeHtml(window.sample_id)}" ${String(window.sample_id) === String(state.selectedWindowId) ? "selected" : ""}>${window.window_start}:${window.window_end} | ${escapeHtml(compactText(window.text, 56))}</option>`
      ).join("");
      const sentenceOptions = (currentWindow()?.sentences || []).map((sentence) =>
        `<option value="${sentence.sentence_id}" ${Number(sentence.sentence_id) === Number(state.selectedSentence) ? "selected" : ""}>Sentence ${sentence.sentence_id}</option>`
      ).join("");
      const featureOptions = [`<option value="">All shortlisted features</option>`].concat(
        (state.scriptView?.top_script_features || []).slice(0, 40).map((feature) => {
          const label = feature.label || ("feature " + feature.feature_id);
          return `<option value="${feature.feature_id}" ${String(feature.feature_id) === String(state.selectedFeatureFilter) ? "selected" : ""}>${escapeHtml(label)} | f${feature.feature_id}</option>`;
        })
      ).join("");
      const focusRows = state.featureLookupRow ? [state.featureLookupRow] : (state.focusRows || []);
      const focusOptions = focusRows.slice(0, 20).map((row) => `
        <option value="${row.feature_id}" ${String(row.feature_id) === String(currentFeatureRow()?.feature_id || "") ? "selected" : ""}>
          ${escapeHtml(row.label || ("feature " + row.feature_id))} | f${row.feature_id}
        </option>
      `).join("");
      container.innerHTML = `
        <div class="toolbar-grid">
          <div class="field">
            <label>Model</label>
            <select id="bundle-select">${bundleOptions}</select>
          </div>
          <div class="field">
            <label>Transcript</label>
            <select id="script-select">${scriptOptions}</select>
          </div>
          <div class="field">
            <label>Layer</label>
            <select id="layer-select">${layerOptions}</select>
          </div>
          <div class="field">
            <label>Window</label>
            <select id="window-select">${windowOptions}</select>
          </div>
          <div class="field">
            <label>Feature Filter</label>
            <select id="feature-filter">${featureOptions}</select>
          </div>
          <div class="field">
            <label>Focus Candidate</label>
            <select id="focus-feature-select">
              <option value="">Current strongest feature</option>
              ${focusOptions}
            </select>
          </div>
          <div class="field">
            <label>Feature Lookup</label>
            <div class="inline-row">
              <input id="feature-lookup-input" value="${escapeHtml(state.featureLookupInput)}" placeholder="feature id">
              <button id="feature-lookup-button" class="pill">Load</button>
              <button id="feature-lookup-clear" class="pill">Clear</button>
            </div>
          </div>
          <div class="field">
            <label>Sentence Jump</label>
            <select id="sentence-select">${sentenceOptions}</select>
          </div>
          <div class="field">
            <label>Search Transcript</label>
            <input id="search-input" value="${escapeHtml(state.search)}" placeholder="word or phrase">
          </div>
          <div class="field">
            <label>Filter Feature List</label>
            <div class="inline-row">
              <input id="top-feature-search" value="${escapeHtml(state.topFeatureSearch)}" placeholder="label or feature id">
              <select id="top-feature-limit">
                ${[8, 10, 16, 24].map((limit) => `<option value="${limit}" ${Number(limit) === Number(state.topFeatureLimit) ? "selected" : ""}>${limit}</option>`).join("")}
              </select>
            </div>
          </div>
          <div class="field">
            <label>Mode</label>
            <div class="mode-row">
              ${["token", "sentence", "span"].map((mode) => `<button class="pill ${state.mode === mode ? "active" : ""}" data-mode="${mode}">${mode}</button>`).join("")}
            </div>
          </div>
          <div class="field">
            <label>Lens</label>
            <div class="lens-row">
              ${[
                ["strongest", "Strongest"],
                ["distinctive", "Distinctive"],
              ].map(([value, label]) => `<button class="pill ${state.lens === value ? "active" : ""}" data-lens="${value}">${label}</button>`).join("")}
            </div>
          </div>
        </div>
      `;
      document.getElementById("bundle-select").onchange = async (event) => {
        state.selectedBundleId = event.target.value;
        const bundle = currentBundleMeta();
        state.selectedScriptId = bundle?.default_script_id || bundle?.scripts?.[0]?.script_id || null;
        state.selectedLayer = bundle?.default_layer || bundle?.scripts?.[0]?.layers?.slice(-1)?.[0] || null;
        state.selectedWindowId = null;
        state.selectedSentence = null;
        state.selectedToken = null;
        state.selectedFeatureFilter = "";
        state.featureLookupRow = null;
        await loadScriptView();
      };
      document.getElementById("script-select").onchange = async (event) => {
        state.selectedScriptId = event.target.value;
        const script = (currentBundleMeta()?.scripts || []).find((item) => item.script_id === state.selectedScriptId);
        state.selectedLayer = (script?.layers || [state.selectedLayer]).slice(-1)[0];
        state.selectedWindowId = null;
        state.selectedSentence = null;
        state.selectedToken = null;
        state.selectedFeatureFilter = "";
        state.featureLookupRow = null;
        await loadScriptView();
      };
      document.getElementById("layer-select").onchange = async (event) => {
        state.selectedLayer = Number(event.target.value);
        state.selectedWindowId = null;
        state.selectedSentence = null;
        state.selectedToken = null;
        state.selectedFeatureFilter = "";
        state.featureLookupRow = null;
        await loadScriptView();
      };
      document.getElementById("window-select").onchange = async (event) => {
        state.selectedWindowId = event.target.value;
        await loadWindowDetail();
        const window = currentWindow();
        state.selectedSentence = window?.sentences?.[0]?.sentence_id ?? null;
        state.selectedToken = window?.token_details?.[0]?.token_position ?? null;
        state.featureLookupRow = null;
        await loadFocus();
        renderAll();
      };
      document.getElementById("feature-filter").onchange = async (event) => {
        state.selectedFeatureFilter = event.target.value;
        state.featureLookupRow = null;
        await loadFocus();
      };
      document.getElementById("focus-feature-select").onchange = async (event) => {
        const value = event.target.value;
        state.featureLookupRow = null;
        state.selectedFeatureFilter = value;
        await loadFocus();
      };
      document.getElementById("feature-lookup-input").oninput = (event) => {
        state.featureLookupInput = event.target.value;
      };
      document.getElementById("feature-lookup-button").onclick = async () => {
        await runFeatureLookup();
      };
      document.getElementById("feature-lookup-clear").onclick = async () => {
        state.featureLookupInput = "";
        state.featureLookupRow = null;
        state.selectedFeatureFilter = "";
        await loadFocus();
        renderAll();
      };
      document.getElementById("feature-lookup-input").onkeydown = async (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          await runFeatureLookup();
        }
      };
      document.getElementById("sentence-select").onchange = async (event) => {
        state.selectedSentence = Number(event.target.value);
        state.mode = "sentence";
        await loadFocus();
        renderAll();
      };
      document.getElementById("search-input").onchange = async (event) => {
        state.search = event.target.value;
        const windows = filteredWindows();
        state.selectedWindowId = windows[0]?.sample_id ?? null;
        await loadWindowDetail();
        await loadFocus();
        renderAll();
      };
      document.getElementById("top-feature-search").oninput = (event) => {
        state.topFeatureSearch = event.target.value;
        renderTopFeatures();
      };
      document.getElementById("top-feature-limit").onchange = (event) => {
        state.topFeatureLimit = Number(event.target.value || 10);
        renderTopFeatures();
      };
      container.querySelectorAll("[data-mode]").forEach((button) => {
        button.onclick = async () => {
          state.mode = button.dataset.mode;
          state.pendingSpanStart = null;
          if (state.mode !== "span") {
            state.spanStart = null;
            state.spanEnd = null;
          }
          await loadFocus();
          renderAll();
        };
      });
      container.querySelectorAll("[data-lens]").forEach((button) => {
        button.onclick = async () => {
          state.lens = button.dataset.lens;
          await loadFocus();
          renderAll();
        };
      });
    }

    function renderStats() {
      const window = currentWindow();
      const container = document.getElementById("stats");
      if (!window) {
        container.innerHTML = "";
        document.getElementById("status").textContent = "No matching windows.";
        return;
      }
      document.getElementById("status").textContent =
        `${currentBundleMeta()?.label || state.selectedBundleId} | ${state.selectedScriptId} | layer ${state.selectedLayer} | window ${window.window_start}:${window.window_end}`;
      const stats = [
        ["Window Tokens", window.token_details?.length ?? window.token_count ?? 0],
        ["Window Features", window.window_features?.length ?? window.window_feature_count ?? 0],
        ["Focus Features", state.focusRows.length],
        ["Sentences", window.sentences?.length ?? window.sentence_count ?? 0],
      ];
      container.innerHTML = stats.map(([label, value]) => `
        <div class="stat">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </div>
      `).join("");
    }

    function tokenSignal(token) {
      if (state.selectedFeatureFilter) {
        const tokenHit = (token.latent_activations || []).find((item) => String(item.latent_id) === String(state.selectedFeatureFilter));
        if (tokenHit) return Number(tokenHit.activation || 0);
        const sentenceHit = (token.sentence_feature_summaries || []).find((item) => String(item.feature_id) === String(state.selectedFeatureFilter));
        if (sentenceHit) return 0.35 * Number(sentenceHit.max_activation || 0);
        const windowHit = (token.window_feature_summaries || []).find((item) => String(item.feature_id) === String(state.selectedFeatureFilter));
        if (windowHit) return 0.2 * Number(windowHit.max_activation || 0);
        return 0;
      }
      return Number(token.signal_total || 0);
    }

    function renderSurface() {
      const container = document.getElementById("surface");
      const window = currentWindow();
      if (!window) {
        container.innerHTML = '<div class="error">No transcript window available.</div>';
        return;
      }
      const tokens = window.token_details || [];
      const maxSignal = Math.max(1e-6, ...tokens.map(tokenSignal));
      const note = state.mode === "span"
        ? `Span mode. ${state.pendingSpanStart === null ? "Click a start token, then click an end token." : "Choose the end token."}`
        : state.mode === "sentence"
        ? `Sentence mode. Click a sentence chip to audit pooled sentence features.`
        : `Token mode. Hover tokens for quick feature hints and click to pin a token.`;
      let html = `<div class="surface-note">${escapeHtml(note)}</div>`;
      for (const sentence of (window.sentences || [])) {
        html += `<div class="sentence"><button class="sentence-chip ${Number(sentence.sentence_id) === Number(state.selectedSentence) && state.mode === "sentence" ? "active" : ""}" data-sentence="${sentence.sentence_id}" title="${escapeHtml(sentence.sentence_text || "")}">S${sentence.sentence_id}</button>`;
        for (let tokenPosition = Number(sentence.sentence_start_token_index); tokenPosition <= Number(sentence.sentence_end_token_index); tokenPosition += 1) {
          const token = tokens[tokenPosition];
          if (!token) continue;
          const intensity = Math.max(0, Math.min(1, tokenSignal(token) / maxSignal));
          const classes = ["token"];
          if (state.mode === "token" && Number(state.selectedToken) === tokenPosition) classes.push("selected");
          if (state.mode === "sentence" && Number(state.selectedSentence) === Number(sentence.sentence_id)) classes.push("selected");
          if (state.mode === "span" && state.spanStart !== null && state.spanEnd !== null && tokenPosition >= state.spanStart && tokenPosition <= state.spanEnd) classes.push("in-span");
          if (state.mode === "span" && state.pendingSpanStart !== null && tokenPosition === state.pendingSpanStart) classes.push("selected");
          const background = `rgba(194, 85, 26, ${0.10 + 0.70 * intensity})`;
          html += `<button class="${classes.join(" ")}" data-token="${tokenPosition}" data-sentence-id="${token.sentence_id}" data-tooltip="${escapeHtml(token.tooltip || "")}" title="${escapeHtml(token.tooltip || "")}" style="background:${background}">${escapeHtml(token.display_token || token.token || "")}</button>`;
        }
        html += `</div>`;
      }
      container.innerHTML = html;
      container.querySelectorAll("[data-sentence]").forEach((button) => {
        button.onclick = async () => {
          state.mode = "sentence";
          state.selectedSentence = Number(button.dataset.sentence);
          await loadFocus();
          renderAll();
        };
      });
      container.querySelectorAll("[data-token]").forEach((button) => {
        button.onclick = async () => {
          const tokenPosition = Number(button.dataset.token);
          const sentenceId = Number(button.dataset.sentenceId);
          if (state.mode === "span") {
            if (state.pendingSpanStart === null) {
              state.pendingSpanStart = tokenPosition;
              state.selectedSentence = sentenceId;
              renderSurface();
              return;
            }
            state.spanStart = Math.min(state.pendingSpanStart, tokenPosition);
            state.spanEnd = Math.max(state.pendingSpanStart, tokenPosition);
            state.pendingSpanStart = null;
            await loadFocus();
            renderAll();
            return;
          }
          state.mode = "token";
          state.selectedToken = tokenPosition;
          state.selectedSentence = sentenceId;
          await loadFocus();
          renderAll();
        };
      });
    }

    function renderSentenceButtons() {
      const container = document.getElementById("sentence-buttons");
      const window = currentWindow();
      if (!window) {
        container.innerHTML = "";
        return;
      }
      container.innerHTML = (window.sentences || []).map((sentence) => `
        <button class="pill ${state.mode === "sentence" && Number(state.selectedSentence) === Number(sentence.sentence_id) ? "active" : ""}" data-sentence="${sentence.sentence_id}" title="${escapeHtml(sentence.sentence_text || "")}">S${sentence.sentence_id}</button>
      `).join("");
      container.querySelectorAll("[data-sentence]").forEach((button) => {
        button.onclick = async () => {
          state.mode = "sentence";
          state.selectedSentence = Number(button.dataset.sentence);
          await loadFocus();
          renderAll();
        };
      });
    }

    function renderInteractiveTokens() {
      const container = document.getElementById("token-buttons");
      const window = currentWindow();
      if (!window) {
        container.innerHTML = "";
        return;
      }
      let tokens = window.token_details || [];
      if (state.selectedSentence !== null) {
        tokens = tokens.filter((token) => Number(token.sentence_id) === Number(state.selectedSentence));
      }
      if (!tokens.length) {
        container.innerHTML = '<div class="muted">No tokens available in this slice.</div>';
        return;
      }
      const limited = tokens.slice(0, 128);
      container.innerHTML = limited.map((token) => `
        <button class="pill ${Number(token.token_position) === Number(state.selectedToken) && state.mode === "token" ? "active" : ""}" data-token="${token.token_position}" title="${escapeHtml(token.tooltip || "")}">
          ${escapeHtml((token.display_token || token.token || "").trim() || "[sp]")}
        </button>
      `).join("");
      container.querySelectorAll("[data-token]").forEach((button) => {
        button.onclick = async () => {
          const windowToken = limited.find((token) => Number(token.token_position) === Number(button.dataset.token));
          state.mode = "token";
          state.selectedToken = Number(button.dataset.token);
          state.selectedSentence = Number(windowToken?.sentence_id ?? state.selectedSentence);
          await loadFocus();
          renderAll();
        };
      });
    }

    function renderTopFeatures() {
      const container = document.getElementById("top-features");
      const rows = filteredTopFeatures();
      const visibleRows = rows.slice(0, state.topFeatureLimit);
      document.getElementById("feature-list-summary").textContent = `${visibleRows.length}/${rows.length} shown`;
      container.innerHTML = visibleRows.map((row) => `
        <div class="feature-row ${String(row.feature_id) === String(state.selectedFeatureFilter) ? "active" : ""}" data-feature="${row.feature_id}">
          <div class="feature-title">
            <span>${escapeHtml(row.label || ("feature " + row.feature_id))}</span>
            <span class="meta">f${row.feature_id}</span>
          </div>
          <div class="meta">rank ${row.transcript_relevance_rank ?? "-"} | max ${formatFloat(row.max_activation)}</div>
          <div class="badge-row">
            ${row.has_judge ? '<span class="badge">judge</span>' : ""}
            ${row.has_dolma ? '<span class="badge">dolma</span>' : ""}
            ${row.has_alignment ? '<span class="badge">alignment</span>' : ""}
          </div>
        </div>
      `).join("");
      container.querySelectorAll("[data-feature]").forEach((row) => {
        row.onclick = async () => {
          state.selectedFeatureFilter = String(row.dataset.feature) === String(state.selectedFeatureFilter) ? "" : String(row.dataset.feature);
          await loadFocus();
          renderAll();
        };
      });
    }

    function renderTokenTable() {
      const container = document.getElementById("token-table");
      const window = currentWindow();
      if (!window) {
        container.innerHTML = "";
        return;
      }
      container.innerHTML = (window.token_details || []).slice(0, 120).map((token) => {
        const tooltipLines = String(token.tooltip || "").split("\\n").slice(2, 5).join(" | ");
        return `
          <div class="token-row">
            <strong>${token.token_position}</strong> ${escapeHtml(token.display_token || token.token || "")}
            <div class="small muted">sentence ${token.sentence_id}</div>
            <div class="small">${escapeHtml(tooltipLines || "no shortlisted features")}</div>
          </div>
        `;
      }).join("");
    }

    function selectionSummary() {
      const window = currentWindow();
      if (!window) return "No selection.";
      if (state.mode === "sentence") {
        const sentence = currentSentence();
        return sentence ? `Sentence ${sentence.sentence_id}: ${sentence.sentence_text}` : "Sentence selection";
      }
      if (state.mode === "span" && state.spanStart !== null && state.spanEnd !== null) {
        const tokens = (window.token_details || [])
          .filter((token) => Number(token.token_position) >= state.spanStart && Number(token.token_position) <= state.spanEnd)
          .map((token) => token.display_token || token.token || "")
          .join("");
        return `Span ${state.spanStart}-${state.spanEnd}: ${tokens.trim()}`;
      }
      const token = (window.token_details || []).find((item) => Number(item.token_position) === Number(state.selectedToken));
      return token ? `Token ${token.token_position}: ${token.display_token || token.token}` : "Token selection";
    }

    function renderProbeStatus() {
      const container = document.getElementById("probe-status-box");
      const row = currentFeatureRow();
      if (!row) {
        container.innerHTML = "";
        return;
      }
      const status = state.probeStatus || {};
      const headline = status.error
        ? `<span class="error">${escapeHtml(status.error)}</span>`
        : status.running
        ? `Probe running${status.pid ? ` (pid ${status.pid})` : ""}`
        : status.has_report
        ? "Probe report saved"
        : status.has_evidence
        ? "Probe partially collected"
        : "No saved probe run yet";
      const summary = status.report_summary?.summary
        ? `<div class="small" style="margin-top:6px;">${escapeHtml(status.report_summary.summary)}</div>`
        : "";
      const actions = `
        <div class="inline-row" style="margin-top:8px;">
          <button id="probe-start-button" class="action-button" ${status.running ? "disabled" : ""}>Probe Selected Feature</button>
          ${status.paths?.root ? `<span class="small muted">${escapeHtml(status.paths.root)}</span>` : ""}
        </div>
      `;
      const details = status.log_tail?.length
        ? `<details><summary>Probe Log</summary><div class="details-body"><div class="codebox">${escapeHtml(status.log_tail.join("\n"))}</div></div></details>`
        : "";
      container.innerHTML = `
        <div class="probe-status">
          <strong>Prober</strong>
          <div style="margin-top:4px;">${headline}</div>
          ${summary}
          ${actions}
        </div>
        ${details}
      `;
      const startButton = document.getElementById("probe-start-button");
      if (startButton) {
        startButton.onclick = async () => {
          await startProbeForCurrentFeature();
        };
      }
    }

    function renderRight() {
      document.getElementById("selection-summary").textContent = state.featureLookupRow
        ? `Feature lookup | ${currentBundleMeta()?.label || state.selectedBundleId} | ${state.selectedScriptId} | layer ${state.selectedLayer}`
        : selectionSummary();
      document.getElementById("source-note").textContent = state.featureLookupRow
        ? `Direct feature lookup for f${state.featureLookupRow.feature_id}`
        : (currentFeatureRow()?.source_note || "");
      const container = document.getElementById("evidence-stack");
      const row = currentFeatureRow();
      renderProbeStatus();
      if (!row) {
        container.innerHTML = '<div class="error">No shortlisted features found for the current selection.</div>';
        return;
      }
      const badges = []
        .concat(row.coverage || [])
        .concat(row.has_judge ? ["judge"] : [])
        .concat(row.has_alignment ? ["alignment"] : [])
        .concat(row.has_dolma ? ["dolma"] : [])
        .concat(row.judge_coverage_status ? [`judge:${row.judge_coverage_status}`] : [])
        .map((badge) => `<span class="badge">${escapeHtml(badge)}</span>`)
        .join("");
      const metrics = row.local_metrics || {};
      const transcriptExamples = (row.top_transcript_examples || []).slice(0, 4).map((item) =>
        `<li>${escapeHtml(item.selection_reason || "example")}: ${escapeHtml((item.snippet_tokens || []).join(" ") || item.token || "")}</li>`
      ).join("");
      const dolmaExamples = (row.top_dolma_contexts || []).slice(0, 5).map((item) => {
        const sentenceSnippet = (item.top_sentence_snippets || [])[0]?.text;
        const spanSnippet = ((item.top_span_snippets || [])[0]?.tokens || []).join(" ");
        const tokenSnippet = ((item.top_token_snippets || [])[0]?.snippet_tokens || []).join(" ");
        const snippet = sentenceSnippet || spanSnippet || tokenSnippet || item.window_text || "context available";
        return `<li>[${escapeHtml(item.dominant_scale || "unspecified")}] ${escapeHtml(item.selection_reason || "context")}: ${escapeHtml(compactText(snippet, 180))}</li>`;
      }).join("");
      const alignments = (row.alignment_summary || []).slice(0, 3).map((item) => {
        const tokenHit = (item.top_token_alignments || [])[0];
        const spanHit = (item.top_span_alignments || [])[0];
        if (tokenHit) return `<li>token ${tokenHit.token_position} (${escapeHtml(tokenHit.method || "")})</li>`;
        if (spanHit) return `<li>span ${spanHit.start_token_position}-${spanHit.end_token_position} (${escapeHtml(spanHit.method || "")})</li>`;
        return `<li>alignment available</li>`;
      }).join("");
      const correlated = (row.top_correlated_features || []).slice(0, 6).map((item) =>
        `<li>f${item.feature_id}: ${formatFloat(item.correlation)}</li>`
      ).join("");
      const judgeFor = ensureList(row.judge_evidence_for).slice(0, 5).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
      const judgeAgainst = ensureList(row.judge_evidence_against).slice(0, 5).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
      const judgeFollowUp = ensureList(row.judge_follow_up).slice(0, 5).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
      const scriptTokenHits = (row.script_token_hits || []).slice(0, 6).map((item) =>
        `<li>tok ${item.token_position} <strong>${escapeHtml(item.display_token || item.token || "")}</strong> | act ${formatFloat(item.activation)}<br><span class="small muted">${escapeHtml(compactText(item.snippet || "", 150))}</span></li>`
      ).join("");
      const scriptSentenceHits = (row.script_sentence_hits || []).slice(0, 5).map((item) =>
        `<li>sent ${item.sentence_id} | total ${formatFloat(item.total_activation)} | max ${formatFloat(item.max_activation)}<br><span class="small muted">${escapeHtml(compactText(item.sentence_text || "", 170))}</span></li>`
      ).join("");
      const scriptWindowHits = (row.script_window_hits || []).slice(0, 4).map((item) =>
        `<li>${item.window_start}:${item.window_end} | total ${formatFloat(item.total_activation)}<br><span class="small muted">${escapeHtml(compactText(item.text || "", 170))}</span></li>`
      ).join("");
      container.innerHTML = `
        <div class="evidence-card">
          <div class="feature-title">
            <span>${escapeHtml(row.label || ("feature " + row.feature_id))}</span>
            <span class="meta">f${row.feature_id}</span>
          </div>
          <div class="meta">${escapeHtml(row.feature_type || "untyped")} | confidence ${formatFloat(row.confidence)} | rank ${row.transcript_relevance_rank ?? "-"}</div>
          <div class="badge-row">${badges}</div>
          ${row.summary ? `<p>${escapeHtml(row.summary)}</p>` : ""}
          ${row.transcript_rationale ? `<p class="small muted">${escapeHtml(row.transcript_rationale)}</p>` : ""}
          <div class="metrics">
            <div class="metric"><div class="label">Span Total</div><div class="value">${formatFloat(metrics.span_total_activation)}</div></div>
            <div class="metric"><div class="label">Peak</div><div class="value">${formatFloat(metrics.token_max_activation || metrics.sentence_max_activation || metrics.window_max_activation)}</div></div>
            <div class="metric"><div class="label">Distinctive</div><div class="value">${formatFloat(row.distinctiveness)}</div></div>
          </div>
          ${row.has_judge ? `
            <details open>
              <summary>Judge Summary</summary>
              <div class="details-body stack">
                <div class="small"><strong>Label:</strong> ${escapeHtml(row.judge_label || row.label || "unlabeled")}</div>
                ${row.judge_summary ? `<div class="small">${escapeHtml(row.judge_summary)}</div>` : ""}
                ${row.judge_uncertainty ? `<div class="small muted"><strong>Uncertainty:</strong> ${escapeHtml(row.judge_uncertainty)}</div>` : ""}
                ${judgeFor ? `<div><div class="small muted">Evidence For</div><ul class="list small">${judgeFor}</ul></div>` : ""}
                ${judgeAgainst ? `<div><div class="small muted">Evidence Against</div><ul class="list small">${judgeAgainst}</ul></div>` : ""}
                ${judgeFollowUp ? `<div><div class="small muted">Follow Up</div><ul class="list small">${judgeFollowUp}</ul></div>` : ""}
              </div>
            </details>
          ` : ""}
          ${scriptTokenHits ? `<details open><summary>Token Hits</summary><div class="details-body"><ul class="list small">${scriptTokenHits}</ul></div></details>` : ""}
          ${scriptSentenceHits ? `<details><summary>Sentence Hits</summary><div class="details-body"><ul class="list small">${scriptSentenceHits}</ul></div></details>` : ""}
          ${scriptWindowHits ? `<details><summary>Window Hits</summary><div class="details-body"><ul class="list small">${scriptWindowHits}</ul></div></details>` : ""}
          ${transcriptExamples ? `<details><summary>Transcript Examples</summary><div class="details-body"><ul class="list small">${transcriptExamples}</ul></div></details>` : ""}
          ${dolmaExamples ? `<details><summary>Dolma Evidence</summary><div class="details-body"><ul class="list small">${dolmaExamples}</ul></div></details>` : ""}
          ${alignments ? `<details><summary>Alignment</summary><div class="details-body"><ul class="list small">${alignments}</ul></div></details>` : ""}
          ${correlated ? `<details><summary>Correlated Features</summary><div class="details-body"><ul class="list small">${correlated}</ul></div></details>` : ""}
        </div>
      `;
    }

    initialize().catch((error) => {
      document.getElementById("status").innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
    });
