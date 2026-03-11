// SelfEvo Dashboard - Frontend Logic

const API = '';
let charts = {};
let refreshInterval = null;

// === Navigation ===
document.querySelectorAll('.nav button').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('page-' + btn.dataset.page).classList.add('active');

        // Load page data
        const page = btn.dataset.page;
        if (page === 'overview') loadOverview();
        else if (page === 'history') loadHistory();
        else if (page === 'trends') loadTrends();
        else if (page === 'compare') loadCompareOptions();
        else if (page === 'control') loadAIConfig();
        else if (page === 'playground') loadPlayground();
    });
});

// === Toast ===
function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// === API Helpers ===
async function apiGet(path) {
    const res = await fetch(API + path);
    return res.json();
}

async function apiPost(path, body) {
    const res = await fetch(API + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
    });
    return res.json();
}

// === Overview Page ===
async function loadOverview() {
    const status = await apiGet('/api/status');

    // Status badge
    const badge = document.getElementById('status-badge');
    badge.textContent = status.status.charAt(0).toUpperCase() + status.status.slice(1);
    badge.className = 'status-badge status-' + status.status;

    // Cards
    document.getElementById('best-val-loss').textContent =
        status.best_val_loss !== null ? status.best_val_loss.toFixed(4) : '--';
    document.getElementById('best-experiment').textContent =
        status.best_experiment || '';
    document.getElementById('total-experiments').textContent = status.total_experiments;
    document.getElementById('today-count').textContent =
        'Today: ' + status.today_experiments;
    document.getElementById('kdc-stats').textContent =
        `${status.keeps} / ${status.discards} / ${status.crashes}`;
    const total = status.keeps + status.discards + status.crashes;
    document.getElementById('keep-rate').textContent =
        total > 0 ? `Keep rate: ${(status.keeps / total * 100).toFixed(1)}%` : '';
    document.getElementById('current-phase').textContent =
        status.phase.charAt(0).toUpperCase() + status.phase.slice(1);
    document.getElementById('data-status').textContent =
        status.data_ready ? 'Data ready' : 'Data not prepared';

    // Running status banner
    const banner = document.getElementById('running-banner');
    const bannerText = document.getElementById('running-banner-text');
    banner.style.display = 'flex';
    banner.className = 'running-banner state-' + status.status;
    const statusMessages = {
        running: `Experiment loop is running... (${status.total_experiments} experiments completed)`,
        training: `Training experiment in progress... (${status.total_experiments} completed)`,
        paused: `Experiment loop is paused. (${status.total_experiments} experiments completed)`,
        idle: `Experiment loop is idle.`,
        error: `Error occurred. Check Control page for details.`,
    };
    bannerText.textContent = statusMessages[status.status] || `Status: ${status.status}`;

    // Update control buttons
    updateControlButtons(status.status);

    // Last experiment
    const lastCard = document.getElementById('last-experiment-card');
    if (status.last_experiment) {
        lastCard.style.display = 'block';
        const e = status.last_experiment;
        document.getElementById('last-experiment-info').innerHTML = `
            <div style="display:flex;gap:20px;margin-top:8px;flex-wrap:wrap">
                <div><strong>ID:</strong> ${e.experiment_id || '--'}</div>
                <div><strong>Class:</strong> ${e.experiment_class || '--'}</div>
                <div><strong>val_loss:</strong> ${e.val_loss !== null ? e.val_loss.toFixed(4) : '--'}</div>
                <div><span class="badge badge-${e.status}">${e.status}</span></div>
            </div>
            <div class="reason-text" style="margin-top:8px;max-width:none">${e.judge_reason || ''}</div>
        `;
    } else {
        lastCard.style.display = 'none';
    }

    // Overview chart
    await loadOverviewChart();
}

async function loadOverviewChart() {
    const trends = await apiGet('/api/trends');
    const data = trends.val_loss_trend || [];

    if (charts.overview) charts.overview.destroy();

    const ctx = document.getElementById('overview-chart').getContext('2d');
    const labels = data.map((_, i) => i + 1);
    const values = data.map(d => d.val_loss);
    const colors = data.map(d => {
        if (d.status === 'keep') return 'rgba(34,197,94,0.8)';
        if (d.status === 'crash') return 'rgba(239,68,68,0.8)';
        return 'rgba(234,179,8,0.8)';
    });

    charts.overview = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'val_loss',
                data: values,
                backgroundColor: colors,
                borderRadius: 2,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: true, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } },
                y: { grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }
            }
        }
    });
}

// === History Page ===
async function loadHistory() {
    const statusFilter = document.getElementById('filter-status').value;
    const classFilter = document.getElementById('filter-class').value;

    let url = '/api/experiments?limit=200';
    if (statusFilter) url += '&status=' + statusFilter;
    if (classFilter) url += '&experiment_class=' + classFilter;

    const result = await apiGet(url);
    const tbody = document.getElementById('history-tbody');
    const empty = document.getElementById('history-empty');

    if (!result.experiments || result.experiments.length === 0) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }

    empty.style.display = 'none';
    tbody.innerHTML = result.experiments.map((e, i) => {
        // Build AI summary: hypothesis + code changes
        const hypothesis = e.hypothesis || '';
        const patch = e.patch_summary || '';
        const summary = hypothesis
            ? `<div class="summary-hypothesis">${escapeHtml(hypothesis)}</div><div class="summary-patch">${escapeHtml(patch)}</div>`
            : `<div class="summary-patch">${escapeHtml(patch)}</div>`;
        return `
        <tr>
            <td>${result.total - i}</td>
            <td>${formatTime(e.timestamp)}</td>
            <td>${e.experiment_class || '--'}</td>
            <td class="summary-cell">${summary}</td>
            <td>${e.val_loss !== null && e.val_loss !== undefined ? e.val_loss.toFixed(4) : '--'}</td>
            <td>${e.train_time_sec !== null && e.train_time_sec !== undefined ? e.train_time_sec.toFixed(1) : '--'}</td>
            <td>${e.peak_mem_mb !== null && e.peak_mem_mb !== undefined ? e.peak_mem_mb.toFixed(0) : '--'}</td>
            <td><span class="badge badge-${e.status}">${e.status}</span></td>
            <td class="reason-text">${e.judge_reason || '--'}</td>
        </tr>`;
    }).join('');
}

document.getElementById('filter-status').addEventListener('change', loadHistory);
document.getElementById('filter-class').addEventListener('change', loadHistory);

// === Trends Page ===
async function loadTrends() {
    const trends = await apiGet('/api/trends');

    // val_loss trend
    if (charts.valloss) charts.valloss.destroy();
    const vlData = trends.val_loss_trend || [];
    const vlCtx = document.getElementById('trend-valloss').getContext('2d');

    // Separate keeps from all
    const allPoints = vlData.map((d, i) => ({ x: i + 1, y: d.val_loss }));
    const keepPoints = vlData.filter(d => d.status === 'keep').map((d, i) => ({
        x: vlData.indexOf(d) + 1,
        y: d.val_loss
    }));

    charts.valloss = new Chart(vlCtx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'All Experiments',
                    data: allPoints,
                    backgroundColor: vlData.map(d => {
                        if (d.status === 'keep') return 'rgba(34,197,94,0.8)';
                        if (d.status === 'crash') return 'rgba(239,68,68,0.8)';
                        return 'rgba(234,179,8,0.5)';
                    }),
                    pointRadius: 5,
                },
                {
                    label: 'Best Progression',
                    data: (trends.best_progression || []).map((d, i) => ({
                        x: vlData.findIndex(v => v.experiment_id === d.experiment_id) + 1,
                        y: d.val_loss
                    })),
                    type: 'line',
                    borderColor: 'rgba(99,102,241,0.8)',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    tension: 0.1,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#8b8d97' } } },
            scales: {
                x: { title: { display: true, text: 'Experiment #', color: '#8b8d97' }, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } },
                y: { title: { display: true, text: 'val_loss', color: '#8b8d97' }, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }
            }
        }
    });

    // Keep rate
    if (charts.keeprate) charts.keeprate.destroy();
    const krData = trends.keep_rate_trend || [];
    charts.keeprate = new Chart(document.getElementById('trend-keeprate').getContext('2d'), {
        type: 'line',
        data: {
            labels: krData.map(d => d.index + 1),
            datasets: [{
                label: 'Keep Rate',
                data: krData.map(d => d.rate),
                borderColor: 'rgba(34,197,94,0.8)',
                backgroundColor: 'rgba(34,197,94,0.1)',
                fill: true,
                tension: 0.3,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } },
                y: { min: 0, max: 1, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }
            }
        }
    });

    // Crash rate
    if (charts.crashrate) charts.crashrate.destroy();
    const crData = trends.crash_rate_trend || [];
    charts.crashrate = new Chart(document.getElementById('trend-crashrate').getContext('2d'), {
        type: 'line',
        data: {
            labels: crData.map(d => d.index + 1),
            datasets: [{
                label: 'Crash Rate',
                data: crData.map(d => d.rate),
                borderColor: 'rgba(239,68,68,0.8)',
                backgroundColor: 'rgba(239,68,68,0.1)',
                fill: true,
                tension: 0.3,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } },
                y: { min: 0, max: 1, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }
            }
        }
    });

    // Class success
    if (charts.classSuccess) charts.classSuccess.destroy();
    const csData = trends.class_success || [];
    charts.classSuccess = new Chart(document.getElementById('trend-class').getContext('2d'), {
        type: 'bar',
        data: {
            labels: csData.map(d => d.class),
            datasets: [
                {
                    label: 'Keep Rate',
                    data: csData.map(d => d.keep_rate),
                    backgroundColor: 'rgba(34,197,94,0.6)',
                },
                {
                    label: 'Crash Rate',
                    data: csData.map(d => d.crash_rate),
                    backgroundColor: 'rgba(239,68,68,0.6)',
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#8b8d97' } } },
            scales: {
                x: { grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } },
                y: { min: 0, max: 1, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }
            }
        }
    });
}

// === Compare Page ===
async function loadCompareOptions() {
    const result = await apiGet('/api/experiments?limit=100');
    const select = document.getElementById('compare-select');

    // Keep first option
    select.innerHTML = '<option value="">Select an experiment to compare...</option>';
    (result.experiments || []).forEach(e => {
        const opt = document.createElement('option');
        opt.value = e.experiment_id;
        opt.textContent = `${e.experiment_id} | ${e.experiment_class} | ${e.status} | val_loss=${e.val_loss !== null ? e.val_loss.toFixed(4) : 'N/A'}`;
        select.appendChild(opt);
    });
}

document.getElementById('compare-select').addEventListener('change', async (ev) => {
    const id = ev.target.value;
    if (!id) {
        document.getElementById('compare-content').style.display = 'none';
        document.getElementById('compare-empty').style.display = 'block';
        return;
    }

    const data = await apiGet('/api/compare/' + id);
    document.getElementById('compare-content').style.display = 'grid';
    document.getElementById('compare-empty').style.display = 'none';

    document.getElementById('compare-baseline').innerHTML = formatExperimentDetail(data.baseline);
    document.getElementById('compare-experiment').innerHTML = formatExperimentDetail(data.experiment);
});

function formatExperimentDetail(e) {
    if (!e) return '<div class="empty-state" style="padding:20px"><p>No baseline available</p></div>';
    return `
        <table style="width:100%;font-size:13px">
            <tr><td style="color:var(--text-muted);padding:4px 8px">ID</td><td style="padding:4px 8px">${e.experiment_id || '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">val_loss</td><td style="padding:4px 8px;font-weight:700">${e.val_loss !== null && e.val_loss !== undefined ? e.val_loss.toFixed(4) : '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Class</td><td style="padding:4px 8px">${e.experiment_class || '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Patch</td><td style="padding:4px 8px">${e.patch_summary || '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Params</td><td style="padding:4px 8px">${e.num_params ? e.num_params.toLocaleString() : '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Train Time</td><td style="padding:4px 8px">${e.train_time_sec ? e.train_time_sec.toFixed(1) + 's' : '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Memory</td><td style="padding:4px 8px">${e.peak_mem_mb ? e.peak_mem_mb.toFixed(0) + ' MB' : '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Status</td><td style="padding:4px 8px"><span class="badge badge-${e.status}">${e.status}</span></td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Reason</td><td style="padding:4px 8px">${e.judge_reason || '--'}</td></tr>
            <tr><td style="color:var(--text-muted);padding:4px 8px">Hypothesis</td><td style="padding:4px 8px">${e.hypothesis || '--'}</td></tr>
        </table>
    `;
}

// === Control Actions ===
let currentStatus = 'idle';

async function controlAction(action) {
    try {
        const result = await apiPost('/api/control/' + action);
        showToast(result.message || 'Action completed');
        setTimeout(loadOverview, 500);
    } catch (err) {
        showToast('Error: ' + err.message);
    }
}

function toggleLoop() {
    if (currentStatus === 'running' || currentStatus === 'training') {
        controlAction('pause');
    } else {
        controlAction('start');
    }
}

function updateControlButtons(status) {
    currentStatus = status;
    const isRunning = (status === 'running' || status === 'training');
    const isPaused = (status === 'paused');

    // Overview toggle button
    const overviewBtn = document.getElementById('overview-toggle-btn');
    if (overviewBtn) {
        if (isRunning) {
            overviewBtn.textContent = '⏸ Pause';
            overviewBtn.className = 'btn btn-sm btn-warning';
        } else if (isPaused) {
            overviewBtn.textContent = '▶ Resume';
            overviewBtn.className = 'btn btn-sm btn-success';
        } else {
            overviewBtn.textContent = '▶ Start';
            overviewBtn.className = 'btn btn-sm btn-success';
        }
    }
}

async function updateSettings() {
    const highRisk = document.getElementById('setting-high-risk').checked;
    const largeChanges = document.getElementById('setting-large-changes').checked;
    await apiPost('/api/settings', {
        allow_high_risk: highRisk,
        allow_large_changes: largeChanges,
    });
    showToast('Settings updated');
}

// AI Model settings
const ALL_MODELS = [
    { value: 'gemini-3.1-pro-preview', label: 'Gemini 3.1 Pro', provider: 'gemini' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro', provider: 'gemini' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash', provider: 'gemini' },
    { value: 'gpt-5.4', label: 'GPT-5.4', provider: 'openai' },
    { value: 'gpt-5.4-pro', label: 'GPT-5.4 Pro', provider: 'openai' },
    { value: 'gpt-4.1', label: 'GPT-4.1', provider: 'openai' },
    { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini', provider: 'openai' },
    { value: 'o4-mini', label: 'o4-mini', provider: 'openai' },
    { value: 'claude-opus-4-6', label: 'Claude Opus 4.6', provider: 'claude' },
    { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4', provider: 'claude' },
];

const PROVIDER_LABELS = { gemini: 'Google Gemini', openai: 'OpenAI', claude: 'Anthropic Claude' };

function populateModelSelect(availableProviders, currentModel) {
    const select = document.getElementById('setting-ai-model');
    if (!select) return;
    select.innerHTML = '<option value="">Auto-detect (default)</option>';

    const groups = {};
    ALL_MODELS.forEach(m => {
        if (!groups[m.provider]) groups[m.provider] = [];
        groups[m.provider].push(m);
    });

    for (const [provider, models] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        const hasKey = availableProviders.includes(provider);
        optgroup.label = PROVIDER_LABELS[provider] || provider;
        if (!hasKey) optgroup.label += ' (no key)';

        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.value;
            opt.textContent = m.label;
            if (!hasKey) { opt.disabled = true; opt.textContent += ' (no key)'; }
            if (m.value === currentModel) opt.selected = true;
            optgroup.appendChild(opt);
        });
        select.appendChild(optgroup);
    }
}

async function loadAIConfig() {
    try {
        const config = await apiGet('/api/ai_config');

        // Populate model dropdown
        populateModelSelect(config.available_providers, config.current_model || '');

        // Model hint
        const modelHint = document.getElementById('ai-model-hint');
        if (modelHint) {
            const active = config.available_providers[0] || 'none';
            const defaultModel = config.defaults[active] || '';
            modelHint.textContent = defaultModel ? `Default: ${defaultModel}` : 'No API key set';
        }

        // API key status hints
        ['gemini', 'openai', 'claude'].forEach(p => {
            const hint = document.getElementById('key-' + p + '-hint');
            if (hint) {
                const source = (config.key_sources || {})[p];
                if (source) {
                    hint.textContent = '✓ Configured';
                    hint.style.color = 'var(--green)';
                } else {
                    hint.textContent = 'Not configured';
                    hint.style.color = 'var(--text-muted)';
                }
            }
        });
    } catch (e) {}
}

async function updateAISettings() {
    const model = document.getElementById('setting-ai-model').value;
    await apiPost('/api/settings', { ai_model: model });
    showToast('AI model updated');
    loadAIConfig();
}

async function saveApiKey(provider) {
    const input = document.getElementById('key-' + provider);
    const key = input ? input.value.trim() : '';
    if (!key) {
        showToast('Please enter an API key');
        return;
    }
    await apiPost('/api/save_key', { provider, key });
    input.value = '';
    showToast(provider.charAt(0).toUpperCase() + provider.slice(1) + ' API key saved');
    loadAIConfig();
}

// === Helpers ===
function formatTime(ts) {
    if (!ts) return '--';
    const d = new Date(ts);
    return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

// === Playground Page ===

// Slider value display
['temperature', 'topk', 'maxtokens'].forEach(param => {
    const slider = document.getElementById('pg-' + param);
    const display = document.getElementById('pg-' + param + '-val');
    if (slider && display) {
        slider.addEventListener('input', () => { display.textContent = slider.value; });
    }
});

async function loadPlayground() {
    // Load model info
    try {
        const info = await apiGet('/api/model_info');
        const bar = document.getElementById('model-info-bar');
        if (!info.has_checkpoint) {
            bar.innerHTML = '<span style="color:var(--yellow)">No checkpoint found.</span> Click "Retrain Model" to create one.';
        } else {
            const parts = [];
            if (info.checkpoint_size_mb) parts.push(`Checkpoint: ${info.checkpoint_size_mb} MB`);
            if (info.config) {
                const cfg = info.config;
                const cfgParts = [];
                if (cfg.n_layer) cfgParts.push(`layers=${cfg.n_layer}`);
                if (cfg.n_head) cfgParts.push(`heads=${cfg.n_head}`);
                if (cfg.n_embd) cfgParts.push(`embd=${cfg.n_embd}`);
                if (cfg.block_size) cfgParts.push(`ctx=${cfg.block_size}`);
                if (cfgParts.length) parts.push(cfgParts.join(', '));
            }
            bar.innerHTML = '<span style="color:var(--green)">Model ready.</span> ' + parts.join(' | ');
        }
    } catch (e) {
        document.getElementById('model-info-bar').textContent = 'Failed to load model info';
    }
}

async function generateText() {
    const btn = document.getElementById('pg-generate-btn');
    const output = document.getElementById('pg-output');
    const statsCard = document.getElementById('pg-stats');

    const prompt = document.getElementById('pg-prompt').value.trim();
    if (!prompt) {
        showToast('Please enter a prompt');
        return;
    }

    const temperature = parseFloat(document.getElementById('pg-temperature').value);
    const top_k = parseInt(document.getElementById('pg-topk').value);
    const max_tokens = parseInt(document.getElementById('pg-maxtokens').value);

    // Show loading
    btn.disabled = true;
    btn.textContent = 'Generating...';
    output.innerHTML = '<span class="pg-placeholder pg-generating">Generating</span>';
    statsCard.style.display = 'none';

    try {
        const result = await apiPost('/api/generate', {
            prompt, temperature, top_k, max_tokens
        });

        if (!result.ok) {
            output.innerHTML = `<span style="color:var(--red)">${result.error}</span>`;
            showToast('Generation failed: ' + result.error);
            return;
        }

        // Display result with prompt highlighted
        const text = result.text || '';
        const promptEnd = text.indexOf(prompt) >= 0
            ? text.indexOf(prompt) + prompt.length
            : prompt.length;

        const promptPart = text.substring(0, promptEnd);
        const generatedPart = text.substring(promptEnd);

        output.innerHTML =
            `<span class="pg-prompt-part">${escapeHtml(promptPart)}</span>` +
            `<span class="pg-generated-part">${escapeHtml(generatedPart)}</span>`;

        // Show stats
        statsCard.style.display = 'block';
        document.getElementById('pg-stats-content').innerHTML = `
            <div style="display:flex;gap:24px;flex-wrap:wrap">
                <div><span style="color:var(--text-muted)">Tokens generated:</span> ${result.tokens_generated}</div>
                <div><span style="color:var(--text-muted)">Model params:</span> ${result.model_params ? result.model_params.toLocaleString() : '--'}</div>
                <div><span style="color:var(--text-muted)">Temperature:</span> ${temperature}</div>
                <div><span style="color:var(--text-muted)">Top-K:</span> ${top_k}</div>
            </div>
        `;
    } catch (e) {
        output.innerHTML = `<span style="color:var(--red)">Error: ${e.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate';
    }
}

async function retrainModel() {
    if (!confirm('This will retrain the model from the current baseline. It may take a few minutes. Continue?')) return;

    showToast('Retraining started... This may take a few minutes.');
    const output = document.getElementById('pg-output');
    output.innerHTML = '<span class="pg-placeholder pg-generating">Retraining model, please wait</span>';

    try {
        const result = await apiPost('/api/retrain');
        if (result.ok) {
            showToast('Model retrained successfully!');
            output.innerHTML = '<span style="color:var(--green)">Model retrained! Try generating text.</span>';
            loadPlayground(); // Refresh model info
        } else {
            showToast('Retrain failed: ' + result.error);
            output.innerHTML = `<span style="color:var(--red)">Retrain failed: ${result.error}</span>`;
        }
    } catch (e) {
        showToast('Retrain error: ' + e.message);
        output.innerHTML = `<span style="color:var(--red)">Error: ${e.message}</span>`;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// === Auto Refresh ===
function startAutoRefresh() {
    refreshInterval = setInterval(async () => {
        // Always update status badge
        await updateStatusBadge();
        // Refresh current page data
        const activePage = document.querySelector('.nav button.active')?.dataset.page;
        if (activePage === 'overview') await loadOverview();
    }, 60000);
}

async function updateStatusBadge() {
    try {
        const status = await apiGet('/api/status');
        const badge = document.getElementById('status-badge');
        const label = status.status.charAt(0).toUpperCase() + status.status.slice(1);
        const extra = status.status === 'running' ? ` (${status.total_experiments} runs)` : '';
        badge.textContent = label + extra;
        badge.className = 'status-badge status-' + status.status;
    } catch (e) {}
}

// === Init ===
loadOverview();
startAutoRefresh();
