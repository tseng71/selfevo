// SelfEvo Static Dashboard - Reads from data.json

let DATA = null;
let charts = {};

async function loadData() {
    try {
        const res = await fetch('data.json?' + Date.now());
        DATA = await res.json();
        return true;
    } catch (e) {
        console.error('Failed to load data.json:', e);
        return false;
    }
}

// Navigation
document.querySelectorAll('.nav button').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('page-' + btn.dataset.page).classList.add('active');
        const page = btn.dataset.page;
        if (page === 'overview') renderOverview();
        else if (page === 'history') renderHistory();
        else if (page === 'trends') renderTrends();
    });
});

// Overview
function renderOverview() {
    if (!DATA) return;
    const s = DATA.status;

    document.getElementById('update-time').textContent =
        'Updated: ' + formatTime(s.last_updated);

    document.getElementById('best-val-loss').textContent =
        s.best_val_loss !== null ? s.best_val_loss.toFixed(4) : '--';
    document.getElementById('best-experiment').textContent = s.best_experiment || '';
    document.getElementById('total-experiments').textContent = s.total_experiments;
    document.getElementById('today-count').textContent = 'Today: ' + s.today_experiments;
    document.getElementById('kdc-stats').textContent =
        `${s.keeps} / ${s.discards} / ${s.crashes}`;

    const total = s.keeps + s.discards + s.crashes;
    document.getElementById('keep-rate').textContent =
        total > 0 ? `Keep rate: ${(s.keeps / total * 100).toFixed(1)}%` : '';

    // Improvement from first to best
    const exps = DATA.experiments;
    if (exps.length > 0 && s.best_val_loss !== null) {
        const first = exps.find(e => e.val_loss !== null);
        if (first) {
            const pct = ((first.val_loss - s.best_val_loss) / first.val_loss * 100).toFixed(1);
            document.getElementById('improvement').textContent = pct + '%';
            document.getElementById('improvement-sub').textContent =
                `${first.val_loss.toFixed(4)} → ${s.best_val_loss.toFixed(4)}`;
        }
    }

    // Last experiment
    const lastCard = document.getElementById('last-experiment-card');
    if (s.last_experiment) {
        lastCard.style.display = 'block';
        const e = s.last_experiment;
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
    renderOverviewChart();
}

function renderOverviewChart() {
    const data = DATA.trends.val_loss_trend || [];
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
            labels,
            datasets: [{ label: 'val_loss', data: values, backgroundColor: colors, borderRadius: 2 }]
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

// History
function renderHistory() {
    if (!DATA) return;
    const statusFilter = document.getElementById('filter-status').value;
    const classFilter = document.getElementById('filter-class').value;

    let exps = [...DATA.experiments].reverse();
    if (statusFilter) exps = exps.filter(e => e.status === statusFilter);
    if (classFilter) exps = exps.filter(e => e.experiment_class === classFilter);

    const tbody = document.getElementById('history-tbody');
    tbody.innerHTML = exps.map((e, i) => {
        const hypothesis = e.hypothesis || '';
        const patch = e.patch_summary || '';
        const summary = hypothesis
            ? `<div class="summary-hypothesis">${escapeHtml(hypothesis)}</div><div class="summary-patch">${escapeHtml(patch)}</div>`
            : `<div class="summary-patch">${escapeHtml(patch)}</div>`;
        return `
        <tr>
            <td>${DATA.experiments.length - DATA.experiments.indexOf(exps.length > i ? DATA.experiments.find(x => x.experiment_id === e.experiment_id) || e : e)}</td>
            <td>${formatTime(e.timestamp)}</td>
            <td>${e.experiment_class || '--'}</td>
            <td class="summary-cell">${summary}</td>
            <td>${e.val_loss !== null && e.val_loss !== undefined ? e.val_loss.toFixed(4) : '--'}</td>
            <td><span class="badge badge-${e.status}">${e.status}</span></td>
            <td class="reason-text">${e.judge_reason || '--'}</td>
        </tr>`;
    }).join('');
}

document.getElementById('filter-status').addEventListener('change', renderHistory);
document.getElementById('filter-class').addEventListener('change', renderHistory);

// Trends
function renderTrends() {
    if (!DATA) return;
    const trends = DATA.trends;

    // val_loss scatter
    if (charts.valloss) charts.valloss.destroy();
    const vlData = trends.val_loss_trend || [];
    const allPoints = vlData.map((d, i) => ({ x: i + 1, y: d.val_loss }));

    charts.valloss = new Chart(document.getElementById('trend-valloss').getContext('2d'), {
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
                    data: (trends.best_progression || []).map(d => ({
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
            datasets: [{ label: 'Keep Rate', data: krData.map(d => d.rate), borderColor: 'rgba(34,197,94,0.8)', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, tension: 0.3 }]
        },
        options: {
            responsive: true, plugins: { legend: { display: false } },
            scales: { x: { grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }, y: { min: 0, max: 1, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } } }
        }
    });

    // Crash rate
    if (charts.crashrate) charts.crashrate.destroy();
    const crData = trends.crash_rate_trend || [];
    charts.crashrate = new Chart(document.getElementById('trend-crashrate').getContext('2d'), {
        type: 'line',
        data: {
            labels: crData.map(d => d.index + 1),
            datasets: [{ label: 'Crash Rate', data: crData.map(d => d.rate), borderColor: 'rgba(239,68,68,0.8)', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: 0.3 }]
        },
        options: {
            responsive: true, plugins: { legend: { display: false } },
            scales: { x: { grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }, y: { min: 0, max: 1, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } } }
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
                { label: 'Keep Rate', data: csData.map(d => d.keep_rate), backgroundColor: 'rgba(34,197,94,0.6)' },
                { label: 'Crash Rate', data: csData.map(d => d.crash_rate), backgroundColor: 'rgba(239,68,68,0.6)' }
            ]
        },
        options: {
            responsive: true, plugins: { legend: { labels: { color: '#8b8d97' } } },
            scales: { x: { grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } }, y: { min: 0, max: 1, grid: { color: '#2d3140' }, ticks: { color: '#8b8d97' } } }
        }
    });
}

// Helpers
function formatTime(ts) {
    if (!ts) return '--';
    const d = new Date(ts);
    return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-refresh every 5 minutes
async function refresh() {
    if (await loadData()) {
        const activePage = document.querySelector('.nav button.active')?.dataset.page;
        if (activePage === 'overview') renderOverview();
        else if (activePage === 'history') renderHistory();
        else if (activePage === 'trends') renderTrends();
    }
}

// Init
(async () => {
    if (await loadData()) {
        renderOverview();
    }
    setInterval(refresh, 300000);
})();
