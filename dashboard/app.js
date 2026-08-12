// State Management for the Dashboard
const state = {
    problemType: null,     // 'regression' or 'classification'
    testBundle: {},        // Map of key -> Plotly JSON for the test set
    trainBundle: {},       // Map of key -> Plotly JSON for the training set
    tables: {},            // Map of test/train split table data
    descriptions: {}       // Map of key -> {title, description, insights}
};

// Independent pagination states for train and test dataset tables
const tableStates = {
    test: {
        currentPage: 1,
        rowsPerPage: 7
    },
    train: {
        currentPage: 1,
        rowsPerPage: 7
    }
};

// DOM Elements
const testGrid = document.getElementById('test-grid');
const trainGrid = document.getElementById('train-grid');
const testSection = document.getElementById('test-section');
const trainSection = document.getElementById('train-section');
const emptyState = document.getElementById('empty-state');
const statusBadge = document.getElementById('status-badge');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toast-message');

// Sizing helper maps for human-readable labels in English
const chartNames = {
    'act_vs_pred': 'Actual vs. Predicted',
    'res_dist': 'Residuals Distribution',
    'res_vs_pred': 'Residuals vs. Predicted',
    'res_vs_act': 'Residuals vs. Actual',
    'qq': 'Normal Q-Q Plot',
    'scale_loc': 'Scale-Location Plot',
    'post_pred': 'Posterior Predictive Density',
    'leverage': 'Residuals vs. Leverage',
    'outliers': 'Outlier Feature Analysis',
    'metrics': 'Performance Metrics Summary',
    'cm': 'Confusion Matrix',
    'class_dist': 'Class Distribution',
    'prob_dist': 'Probability Confidence Distribution',
    'roc': 'Receiver Operating Characteristic (ROC)',
    'pr': 'Precision-Recall Curve',
    'misclass': 'Misclassification Feature Analysis',
    'calibration': 'Calibration Curve (Reliability)',
    'threshold': 'Decision Threshold Analysis',
    'shap_summary': 'Global Feature Importance (SHAP)',
    'shap_beeswarm': 'SHAP Beeswarm Plot',
    'shap_waterfall': 'Local Explanation (Waterfall)'
};

// Initialize listeners & Load dynamic bundle
function init() {
    loadActiveBundle();
}

// Show Custom Toast alerts
function showToast(message) {
    toastMessage.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Load Active Bundle from disk (generated via Python's `save_dashboard_bundle()`)
async function loadActiveBundle() {
    // Load graph descriptions metadata in parallel
    try {
        const descResponse = await fetch('graph_descriptions.json?t=' + Date.now());
        if (descResponse.ok) {
            state.descriptions = await descResponse.json();
        }
    } catch (err) {
        console.warn("Could not load graph_descriptions.json: ", err);
    }

    try {
        const response = await fetch('active_bundle.json?t=' + Date.now());
        if (!response.ok) {
            throw new Error(`Failed to load file: ${response.statusText}`);
        }
        const data = await response.json();
        parseAndLoadBundle(data);
        showToast("Dashboard successfully synced with active model!");
    } catch (err) {
        console.warn("Could not load active_bundle.json: ", err);
        // Show instruction empty state if bundle isn't available
        emptyState.style.display = 'flex';
        testSection.style.display = 'none';
        trainSection.style.display = 'none';
        statusBadge.textContent = "Status: Waiting for Bundle";
    }
}

// Centralized parsing: supports both flat (legacy) bundles and nested full-subsets bundles
function parseAndLoadBundle(bundle) {
    state.testBundle = {};
    state.trainBundle = {};

    // 1. Detect Problem Type
    if (bundle.problem_type) {
        state.problemType = bundle.problem_type;
    } else {
        // Fallback: detect type by scanning keys
        const keys = Object.keys(bundle);
        const isClf = keys.includes('cm') || keys.includes('roc') || keys.includes('pr') || keys.includes('class_dist');
        state.problemType = isClf ? 'classification' : 'regression';
    }

    // 2. Parse Test vs Train plots
    if (bundle.test && bundle.train) {
        // Nested structure format (double-subset bundle)
        state.testBundle = bundle.test;
        state.trainBundle = bundle.train;
    } else {
        // Flat structure format: Treat all keys as test set plots, and train is empty
        Object.entries(bundle).forEach(([key, value]) => {
            if (key !== 'problem_type') {
                state.testBundle[key] = value;
            }
        });
        state.trainBundle = {};
    }

    // Parse tables
    state.tables = bundle.tables || {};

    // Update status badge
    const ptLabel = state.problemType === 'regression' ? 'Regression' : 'Classification';
    statusBadge.textContent = `Active Model: ${ptLabel}`;
    statusBadge.style.background = state.problemType === 'regression' ? 'rgba(59, 130, 246, 0.08)' : 'rgba(168, 85, 247, 0.08)';
    statusBadge.style.borderColor = state.problemType === 'regression' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(168, 85, 247, 0.2)';
    statusBadge.style.color = state.problemType === 'regression' ? '#3b82f6' : '#a855f7';

    // 3. Render Dashboard grids
    renderGrids();
}

// Render test and training grids
function renderGrids() {
    const hasTest = Object.keys(state.testBundle).length > 0;
    const hasTrain = Object.keys(state.trainBundle).length > 0;

    if (!hasTest && !hasTrain) {
        emptyState.style.display = 'flex';
        testSection.style.display = 'none';
        trainSection.style.display = 'none';
        return;
    }

    emptyState.style.display = 'none';
    testSection.style.display = 'flex';

    // Build Test Grid
    buildGrid(testGrid, state.testBundle, 'test');

    // Build Training Grid if we have training data plots available
    if (hasTrain) {
        trainSection.style.display = 'flex';
        buildGrid(trainGrid, state.trainBundle, 'train');
    } else {
        trainSection.style.display = 'none';
    }

    // Render the inline tables
    renderSubsetTable('test');
    renderSubsetTable('train');

    // Dispatch resize to let Plotly fit itself to the 2-column container immediately
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 150);
}

// Helper to construct and plot inside a grid element
function buildGrid(gridContainer, subsetBundle, subsetName) {
    gridContainer.innerHTML = '';

    // Order of plots for neat comparisons
    const orderedKeys = state.problemType === 'regression'
        ? ['metrics', 'act_vs_pred', 'res_dist', 'res_vs_pred', 'res_vs_act', 'qq', 'scale_loc', 'post_pred', 'leverage', 'outliers', 'shap_summary', 'shap_beeswarm']
        : ['metrics', 'cm', 'class_dist', 'prob_dist', 'roc', 'pr', 'calibration', 'threshold', 'misclass', 'shap_summary', 'shap_beeswarm'];

    orderedKeys.forEach(key => {
        const plotData = subsetBundle[key];
        if (!plotData || plotData.error) return; // Skip if plot wasn't computed

        const id = `${subsetName}-${key}`;
        const descMeta = state.descriptions[key] || {};
        const title = descMeta.title || chartNames[key] || key.toUpperCase();
        const descriptionText = descMeta.description || '';
        const insightsText = descMeta.insights || '';
        const exampleText = descMeta.example || '';
        const hasInfo = Boolean(descriptionText || insightsText || exampleText);

        // Create Card HTML
        const card = document.createElement('div');
        card.className = 'chart-card';
        card.innerHTML = `
            <div class="card-header">
                <span class="card-title">
                    ${title}
                    ${hasInfo ? `
                    <span class="info-btn-wrapper">
                        <button class="info-btn" aria-label="Chart information">i</button>
                        <div class="info-tooltip-popover">
                            <div class="tooltip-block">
                                ${descriptionText ? `
                                <div class="tooltip-section">
                                    <span class="tooltip-label">What it represents:</span>
                                    <span>${descriptionText}</span>
                                </div>` : ''}
                                ${(descriptionText && insightsText) ? `<div class="tooltip-divider"></div>` : ''}
                                ${insightsText ? `
                                <div class="tooltip-section">
                                    <span class="tooltip-label">Information & Insights:</span>
                                    <span>${insightsText}</span>
                                </div>` : ''}
                                ${exampleText ? `
                                <div class="tooltip-divider"></div>
                                <div class="tooltip-section">
                                    <span class="tooltip-label">Practical Example:</span>
                                    <div class="tooltip-example">${exampleText}</div>
                                </div>` : ''}
                            </div>
                        </div>
                    </span>` : ''}
                </span>
            </div>
            <div class="chart-container" id="plotly-${id}"></div>
        `;
        gridContainer.appendChild(card);

        // Render Plotly
        try {
            const layout = plotData.layout || {};

            // Clear redundant nested canvas titles to avoid overlap with beautiful HTML card titles
            if (layout.title) {
                layout.title = '';
            }

            // Adjust design configurations to blend seamlessly in our dashboard theme
            layout.paper_bgcolor = 'transparent';
            layout.plot_bgcolor = 'transparent';
            layout.autosize = true;

            if (key === 'shap_summary' && subsetBundle.shap_raw) {
                renderShapSummary(`plotly-${id}`, subsetBundle.shap_raw, null);
            } else if (key === 'shap_beeswarm' && subsetBundle.shap_raw) {
                renderShapBeeswarm(`plotly-${id}`, subsetBundle.shap_raw, null);
            } else {
                Plotly.newPlot(`plotly-${id}`, plotData.data, layout, {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    scrollZoom: false
                });
            }

            // Set up cross-filtering on ALL Plotly charts (box/lasso select)
            // Skip SHAP charts — they respond to selection but don't emit it
            if (key !== 'shap_summary' && key !== 'shap_beeswarm') {
                const graphDiv = document.getElementById(`plotly-${id}`);
                if (!graphDiv) return;
                graphDiv.on('plotly_selected', function (eventData) {
                    if (eventData && eventData.points && eventData.points.length > 0) {
                        const selectedIndices = [];
                        eventData.points.forEach(p => {
                            // Use customdata if available, otherwise fallback to pointIndex
                            const idx = (p.customdata !== undefined && p.customdata !== null)
                                ? p.customdata
                                : p.pointIndex;
                            selectedIndices.push(idx);
                        });
                        applyCrossFilter(subsetName, selectedIndices, `plotly-${id}`);
                    } else {
                        applyCrossFilter(subsetName, null, `plotly-${id}`);
                    }
                });

                graphDiv.on('plotly_deselect', function () {
                    applyCrossFilter(subsetName, null, `plotly-${id}`);
                });
            }

        } catch (err) {
            console.error(`Failed to plot: ${id}`, err);
            document.getElementById(`plotly-${id}`).innerHTML = `
                <div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:0.85rem">
                    Failed to render interactive chart.
                </div>`;
        }
    });
}

// Generic cross-filtering synchronization
function applyCrossFilter(subsetName, selectedIndices, sourceChartId) {
    const prefix = `${subsetName}-`;
    const bundle = subsetName === 'test' ? state.testBundle : state.trainBundle;

    Object.keys(bundle).forEach(key => {
        const chartId = `plotly-${prefix}${key}`;
        if (chartId === sourceChartId) return; // Skip source chart to prevent feedback loops

        const graphDiv = document.getElementById(chartId);
        if (!graphDiv || !graphDiv.data) return;

        let hasChanges = false;
        const selectedPointsArray = [];

        graphDiv.data.forEach((trace, traceIdx) => {
            let selectedpoints = null;
            if (trace.type === 'scatter' && (trace.mode || '').includes('markers') && trace.customdata) {
                if (selectedIndices && selectedIndices.length > 0) {
                    const localIndices = [];
                    const customdata = trace.customdata || [];
                    for (let i = 0; i < customdata.length; i++) {
                        if (selectedIndices.includes(customdata[i])) {
                            localIndices.push(i);
                        }
                    }
                    selectedpoints = localIndices;
                }
                hasChanges = true;
            }
            selectedPointsArray.push(selectedpoints);
        });

        if (hasChanges) {
            // Programmatically apply selection states to each trace using Plotly.restyle
            selectedPointsArray.forEach((val, idx) => {
                Plotly.restyle(graphDiv, { selectedpoints: [val] }, [idx]);
            });
        }
    });

    // Update the inline dataset table with the selected indices
    if (tableStates[subsetName]) {
        tableStates[subsetName].selectedIndices = selectedIndices;
        tableStates[subsetName].currentPage = 1; // Reset to page 1 for the filtered rows
        renderSubsetTable(subsetName);
    }

    // Update SHAP charts with the current selection
    if (bundle.shap_raw) {
        const indexMap = buildShapIndexMap(bundle.shap_raw);
        renderShapSummary(`plotly-${prefix}shap_summary`, bundle.shap_raw, selectedIndices, indexMap);
        renderShapBeeswarm(`plotly-${prefix}shap_beeswarm`, bundle.shap_raw, selectedIndices, indexMap);
    }
}

// Render dynamic inline tables with columnar orientation
function renderSubsetTable(subsetName) {
    const tableCard = document.getElementById(`${subsetName}-table-card`);
    const tableElement = document.getElementById(`${subsetName}-dataset-table`);
    const footerElement = document.getElementById(`${subsetName}-table-footer`);

    if (!tableCard || !tableElement || !footerElement) return;

    const tableData = state.tables ? state.tables[subsetName] : null;
    if (!tableData || !tableData.columns || !tableData.data || tableData.data.length === 0) {
        tableCard.style.display = 'none';
        return;
    }

    tableCard.style.display = 'flex';

    const tState = tableStates[subsetName];

    // Filter rows based on Plotly active selected indices if any are active
    let activeIndex = tableData.index;
    let activeData = tableData.data;

    if (tState.selectedIndices && tState.selectedIndices.length > 0) {
        const filteredIndex = [];
        const filteredData = [];
        for (let i = 0; i < tableData.index.length; i++) {
            // Compare against the actual row index value, not array position
            if (tState.selectedIndices.includes(tableData.index[i])) {
                filteredIndex.push(tableData.index[i]);
                filteredData.push(tableData.data[i]);
            }
        }
        activeIndex = filteredIndex;
        activeData = filteredData;
    }

    const totalRows = activeData.length;
    const totalPages = Math.ceil(totalRows / tState.rowsPerPage);

    // Clamp current page
    if (tState.currentPage > totalPages) tState.currentPage = totalPages;
    if (tState.currentPage < 1) tState.currentPage = 1;

    const startIndex = (tState.currentPage - 1) * tState.rowsPerPage;
    const endIndex = Math.min(startIndex + tState.rowsPerPage, totalRows);
    const pageData = activeData.slice(startIndex, endIndex);

    // 1. Build Table Headers
    let headerHTML = '<thead><tr>';
    headerHTML += '<th style="width: 80px;">Index</th>';

    // Track column indexes for targets, predictions, residuals, errors, match badges
    let actualIdx = -1;
    let predictedIdx = -1;
    let residualIdx = -1;
    let absErrorIdx = -1;
    let correctIdx = -1;

    tableData.columns.forEach((col, idx) => {
        if (col === '__actual__') {
            actualIdx = idx;
            headerHTML += '<th>Actual Target</th>';
        } else if (col === '__predicted__') {
            predictedIdx = idx;
            headerHTML += '<th>Prediction</th>';
        } else if (col === '__residual__') {
            residualIdx = idx;
            headerHTML += '<th>Residual</th>';
        } else if (col === '__abs_error__') {
            absErrorIdx = idx;
            headerHTML += '<th>|Error|</th>';
        } else if (col === '__correct__') {
            correctIdx = idx;
            headerHTML += '<th>Match</th>';
        } else {
            // Feature column
            headerHTML += `<th>${col}</th>`;
        }
    });
    headerHTML += '</tr></thead>';

    // 2. Build Table Body
    let bodyHTML = '<tbody>';
    pageData.forEach((row, rowIdx) => {
        const originalIndex = activeIndex[startIndex + rowIdx];
        const isSelected = tState.selectedIndices && tState.selectedIndices.length === 1 && tState.selectedIndices[0] === originalIndex;
        const highlightStyle = isSelected ? 'background: rgba(59,130,246,0.12); outline: 1px solid rgba(59,130,246,0.4);' : '';
        bodyHTML += `<tr style="${highlightStyle}">`;
        bodyHTML += `<td class="row-index">${originalIndex}</td>`;

        row.forEach((val, colIdx) => {
            let formattedVal = val;
            if (typeof val === 'number') {
                formattedVal = val.toFixed(4);
            } else if (val === null || val === undefined) {
                formattedVal = '-';
            }

            if (colIdx === actualIdx) {
                bodyHTML += `<td class="col-actual">${formattedVal}</td>`;
            } else if (colIdx === predictedIdx) {
                bodyHTML += `<td class="col-predicted">${formattedVal}</td>`;
            } else if (colIdx === residualIdx) {
                const className = val >= 0 ? 'col-residual-pos' : 'col-residual-neg';
                const sign = val >= 0 ? '+' : '';
                bodyHTML += `<td class="${className}">${sign}${formattedVal}</td>`;
            } else if (colIdx === absErrorIdx) {
                // Background error intensity heatmap (max out at soft-red opacity 0.25)
                const absVal = Math.abs(val);
                const opacity = Math.min(absVal * 0.1, 0.25);
                bodyHTML += `<td class="col-error-heat" style="background: rgba(239, 68, 68, ${opacity});">${formattedVal}</td>`;
            } else if (colIdx === correctIdx) {
                const badgeClass = val ? 'badge-correct' : 'badge-incorrect';
                const badgeLabel = val ? '✅ Correct' : '❌ Incorrect';
                bodyHTML += `<td><span class="${badgeClass}">${badgeLabel}</span></td>`;
            } else {
                bodyHTML += `<td>${formattedVal}</td>`;
            }
        });
        bodyHTML += '</tr>';
    });
    bodyHTML += '</tbody>';

    tableElement.innerHTML = headerHTML + bodyHTML;

    // 3. Build Table Footer (Pagination)
    const isFiltered = tState.selectedIndices && tState.selectedIndices.length > 0;
    let footerHTML = `
        <div style="display:flex;align-items:center;gap:8px;">
            <span>Showing <b>${startIndex + 1}</b> to <b>${endIndex}</b> of <b>${totalRows}</b> samples</span>
            ${isFiltered ? `<button class="pagination-btn" style="background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.3);color:#ef4444;font-size:0.75rem;padding:2px 8px;" onclick="applyCrossFilter('${subsetName}', null, 'table')">✕ Clear Filter</button>` : ''}
        </div>
        <div class="pagination-controls">
            <button class="pagination-btn" id="${subsetName}-btn-prev" ${tState.currentPage === 1 ? 'disabled' : ''}>Previous</button>
            <span class="pagination-info-pages">Page <b>${tState.currentPage}</b> of <b>${totalPages}</b></span>
            <button class="pagination-btn" id="${subsetName}-btn-next" ${tState.currentPage === totalPages ? 'disabled' : ''}>Next</button>
        </div>
    `;
    footerElement.innerHTML = footerHTML;

    // 4. Hook up Pagination Button Event Listeners
    document.getElementById(`${subsetName}-btn-prev`).addEventListener('click', () => {
        if (tState.currentPage > 1) {
            tState.currentPage--;
            renderSubsetTable(subsetName);
        }
    });

    document.getElementById(`${subsetName}-btn-next`).addEventListener('click', () => {
        if (tState.currentPage < totalPages) {
            tState.currentPage++;
            renderSubsetTable(subsetName);
        }
    });
}

// Auto-boot application
document.addEventListener('DOMContentLoaded', init);

// =============================================================
// SHAP Client-Side Rendering (Plotly from bundle shap_raw)
// =============================================================

/** Build map: original dataset index → position in shap_values array. */
function buildShapIndexMap(shapRaw) {
    const map = {};
    if (shapRaw.eval_index) {
        shapRaw.eval_index.forEach((origIdx, pos) => { map[origIdx] = pos; });
    }
    return map;
}

/** Resolve selected original indices to positions inside the shap_values array. */
function resolveShapPositions(shapRaw, selectedIndices, indexMap) {
    if (!selectedIndices || selectedIndices.length === 0) return null;
    if (!shapRaw.eval_index || shapRaw.eval_index.length === 0) return selectedIndices;
    const positions = [];
    selectedIndices.forEach(origIdx => {
        const pos = indexMap[origIdx];
        if (pos !== undefined) positions.push(pos);
    });
    return positions.length > 0 ? positions : null;
}

/**
 * Render SHAP Global Feature Importance (bar chart).
 * When selectedIndices is provided, recomputes mean|SHAP| only for those rows.
 */
function renderShapSummary(containerId, shapRaw, selectedIndices, indexMap) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const numFeatures = shapRaw.feature_names.length;
    let meanAbsShap;

    const positions = resolveShapPositions(shapRaw, selectedIndices, indexMap || buildShapIndexMap(shapRaw));

    if (positions && positions.length > 0) {
        meanAbsShap = new Array(numFeatures).fill(0);
        let count = 0;
        positions.forEach(pos => {
            if (pos >= 0 && pos < shapRaw.shap_values.length && shapRaw.shap_values[pos]) {
                for (let f = 0; f < numFeatures; f++) {
                    meanAbsShap[f] += Math.abs(shapRaw.shap_values[pos][f]);
                }
                count++;
            }
        });
        if (count > 0) for (let f = 0; f < numFeatures; f++) meanAbsShap[f] /= count;
    } else {
        meanAbsShap = shapRaw.mean_abs_shap;
    }

    // Top 10, ascending (SHAP convention: largest bar at the bottom)
    let items = shapRaw.feature_names.map((name, i) => ({ name, val: meanAbsShap[i] }));
    items.sort((a, b) => a.val - b.val);
    items = items.slice(-10);

    const traces = [{
        type: 'bar',
        x: items.map(d => d.val),
        y: items.map(d => d.name),
        orientation: 'h',
        marker: { color: '#008bfb' },
        hovertemplate: '<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
    }];

    const layout = {
        margin: { l: 10, r: 20, t: 10, b: 50 },
        xaxis: {
            title: { text: 'Mean |SHAP value| (average impact on model output)', font: { size: 11 } },
            zeroline: true, zerolinecolor: '#cccccc', zerolinewidth: 1,
            gridcolor: '#eeeeee', tickfont: { color: '#444444' }
        },
        yaxis: {
            automargin: true, gridcolor: '#eeeeee',
            tickfont: { color: '#444444' }, zeroline: false
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#ffffff',
        autosize: true,
        font: { color: '#333333', size: 11, family: 'Arial, sans-serif' },
        showlegend: false
    };

    try { Plotly.purge(containerId); } catch (e) {}
    Plotly.newPlot(containerId, traces, layout, { staticPlot: true, responsive: true });
}

// Stable per-chart jitter so dots don't jump on every filter update
const _beeswarmJitterCache = {};
function _getJitter(key, n) {
    if (!_beeswarmJitterCache[key]) {
        const a = new Float32Array(n);
        for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 0.35;
        _beeswarmJitterCache[key] = a;
    }
    return _beeswarmJitterCache[key];
}

/**
 * Render SHAP Beeswarm Plot.
 * Each row = one feature; each dot = one sample at x=SHAP value, coloured by normalised feature value.
 * Selected samples are opaque; non-selected are heavily dimmed.
 */
function renderShapBeeswarm(containerId, shapRaw, selectedIndices, indexMap) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const numSamples = shapRaw.shap_values.length;
    const jitter = _getJitter(containerId, numSamples);

    // Top 10 features by mean abs shap, ascending
    let featureOrder = shapRaw.feature_names.map((name, i) => ({ name, idx: i, score: shapRaw.mean_abs_shap[i] }));
    featureOrder.sort((a, b) => a.score - b.score);
    featureOrder = featureOrder.slice(-10);

    const imap = indexMap || buildShapIndexMap(shapRaw);
    const selectedPositions = new Set();
    if (selectedIndices && selectedIndices.length > 0) {
        const pos = resolveShapPositions(shapRaw, selectedIndices, imap);
        if (pos) pos.forEach(p => selectedPositions.add(p));
    }
    const hasSelection = selectedPositions.size > 0;

    const traces = [];
    featureOrder.forEach((feat, yPos) => {
        // Per-feature min/max normalisation (SHAP library convention)
        let featMin = Infinity, featMax = -Infinity;
        for (let s = 0; s < numSamples; s++) {
            const v = shapRaw.feature_values ? shapRaw.feature_values[s][feat.idx] : shapRaw.shap_values[s][feat.idx];
            if (v < featMin) featMin = v;
            if (v > featMax) featMax = v;
        }
        const featRange = featMax - featMin || 1;

        const xSel = [], ySel = [], cSel = [], tipSel = [];
        const xDim = [], yDim = [], cDim = [], tipDim = [];

        for (let s = 0; s < numSamples; s++) {
            const sv  = shapRaw.shap_values[s][feat.idx];
            const fv  = shapRaw.feature_values ? shapRaw.feature_values[s][feat.idx] : sv;
            const yv  = yPos + jitter[s];
            const cv  = (fv - featMin) / featRange;
            const tip = `Row ${shapRaw.eval_index ? shapRaw.eval_index[s] : s}<br>${feat.name}: SHAP=${sv.toFixed(4)}, Val=${fv.toFixed ? fv.toFixed(4) : fv}`;

            if (!hasSelection || selectedPositions.has(s)) {
                xSel.push(sv); ySel.push(yv); cSel.push(cv); tipSel.push(tip);
            } else {
                xDim.push(sv); yDim.push(yv); cDim.push(cv); tipDim.push(tip);
            }
        }

        const markerBase = {
            size: 5, cmin: 0, cmax: 1,
            colorscale: [[0, '#008bfb'], [1, '#ff0052']],
            reversescale: false, line: { width: 0 }
        };

        if (xDim.length > 0) {
            traces.push({
                type: 'scatter', mode: 'markers', x: xDim, y: yDim,
                marker: { ...markerBase, color: cDim, opacity: 0.07, showscale: false },
                text: tipDim, hovertemplate: '%{text}<extra></extra>', showlegend: false
            });
        }
        if (xSel.length > 0) {
            traces.push({
                type: 'scatter', mode: 'markers', x: xSel, y: ySel,
                marker: {
                    ...markerBase, color: cSel, opacity: 0.85,
                    showscale: yPos === 0,
                    colorbar: yPos === 0 ? {
                        title: { text: 'Feature value', font: { size: 10, color: '#444444' } },
                        ticks: 'outside', tickvals: [0, 1], ticktext: ['Low', 'High'],
                        thickness: 10, len: 0.5, outlinewidth: 0,
                        tickfont: { color: '#444444', size: 10 }
                    } : undefined
                },
                text: tipSel, hovertemplate: '%{text}<extra></extra>', showlegend: false
            });
        }
    });

    const layout = {
        margin: { l: 10, r: 60, t: 10, b: 50 },
        xaxis: {
            title: { text: 'SHAP value (impact on model output)', font: { size: 11 } },
            zeroline: true, zerolinecolor: '#aaaaaa', zerolinewidth: 1.5,
            gridcolor: '#eeeeee', tickfont: { color: '#444444' }
        },
        yaxis: {
            tickvals: featureOrder.map((_, i) => i),
            ticktext: featureOrder.map(f => f.name),
            automargin: true, zeroline: false, showgrid: false,
            tickfont: { color: '#333333' }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#ffffff',
        autosize: true,
        font: { color: '#333333', size: 11, family: 'Arial, sans-serif' },
        hovermode: 'closest'
    };

    try { Plotly.purge(containerId); } catch (e) {}
    Plotly.newPlot(containerId, traces, layout, { staticPlot: true, responsive: true });
}
