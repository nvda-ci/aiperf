// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AIPerf Interactive HTML Export - Main Application
 *
 * This module provides client-side interactivity for the exported HTML report.
 * All data is embedded in the HTML file, enabling fully offline operation.
 */

(function () {
    'use strict';

    // Theme colors
    const LIGHT_COLORS = {
        primary: '#76B900',
        background: '#FFFFFF',
        paper: '#FFFFFF',
        text: '#0a0a0a',
        grid: '#CCCCCC',
        border: '#CCCCCC',
    };

    const DARK_COLORS = {
        primary: '#76B900',
        background: '#1a1a1a',
        paper: '#252525',
        text: '#E8E8E8',
        grid: '#333333',
        border: '#333333',
    };

    const FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif";

    // Semantic colors for experiment classification (matches Python plot_generator.py)
    const NVIDIA_GREEN = '#76B900';
    const NVIDIA_GRAY = '#999999';

    // Seaborn "bright" palette for additional treatments
    const BRIGHT_PALETTE = [
        '#023EFF', // blue
        '#FF7C00', // orange
        '#1AC938', // green
        '#E8000B', // red
        '#8B2BE2', // purple
        '#9F4800', // brown
        '#F14CC1', // pink
        '#FFC400', // yellow
        '#00D7FF', // cyan
    ];

    // =========================================================================
    // State Manager
    // =========================================================================
    const StateManager = {
        _state: {
            mode: 'multi',
            runs: [],
            selectedRuns: [],
            theme: 'light',
            plots: [],
            availableMetrics: {},
            statLabels: {},
            statKeys: [],
            sweptParameters: [],
            _precomputedFigures: {},
        },

        _listeners: [],

        init(data, config) {
            this._state.mode = data.mode || 'multi';
            this._state.theme = config.theme || 'light';
            this._state.availableMetrics = data.availableMetrics || {};
            this._state.statLabels = data.statLabels || {};
            this._state.statKeys = data.statKeys || ['p50', 'avg', 'p90', 'p95', 'p99'];
            this._state.sweptParameters = data.sweptParameters || [];

            if (data.mode === 'multi') {
                this._state.runs = data.runs || [];
                this._state.selectedRuns = data.runs ? data.runs.map((_, i) => i) : [];
            } else {
                this._state.runs = [data];
                this._state.selectedRuns = [0];
            }

            return this;
        },

        getState() {
            return { ...this._state };
        },

        setState(updates) {
            Object.assign(this._state, updates);
            this._notifyListeners();
        },

        subscribe(listener) {
            this._listeners.push(listener);
            return () => {
                this._listeners = this._listeners.filter((l) => l !== listener);
            };
        },

        _notifyListeners() {
            this._listeners.forEach((fn) => fn(this._state));
        },

        getSelectedRuns() {
            return this._state.selectedRuns.map((i) => this._state.runs[i]);
        },

        getRunByIndex(idx) {
            return this._state.runs[idx];
        },

        getPlot(plotId) {
            return this._state.plots.find((p) => p.id === plotId);
        },

        updatePlot(plotId, updates) {
            const plots = this._state.plots.map((p) => {
                if (p.id === plotId) {
                    return { ...p, config: { ...p.config, ...updates } };
                }
                return p;
            });
            this.setState({ plots });
        },

        addPlot(plotConfig) {
            const newId = `custom-${Date.now()}`;
            const newPlot = { id: newId, config: plotConfig, visible: true };
            this.setState({ plots: [...this._state.plots, newPlot] });
            return newId;
        },

        removePlot(plotId) {
            const plots = this._state.plots.filter((p) => p.id !== plotId);
            this.setState({ plots });
        },

        togglePlotVisibility(plotId) {
            const plots = this._state.plots.map((p) => {
                if (p.id === plotId) {
                    return { ...p, visible: !p.visible };
                }
                return p;
            });
            this.setState({ plots });
        },
    };

    // =========================================================================
    // Plot Manager
    // =========================================================================
    const PlotManager = {
        _state: null,
        _plotCache: new Map(),

        init(stateManager) {
            this._state = stateManager;
            return this;
        },

        renderPlot(plotConfig, forceRender = false) {
            const containerId = `plot-${plotConfig.id}`;
            const container = document.getElementById(containerId);

            if (!container) {
                console.error(`Container not found: ${containerId}`);
                return;
            }

            // Check if plot is already rendered (pre-rendered HTML from Python)
            // Skip UNLESS forceRender is true (used when updating plot settings)
            if (!forceRender && (container.classList.contains('js-plotly-plot') || container.querySelector('.js-plotly-plot'))) {
                console.log(`Plot ${plotConfig.id} already rendered, skipping`);
                return;
            }

            // Dynamic generation (for custom plots added by user)
            console.log(`[DEBUG] Generating plot dynamically for: ${plotConfig.id}`);
            const plotData = this._generatePlotData(plotConfig);
            const layout = this._generateLayout(plotConfig);
            const config = this._getPlotlyConfig();

            const hasData = plotData.some(trace => trace.x && trace.x.length > 0);
            if (!hasData) {
                console.warn(`[WARN] No data for plot: ${plotConfig.id}`);
                layout.annotations = layout.annotations || [];
                layout.annotations.push({
                    text: 'No data available for this metric',
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.5,
                    y: 0.5,
                    showarrow: false,
                    font: { size: 14, color: '#999' },
                });
            }

            Plotly.react(containerId, plotData, layout, config);
        },

        updatePlot(plotId, updates) {
            this._state.updatePlot(plotId, updates);
            const plot = this._state.getPlot(plotId);
            if (plot) {
                this.renderPlot(plot, true);  // Force re-render with new settings
            }
        },

        refreshAllPlots() {
            const plots = this._state.getState().plots;
            plots.forEach((plot) => {
                if (plot.visible) {
                    this.renderPlot(plot);
                }
            });
        },

        _generatePlotData(plotConfig) {
            const state = this._state.getState();
            const selectedRuns = this._state.getSelectedRuns();
            const config = plotConfig.config;

            if (state.mode === 'multi') {
                return this._generateMultiRunData(selectedRuns, config);
            } else {
                return this._generateSingleRunData(selectedRuns[0], config);
            }
        },

        _generateMultiRunData(runs, config) {
            if (!runs || runs.length === 0) {
                return [];
            }

            const xMetric = config.xMetric || 'concurrency';
            const yMetric = config.yMetric || 'request_throughput';
            const xStat = config.xStat || 'p50';
            const yStat = config.yStat || 'avg';
            const groupBy = config.groupBy || null;
            const labelBy = config.labelBy || 'concurrency';
            const showLabels = labelBy && labelBy !== 'none';

            const data = runs.map((run) => {
                const xVal = this._getMetricValue(run, xMetric, xStat);
                const yVal = this._getMetricValue(run, yMetric, yStat);
                const group = groupBy ? this._getGroupValue(run, groupBy) : 'All Runs';
                const label = this._getLabelValue(run, labelBy);
                const experimentType = run.metadata.experimentType || null;

                return { x: xVal, y: yVal, group, label, metadata: run.metadata, experimentType };
            });

            const groups = {};
            data.forEach((d) => {
                if (!groups[d.group]) {
                    groups[d.group] = { data: [], experimentType: d.experimentType };
                }
                groups[d.group].data.push(d);
            });

            const groupColors = this._getGroupColors(groups, groupBy);
            const orderedGroupNames = this._getOrderedGroupNames(groups, groupBy);

            const traces = [];
            orderedGroupNames.forEach((groupName) => {
                const groupInfo = groups[groupName];
                const groupData = groupInfo.data;
                const sorted = groupData.sort((a, b) => (a.x || 0) - (b.x || 0));
                const color = groupColors[groupName];

                // Determine plot mode based on plotType and whether labels should be shown
                let mode = config.plotType === 'scatter' ? 'markers' : 'lines+markers';
                if (showLabels) {
                    mode += '+text';
                }

                traces.push({
                    x: sorted.map((d) => d.x),
                    y: sorted.map((d) => d.y),
                    mode,
                    name: groupName,
                    text: sorted.map((d) => d.label),
                    textposition: 'top center',
                    customdata: sorted.map((d) => d.metadata),
                    hovertemplate:
                        '<b>%{text}</b><br>' +
                        'X: %{x}<br>' +
                        'Y: %{y}<br>' +
                        '<extra>%{fullData.name}</extra>',
                    marker: { color, size: 10 },
                    line: { color, width: 2 },
                });
            });

            return traces;
        },

        _getGroupColors(groups, groupBy) {
            const classification = window.__AIPERF_DATA__?.classification;
            const groupNames = Object.keys(groups);
            const groupColors = {};

            // Use semantic coloring when grouping by experimentGroup and classification exists
            const isExperimentGroup = groupBy === 'experimentGroup' || groupBy === 'experiment_group';
            if (isExperimentGroup && classification) {
                const baselines = classification.baselines || [];
                const treatments = classification.treatments || [];

                // Assign gray to all baselines
                baselines.forEach((name) => {
                    if (groups[name]) {
                        groupColors[name] = NVIDIA_GRAY;
                    }
                });

                // Assign green to first treatment, bright palette to rest
                treatments.forEach((name, idx) => {
                    if (groups[name]) {
                        if (idx === 0) {
                            groupColors[name] = NVIDIA_GREEN;
                        } else {
                            groupColors[name] = BRIGHT_PALETTE[(idx - 1) % BRIGHT_PALETTE.length];
                        }
                    }
                });

                // Handle any groups not in classification (shouldn't happen, but fallback)
                const fallbackColors = this._getColorPalette();
                let fallbackIdx = 0;
                groupNames.forEach((name) => {
                    if (!groupColors[name]) {
                        groupColors[name] = fallbackColors[fallbackIdx % fallbackColors.length];
                        fallbackIdx++;
                    }
                });

                return groupColors;
            }

            // Default: use standard color palette
            const colors = this._getColorPalette();
            groupNames.forEach((name, idx) => {
                groupColors[name] = colors[idx % colors.length];
            });

            return groupColors;
        },

        _getOrderedGroupNames(groups, groupBy) {
            const classification = window.__AIPERF_DATA__?.classification;
            const groupNames = Object.keys(groups);

            // Order by classification when grouping by experimentGroup
            const isExperimentGroup = groupBy === 'experimentGroup' || groupBy === 'experiment_group';
            if (isExperimentGroup && classification) {
                const baselines = (classification.baselines || []).filter((n) => groups[n]).sort();
                const treatments = (classification.treatments || []).filter((n) => groups[n]).sort();
                const others = groupNames.filter(
                    (n) => !baselines.includes(n) && !treatments.includes(n)
                ).sort();

                return [...baselines, ...treatments, ...others];
            }

            return groupNames.sort();
        },

        _generateSingleRunData(run, config) {
            const plotType = config.plotType || 'scatter';
            const yMetric = config.yMetric || 'time_to_first_token';
            const xMetric = config.xMetric || 'request_number';
            const state = this._state.getState();
            const theme = state.theme;
            const primaryColor = theme === 'dark' ? DARK_COLORS.primary : LIGHT_COLORS.primary;
            const palette = this._getColorPalette();

            // Handle timeslice plot type
            if (plotType === 'timeslice' && run.timeslices?.data) {
                return this._generateTimesliceData(run.timeslices, yMetric, config.yStat, primaryColor);
            }

            // Handle request data based plots
            if (!run || !run.requests || !run.requests.data) {
                return [];
            }

            const data = run.requests.data;
            const columns = run.requests.columns || [];

            if (yMetric !== 'request_number' && !columns.includes(yMetric)) {
                console.warn(`Y-metric "${yMetric}" not found in data`);
                return [];
            }

            const xValues = [];
            const yValues = [];

            data.forEach((row, idx) => {
                let xVal = xMetric === 'request_number' ? idx + 1 : row[xMetric];
                let yVal = row[yMetric];

                if (xVal !== null && xVal !== undefined && yVal !== null && yVal !== undefined) {
                    xValues.push(xVal);
                    yValues.push(yVal);
                }
            });

            if (plotType === 'area') {
                return [{
                    x: xValues,
                    y: yValues,
                    mode: 'lines',
                    fill: 'tozeroy',
                    type: 'scatter',
                    name: yMetric,
                    line: { color: primaryColor, width: 1 },
                    fillcolor: primaryColor + '40',
                    hovertemplate: `${xMetric}: %{x}<br>${yMetric}: %{y}<extra></extra>`,
                }];
            }

            if (plotType === 'scatter_with_percentiles') {
                return this._generateScatterWithPercentilesData(xValues, yValues, yMetric, xMetric, primaryColor, palette);
            }

            // Default scatter plot
            return [{
                x: xValues,
                y: yValues,
                mode: 'markers',
                type: 'scatter',
                name: yMetric,
                marker: { color: primaryColor, size: 6, opacity: 0.7 },
                hovertemplate: `${xMetric}: %{x}<br>${yMetric}: %{y}<extra></extra>`,
            }];
        },

        _generateTimesliceData(timeslices, yMetric, yStat, color) {
            const data = timeslices.data || [];
            const statSuffix = yStat || 'avg';
            const columnName = `${yMetric}_${statSuffix}`;

            const xValues = [];
            const yValues = [];

            data.forEach((row, idx) => {
                const yVal = row[columnName] ?? row[yMetric];
                if (yVal !== null && yVal !== undefined) {
                    xValues.push(idx);
                    yValues.push(yVal);
                }
            });

            return [{
                x: xValues,
                y: yValues,
                mode: 'lines+markers',
                type: 'scatter',
                name: `${yMetric} (${statSuffix})`,
                line: { color, width: 2 },
                marker: { color, size: 6 },
            }];
        },

        _generateScatterWithPercentilesData(xValues, yValues, yMetric, xMetric, primaryColor, palette) {
            const traces = [];

            // Main scatter trace
            traces.push({
                x: xValues,
                y: yValues,
                mode: 'markers',
                type: 'scatter',
                name: yMetric,
                marker: { color: primaryColor, size: 5, opacity: 0.5 },
                hovertemplate: `${xMetric}: %{x}<br>${yMetric}: %{y}<extra></extra>`,
            });

            // Calculate rolling percentiles (window of 50 points)
            const windowSize = Math.min(50, Math.floor(xValues.length / 10) || 10);
            if (xValues.length >= windowSize) {
                const percentiles = [
                    { p: 50, name: 'p50', color: palette[1] },
                    { p: 95, name: 'p95', color: palette[2] },
                    { p: 99, name: 'p99', color: palette[3] },
                ];

                percentiles.forEach(({ p, name, color }) => {
                    const rollingValues = this._calculateRollingPercentile(yValues, windowSize, p);
                    traces.push({
                        x: xValues,
                        y: rollingValues,
                        mode: 'lines',
                        type: 'scatter',
                        name: `Rolling ${name}`,
                        line: { color, width: 2 },
                    });
                });
            }

            return traces;
        },

        _calculateRollingPercentile(values, windowSize, percentile) {
            const result = [];
            for (let i = 0; i < values.length; i++) {
                const start = Math.max(0, i - windowSize + 1);
                const window = values.slice(start, i + 1);
                window.sort((a, b) => a - b);
                const idx = Math.floor((window.length - 1) * percentile / 100);
                result.push(window[idx]);
            }
            return result;
        },

        _getMetricValue(run, metricName, stat) {
            if (metricName === 'concurrency') {
                const val = run.metadata?.concurrency;
                console.log('[DEBUG] _getMetricValue concurrency:', val);
                return val;
            }

            const metric = run.aggregated?.[metricName];
            if (!metric) {
                console.warn('[DEBUG] _getMetricValue: metric not found:', metricName, 'in', run.aggregated ? Object.keys(run.aggregated) : 'no aggregated');
                return null;
            }

            if (metric[stat] !== undefined && metric[stat] !== null) {
                console.log('[DEBUG] _getMetricValue:', metricName, stat, '=', metric[stat]);
                return metric[stat];
            }
            if (metric.avg !== undefined) {
                console.log('[DEBUG] _getMetricValue:', metricName, 'avg =', metric.avg);
                return metric.avg;
            }
            if (metric.value !== undefined) {
                console.log('[DEBUG] _getMetricValue:', metricName, 'value =', metric.value);
                return metric.value;
            }

            console.warn('[DEBUG] _getMetricValue: no value found for', metricName, 'metric:', metric);
            return null;
        },

        _getGroupValue(run, groupBy) {
            if (groupBy === 'model') {
                return run.metadata.model || 'Unknown';
            }
            if (groupBy === 'experimentGroup' || groupBy === 'experiment_group') {
                return run.metadata.experimentGroup || 'Default';
            }
            if (groupBy === 'concurrency') {
                return `C${run.metadata.concurrency || '?'}`;
            }
            return 'All Runs';
        },

        _getLabelValue(run, labelBy) {
            if (!labelBy || labelBy === 'none') {
                return '';
            }
            if (labelBy === 'concurrency') {
                return String(run.metadata.concurrency || '');
            }
            if (labelBy === 'model') {
                return run.metadata.model || '';
            }
            if (labelBy === 'run_name') {
                return run.metadata.runName || '';
            }
            if (labelBy === 'experimentGroup' || labelBy === 'experiment_group') {
                return run.metadata.experimentGroup || '';
            }
            // For swept parameters, check metadata or aggregated data
            if (run.metadata[labelBy] !== undefined) {
                return String(run.metadata[labelBy]);
            }
            if (run.aggregated?.[labelBy]?.value !== undefined) {
                return String(run.aggregated[labelBy].value);
            }
            return '';
        },

        _generateLayout(plotConfig) {
            const state = this._state.getState();
            const theme = state.theme;
            const colors = theme === 'dark' ? DARK_COLORS : LIGHT_COLORS;
            const config = plotConfig.config;

            const xLabel = this._getMetricLabel(config.xMetric, config.xStat);
            const yLabel = this._getMetricLabel(config.yMetric, config.yStat);

            // Title uses display names only (matches dashboard behavior)
            const xDisplayName = this._getMetricDisplayName(config.xMetric);
            const yDisplayName = this._getMetricDisplayName(config.yMetric);
            const autoTitle = `${yDisplayName} vs ${xDisplayName}`;

            return {
                title: {
                    text: config.title || autoTitle,
                    font: { size: 16, color: colors.text },
                },
                xaxis: {
                    title: { text: xLabel, font: { color: colors.text } },
                    gridcolor: colors.grid,
                    linecolor: colors.border,
                    tickfont: { color: colors.text },
                    type: config.logScaleX ? 'log' : 'linear',
                    zeroline: false,
                },
                yaxis: {
                    title: { text: yLabel, font: { color: colors.text } },
                    gridcolor: colors.grid,
                    linecolor: colors.border,
                    tickfont: { color: colors.text },
                    type: config.logScaleY ? 'log' : 'linear',
                    zeroline: false,
                },
                plot_bgcolor: colors.background,
                paper_bgcolor: colors.paper,
                font: { family: FONT_FAMILY, color: colors.text },
                legend: {
                    x: 1.02,
                    y: 1,
                    xanchor: 'left',
                    bgcolor: 'rgba(0,0,0,0)',
                },
                margin: { l: 80, r: 150, t: 60, b: 80 },
                hovermode: 'closest',
            };
        },

        _getMetricDisplayName(metricName) {
            if (!metricName) return '';

            const state = this._state.getState();
            const metrics = state.availableMetrics;

            if (metrics[metricName]) {
                return metrics[metricName].displayName || metricName;
            }
            return metricName;
        },

        _getMetricLabel(metricName, stat) {
            if (!metricName) return '';

            const state = this._state.getState();
            const metrics = state.availableMetrics;
            const statLabels = state.statLabels;

            let label = this._getMetricDisplayName(metricName);

            if (stat && stat !== 'value' && metricName !== 'concurrency') {
                const statLabel = statLabels[stat] || stat.toUpperCase();
                label = `${label} (${statLabel})`;
            }

            const unit = metrics[metricName]?.unit;
            if (unit) {
                label = `${label} [${unit}]`;
            }

            return label;
        },

        _getPlotlyConfig() {
            return {
                displayModeBar: true,
                responsive: true,
                modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'aiperf_plot',
                    width: 1600,
                    height: 800,
                    scale: 2,
                },
            };
        },

        _getColorPalette() {
            const state = this._state.getState();
            const theme = state.theme;

            // Seaborn "deep" palette for light theme (professional, subdued)
            const seabornDeep = [
                '#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
                '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd',
            ];

            // Seaborn "bright" palette for dark theme (vibrant contrast)
            const seabornBright = [
                '#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2',
                '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff',
            ];

            if (theme === 'dark') {
                // Dark theme: NVIDIA brand colors + seaborn bright
                return ['#76B900', '#F4E5C3', ...seabornBright];
            } else {
                // Light theme: seaborn deep (no brand colors)
                return seabornDeep;
            }
        },

        exportPlot(plotId) {
            Plotly.downloadImage(`plot-${plotId}`, {
                format: 'png',
                width: 1600,
                height: 800,
                scale: 2,
                filename: `aiperf_plot_${plotId}`,
            });
        },
    };

    // =========================================================================
    // UI Controls
    // =========================================================================
    const UIControls = {
        _state: null,
        _plotManager: null,

        init(stateManager, plotManager) {
            this._state = stateManager;
            this._plotManager = plotManager;
            return this;
        },

        setupControls() {
            this._setupRunSelector();
            this._setupGlobalStatSelector();
            this._setupPlotControls();
            this._setupAddPlotButton();
            this._setupSidebarToggle();
            this._setupModals();
            this._setupCollapsibleSections();
            this._movePlotTypeToTop();
            this._filterMetricsForMode();
        },

        _movePlotTypeToTop() {
            const plotTypeSelects = ['new-plot-type', 'settings-plot-type'];
            plotTypeSelects.forEach((selectId) => {
                const select = document.getElementById(selectId);
                if (!select) return;

                const formGroup = select.closest('.form-group');
                const modalBody = formGroup?.closest('.modal-body');
                if (formGroup && modalBody && modalBody.firstChild !== formGroup) {
                    modalBody.insertBefore(formGroup, modalBody.firstChild);
                }
            });
        },

        _setupCollapsibleSections() {
            document.querySelectorAll('.section-header[data-section]').forEach((header) => {
                header.addEventListener('click', () => {
                    const sectionId = header.dataset.section;
                    const content = document.querySelector(`.section-content[data-section="${sectionId}"]`);

                    if (content) {
                        header.classList.toggle('collapsed');
                        content.classList.toggle('collapsed');

                        const arrow = header.querySelector('.section-arrow');
                        if (arrow) {
                            arrow.textContent = header.classList.contains('collapsed') ? '▶' : '▼';
                        }
                    }
                });
            });
        },

        _setupRunSelector() {
            const container = document.getElementById('run-selector');
            if (!container) return;

            const checkboxes = container.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.addEventListener('change', () => {
                    const selected = [];
                    container.querySelectorAll('input[type="checkbox"]:checked').forEach((checked) => {
                        selected.push(parseInt(checked.value, 10));
                    });
                    this._state.setState({ selectedRuns: selected });
                    this._plotManager.refreshAllPlots();
                });
            });
        },

        _setupGlobalStatSelector() {
            const selector = document.getElementById('global-stat-selector');
            const applyBtn = document.getElementById('apply-global-stat-btn');

            if (!selector || !applyBtn) return;

            applyBtn.addEventListener('click', () => {
                const stat = selector.value;
                const plots = this._state.getState().plots;

                plots.forEach((plot) => {
                    this._state.updatePlot(plot.id, { xStat: stat, yStat: stat });
                });

                this._plotManager.refreshAllPlots();
                this._showToast('Applied stat to all plots', 'success');
            });
        },

        _setupPlotControls() {
            document.querySelectorAll('.x-metric-select').forEach((select) => {
                select.addEventListener('change', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.updatePlot(plotId, { xMetric: e.target.value });
                });
            });

            document.querySelectorAll('.y-metric-select').forEach((select) => {
                select.addEventListener('change', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.updatePlot(plotId, { yMetric: e.target.value });
                });
            });

            document.querySelectorAll('.x-stat-select').forEach((select) => {
                select.addEventListener('change', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.updatePlot(plotId, { xStat: e.target.value });
                });
            });

            document.querySelectorAll('.y-stat-select').forEach((select) => {
                select.addEventListener('change', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.updatePlot(plotId, { yStat: e.target.value });
                });
            });

            document.querySelectorAll('.log-scale-x').forEach((cb) => {
                cb.addEventListener('change', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.updatePlot(plotId, { logScaleX: e.target.checked });
                });
            });

            document.querySelectorAll('.log-scale-y').forEach((cb) => {
                cb.addEventListener('change', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.updatePlot(plotId, { logScaleY: e.target.checked });
                });
            });

            document.querySelectorAll('.plot-type-select').forEach((select) => {
                select.addEventListener('change', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.updatePlot(plotId, { plotType: e.target.value });
                });
            });

            document.querySelectorAll('.settings-btn').forEach((btn) => {
                btn.addEventListener('click', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._openSettingsModal(plotId);
                });
            });

            document.querySelectorAll('.hide-btn').forEach((btn) => {
                btn.addEventListener('click', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._togglePlotVisibility(plotId);
                });
            });

            document.querySelectorAll('.export-btn').forEach((btn) => {
                btn.addEventListener('click', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._plotManager.exportPlot(plotId);
                });
            });

            document.querySelectorAll('.remove-btn').forEach((btn) => {
                btn.addEventListener('click', (e) => {
                    const plotId = e.target.dataset.plotId;
                    this._removePlot(plotId);
                });
            });
        },

        _setupAddPlotButton() {
            const btn = document.getElementById('add-plot-btn');
            if (btn) {
                btn.addEventListener('click', () => {
                    this._openAddPlotModal();
                });
            }

            const addSlot = document.getElementById('add-plot-slot');
            if (addSlot) {
                addSlot.addEventListener('click', () => {
                    this._openAddPlotModal();
                });
            }
        },

        _setupSidebarToggle() {
            const btn = document.getElementById('sidebar-toggle-btn');
            const sidebar = document.getElementById('sidebar');

            if (!btn || !sidebar) return;

            btn.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                btn.textContent = sidebar.classList.contains('collapsed') ? '☰' : '✕';
            });
        },

        _setupModals() {
            document.querySelectorAll('.modal-overlay').forEach((overlay) => {
                overlay.addEventListener('click', (e) => {
                    if (e.target === overlay) {
                        overlay.classList.remove('active');
                    }
                });
            });

            document.querySelectorAll('.modal-close').forEach((btn) => {
                btn.addEventListener('click', () => {
                    btn.closest('.modal-overlay').classList.remove('active');
                });
            });

            const addPlotForm = document.getElementById('add-plot-form');
            if (addPlotForm) {
                addPlotForm.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this._handleAddPlotSubmit();
                });
            }

            const settingsApplyBtn = document.getElementById('settings-apply-btn');
            if (settingsApplyBtn) {
                settingsApplyBtn.addEventListener('click', () => {
                    this._handleSettingsApply();
                });
            }

            const hideBtn = document.getElementById('settings-hide-btn');
            if (hideBtn) {
                hideBtn.addEventListener('click', () => {
                    const modal = document.getElementById('settings-modal');
                    const plotId = modal?.dataset.plotId;
                    if (plotId) {
                        this._hidePlot(plotId);
                        modal.classList.remove('active');
                    }
                });
            }

            const saveAsNewBtn = document.getElementById('settings-save-as-new-btn');
            if (saveAsNewBtn) {
                saveAsNewBtn.addEventListener('click', () => {
                    this._handleSaveAsNew();
                });
            }

            // Setup stat filtering based on selected metrics
            this._setupStatFiltering();
        },

        _setupStatFiltering() {
            // Add Plot modal - X metric change
            const newXMetric = document.getElementById('new-x-metric');
            if (newXMetric) {
                newXMetric.addEventListener('change', () => {
                    this._updateStatOptions('new-x-metric', 'new-x-stat', 'p50');
                });
            }

            // Add Plot modal - Y metric change
            const newYMetric = document.getElementById('new-y-metric');
            if (newYMetric) {
                newYMetric.addEventListener('change', () => {
                    this._updateStatOptions('new-y-metric', 'new-y-stat', 'avg');
                });
            }

            // Settings modal - X metric change
            const settingsXMetric = document.getElementById('settings-x-metric');
            if (settingsXMetric) {
                settingsXMetric.addEventListener('change', () => {
                    this._updateStatOptions('settings-x-metric', 'settings-x-stat', 'p50');
                });
            }

            // Settings modal - Y metric change
            const settingsYMetric = document.getElementById('settings-y-metric');
            if (settingsYMetric) {
                settingsYMetric.addEventListener('change', () => {
                    this._updateStatOptions('settings-y-metric', 'settings-y-stat', 'avg');
                });
            }
        },

        _updateStatOptions(metricSelectId, statSelectId, preferredStat) {
            const metricSelect = document.getElementById(metricSelectId);
            const statSelect = document.getElementById(statSelectId);

            if (!metricSelect || !statSelect) return;

            const metric = metricSelect.value;
            if (!metric) return;

            // Get available stats for this metric
            const availableStats = this._getAvailableStatsForMetric(metric);
            const currentStat = statSelect.value;

            // Rebuild stat dropdown options
            const statLabels = this._state.getState().statLabels || {};
            statSelect.innerHTML = '<option value="">Select statistic</option>';

            availableStats.forEach((stat) => {
                const option = document.createElement('option');
                option.value = stat;
                option.textContent = statLabels[stat] || stat;
                statSelect.appendChild(option);
            });

            // Smart default selection
            let newValue = '';
            if (availableStats.length === 1) {
                newValue = availableStats[0];
            } else if (currentStat && availableStats.includes(currentStat)) {
                newValue = currentStat;
            } else if (availableStats.includes(preferredStat)) {
                newValue = preferredStat;
            } else if (availableStats.length > 0) {
                newValue = availableStats[0];
            }

            statSelect.value = newValue;
        },

        _getAvailableStatsForMetric(metricName) {
            const state = this._state.getState();
            const statKeys = state.statKeys || ['p50', 'avg', 'p90', 'p95', 'p99', 'min', 'max', 'std'];

            // Special case: concurrency is metadata
            if (metricName === 'concurrency') {
                return ['value'];
            }

            // Check first run for available stats
            if (state.mode === 'multi' && state.runs && state.runs.length > 0) {
                const firstRun = state.runs[0];
                const metric = firstRun.aggregated?.[metricName];

                if (!metric) {
                    return statKeys;  // Fallback to all stats
                }

                // Return keys that exist and aren't 'unit'
                return Object.keys(metric).filter(k => k !== 'unit' && metric[k] !== null && metric[k] !== undefined);
            } else if (state.mode === 'single' && state.requests) {
                // For single-run, check if column exists
                const columns = state.requests.columns || [];
                return columns.includes(metricName) ? statKeys : [];
            }

            return statKeys;
        },

        _filterMetricsForMode() {
            const state = this._state.getState();
            if (state.mode !== 'single') return;

            const data = window.__AIPERF_DATA__;
            const perRequestColumns = data?.perRequestColumns || [];
            if (perRequestColumns.length === 0) return;

            const metricSelectIds = [
                'new-x-metric',
                'new-y-metric',
                'settings-x-metric',
                'settings-y-metric',
            ];

            metricSelectIds.forEach((selectId) => {
                const select = document.getElementById(selectId);
                if (!select) return;

                Array.from(select.options).forEach((option) => {
                    if (option.value === '' || option.value === 'request_number') {
                        return;
                    }
                    if (!perRequestColumns.includes(option.value)) {
                        option.style.display = 'none';
                        option.disabled = true;
                    }
                });
            });

            this._adaptModalsForSingleRun();
        },

        _adaptModalsForSingleRun() {
            // Hide X-axis stat dropdown (single-run doesn't use X stats)
            const xStatFields = ['new-x-stat', 'settings-x-stat'];
            xStatFields.forEach((id) => {
                const field = document.getElementById(id);
                if (field) {
                    const formGroup = field.closest('.form-group');
                    if (formGroup) formGroup.style.display = 'none';
                }
            });

            // Keep Y-axis stat visible and update label
            ['new-y-stat', 'settings-y-stat'].forEach((id) => {
                const field = document.getElementById(id);
                if (field) {
                    const formGroup = field.closest('.form-group');
                    const label = formGroup?.querySelector('label');
                    if (label) label.textContent = 'Y-Axis Statistic';
                }
            });

            // Hide multi-run-only fields (labelBy, groupBy)
            const multiRunFields = ['new-label-by', 'new-group-by'];
            multiRunFields.forEach((id) => {
                const field = document.getElementById(id);
                if (field) {
                    const formGroup = field.closest('.form-group');
                    if (formGroup) formGroup.style.display = 'none';
                }
            });

            // Update plot type options for single-run (matching interactive dashboard)
            const plotTypeSelects = ['new-plot-type', 'settings-plot-type'];
            const data = window.__AIPERF_DATA__;
            const hasTimeslices = data?.timeslices?.data?.length > 0;
            const hasGpuTelemetry = data?.gpuTelemetry?.data?.length > 0;

            const singleRunPlotTypes = [
                { value: 'scatter', label: 'Per-Request Scatter' },
                { value: 'scatter_with_percentiles', label: 'Scatter with Trends' },
                { value: 'request_timeline', label: 'Request Phase Breakdown' },
            ];
            if (hasTimeslices) {
                singleRunPlotTypes.push({ value: 'timeslice', label: 'Time Window Summary' });
            }
            singleRunPlotTypes.push({ value: 'area', label: 'Area' });
            if (hasGpuTelemetry) {
                singleRunPlotTypes.push({ value: 'dual_axis', label: 'Dual Axis (GPU)' });
            }

            plotTypeSelects.forEach((selectId) => {
                const select = document.getElementById(selectId);
                if (!select) return;

                select.innerHTML = '<option value="">Select plot type</option>';
                singleRunPlotTypes.forEach((type) => {
                    const option = document.createElement('option');
                    option.value = type.value;
                    option.textContent = type.label;
                    select.appendChild(option);
                });
            });

            // Replace X-axis metric dropdown with fixed single-run options
            const xAxisConfigs = [
                { selectId: 'new-x-metric', labelText: 'X-Axis' },
                { selectId: 'settings-x-metric', labelText: 'X-Axis' },
            ];
            const defaultXAxisOptions = [
                { value: 'request_number', label: 'Request Number' },
                { value: 'timestamp_s', label: 'Timestamp (s)' },
            ];

            xAxisConfigs.forEach(({ selectId, labelText }) => {
                const select = document.getElementById(selectId);
                if (!select) return;

                const formGroup = select.closest('.form-group');
                const label = formGroup?.querySelector('label');
                if (label) label.textContent = labelText;

                select.innerHTML = '';
                defaultXAxisOptions.forEach((opt) => {
                    const option = document.createElement('option');
                    option.value = opt.value;
                    option.textContent = opt.label;
                    select.appendChild(option);
                });
                select.value = 'request_number';
            });

            // Update Y-axis label
            ['new-y-metric', 'settings-y-metric'].forEach((selectId) => {
                const select = document.getElementById(selectId);
                if (!select) return;
                const formGroup = select.closest('.form-group');
                const label = formGroup?.querySelector('label');
                if (label) label.textContent = 'Y-Axis Metric';
            });

            // Setup plot type change listener to update X-axis options
            this._setupSingleRunPlotTypeListeners();
        },

        _setupSingleRunPlotTypeListeners() {
            const configs = [
                { plotTypeId: 'new-plot-type', xAxisId: 'new-x-metric' },
                { plotTypeId: 'settings-plot-type', xAxisId: 'settings-x-metric' },
            ];

            configs.forEach(({ plotTypeId, xAxisId }) => {
                const plotTypeSelect = document.getElementById(plotTypeId);
                const xAxisSelect = document.getElementById(xAxisId);
                if (!plotTypeSelect || !xAxisSelect) return;

                plotTypeSelect.addEventListener('change', () => {
                    const plotType = plotTypeSelect.value;
                    this._updateXAxisOptionsForPlotType(xAxisSelect, plotType);
                });
            });
        },

        _updateXAxisOptionsForPlotType(xAxisSelect, plotType) {
            let options;
            let defaultValue;

            if (plotType === 'request_timeline' || plotType === 'dual_axis') {
                options = [{ value: 'timestamp_s', label: 'Timestamp (s)' }];
                defaultValue = 'timestamp_s';
            } else if (plotType === 'timeslice') {
                const data = window.__AIPERF_DATA__;
                const sliceDuration = data?.sliceDuration || '';
                const label = sliceDuration ? `Timeslice (${sliceDuration}s)` : 'Timeslice';
                options = [{ value: 'Timeslice', label }];
                defaultValue = 'Timeslice';
            } else {
                options = [
                    { value: 'request_number', label: 'Request Number' },
                    { value: 'timestamp_s', label: 'Timestamp (s)' },
                ];
                defaultValue = 'request_number';
            }

            xAxisSelect.innerHTML = '';
            options.forEach((opt) => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                xAxisSelect.appendChild(option);
            });
            xAxisSelect.value = defaultValue;
        },

        _handleSettingsApply() {
            const modal = document.getElementById('settings-modal');
            if (!modal) return;

            const plotId = modal.dataset.plotId;
            if (!plotId) return;

            const xMetric = modal.querySelector('#settings-x-metric')?.value;
            const yMetric = modal.querySelector('#settings-y-metric')?.value;
            const xStat = modal.querySelector('#settings-x-stat')?.value || 'p50';
            const yStat = modal.querySelector('#settings-y-stat')?.value || 'avg';
            const logScaleX = modal.querySelector('#settings-x-log')?.value === 'true';
            const logScaleY = modal.querySelector('#settings-y-log')?.value === 'true';
            const plotType = modal.querySelector('#settings-plot-type')?.value || 'scatter_line';
            const title = modal.querySelector('#settings-title')?.value || '';
            const xLabel = modal.querySelector('#settings-x-label')?.value || '';
            const yLabel = modal.querySelector('#settings-y-label')?.value || '';
            const xAutoscale = modal.querySelector('#settings-x-autoscale')?.value === 'true';
            const yAutoscale = modal.querySelector('#settings-y-autoscale')?.value === 'true';

            const updates = {
                xMetric,
                yMetric,
                xStat,
                yStat,
                logScaleX,
                logScaleY,
                plotType,
                title,
                xLabel,
                yLabel,
                autoscaleX: xAutoscale,
                autoscaleY: yAutoscale,
            };

            this._plotManager.updatePlot(plotId, updates);
            modal.classList.remove('active');
            this._showToast('Settings applied', 'success');
        },

        _handleSaveAsNew() {
            const modal = document.getElementById('settings-modal');
            if (!modal) return;

            const plotId = modal.dataset.plotId;
            const currentPlot = this._state.getPlot(plotId);
            if (!currentPlot) return;

            const xMetric = modal.querySelector('#settings-x-metric')?.value;
            const yMetric = modal.querySelector('#settings-y-metric')?.value;
            const xStat = modal.querySelector('#settings-x-stat')?.value || 'p50';
            const yStat = modal.querySelector('#settings-y-stat')?.value || 'avg';
            const plotType = modal.querySelector('#settings-plot-type')?.value || 'scatter_line';
            const logScaleX = modal.querySelector('#settings-x-log')?.value === 'true';
            const logScaleY = modal.querySelector('#settings-y-log')?.value === 'true';
            const title = modal.querySelector('#settings-title')?.value || '';
            const xLabel = modal.querySelector('#settings-x-label')?.value || '';
            const yLabel = modal.querySelector('#settings-y-label')?.value || '';
            const xAutoscale = modal.querySelector('#settings-x-autoscale')?.value === 'true';
            const yAutoscale = modal.querySelector('#settings-y-autoscale')?.value === 'true';

            // Generate auto-title using display names (matches dashboard behavior)
            const xDisplayName = this._plotManager._getMetricDisplayName(xMetric);
            const yDisplayName = this._plotManager._getMetricDisplayName(yMetric);
            const autoTitle = `${yDisplayName} vs ${xDisplayName}`;

            const newConfig = {
                xMetric,
                yMetric,
                xStat,
                yStat,
                plotType,
                logScaleX,
                logScaleY,
                title: title || autoTitle,
                xLabel,
                yLabel,
                autoscaleX: xAutoscale,
                autoscaleY: yAutoscale,
                labelBy: currentPlot.config.labelBy || 'concurrency',
                groupBy: currentPlot.config.groupBy || 'model',
            };

            const newId = this._state.addPlot(newConfig);
            this._renderNewPlotContainer(newId, newConfig);
            this._plotManager.renderPlot({ id: newId, config: newConfig });

            modal.classList.remove('active');
            this._showToast('Plot copied successfully', 'success');
        },

        _openAddPlotModal() {
            const modal = document.getElementById('add-plot-modal');
            if (!modal) return;

            // Default groupBy to experiment_group when classification is enabled
            const classification = window.__AIPERF_DATA__?.classification;
            const groupBySelect = modal.querySelector('#new-group-by');
            if (groupBySelect && classification) {
                groupBySelect.value = 'experiment_group';
            }

            modal.classList.add('active');
        },

        _openSettingsModal(plotId) {
            const plot = this._state.getPlot(plotId);
            if (!plot) return;

            const modal = document.getElementById('settings-modal');
            if (!modal) return;

            modal.dataset.plotId = plotId;

            const xMetricSelect = modal.querySelector('#settings-x-metric');
            const yMetricSelect = modal.querySelector('#settings-y-metric');
            const xStatSelect = modal.querySelector('#settings-x-stat');
            const yStatSelect = modal.querySelector('#settings-y-stat');
            const logScaleX = modal.querySelector('#settings-x-log');
            const logScaleY = modal.querySelector('#settings-y-log');
            const plotType = modal.querySelector('#settings-plot-type');
            const title = modal.querySelector('#settings-title');
            const xLabel = modal.querySelector('#settings-x-label');
            const yLabel = modal.querySelector('#settings-y-label');
            const xAutoscale = modal.querySelector('#settings-x-autoscale');
            const yAutoscale = modal.querySelector('#settings-y-autoscale');

            if (xMetricSelect) xMetricSelect.value = plot.config.xMetric || '';
            if (yMetricSelect) yMetricSelect.value = plot.config.yMetric || '';

            // Filter stat options based on selected metrics
            if (plot.config.xMetric) {
                this._updateStatOptions('settings-x-metric', 'settings-x-stat', 'p50');
            }
            if (plot.config.yMetric) {
                this._updateStatOptions('settings-y-metric', 'settings-y-stat', 'avg');
            }

            // Set stat values after filtering
            if (xStatSelect) xStatSelect.value = plot.config.xStat || '';
            if (yStatSelect) yStatSelect.value = plot.config.yStat || '';

            if (logScaleX) logScaleX.value = plot.config.logScaleX ? 'true' : 'false';
            if (logScaleY) logScaleY.value = plot.config.logScaleY ? 'true' : 'false';
            if (plotType) plotType.value = plot.config.plotType || 'scatter_line';
            if (title) title.value = plot.config.title || '';
            if (xLabel) xLabel.value = plot.config.xLabel || '';
            if (yLabel) yLabel.value = plot.config.yLabel || '';
            if (xAutoscale) xAutoscale.value = plot.config.autoscaleX ? 'true' : 'false';
            if (yAutoscale) yAutoscale.value = plot.config.autoscaleY ? 'true' : 'false';

            modal.classList.add('active');
        },

        _handleAddPlotSubmit() {
            const modal = document.getElementById('add-plot-modal');
            if (!modal) return;

            const xMetric = modal.querySelector('#new-x-metric')?.value;
            const yMetric = modal.querySelector('#new-y-metric')?.value;
            const xStat = modal.querySelector('#new-x-stat')?.value || 'p50';
            const yStat = modal.querySelector('#new-y-stat')?.value || 'avg';
            const title = modal.querySelector('#new-plot-title')?.value || '';
            const plotType = modal.querySelector('#new-plot-type')?.value || 'scatter_line';
            const labelBy = modal.querySelector('#new-label-by')?.value || 'concurrency';
            const groupBy = modal.querySelector('#new-group-by')?.value || 'model';
            const xLabel = modal.querySelector('#new-x-label')?.value || '';
            const yLabel = modal.querySelector('#new-y-label')?.value || '';
            const xLog = modal.querySelector('#new-x-log')?.value === 'true';
            const yLog = modal.querySelector('#new-y-log')?.value === 'true';
            const xAutoscale = modal.querySelector('#new-x-autoscale')?.value === 'true';
            const yAutoscale = modal.querySelector('#new-y-autoscale')?.value === 'true';

            if (!xMetric || !yMetric) {
                this._showToast('Please select both X and Y metrics', 'error');
                return;
            }

            // Generate auto-title using display names (matches dashboard behavior)
            const xDisplayName = this._plotManager._getMetricDisplayName(xMetric);
            const yDisplayName = this._plotManager._getMetricDisplayName(yMetric);
            const autoTitle = `${yDisplayName} vs ${xDisplayName}`;

            const plotConfig = {
                xMetric,
                yMetric,
                xStat,
                yStat,
                title: title || autoTitle,
                plotType,
                labelBy,
                groupBy,
                xLabel,
                yLabel,
                logScaleX: xLog,
                logScaleY: yLog,
                autoscaleX: xAutoscale,
                autoscaleY: yAutoscale,
            };

            const newId = this._state.addPlot(plotConfig);
            this._renderNewPlotContainer(newId, plotConfig);
            this._plotManager.renderPlot({ id: newId, config: plotConfig });

            modal.classList.remove('active');
            this._showToast('Plot added successfully', 'success');
        },

        _renderNewPlotContainer(plotId, config) {
            const grid = document.getElementById('plot-grid');
            const addSlot = document.getElementById('add-plot-slot');

            if (!grid) return;

            const container = document.createElement('div');
            container.id = `plot-container-${plotId}`;
            container.className = 'plot-container size-half';

            container.innerHTML = `
                <div class="plot-toolbar">
                    <button class="settings-btn" data-plot-id="${plotId}" title="Settings">⚙</button>
                    <button class="export-btn" data-plot-id="${plotId}" title="Export">📥</button>
                    <button class="hide-btn" data-plot-id="${plotId}" title="Hide">👁</button>
                    <button class="remove-btn" data-plot-id="${plotId}" title="Remove">✕</button>
                </div>
                <div id="plot-${plotId}" class="plot-area"></div>
            `;

            if (addSlot) {
                grid.insertBefore(container, addSlot);
            } else {
                grid.appendChild(container);
            }

            container.querySelector('.settings-btn').addEventListener('click', () => {
                this._openSettingsModal(plotId);
            });
            container.querySelector('.export-btn').addEventListener('click', () => {
                this._plotManager.exportPlot(plotId);
            });
            container.querySelector('.hide-btn').addEventListener('click', () => {
                this._togglePlotVisibility(plotId);
            });
            container.querySelector('.remove-btn').addEventListener('click', () => {
                this._removePlot(plotId);
            });
        },

        _togglePlotVisibility(plotId) {
            const container = document.getElementById(`plot-container-${plotId}`);
            if (container) {
                container.classList.toggle('hidden');
            }
            this._state.togglePlotVisibility(plotId);
        },

        _removePlot(plotId) {
            const container = document.getElementById(`plot-container-${plotId}`);
            if (container) {
                container.remove();
            }
            this._state.removePlot(plotId);
            this._showToast('Plot removed', 'success');
        },

        _showToast(message, type = 'success') {
            let container = document.querySelector('.toast-container');
            if (!container) {
                container = document.createElement('div');
                container.className = 'toast-container';
                document.body.appendChild(container);
            }

            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            container.appendChild(toast);

            setTimeout(() => {
                toast.remove();
            }, 3000);
        },
    };

    // =========================================================================
    // Theme Manager
    // =========================================================================
    const ThemeManager = {
        _state: null,

        init(stateManager) {
            this._state = stateManager;
            this._setupThemeToggle();
            return this;
        },

        _setupThemeToggle() {
            const toggle = document.getElementById('theme-toggle');
            if (!toggle) return;

            toggle.addEventListener('change', () => {
                const newTheme = toggle.checked ? 'dark' : 'light';
                this.applyTheme(newTheme);
            });
        },

        applyTheme(theme) {
            this._state.setState({ theme });

            document.body.classList.remove('theme-light', 'theme-dark');
            document.body.classList.add(`theme-${theme}`);

            const toggle = document.getElementById('theme-toggle');
            if (toggle) {
                toggle.checked = theme === 'dark';
            }

            this._updatePlotThemes(theme);
        },

        _updatePlotThemes(theme) {
            const colors = theme === 'dark' ? DARK_COLORS : LIGHT_COLORS;
            const plots = this._state.getState().plots;

            plots.forEach((plot) => {
                const containerId = `plot-${plot.id}`;
                const container = document.getElementById(containerId);

                if (container) {
                    const layoutUpdate = {
                        plot_bgcolor: colors.background,
                        paper_bgcolor: colors.paper,
                        font: { color: colors.text },
                        'xaxis.gridcolor': colors.grid,
                        'xaxis.linecolor': colors.border,
                        'xaxis.tickfont.color': colors.text,
                        'xaxis.title.font.color': colors.text,
                        'yaxis.gridcolor': colors.grid,
                        'yaxis.linecolor': colors.border,
                        'yaxis.tickfont.color': colors.text,
                        'yaxis.title.font.color': colors.text,
                        'title.font.color': colors.text,
                    };

                    Plotly.relayout(containerId, layoutUpdate);
                }
            });
        },
    };

    // =========================================================================
    // Main Application Entry Point
    // =========================================================================
    function initApp() {
        console.log('[DEBUG] initApp starting');
        const DATA = window.__AIPERF_DATA__;
        const INITIAL_PLOTS = window.__AIPERF_INITIAL_PLOTS__ || [];
        const CONFIG = window.__AIPERF_CONFIG__ || {};

        console.log('[DEBUG] __AIPERF_DATA__:', DATA);
        console.log('[DEBUG] __AIPERF_INITIAL_PLOTS__:', INITIAL_PLOTS);
        console.log('[DEBUG] __AIPERF_CONFIG__:', CONFIG);
        console.log('[DEBUG] Data mode:', DATA?.mode);

        if (!DATA) {
            console.error('AIPerf data not found. Ensure __AIPERF_DATA__ is defined.');
            return;
        }

        StateManager.init(DATA, CONFIG);
        console.log('[DEBUG] StateManager initialized, state:', StateManager.getState());

        INITIAL_PLOTS.forEach((plot) => {
            StateManager._state.plots.push(plot);
        });

        PlotManager.init(StateManager);
        UIControls.init(StateManager, PlotManager);
        ThemeManager.init(StateManager);

        UIControls.setupControls();

        ThemeManager.applyTheme(CONFIG.theme || 'light');

        INITIAL_PLOTS.forEach((plot) => {
            PlotManager.renderPlot(plot);
        });

        console.log('AIPerf Interactive HTML initialized successfully');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initApp);
    } else {
        initApp();
    }
})();
