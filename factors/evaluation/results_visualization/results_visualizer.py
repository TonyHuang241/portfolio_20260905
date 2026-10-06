"""按因子合并原始值和各滚动均值列的每日评估结果，生成一份离线 HTML 报告。"""

import argparse
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class ResultsVisualizer:
    COLORS = ["#2563eb", "#06a6a0", "#e7a32e", "#9764d9", "#ed6976"]
    # 超过五组时用蓝红分歧色：G1–G5 蓝色由深到浅，G6–G10 红色由浅到深，两端（多空组）颜色最深。
    DIVERGING_COLORS = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#ea9a93", "#dd716a", "#c74845", "#9e3432", "#762221"]
    # 窗口对比图的颜色按因子文件中的列顺序固定分配（原始值、由短到长的滚动均值列），相邻颜色已校验色盲可分。
    WINDOW_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

    @staticmethod
    def _default_output_dir():
        project_dir = Path(__file__).resolve().parents[3]
        local_config = json.loads((project_dir / "config_local.json").read_text(encoding="utf-8"))
        config = json.loads((project_dir / "config/config_factor_evaluation.json").read_text(encoding="utf-8"))
        return Path(local_config["root_dir"]).expanduser() / config["visualization_output_dir"]

    def __init__(self, factor_results, factor_name, start_date=None, output_dir=None,
                 periods_per_year=252, risk_free_rate=0.0):
        """Accept a DataFrame or CSV path; returns must be daily decimal returns.

        The date column/index labels the beginning of the return period, matching
        SingleFactorEvaluation. risk_free_rate is an effective annual rate.
        """
        if periods_per_year <= 0 or risk_free_rate <= -1:
            raise ValueError("periods_per_year must be positive and risk_free_rate must exceed -1.")
        self.factor_name = str(factor_name)
        self.start_date = start_date
        self.output_dir = Path(output_dir) if output_dir is not None else self._default_output_dir()
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        data = factor_results.copy() if isinstance(factor_results, pd.DataFrame) else pd.read_csv(factor_results)
        if "date" in data.columns:
            data = data.set_index("date")
        # Casting integer YYYYMMDD dates to strings avoids nanosecond timestamps.
        data.index = pd.to_datetime(data.index.astype(str))
        data = data.sort_index()
        # 分组数由结果中的 group_N 列确定，五组和十组的评估结果都可以直接生成报告。
        group_number = len([column for column in data.columns if column.startswith("group_") and column[6:].isdigit()])
        self.GROUPS = [f"group_{group}" for group in range(1, group_number + 1)]
        self.group_colors = self.COLORS if group_number <= len(self.COLORS) else self.DIVERGING_COLORS
        if data.index.has_duplicates or data.index.hasnans:
            raise ValueError("Evaluation dates must be unique and non-null.")
        # 持有期由结果中的 IC_{持有期}d 列确定，持有 1 天的列不带后缀；汇总指标和多空方向只用持有 1 天的结果。
        self.holding_periods = [1]
        for column in data.columns:
            if column.startswith("IC_") and column.endswith("d") and column[3:-1].isdigit():
                self.holding_periods.append(int(column[3:-1]))
        return_columns = []
        ic_columns = []
        for holding_period in self.holding_periods:
            suffix = "" if holding_period == 1 else f"_{holding_period}d"
            return_columns += [f"{group}{suffix}" for group in self.GROUPS]
            ic_columns += [f"IC{suffix}", f"rankIC{suffix}"]
        data = data[return_columns + ic_columns + [column for column in data.columns if column.endswith(("_count", "_turnover"))]].astype(float)
        if start_date is not None:
            data = data.loc[data.index >= pd.to_datetime(str(start_date))]
        if data.empty:
            raise ValueError("No evaluation results in the selected date range.")
        if np.isinf(data.to_numpy()).any():
            raise ValueError("Evaluation results contain infinite values.")
        if (data[return_columns] < -1).any().any():
            raise ValueError("Long-only group returns cannot be below -100%.")
        if (data[ic_columns].abs() > 1).any().any():
            raise ValueError("IC and RankIC must be in [-1, 1].")
        if data["IC"].notna().sum() == 0:
            raise ValueError("At least one valid IC value is required to select the long-short direction.")
        self.factor_results = data
        self.group_returns = data[self.GROUPS].dropna()
        if self.group_returns.empty:
            raise ValueError("No dates with valid returns for all five groups.")
        self.ic_mean = data["IC"].mean()
        self.preferred_group = self.GROUPS[-1] if self.ic_mean > 0 else self.GROUPS[0]
        self.short_group = self.GROUPS[0] if self.ic_mean > 0 else self.GROUPS[-1]
        self.long_short = self.group_returns[self.preferred_group] - self.group_returns[self.short_group]
        if (self.long_short < -1).any():
            raise ValueError("Long-short daily loss exceeds 100%; compounded NAV is undefined for this report.")

    def _performance(self, returns):
        returns = returns.dropna()
        wealth = (1 + returns).cumprod()
        total_return = wealth.iloc[-1] - 1
        annual_return = wealth.iloc[-1] ** (self.periods_per_year / len(returns)) - 1
        daily_volatility = returns.std(ddof=1)
        annual_volatility = daily_volatility * np.sqrt(self.periods_per_year)
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / self.periods_per_year) - 1
        sharpe = (returns.mean() - daily_risk_free) / daily_volatility * np.sqrt(self.periods_per_year) if daily_volatility > 0 else np.nan
        # Include starting NAV=1 so a loss on the first date counts as drawdown.
        drawdown = wealth / wealth.cummax().clip(lower=1) - 1
        return [len(returns), total_return, annual_return, annual_volatility, drawdown.min(), sharpe]

    def summary(self):
        """返回与报告口径一致的指标；收益和换手率使用小数。"""
        deviation = self.factor_results["IC"].std(ddof=1)
        return {
            "因子": self.factor_name,
            "IC": self.ic_mean,
            "RankIC": self.factor_results["rankIC"].mean(),
            "ICIR": self.ic_mean / deviation if deviation > 0 else np.nan,
            "多空年化收益": self._performance(self.long_short)[2],
            "多头年化收益": self._performance(self.group_returns[self.preferred_group])[2],
            "多头换手率": self.factor_results.reindex(index=self.group_returns.index, columns=[f"{self.preferred_group}_turnover"]).iloc[:, 0].mean(),
        }

    @staticmethod
    def _format(value, percent=False):
        if pd.isna(value):
            return "—"
        return f"{value:.2%}" if percent else f"{value:.3f}"

    @staticmethod
    def _report_script():
        """Keep interactive calculations local so the exported report works offline."""
        return r"""
class FactorReport {
    constructor(data) {
        this.data = data;
        this.dates = data.dates;
        this.names = data.names;
        this.holdings = data.holdings;
        // 每个窗口的数据按列展开，series[列名] 与 dates 一一对应，缺失为 null。
        this.windows = data.windows.map(item => {
            const series = {};
            data.columns.forEach((column, index) => series[column] = item.data.map(row => row[index]));
            return {label: item.label, column: item.column, color: item.color, series};
        });
        this.current = 0;
        this.holding = 0;
        this.page = 'overview';
        // 嵌入多因子报告时（地址带 #embed）隐藏标题栏。
        if (location.hash === '#embed') document.body.classList.add('embed');
        this.charts = {};
        this.tooltip = document.getElementById('tooltip');
        this.start = document.getElementById('range-start');
        this.end = document.getElementById('range-end');
        this.pan = document.getElementById('range-pan');
        this.start.max = this.end.max = this.dates.length - 1;
        this.end.value = this.dates.length - 1;
        for (const slider of [this.start, this.end]) {
            slider.addEventListener('input', () => {
                if (+this.start.value > +this.end.value) {
                    (slider === this.start ? this.end : this.start).value = slider.value;
                }
                this.schedule();
            });
        }
        this.pan.addEventListener('input', () => {
            const width = +this.end.value - +this.start.value;
            this.start.value = this.pan.value;
            this.end.value = +this.pan.value + width;
            this.schedule();
        });
        document.getElementById('range-reset').addEventListener('click', () => {
            this.start.value = 0;
            this.end.value = this.dates.length - 1;
            this.schedule();
        });
        for (const button of document.querySelectorAll('[role="tab"]')) {
            button.addEventListener('click', () => this.show(button.dataset.page));
        }
        // 顶栏按钮和对比表中的名称都带 data-window / data-holding；点击对比表中的名称还会跳到窗口详情页。
        document.addEventListener('click', event => {
            const button = event.target.closest('[data-window], [data-holding]');
            if (!button) return;
            if (button.dataset.window !== undefined) this.current = +button.dataset.window;
            if (button.dataset.holding !== undefined) this.holding = +button.dataset.holding;
            if (button.classList.contains('window-link')) this.show('window'); else this.render();
        });
        // 打印前渲染全部页面，隐藏页也有完整图表。
        addEventListener('beforeprint', () => this.render(true));
        this.render();
    }

    show(page) {
        this.page = page;
        for (const tab of document.querySelectorAll('[role="tab"]')) {
            const active = tab.dataset.page === page;
            tab.setAttribute('aria-selected', String(active));
            document.getElementById('page-' + tab.dataset.page).hidden = !active;
        }
        document.getElementById('window-picker').hidden = page === 'overview';
        document.getElementById('holding-picker').hidden = page === 'holdings';
        document.getElementById('range-controls').hidden = page === 'years';
        this.render();
        // 从页面下方切换时回到导航栏位置，新页面从头开始显示。
        const header = document.querySelector('header');
        if (scrollY > header.offsetTop + header.offsetHeight) scrollTo({top: header.offsetTop + header.offsetHeight});
    }

    schedule() {
        cancelAnimationFrame(this.frame);
        this.frame = requestAnimationFrame(() => this.render());
    }

    render(all = false) {
        const start = +this.start.value;
        const end = +this.end.value;
        this.pan.max = this.dates.length - (end - start + 1);
        this.pan.value = start;
        this.pan.disabled = +this.pan.max === 0;
        document.getElementById('start-date').textContent = this.dates[start];
        document.getElementById('end-date').textContent = this.dates[end];
        document.getElementById('range-summary').textContent = `${this.dates[start]} — ${this.dates[end]} · ${end - start + 1} 个评估日`;
        for (const button of document.querySelectorAll('#window-picker [data-window]')) {
            button.setAttribute('aria-pressed', String(+button.dataset.window === this.current));
        }
        for (const button of document.querySelectorAll('#holding-picker [data-holding]')) {
            button.setAttribute('aria-pressed', String(+button.dataset.holding === this.holding));
        }
        for (const element of document.querySelectorAll('.window-name')) {
            element.textContent = this.windows[this.current].label;
        }
        for (const element of document.querySelectorAll('.holding-name')) {
            element.textContent = this.holdings[this.holding].label;
        }
        if (all || this.page === 'overview') this.renderOverview(start, end);
        if (all || this.page === 'holdings') this.renderHoldings(start, end);
        if (all || this.page === 'window') this.renderWindow(start, end);
        if (all || this.page === 'years') this.renderYears();
    }

    mean(values) {
        const valid = values.filter(value => value !== null && Number.isFinite(value));
        return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : NaN;
    }

    deviation(values) {
        const valid = values.filter(value => value !== null && Number.isFinite(value));
        if (valid.length < 2) return NaN;
        const average = this.mean(valid);
        return Math.sqrt(valid.reduce((sum, value) => sum + (value - average) ** 2, 0) / (valid.length - 1));
    }

    format(value, percent = false, digits = null) {
        if (value === null || value === undefined || !Number.isFinite(value)) return '—';
        const places = digits === null ? (percent ? 2 : 3) : digits;
        return (value * (percent ? 100 : 1)).toFixed(places) + (percent ? '%' : '');
    }

    escape(text) {
        return String(text).replace(/[&<>"]/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[character]);
    }

    // 输入与评估日对齐、缺失为 null 的日收益；指标只用有效日，曲线在缺失日保持持平，首项为区间起点。
    performance(values) {
        const returns = values.filter(value => value !== null);
        if (!returns.length || returns.some(value => value < -1)) {
            return {metrics: [returns.length, NaN, NaN, NaN, NaN, NaN], cumulative: [], drawdown: []};
        }
        let wealth = 1;
        let peak = 1;
        let minimum = 0;
        const cumulative = [0];
        const drawdown = [0];
        for (const value of values) {
            if (value !== null) {
                wealth *= 1 + value;
                peak = Math.max(peak, wealth);
                minimum = Math.min(minimum, wealth / peak - 1);
            }
            cumulative.push(wealth - 1);
            drawdown.push(wealth / peak - 1);
        }
        const volatility = this.deviation(returns);
        const scale = Math.sqrt(this.data.periods);
        const dailyRiskFree = (1 + this.data.riskFree) ** (1 / this.data.periods) - 1;
        return {
            metrics: [returns.length, wealth - 1, wealth ** (this.data.periods / returns.length) - 1,
                volatility * scale, minimum, volatility > 0 ? (this.mean(returns) - dailyRiskFree) / volatility * scale : NaN],
            cumulative, drawdown
        };
    }

    cumulativeSum(values) {
        let total = 0;
        const result = [0];
        for (const value of values) {
            if (value !== null) total += value;
            result.push(total);
        }
        return result;
    }

    // 返回 [有效观测数, 均值, 标准差, IR, 年化 IR, 大于零占比]；持有 days 天的 IC 相邻日期重叠，年化 IR 按 √(年化交易日数 / days) 折算。
    icStats(values, days) {
        const valid = values.filter(value => value !== null);
        const deviation = this.deviation(valid);
        const ratio = deviation > 0 ? this.mean(valid) / deviation : NaN;
        return [valid.length, this.mean(valid), deviation, ratio, ratio * Math.sqrt(this.data.periods / days),
            valid.length ? valid.filter(value => value > 0).length / valid.length : NaN];
    }

    // 计算一个窗口在 [start, end] 内、第 holding 个持有期的全部结果；多空方向由区间 IC 均值确定，任一组收益缺失的日期在所有组中共同剔除。
    analyze(item, start, end, holding) {
        const suffix = this.holdings[holding].suffix;
        const pick = column => (item.series[column] || []).slice(start, end + 1);
        const groupColumns = this.names.map((_, index) => 'group_' + (index + 1) + suffix);
        const dates = this.dates.slice(start, end + 1);
        const raw = groupColumns.map(pick);
        const complete = dates.map((_, day) => raw.every(values => values[day] !== null && values[day] !== undefined));
        const onComplete = column => {
            const values = pick(column);
            return dates.map((_, day) => complete[day] && values[day] !== undefined ? values[day] : null);
        };
        const ic = pick('IC' + suffix);
        const icMean = this.mean(ic);
        const preferred = Number.isFinite(icMean) ? (icMean > 0 ? this.names.length - 1 : 0) : null;
        const short = preferred === null ? null : this.names.length - 1 - preferred;
        const groups = groupColumns.map(onComplete);
        const spread = groups[0].map((value, day) => preferred === null || value === null ? null : groups[preferred][day] - groups[short][day]);
        return {
            dates, ic, rankIC: pick('rankIC' + suffix), days: this.holdings[holding].days, preferred, short, complete, groups, spread,
            counts: groupColumns.map(column => onComplete(column + '_count')),
            turnovers: groupColumns.map(column => onComplete(column + '_turnover')),
            performances: groups.map(values => this.performance(values)),
            longShort: this.performance(spread)
        };
    }

    table(headers, rows) {
        return '<table class="metrics"><thead><tr>' + headers.map(value => `<th>${value}</th>`).join('')
            + '</tr></thead><tbody>' + rows.map(row => '<tr>' + row.map(value => `<td>${value}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
    }

    metricRow(label, returns, counts, turnover) {
        const metrics = this.performance(returns).metrics;
        return [label, metrics[0], this.format(this.mean(counts), false, 1), this.format(this.mean(turnover), true)]
            .concat(metrics.slice(1).map((value, index) => this.format(value, index < 4)));
    }

    // 热力表：正值为红、负值为蓝，颜色深浅与表内最大绝对值成比例；row.long 为需要加框的列序号。
    heatTable(corner, headers, rows, percent) {
        let scale = 0;
        for (const row of rows) {
            for (const value of row.values) {
                if (Number.isFinite(value)) scale = Math.max(scale, Math.abs(value));
            }
        }
        let html = `<table class="metrics heatmap"><thead><tr><th>${corner}</th>` + headers.map(value => `<th>${this.escape(value)}</th>`).join('') + '</tr></thead><tbody>';
        for (const row of rows) {
            html += `<tr><td>${this.escape(row.label)}</td>`;
            row.values.forEach((value, index) => {
                const text = this.format(value, percent);
                const alpha = Number.isFinite(value) && scale > 0 ? Math.abs(value) / scale * 0.85 : 0;
                const color = value > 0 ? '199, 72, 69' : '28, 92, 171';
                html += `<td class="heat${index === row.long ? ' long' : ''}" style="background: rgba(${color}, ${alpha.toFixed(3)}); color: ${alpha > 0.5 ? '#ffffff' : '#172b4d'}" title="${this.escape(row.label)} · ${this.escape(headers[index])}：${text}">${text}</td>`;
            });
            html += '</tr>';
        }
        return html + '</tbody></table>';
    }

    // 直接绘制 SVG，不依赖 CDN 或外部图表库；折线图带十字准线和提示框，两条以上的线显示图例。
    chart(target, labels, series, options = {}) {
        const container = document.getElementById(target);
        const percent = options.percent !== false;
        let low = 0;
        let high = 0;
        let count = 0;
        for (const item of series) {
            for (const value of item.values) {
                if (!Number.isFinite(value)) continue;
                low = Math.min(low, value);
                high = Math.max(high, value);
                count++;
            }
        }
        if (!count) {
            container.innerHTML = '<p class="empty">所选区间没有可用数据。</p>';
            delete this.charts[target];
            return;
        }
        const padding = (high - low || 0.01) * (options.bars ? 0.15 : 0.04);
        low -= padding;
        high += padding;
        // 纵轴刻度间隔取 1、2、5 × 10^k，上下界落在整刻度上。
        const rough = (high - low) / 4;
        const magnitude = 10 ** Math.floor(Math.log10(rough));
        const step = [1, 2, 5, 10].map(multiple => multiple * magnitude).find(value => value >= rough);
        const digits = Math.max(0, -Math.floor(Math.log10(step * (percent ? 100 : 1)) + 1e-9));
        low = Math.floor(low / step) * step;
        high = Math.ceil(high / step) * step;
        const left = 84, top = 14, width = 990, height = 270;
        const x = index => left + (options.bars ? (index + 0.5) / labels.length : index / Math.max(labels.length - 1, 1)) * width;
        const y = value => top + (high - value) / (high - low) * height;
        let legend = '';
        if (series.length > 1) {
            legend = '<div class="legend">' + series.map(item => `<span><i class="key" style="background: ${item.color}"></i>${this.escape(item.name)}</span>`).join('') + '</div>';
        }
        let svg = `<svg viewBox="0 0 1120 320" role="img" aria-label="${this.escape(container.dataset.label || '')}">`;
        for (let index = 0; index <= Math.round((high - low) / step); index++) {
            const value = Math.abs(low + index * step) < step / 1e6 ? 0 : low + index * step;
            svg += `<line class="grid" x1="${left}" x2="${left + width}" y1="${y(value)}" y2="${y(value)}"/>`;
            svg += `<text x="${left - 12}" y="${y(value) + 4}" text-anchor="end">${this.format(value, percent, digits)}</text>`;
        }
        svg += `<line class="zero" x1="${left}" x2="${left + width}" y1="${y(0)}" y2="${y(0)}"/>`;
        const ticks = options.bars ? labels.length : Math.min(labels.length, 7);
        for (let tick = 0; tick < ticks; tick++) {
            const index = ticks === 1 ? 0 : Math.round(tick * (labels.length - 1) / (ticks - 1));
            svg += `<text x="${x(index)}" y="${top + height + 28}" text-anchor="middle">${this.escape(labels[index])}</text>`;
        }
        if (options.bars) {
            const barWidth = Math.min(72, width / labels.length * 0.6);
            series[0].values.forEach((value, index) => {
                if (!Number.isFinite(value)) return;
                svg += `<rect x="${x(index) - barWidth / 2}" y="${Math.min(y(0), y(value))}" width="${barWidth}" height="${Math.max(Math.abs(y(value) - y(0)), 1)}" rx="3" fill="${this.data.colors[index % this.data.colors.length]}"><title>${this.escape(labels[index])}：${this.format(value, percent)}</title></rect>`;
                svg += `<text class="value" x="${x(index)}" y="${value >= 0 ? y(value) - 8 : y(value) + 17}" text-anchor="middle">${this.format(value, percent)}</text>`;
            });
        } else {
            for (const item of series) {
                let path = '';
                let move = true;
                item.values.forEach((value, index) => {
                    if (!Number.isFinite(value)) { move = true; return; }
                    path += `${move ? 'M' : 'L'}${x(index).toFixed(2)},${y(value).toFixed(2)} `;
                    move = false;
                });
                svg += `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="2" stroke-linejoin="round"/>`;
            }
            svg += `<line class="crosshair" x1="0" x2="0" y1="${top}" y2="${top + height}" visibility="hidden"/>`;
        }
        container.innerHTML = legend + svg + '</svg>';
        this.charts[target] = {labels, series, percent, left, width};
        if (!options.bars && !container.dataset.hover) {
            container.dataset.hover = '1';
            container.addEventListener('pointermove', event => this.hover(container, event));
            container.addEventListener('pointerleave', () => {
                this.tooltip.hidden = true;
                const line = container.querySelector('.crosshair');
                if (line) line.setAttribute('visibility', 'hidden');
            });
        }
    }

    // 准线吸附到最近的评估日，提示框列出该日全部曲线的数值；名称用 textContent 写入。
    hover(container, event) {
        const chart = this.charts[container.id];
        const svg = container.querySelector('svg');
        if (!chart || !svg) return;
        const box = svg.getBoundingClientRect();
        const count = chart.labels.length;
        const position = ((event.clientX - box.left) / box.width * 1120 - chart.left) / chart.width;
        const index = Math.max(0, Math.min(count - 1, Math.round(position * (count - 1))));
        const x = chart.left + index / Math.max(count - 1, 1) * chart.width;
        const line = svg.querySelector('.crosshair');
        line.setAttribute('x1', x);
        line.setAttribute('x2', x);
        line.setAttribute('visibility', 'visible');
        const title = document.createElement('div');
        title.className = 'tooltip-title';
        title.textContent = chart.labels[index];
        this.tooltip.replaceChildren(title);
        for (const item of chart.series) {
            const row = document.createElement('div');
            const key = document.createElement('i');
            key.className = 'key';
            key.style.background = item.color;
            const value = document.createElement('strong');
            value.textContent = this.format(item.values[index], chart.percent);
            const name = document.createElement('span');
            name.textContent = item.name;
            row.append(key, value, name);
            this.tooltip.append(row);
        }
        this.tooltip.hidden = false;
        const size = this.tooltip.getBoundingClientRect();
        this.tooltip.style.left = (event.clientX + 16 + size.width > innerWidth ? event.clientX - 16 - size.width : event.clientX + 16) + 'px';
        this.tooltip.style.top = Math.min(event.clientY + 16, innerHeight - size.height - 8) + 'px';
    }

    // 对比表：results 与 rows 一一对应，每行一个窗口或持有期，rows[i].attribute 为点击行名时切换的选择；各列最优值加粗。
    compareTable(target, header, rows, results) {
        // 每列依次为：列名、是否百分比、最优值规则（abs 绝对值最大、max 最大、min 最小、空为不比较）。
        const columns = [['IC 均值', false, 'abs'], ['RankIC 均值', false, 'abs'], ['ICIR', false, 'abs'], ['RankICIR', false, 'abs'], ['IC &gt; 0 占比', true, ''],
            ['多空年化收益', true, 'max'], ['多空 Sharpe', false, 'max'], ['多空最大回撤', true, 'max'], ['多头年化收益', true, 'max'], ['多头 Sharpe', false, 'max'], ['多头平均换手率', true, 'min']];
        const values = results.map(result => {
            const ic = this.icStats(result.ic, result.days);
            const rankIC = this.icStats(result.rankIC, result.days);
            const long = result.preferred === null ? [] : result.performances[result.preferred].metrics;
            const turnover = result.preferred === null ? NaN : this.mean(result.turnovers[result.preferred]);
            return [ic[1], rankIC[1], ic[3], rankIC[3], ic[5], result.longShort.metrics[2], result.longShort.metrics[5], result.longShort.metrics[4], long[2], long[5], turnover];
        });
        const best = columns.map((column, index) => {
            let bestRow = -1;
            let bestScore = -Infinity;
            values.forEach((row, rowIndex) => {
                if (!column[2] || !Number.isFinite(row[index])) return;
                const score = column[2] === 'abs' ? Math.abs(row[index]) : column[2] === 'min' ? -row[index] : row[index];
                if (score > bestScore) {
                    bestScore = score;
                    bestRow = rowIndex;
                }
            });
            return values.length > 1 ? bestRow : -1;
        });
        let html = `<table class="metrics compare"><thead><tr><th>${header}</th><th>完整收益日</th><th>多头组</th>` + columns.map(column => `<th>${column[0]}</th>`).join('') + '</tr></thead><tbody>';
        results.forEach((result, rowIndex) => {
            const row = rows[rowIndex];
            html += `<tr><td><button type="button" class="window-link" ${row.attribute} title="${this.escape(row.title)}"><i class="key" style="background: ${row.color}"></i>${this.escape(row.label)}</button></td>`;
            html += `<td>${result.complete.filter(Boolean).length}</td><td>${result.preferred === null ? '—' : this.names[result.preferred]}</td>`;
            values[rowIndex].forEach((value, index) => {
                html += `<td${best[index] === rowIndex ? ' class="best"' : ''}>${this.format(value, columns[index][1])}</td>`;
            });
            html += '</tr>';
        });
        document.getElementById(target).innerHTML = html + '</tbody></table>';
    }

    renderOverview(start, end) {
        const results = this.windows.map(item => this.analyze(item, start, end, this.holding));
        this.compareTable('compare-table', '窗口', this.windows.map((item, window) => ({label: item.label, color: item.color, title: item.column, attribute: `data-window="${window}"`})), results);
        document.getElementById('group-heatmap').innerHTML = this.heatTable('窗口', this.names, results.map((result, window) => ({
            label: this.windows[window].label, values: result.performances.map(performance => performance.metrics[2]), long: result.preferred
        })), true);
        const labels = ['起点', ...results[0].dates];
        const windowSeries = values => results.map((result, window) => ({name: this.windows[window].label, color: this.windows[window].color, values: values(result)}));
        this.chart('compare-ls', labels, windowSeries(result => result.longShort.cumulative));
        this.chart('compare-long', labels, windowSeries(result => result.preferred === null ? [] : result.performances[result.preferred].cumulative));
        this.chart('compare-ic', labels, windowSeries(result => this.cumulativeSum(result.ic)), {percent: false});
        this.chart('compare-rankic', labels, windowSeries(result => this.cumulativeSum(result.rankIC)), {percent: false});
    }

    // 持有期对比：当前窗口在各持有期下的结果，各持有期分别由区间 IC 均值确定多头组。
    renderHoldings(start, end) {
        const item = this.windows[this.current];
        const results = this.holdings.map((_, holding) => this.analyze(item, start, end, holding));
        this.compareTable('holding-table', '持有期', this.holdings.map((holding, index) => ({label: holding.label, color: holding.color, title: holding.label, attribute: `data-holding="${index}"`})), results);
        document.getElementById('holding-heatmap').innerHTML = this.heatTable('持有期', this.names, results.map((result, holding) => ({
            label: this.holdings[holding].label, values: result.performances.map(performance => performance.metrics[2]), long: result.preferred
        })), true);
        const labels = ['起点', ...results[0].dates];
        const holdingSeries = values => results.map((result, holding) => ({name: this.holdings[holding].label, color: this.holdings[holding].color, values: values(result)}));
        this.chart('holding-ls', labels, holdingSeries(result => result.longShort.cumulative));
        this.chart('holding-long', labels, holdingSeries(result => result.preferred === null ? [] : result.performances[result.preferred].cumulative));
    }

    renderWindow(start, end) {
        const item = this.windows[this.current];
        const result = this.analyze(item, start, end, this.holding);
        const preferred = result.preferred;
        const short = result.short;
        const long = preferred === null ? [] : result.performances[preferred].metrics;
        const ic = this.icStats(result.ic, result.days);
        const rankIC = this.icStats(result.rankIC, result.days);
        [ic[1], rankIC[1], ic[3], result.longShort.metrics[2], long[2], long[5]].forEach((value, index) => {
            document.getElementById('card-' + index).textContent = this.format(value, index === 3 || index === 4);
        });
        const completeDays = result.complete.filter(Boolean).length;
        document.getElementById('window-summary').textContent = `因子列 ${item.column} · ${this.holdings[this.holding].label} · ${result.dates.length} 个评估日 · ${completeDays} 个完整收益日 · 剔除 ${result.dates.length - completeDays} 个收益缺失日`;
        document.getElementById('direction-note').textContent = preferred === null
            ? '所选区间没有有效 IC，无法确定多头组和多空方向。'
            : `当前方向：${this.names[preferred]} − ${this.names[short]}，由所选区间 IC 均值确定。` + (result.spread.some(value => value !== null && value < -1) ? '该区间多空单日亏损超过 100%，不计算多空复利指标和曲线。' : '');
        const records = result.groups.map((returns, index) => this.metricRow(this.names[index] + (index === preferred ? ' · 多头' : ''), returns, result.counts[index], result.turnovers[index]));
        records.push(this.metricRow('Long-short', result.spread, result.counts[0].map((_, day) => {
            if (preferred === null || result.counts[preferred][day] === null || result.counts[short][day] === null) return null;
            return result.counts[preferred][day] + result.counts[short][day];
        }), []));
        document.getElementById('group-table').innerHTML = this.table(
            ['组合', '有效交易日', '日均股票数量', '平均换手率', '区间收益', '年化收益', '年化波动率', '最大回撤', 'Sharpe'], records);
        document.getElementById('ic-table').innerHTML = this.table(['指标', '有效观测数', '均值', '标准差', 'IR（均值 / 标准差）', '年化 IR', '大于零占比'],
            [['IC', ic], ['RankIC', rankIC]].map(([label, stats]) => [label, stats[0]].concat(stats.slice(1).map((value, index) => this.format(value, index === 4)))));
        const labels = ['起点', ...result.dates];
        const groupSeries = key => result.performances.map((performance, index) => ({name: this.names[index], color: this.data.colors[index], values: performance[key]}));
        this.chart('group-cumulative', labels, groupSeries('cumulative'));
        this.chart('group-drawdown', labels, groupSeries('drawdown'));
        this.chart('annual-returns', this.names, [{name: '年化收益', values: result.performances.map(performance => performance.metrics[2])}], {bars: true});
        this.chart('ls-cumulative', labels, [{name: 'Long-short', color: item.color, values: result.longShort.cumulative}]);
        this.chart('ls-drawdown', labels, [{name: 'Long-short', color: item.color, values: result.longShort.drawdown}]);
        this.chart('window-ic', labels, [{name: '累积 IC', color: '#2a78d6', values: this.cumulativeSum(result.ic)},
            {name: '累积 RankIC', color: '#eb6834', values: this.cumulativeSum(result.rankIC)}], {percent: false});
    }

    // 分年页始终使用完整区间，各窗口方向由完整区间 IC 均值确定，持有 1 天时与持仓清单一致。
    renderYears() {
        const results = this.windows.map(item => this.analyze(item, 0, this.dates.length - 1, this.holding));
        const years = [];
        this.dates.forEach((date, index) => {
            if (!years.length || years[years.length - 1].year !== date.slice(0, 4)) years.push({year: date.slice(0, 4), start: index});
            years[years.length - 1].end = index;
        });
        const slice = (values, year) => values.slice(year.start, year.end + 1);
        const headers = this.windows.map(item => item.label);
        document.getElementById('yearly-ls').innerHTML = this.heatTable('年份', headers, years.map(year => ({
            label: year.year, values: results.map(result => this.performance(slice(result.spread, year)).metrics[1])
        })), true);
        document.getElementById('yearly-ic').innerHTML = this.heatTable('年份', headers, years.map(year => ({
            label: year.year, values: results.map(result => this.mean(slice(result.ic, year)))
        })), false);

        const result = results[this.current];
        const preferred = result.preferred;
        document.getElementById('yearly-long').textContent = preferred === null ? '无有效 IC' : `多头组 ${this.names[preferred]}`;
        document.getElementById('yearly-table').innerHTML = preferred === null ? '<p class="empty">没有有效 IC，无法确定多头组。</p>' : this.table(
            ['年份', '有效交易日', '日均股票数量', '平均换手率', '区间收益', '年化收益', '年化波动率', '最大回撤', 'Sharpe'],
            years.map(year => this.metricRow(year.year, slice(result.groups[preferred], year), slice(result.counts[preferred], year), slice(result.turnovers[preferred], year))));
        const container = document.getElementById('yearly-curves');
        container.replaceChildren();
        for (const year of years) {
            const heading = document.createElement('h3');
            heading.textContent = year.year + ' · 各组累计收益';
            const chart = document.createElement('div');
            chart.id = 'year-' + year.year;
            chart.className = 'chart';
            chart.dataset.label = heading.textContent;
            container.append(heading, chart);
            this.chart(chart.id, ['起点', ...this.dates.slice(year.start, year.end + 1)], result.groups.map((returns, index) => ({
                name: this.names[index], color: this.data.colors[index], values: this.performance(slice(returns, year)).cumulative
            })));
        }
    }
}
new FactorReport(JSON.parse(document.getElementById('report-data').textContent));
"""

    @staticmethod
    def plot_results_html(factor_path, results_dir, start_date=None, output_dir=None):
        """将一个因子文件中原始值和各滚动均值列的评估结果合并为一份离线 HTML 报告，每个因子只输出一份，写入 output_dir/single_factor_evaluation。"""
        factor_path = Path(factor_path)
        factor_name = factor_path.stem
        # 窗口顺序沿用因子文件中的列顺序（原始值在前，滚动窗口由短到长），只合并已有评估结果的列。
        visualizers = []
        for column in pq.read_schema(factor_path).names:
            if column not in ("code", "date") and (Path(results_dir) / f"{column}.csv").is_file():
                visualizers.append(ResultsVisualizer(Path(results_dir) / f"{column}.csv", column, start_date, output_dir))
        if not visualizers:
            raise ValueError(f"No evaluation results for factor {factor_name} in {results_dir}.")
        output_dir = visualizers[0].output_dir
        groups = visualizers[0].GROUPS

        # 持有期取各窗口结果的并集，由短到长排列；每个持有期一组列，持有 1 天的列不带后缀。
        holding_periods = []
        for visualizer in visualizers:
            for holding_period in visualizer.holding_periods:
                if holding_period not in holding_periods:
                    holding_periods.append(holding_period)
        holding_periods.sort()
        columns = []
        holdings = []
        holding_picker = ""
        for index, holding_period in enumerate(holding_periods):
            suffix = "" if holding_period == 1 else f"_{holding_period}d"
            columns += [f"{group}{suffix}" for group in groups] + [f"IC{suffix}", f"rankIC{suffix}"] + [f"{group}{suffix}_count" for group in groups] + [f"{group}{suffix}_turnover" for group in groups]
            holdings.append({"days": holding_period, "suffix": suffix, "label": f"持有 {holding_period} 天", "color": ResultsVisualizer.WINDOW_COLORS[index % len(ResultsVisualizer.WINDOW_COLORS)]})
            holding_picker += f'<button type="button" data-holding="{index}" aria-pressed="{str(index == 0).lower()}">{holding_period} 天</button>'

        # 各窗口按评估日期的并集对齐，某个窗口缺失的日期留空。
        dates = visualizers[0].factor_results.index
        for visualizer in visualizers[1:]:
            dates = dates.union(visualizer.factor_results.index)
        windows = []
        picker = ""
        for index, visualizer in enumerate(visualizers):
            suffix = visualizer.factor_name.removeprefix(f"{factor_name}_")
            if visualizer.factor_name == factor_name:
                label = "原始值"
            elif suffix.endswith("_m"):
                label = f"{suffix[:-2]} 日均值"
            else:
                label = suffix
            color = ResultsVisualizer.WINDOW_COLORS[index % len(ResultsVisualizer.WINDOW_COLORS)]
            # 数值保留 6 位小数（日收益精确到 0.0001%），控制多窗口、多持有期报告的文件大小。
            payload = visualizer.factor_results.reindex(index=dates, columns=columns)
            windows.append({"label": label, "column": visualizer.factor_name, "color": color, "data": json.loads(payload.to_json(orient="values", double_precision=6))})
            picker += f'<button type="button" data-window="{index}" aria-pressed="{str(index == 0).lower()}" title="{escape(visualizer.factor_name)}"><i class="key" style="background: {color}"></i>{escape(label)}</button>'

        report_data = json.dumps({"dates": list(dates.strftime("%Y-%m-%d")), "columns": columns, "windows": windows, "holdings": holdings,
                                  "periods": visualizers[0].periods_per_year, "riskFree": visualizers[0].risk_free_rate,
                                  "colors": visualizers[0].group_colors, "names": [f"G{group}" for group in range(1, len(groups) + 1)]}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        cards = "".join(f'<div class="stat"><span>{label}</span><strong id="card-{index}">—</strong></div>'
                        for index, label in enumerate(["IC 均值", "RankIC 均值", "ICIR", "多空年化收益", "多头年化收益", "多头 Sharpe"]))
        title = escape(factor_name)
        window_labels = escape("、".join(window["label"] for window in windows))
        holding_labels = "、".join(str(holding_period) for holding_period in holding_periods)
        last_group = f"G{len(groups)}"
        html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 因子评估报告</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif; color: #172b4d; background: #f2f5fa; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; background: #f2f5fa; }} main {{ max-width: 1280px; margin: auto; padding: 24px 28px; }} [hidden] {{ display: none !important; }}
header {{ padding: 30px 32px; background: linear-gradient(120deg, #172c52, #28578c); color: white; border-radius: 20px; }}
.eyebrow {{ color: #9edcfa; font-size: 12px; letter-spacing: 3px; }} h1 {{ font-size: 30px; margin: 14px 0 10px; overflow-wrap: anywhere; }} header p {{ margin: 0; color: #d2dfef; line-height: 1.8; }}
.toolbar {{ position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px 18px; padding: 14px 0; background: #f2f5fa; }}
.tabs, .picker {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }} .picker > span {{ font-size: 13px; color: #6b7c94; margin-right: 2px; }}
button {{ font: inherit; cursor: pointer; }} button:focus-visible, input:focus-visible {{ outline: 3px solid #06a6a0; outline-offset: 3px; }}
.tabs button {{ border: 1px solid #dce3ed; border-radius: 10px; padding: 11px 20px; background: white; color: #28578c; }}
.tabs button[aria-selected="true"] {{ color: white; background: #28578c; border-color: #28578c; }}
.picker button {{ border: 1px solid #dce3ed; border-radius: 999px; padding: 6px 13px; background: white; color: #28578c; font-size: 13px; }}
.picker button[aria-pressed="true"] {{ color: white; background: #172c52; border-color: #172c52; }}
i.key {{ display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 7px; vertical-align: middle; }}
section, .stat {{ background: white; border: 1px solid #e3e9f2; border-radius: 16px; box-shadow: 0 5px 20px #23395605; }}
section {{ padding: 24px 26px; margin-bottom: 20px; }} h2 {{ margin: 0; font-size: 20px; }} h3 {{ margin: 24px 0 4px; font-size: 15px; }}
section p, footer {{ font-size: 13px; color: #6b7c94; line-height: 1.9; }} .content, .chart {{ overflow-x: auto; }} .empty {{ text-align: center; padding: 40px; }}
.section-head {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; }}
.ghost {{ border: 1px solid #dce3ed; border-radius: 8px; padding: 7px 14px; background: #f4f7fb; color: #28578c; font-size: 13px; }}
.range-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 28px; margin: 6px 0 4px; }}
.range-grid label {{ display: flex; justify-content: space-between; gap: 8px; font-size: 13px; color: #6b7c94; margin-bottom: 4px; }} output {{ color: #2563a4; font-weight: 600; }}
input[type="range"] {{ width: 100%; accent-color: #2563eb; cursor: ew-resize; }}
.window-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; margin: 4px 0 14px; }} .window-head h2 {{ font-size: 24px; }} .window-head p {{ margin: 0; font-size: 13px; color: #6b7c94; }}
.stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 20px; }}
.stat {{ padding: 18px 20px; }} .stat span {{ display: block; color: #718096; font-size: 13px; }} .stat strong {{ display: block; margin-top: 10px; font-size: 24px; color: #2563a4; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 28px; }}
svg {{ display: block; width: 100%; height: auto; min-width: 560px; }} svg text {{ font-family: inherit; font-size: 12px; fill: #64748b; }} svg text.value {{ fill: #172b4d; }}
svg .grid {{ stroke: #e8edf5; }} svg .zero {{ stroke: #94a3b8; stroke-dasharray: 4 4; }} svg .crosshair {{ stroke: #475569; stroke-width: 1; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 8px 0 2px; font-size: 12px; color: #52607a; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }} th {{ background: #f4f7fb; color: #61718b; font-weight: 500; }}
th, td {{ padding: 12px 13px; text-align: right; border-bottom: 1px solid #edf1f6; }} th:first-child, td:first-child {{ text-align: left; }} tbody tr:hover {{ background: #f4f8ff; }}
table.compare th, table.compare td {{ padding: 12px 8px; }} table.compare th {{ white-space: normal; vertical-align: bottom; }} table.compare td.best {{ font-weight: 700; color: #1c5cab; background: #eef4fd; }}
.window-link {{ border: 0; padding: 0; background: none; color: #28578c; font-weight: 600; text-decoration: underline dotted #94a3b8; text-underline-offset: 4px; }}
table.heatmap td.heat {{ text-align: center; border: 1px solid white; min-width: 72px; }} table.heatmap td.long {{ box-shadow: inset 0 0 0 2px #172b4d; font-weight: 700; }}
#tooltip {{ position: fixed; z-index: 20; pointer-events: none; min-width: 150px; padding: 10px 12px; background: white; border: 1px solid #dce3ed; border-radius: 10px; box-shadow: 0 8px 24px #1729521f; font-size: 12px; color: #52607a; }}
#tooltip div {{ display: flex; align-items: center; gap: 6px; line-height: 1.7; }} #tooltip strong {{ min-width: 64px; color: #172b4d; font-variant-numeric: tabular-nums; }} #tooltip i.key {{ margin-right: 0; }}
#tooltip .tooltip-title {{ margin-bottom: 4px; color: #172b4d; font-weight: 600; }}
footer {{ padding: 4px 12px 20px; }} .pickers {{ display: flex; flex-wrap: wrap; gap: 10px 22px; }}
body.embed header {{ display: none; }} body.embed main {{ padding-top: 4px; }}
@media(max-width: 1000px) {{ .stats {{ grid-template-columns: repeat(3, 1fr); }} .range-grid, .grid-2 {{ grid-template-columns: 1fr; }} }}
@media(max-width: 700px) {{ main {{ padding: 12px 16px; }} header, section {{ padding: 18px; }} h1 {{ font-size: 24px; }} .stats {{ grid-template-columns: repeat(2, 1fr); gap: 10px; }} .stat strong {{ font-size: 21px; }} .tabs {{ width: 100%; gap: 6px; }} .tabs button {{ flex: 1; padding: 11px 2px; font-size: 14px; }} .toolbar {{ position: static; }} }}
@media print {{ main {{ padding: 0; }} .toolbar, #range-controls, #tooltip {{ display: none !important; }} [role="tabpanel"][hidden] {{ display: block !important; }} section {{ break-inside: avoid; }} .content, .chart {{ overflow: visible; }} svg {{ min-width: 0; }} }}
</style></head><body><main>
<header><div class="eyebrow">FACTOR RESEARCH / PERFORMANCE REPORT</div><h1>{title} · 因子评估报告</h1>
<p>{dates.min():%Y-%m-%d} — {dates.max():%Y-%m-%d} · {len(dates):,} 个评估日 · 每日分 {len(groups)} 组 · {len(windows)} 个窗口：{window_labels} · 持有期：{holding_labels} 天</p></header>
<nav class="toolbar" aria-label="报告导航">
<div class="tabs" role="tablist"><button type="button" role="tab" data-page="overview" aria-controls="page-overview" aria-selected="true">窗口对比</button><button type="button" role="tab" data-page="holdings" aria-controls="page-holdings" aria-selected="false">持有期对比</button><button type="button" role="tab" data-page="window" aria-controls="page-window" aria-selected="false">窗口详情</button><button type="button" role="tab" data-page="years" aria-controls="page-years" aria-selected="false">分年表现</button></div>
<div class="pickers"><div class="picker" id="window-picker" hidden><span>窗口</span>{picker}</div><div class="picker" id="holding-picker"><span>持有期</span>{holding_picker}</div></div>
</nav>
<noscript><p>请启用 JavaScript，以使用页面切换、时间滑块和交互收益图表。</p></noscript>
<section id="range-controls"><div class="section-head"><h2>分析时间区间</h2><button type="button" id="range-reset" class="ghost">恢复完整区间</button></div>
<p id="range-summary" aria-live="polite"></p>
<div class="range-grid">
<div><label for="range-start"><span>开始日期</span><output id="start-date" for="range-start"></output></label><input id="range-start" type="range" min="0" value="0" step="1"></div>
<div><label for="range-end"><span>结束日期</span><output id="end-date" for="range-end"></output></label><input id="range-end" type="range" min="0" value="0" step="1"></div>
<div><label for="range-pan"><span>平移整个区间</span><span>评估日数不变</span></label><input id="range-pan" type="range" min="0" value="0" step="1"></div>
</div><p>拖动滑块后，窗口对比、持有期对比和窗口详情三页的指标与曲线立即重算；分年表现页始终使用完整区间。</p></section>

<div id="page-overview" role="tabpanel" aria-label="窗口对比">
<section><h2>窗口指标对比 · <span class="holding-name"></span></h2><p>每行一个窗口（原始值或滚动均值列），全部指标按所选区间和顶栏选择的持有期计算，各窗口分别由区间 IC 均值确定多头组。蓝底加粗为该列最优：IC 类取绝对值最大，换手率取最小，其余取最大。点击窗口名查看该窗口详情。</p><div id="compare-table" class="content"></div></section>
<section><h2>各窗口分组年化收益 · <span class="holding-name"></span></h2><p>每行一个窗口，从左到右为 G1（因子值最低）到 {last_group}（因子值最高）。红色为正、蓝色为负，颜色越深绝对值越大；加框为该窗口的多头组。用于比较不同窗口的分组单调性。</p><div id="group-heatmap" class="content"></div></section>
<section><h2>各窗口累计收益 · <span class="holding-name"></span></h2><p>按日复利累计，所选区间起点收益为 0；各窗口使用各自的多空方向。横轴为评估日，某窗口收益缺失的日期净值持平。</p>
<h3>多空组合</h3><div id="compare-ls" class="chart" data-label="各窗口多空累计收益"></div><h3>多头组合</h3><div id="compare-long" class="chart" data-label="各窗口多头累计收益"></div></section>
<section><h2>各窗口累积 IC · <span class="holding-name"></span></h2><p>每日 IC / RankIC 的算术累加，所选区间起点为 0；缺失日期不累加。负向因子的曲线向下，斜率越陡越有效。持有期大于 1 天时为该持有期收益的 IC，相邻日期的收益区间重叠。</p>
<h3>累积 IC</h3><div id="compare-ic" class="chart" data-label="各窗口累积 IC"></div><h3>累积 RankIC</h3><div id="compare-rankic" class="chart" data-label="各窗口累积 RankIC"></div></section>
</div>

<div id="page-holdings" role="tabpanel" aria-label="持有期对比" hidden>
<section><h2><span class="window-name"></span> · 持有期指标对比</h2><p>每行一个持有期，均为顶栏所选窗口的结果；指标按所选区间计算，各持有期分别由区间 IC 均值确定多头组。持有 N 天的 IC 为因子值与之后 N 天累计收益的相关系数，可用来观察因子预测力随时间的衰减。收益为日收益口径，可直接比较不同持有期；换手率为买入批与卖出批的成分变动比例，不折算为日均。点击持有期查看详情。</p><div id="holding-table" class="content"></div></section>
<section><h2><span class="window-name"></span> · 各持有期分组年化收益</h2><p>每行一个持有期，从左到右为 G1 到 {last_group}；红色为正、蓝色为负，加框为该持有期的多头组。</p><div id="holding-heatmap" class="content"></div></section>
<section><h2><span class="window-name"></span> · 各持有期累计收益</h2><p>按日复利累计，所选区间起点收益为 0；各持有期使用各自的多空方向。</p>
<h3>多空组合</h3><div id="holding-ls" class="chart" data-label="各持有期多空累计收益"></div><h3>多头组合</h3><div id="holding-long" class="chart" data-label="各持有期多头累计收益"></div></section>
</div>

<div id="page-window" role="tabpanel" aria-label="窗口详情" hidden>
<div class="window-head"><h2><span class="window-name"></span> · <span class="holding-name"></span> · 窗口详情</h2><p id="window-summary"></p></div>
<div class="stats">{cards}</div>
<section><h2>分组绩效</h2><p id="direction-note"></p><p>G1 为因子值最低组，{last_group} 为最高组。各组与多空组合使用相同的完整收益日；股票数量与换手率按这些日期取均值，缺失值不参与均值。多空股票数量为两端之和，不展示多空换手率。</p><div id="group-table" class="content"></div></section>
<section><h2>各组累计收益和动态回撤</h2><p>按日复利累计；所选区间起点收益为 0、净值为 1。动态回撤 = 当前净值 / 区间内历史最高净值 − 1，包含初始净值。</p>
<h3>各组累计收益</h3><div id="group-cumulative" class="chart" data-label="各组累计收益"></div><h3>各组动态回撤</h3><div id="group-drawdown" class="chart" data-label="各组动态回撤"></div></section>
<section><h2>各组年化收益 · 单调性</h2><p>按因子值从低到高排列 G1 → {last_group}。年化收益按所选区间的有效收益日数折算；正向因子观察是否递增，负向因子观察是否递减。</p><div id="annual-returns" class="chart" data-label="各组年化收益"></div></section>
<section><h2>Long-short 累计收益和动态回撤</h2><p>每日多头收益减空头收益，再复利累计；累计收益与回撤均在所选区间起点重置。</p>
<h3>Long-short 累计收益</h3><div id="ls-cumulative" class="chart" data-label="Long-short 累计收益"></div><h3>Long-short 动态回撤</h3><div id="ls-drawdown" class="chart" data-label="Long-short 动态回撤"></div></section>
<section><h2>IC / RankIC</h2><p>所选区间的截面相关性统计，IR 保留方向符号；累积曲线为每日值的算术累加，缺失日期不累加。</p><div id="ic-table" class="content"></div>
<h3>累积 IC 与 RankIC</h3><div id="window-ic" class="chart" data-label="累积 IC 与 RankIC"></div></section>
</div>

<div id="page-years" role="tabpanel" aria-label="分年表现" hidden>
<section><h2>各窗口分年对比 · <span class="holding-name"></span></h2><p>使用完整区间，各窗口的多空方向由完整区间 IC 均值确定。多空收益为该年实际覆盖日期的复利收益，首尾年份可能不完整；红色为正、蓝色为负，颜色深浅在每张表内单独标定。</p>
<div class="grid-2"><div><h3>分年多空收益</h3><div id="yearly-ls" class="content"></div></div><div><h3>分年 IC 均值</h3><div id="yearly-ic" class="content"></div></div></div></section>
<section><h2><span class="window-name"></span> · <span class="holding-name"></span> · 分年绩效（<span id="yearly-long"></span>）</h2><p>多头组由完整区间 IC 均值确定。区间收益为该年实际覆盖日期的复利收益；回撤每年重置。</p><div id="yearly-table" class="content"></div></section>
<section><h2><span class="window-name"></span> · <span class="holding-name"></span> · 分年各组累计收益</h2><p>各年从 0 重新开始，净值从 1 按该年有效日收益复利累计，不继承上一年净值。</p><div id="yearly-curves"></div></section>
</div>
<footer><b>计算口径</b><br>
窗口为因子文件中的各列：原始值和按交易日滚动的均值列（如 5 日均值对应 {title}_5_m），每个窗口单独分组回测，互不影响。<br>
持有期：因子在 t 日收盘计算，t+1 日 10 点按分组买入；持有 1 天在 t+2 日 10 点卖出，持有 N 天在 t+1+N 日 10 点卖出。持有 N 天时每个交易日买入一批、同时持有最近 N 批，各批等权，组合日收益为各批当日收益的均值（重叠持仓，结果不依赖调仓起始日）；起始阶段只平均已买入的批次。
每批只买入当日可交易（有收益）的股票，持有期内停牌的日收益记 0。持有 N 天的 IC / RankIC 为因子值与之后 N 天复利累计收益的截面相关系数，最后 N − 1 个评估日的收益尚未实现，记为空。
持有 N 天的换手率为当日买入批与当日卖出批（N 天前买入）的成分变动比例，不除以 N，前 N 个评估日为空；股票数量为各批的平均股票数。<br>
日收益使用小数；年化交易日数 {visualizers[0].periods_per_year:g}，年化无风险利率 {visualizers[0].risk_free_rate:.2%}。
年化收益 = ∏(1 + 日收益)^(年化交易日数 / 有效交易日数) − 1；年化波动率 = 日收益样本标准差 × √年化交易日数。
Sharpe = (平均日收益 − 等效日无风险利率) / 日收益样本标准差 × √年化交易日数。
ICIR / RankICIR = 均值 / 样本标准差；年化 IR 再乘 √(年化交易日数 / 持有天数)。标准差为零或样本不足时显示「—」。<br>
IC 均值 &gt; 0 时做多 {last_group}、做空 G1，否则做多 G1、做空 {last_group}。每个窗口、每个持有期分别确定方向：窗口对比、持有期对比与窗口详情按所选区间确定，分年表现按完整区间确定，属于事后分析。
多空按多头 100%、空头 100% 的收益差计算，未除以 2；不计手续费、滑点和融券成本。<br>
缺失 IC / RankIC 各自剔除；任一组缺失收益的日期从该窗口的全部收益统计中共同剔除，不填充为零。收益图横轴为评估日，收益缺失日净值持平。
平均换手率沿用原始日换手率，在有效收益日内求均值；拖动区间不重建持仓，首日沿用已有值。<br>
日期沿用评估结果的收益起始日标签，跨年收益按该标签归属年份。所有数据与图表已嵌入，可离线查看。生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}。
</footer></main>
<div id="tooltip" role="tooltip" hidden></div>
<script id="report-data" type="application/json">{report_data}</script><script>{ResultsVisualizer._report_script()}</script></body></html>'''
        # 单因子报告统一放在 single_factor_evaluation 文件夹，与多因子报告 multi_factor_evaluation_report.html 同级；删除旧版直接放在 output_dir 下的报告（按因子或按列输出）。
        (output_dir / "single_factor_evaluation").mkdir(parents=True, exist_ok=True)
        for visualizer in visualizers:
            (output_dir / f"{visualizer.factor_name}_report.html").unlink(missing_ok=True)
        safe_name = "".join(character if character.isalnum() or character in "-_." else "_" for character in factor_name).strip(".") or "factor"
        output_path = output_dir / "single_factor_evaluation" / f"{safe_name}_report.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-name", help="因子文件名（不含滚动窗口后缀）；默认取 specified_column 所属的因子。")
    parser.add_argument("--start-date")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[3] / "config/config_factor_evaluation.json")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from main import load_config

    config = load_config(args.config)
    factor_name = (args.factor_name or config["specified_column"]).split("_")[0]
    output_dir = args.output_dir or config.get("visualization_output_dir")
    print(ResultsVisualizer.plot_results_html(Path(config["factor_data_dir"]) / f"{factor_name}.parquet", config["output_dir"], args.start_date or config["start_date"], output_dir))


if __name__ == "__main__":
    main()
