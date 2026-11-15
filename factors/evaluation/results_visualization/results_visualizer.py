"""汇总全部因子的每日评估结果，生成一份离线 HTML 因子评估报告。"""

import base64
import gzip
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from factors.evaluation.factor_correlation_analysis.factor_correlation_analyzer import FactorCorrelationAnalyzer


class ResultsVisualizer:
    COLORS = ["#2563eb", "#06a6a0", "#e7a32e", "#9764d9", "#ed6976"]
    # 超过五组时用蓝红分歧色：G1–G5 蓝色由深到浅，G6–G10 红色由浅到深，两端（多空组）颜色最深。
    DIVERGING_COLORS = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#ea9a93", "#dd716a", "#c74845", "#9e3432", "#762221"]
    # 因子对比图的颜色和线型按因子顺序固定分配：每 8 个因子用一轮颜色（相邻颜色已校验色盲可分），第一轮为实线，之后依次为虚线、点线、点划线。
    FACTOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    FACTOR_DASHES = ["", "7 4", "2 3", "9 3 2 3"]

    def __init__(self, factor_results, factor_name, start_date=None, periods_per_year=252, risk_free_rate=0.0):
        """Accept a DataFrame or CSV path; returns must be daily decimal returns.

        The date column/index labels the beginning of the return period, matching
        SingleFactorEvaluation. risk_free_rate is an effective annual rate.
        """
        if periods_per_year <= 0 or risk_free_rate <= -1:
            raise ValueError("periods_per_year must be positive and risk_free_rate must exceed -1.")
        self.factor_name = str(factor_name)
        self.start_date = start_date
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
// 因子较多时，对比曲线默认只显示完整区间 |ICIR| 最大的 MAX_VISIBLE 个因子。
const MAX_VISIBLE = 8;

class FactorReport {
    constructor(data) {
        this.data = data;
        this.dates = data.dates;
        this.names = data.names;
        this.holdings = data.holdings;
        // series[持有期序号][列名] 与 dates 一一对应，缺失为 null；某个持有期首次用到时才解压。
        this.factors = data.factors.map(item => ({name: item.name, color: item.color, dash: item.dash, series: []}));
        this.current = 0;
        this.holding = 0;
        this.page = 'detail';
        // 对比表的排序列，null 为因子原始顺序；分组年化收益和分年对比的行顺序与对比表一致。
        this.sort = null;
        this.order = this.factors.map((_, index) => index);
        this.visible = new Set();
        // 因子对比的累计收益图默认用对数纵轴，可切换为线性。
        this.log = true;
        this.full = {};
        this.charts = {};
        // 每个年份在 dates 中的起止位置，分年统计使用。
        this.years = [];
        this.dates.forEach((date, index) => {
            if (!this.years.length || this.years[this.years.length - 1].year !== date.slice(0, 4)) this.years.push({year: date.slice(0, 4), start: index});
            this.years[this.years.length - 1].end = index;
        });
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
        document.getElementById('factor-picker').innerHTML = this.factors.map((factor, index) => `<button type="button" data-factor="${index}" aria-pressed="false">${this.key(factor)}${this.escape(factor.name)}</button>`).join('');
        document.addEventListener('click', event => this.click(event));
        // 悬停图例时突出该因子的曲线。
        for (const legend of document.querySelectorAll('.legend-toggles')) {
            legend.addEventListener('pointerover', event => this.focus(event.target.closest('[data-toggle]')));
            legend.addEventListener('pointerleave', () => this.focus(null));
        }
        // 已画过的图离开视口附近后只记下最新数据，滚动到附近时再画，拖动滑块时只重画看得见的图。
        this.pending = {};
        this.nearby = new Set();
        this.observer = new IntersectionObserver(entries => {
            for (const entry of entries) {
                if (!entry.isIntersecting) {
                    this.nearby.delete(entry.target.id);
                    continue;
                }
                this.nearby.add(entry.target.id);
                if (this.pending[entry.target.id]) this.chart(entry.target.id, ...this.pending[entry.target.id]);
            }
        }, {rootMargin: '300px 0px'});
        for (const element of document.querySelectorAll('.chart')) {
            this.observer.observe(element);
        }
        // 打印前渲染全部页面并画出全部图表，隐藏页也有完整图表。
        addEventListener('beforeprint', () => {
            this.printing = true;
            this.render(true);
            this.printing = false;
        });
        this.load(0).then(() => {
            this.visible = this.defaultVisible();
            this.renderLegend();
            this.render();
        }).catch(error => {
            const notice = document.getElementById('load-error');
            notice.textContent = '报告数据解压失败，请使用新版 Chrome、Edge、Firefox 或 Safari 打开。' + error;
            notice.hidden = false;
        });
    }

    // 每个因子、每个持有期的数据以 gzip + base64 嵌入页面；某个持有期首次用到时，解压全部因子在该持有期的数据并缓存。
    load(holding) {
        return Promise.all(this.factors.map(async (factor, index) => {
            if (factor.series[holding]) return;
            const text = atob(document.getElementById(`data-${index}-${holding}`).textContent);
            const bytes = new Uint8Array(text.length);
            for (let position = 0; position < text.length; position++) bytes[position] = text.charCodeAt(position);
            const values = JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
            factor.series[holding] = {};
            this.data.columns.forEach((column, position) => factor.series[holding][column] = values[position]);
        }));
    }

    // 页面上的按钮都带 data-* 属性，统一在这里处理：持有期、因子、标签页、对比表排序、累计收益图的纵轴和图例开关。
    click(event) {
        const target = event.target.closest('[data-holding], [data-factor], [data-page], [data-sort], [data-scale], [data-toggle], [data-legend]');
        if (!target) return;
        const set = target.dataset;
        if (set.holding !== undefined) {
            this.holding = +set.holding;
            this.load(this.holding).then(() => this.render());
        } else if (set.factor !== undefined) {
            this.current = +set.factor;
            // 点击对比表中的因子名时切换到因子详情，并跳到单因子表现。
            if (target.classList.contains('factor-link')) {
                this.show('detail');
                document.getElementById('single').scrollIntoView();
            } else {
                this.render();
            }
        } else if (set.page !== undefined) {
            this.show(set.page);
        } else if (set.sort !== undefined) {
            const column = +set.sort;
            this.sort = column < 0 ? null : {column, descending: this.sort && this.sort.column === column ? !this.sort.descending : true};
            this.render();
        } else if (set.scale !== undefined) {
            this.log = set.scale === 'log';
            this.renderCurves();
        } else if (set.toggle !== undefined) {
            const index = +set.toggle;
            if (this.visible.has(index)) this.visible.delete(index); else this.visible.add(index);
            this.renderLegend();
            this.renderCurves();
        } else {
            this.visible = set.legend === 'all' ? new Set(this.factors.map((_, index) => index)) : set.legend === 'none' ? new Set() : this.defaultVisible();
            this.renderLegend();
            this.renderCurves();
        }
    }

    // 只突出已显示的因子；移出图例时恢复。
    focus(button) {
        const index = button && this.visible.has(+button.dataset.toggle) ? button.dataset.toggle : null;
        for (const path of document.querySelectorAll('.curves path.series')) path.classList.toggle('faded', index !== null && path.dataset.series !== index);
    }

    show(page) {
        this.page = page;
        for (const tab of document.querySelectorAll('[role="tab"]')) {
            const active = tab.dataset.page === page;
            tab.setAttribute('aria-selected', String(active));
            document.getElementById('page-' + tab.dataset.page).hidden = !active;
        }
        this.render();
    }

    schedule() {
        cancelAnimationFrame(this.frame);
        this.frame = requestAnimationFrame(() => this.render());
    }

    render(all = false) {
        if (this.factors.some(factor => !factor.series[this.holding])) return;
        const start = +this.start.value;
        const end = +this.end.value;
        this.pan.max = this.dates.length - (end - start + 1);
        this.pan.value = start;
        this.pan.disabled = +this.pan.max === 0;
        document.getElementById('start-date').textContent = this.dates[start];
        document.getElementById('end-date').textContent = this.dates[end];
        document.getElementById('range-summary').textContent = `${this.dates[start]} — ${this.dates[end]} · ${end - start + 1} 个评估日`;
        document.getElementById('toolbar-range').textContent = `分析区间 ${this.dates[start]} — ${this.dates[end]}`;
        for (const button of document.querySelectorAll('[data-holding]')) {
            button.setAttribute('aria-pressed', String(+button.dataset.holding === this.holding));
        }
        for (const button of document.querySelectorAll('#factor-picker [data-factor]')) {
            button.setAttribute('aria-pressed', String(+button.dataset.factor === this.current));
        }
        for (const element of document.querySelectorAll('.holding-name')) {
            element.textContent = this.holdings[this.holding].label;
        }
        for (const element of document.querySelectorAll('.factor-name')) {
            element.textContent = this.factors[this.current].name;
        }
        // 全部因子在所选区间、所选持有期的结果，对比部分和因子详情共用；只有当前因子计算全部明细。
        this.results = this.factors.map((_, index) => this.analyze(index, start, end, this.holding, index === this.current));
        this.renderCompare(all);
        if (all || this.page === 'detail') this.renderDetail(this.results[this.current]);
        if (all || this.page === 'years') this.renderYears(all);
    }

    // 有效值的算术平均，缺失值（null）不参与。
    mean(values) {
        let count = 0;
        let sum = 0;
        for (const value of values) {
            if (value === null || !Number.isFinite(value)) continue;
            count++;
            sum += value;
        }
        return count ? sum / count : NaN;
    }

    format(value, percent = false, digits = null) {
        if (value === null || value === undefined || !Number.isFinite(value)) return '—';
        const places = digits === null ? (percent ? 2 : 3) : digits;
        return (value * (percent ? 100 : 1)).toFixed(places) + (percent ? '%' : '');
    }

    escape(text) {
        return String(text).replace(/[&<>"]/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[character]);
    }

    // 图例线段：颜色和线型与曲线一致。
    key(item) {
        return `<svg class="key" viewBox="0 0 22 6" aria-hidden="true"><line x1="0" y1="3" x2="22" y2="3" stroke="${item.color}" stroke-width="3"${item.dash ? ` stroke-dasharray="${item.dash}"` : ''}/></svg>`;
    }

    // 输入与评估日对齐、缺失为 null 的日收益；指标只用有效日，曲线在缺失日保持持平，首项为区间起点；curves 为假时只算指标，不生成曲线。
    // 拖动滑块时每个因子每组都要重算，因此在一次遍历中同时累计净值、回撤和收益之和，再用样本标准差计算波动率。
    performance(values, curves = true) {
        let count = 0;
        let sum = 0;
        let wealth = 1;
        let peak = 1;
        let minimum = 0;
        let valid = true;
        const cumulative = curves ? new Array(values.length + 1) : [];
        const drawdown = curves ? new Array(values.length + 1) : [];
        if (curves) {
            cumulative[0] = 0;
            drawdown[0] = 0;
        }
        for (let day = 0; day < values.length; day++) {
            const value = values[day];
            if (value !== null) {
                if (value < -1) valid = false;
                count++;
                sum += value;
                wealth *= 1 + value;
                peak = Math.max(peak, wealth);
                minimum = Math.min(minimum, wealth / peak - 1);
            }
            if (curves) {
                cumulative[day + 1] = wealth - 1;
                drawdown[day + 1] = wealth / peak - 1;
            }
        }
        if (!count || !valid) {
            return {metrics: [count, NaN, NaN, NaN, NaN, NaN], cumulative: [], drawdown: []};
        }
        const average = sum / count;
        let squares = 0;
        for (const value of values) {
            if (value !== null) squares += (value - average) ** 2;
        }
        const volatility = count > 1 ? Math.sqrt(squares / (count - 1)) : NaN;
        const scale = Math.sqrt(this.data.periods);
        const dailyRiskFree = (1 + this.data.riskFree) ** (1 / this.data.periods) - 1;
        return {
            metrics: [count, wealth - 1, wealth ** (this.data.periods / count) - 1,
                volatility * scale, minimum, volatility > 0 ? (average - dailyRiskFree) / volatility * scale : NaN],
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
    // 对比表每次重算都要对每个因子调用，因此直接遍历累计；标准差为样本标准差。
    icStats(values, days) {
        let count = 0;
        let sum = 0;
        let positive = 0;
        for (const value of values) {
            if (value === null) continue;
            count++;
            sum += value;
            if (value > 0) positive++;
        }
        const average = count ? sum / count : NaN;
        let squares = 0;
        for (const value of values) {
            if (value !== null) squares += (value - average) ** 2;
        }
        const deviation = count > 1 ? Math.sqrt(squares / (count - 1)) : NaN;
        const ratio = deviation > 0 ? average / deviation : NaN;
        return [count, average, deviation, ratio, ratio * Math.sqrt(this.data.periods / days), count ? positive / count : NaN];
    }

    // 计算第 factor 个因子在 [start, end] 内、第 holding 个持有期的结果；多空方向由区间 IC 均值确定，任一组收益缺失的日期在所有组中共同剔除。
    // 对比部分只用到各组指标以及多头组、多空的曲线和换手率；detail 为真时（当前因子和完整区间）才计算全部组的曲线、股票数和换手率。
    analyze(factor, start, end, holding, detail) {
        const series = this.factors[factor].series[holding];
        const pick = column => series[column].slice(start, end + 1);
        const groupColumns = this.names.map((_, index) => 'group_' + (index + 1));
        const dates = this.dates.slice(start, end + 1);
        const raw = groupColumns.map(pick);
        const complete = dates.map((_, day) => raw.every(values => values[day] !== null));
        // 每天都有完整收益时直接使用截取的数据，不再逐日复制。
        const allComplete = complete.every(Boolean);
        const onComplete = values => allComplete ? values : values.map((value, day) => complete[day] ? value : null);
        const ic = pick('IC');
        const icMean = this.mean(ic);
        const preferred = Number.isFinite(icMean) ? (icMean > 0 ? this.names.length - 1 : 0) : null;
        const short = preferred === null ? null : this.names.length - 1 - preferred;
        const groups = raw.map(onComplete);
        const spread = groups[0].map((value, day) => preferred === null || value === null ? null : groups[preferred][day] - groups[short][day]);
        const detailed = index => detail || index === preferred;
        return {
            dates, ic, rankIC: pick('rankIC'), days: this.holdings[holding].days, preferred, short, complete, groups, spread,
            counts: groupColumns.map((column, index) => detailed(index) ? onComplete(pick(column + '_count')) : null),
            turnovers: groupColumns.map((column, index) => detailed(index) ? onComplete(pick(column + '_turnover')) : null),
            performances: groups.map((values, index) => this.performance(values, detailed(index))),
            longShort: this.performance(spread)
        };
    }

    // 完整区间的结果不随时间滑块变化，按因子和持有期缓存，供分年对比、分年表现和默认显示的因子使用。
    fullResult(factor, holding) {
        const key = factor + '-' + holding;
        if (!this.full[key]) this.full[key] = this.analyze(factor, 0, this.dates.length - 1, holding, true);
        return this.full[key];
    }

    // 因子不超过 MAX_VISIBLE 个时全部显示，否则显示当前持有期完整区间 |ICIR| 最大的 MAX_VISIBLE 个。
    defaultVisible() {
        const indices = this.factors.map((_, index) => index);
        if (indices.length <= MAX_VISIBLE) return new Set(indices);
        const scores = indices.map(index => Math.abs(this.icStats(this.fullResult(index, this.holding).ic, 1)[3]));
        indices.sort((a, b) => (Number.isFinite(scores[b]) ? scores[b] : -1) - (Number.isFinite(scores[a]) ? scores[a] : -1));
        return new Set(indices.slice(0, MAX_VISIBLE));
    }

    table(headers, rows) {
        return '<table class="metrics"><thead><tr>' + headers.map(value => `<th>${value}</th>`).join('')
            + '</tr></thead><tbody>' + rows.map(row => '<tr>' + row.map(value => `<td>${value}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
    }

    metricRow(label, returns, counts, turnover) {
        const metrics = this.performance(returns, false).metrics;
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

    // 对数坐标的纵轴刻度，low、high 为净值的常用对数，返回刻度对应的累计收益和标注：净值取 1、2、5 × 10^k，超过 8 个时只取 10^k（数量级超过 8 个时每隔几个取一个）；
    // 区间较窄、整刻度不足 3 个时，改为在净值上按 1、2、5 × 10^k 的等间隔取刻度。
    logTicks(low, high) {
        let gridlines = [];
        for (let power = Math.floor(low); power <= Math.ceil(high); power++) {
            for (const multiple of [1, 2, 5]) {
                const level = Math.log10(multiple * 10 ** power);
                if (level >= low && level <= high) gridlines.push({value: multiple * 10 ** power - 1, label: this.format(multiple * 10 ** power - 1, true, Math.max(0, -power - 2)), multiple, power});
            }
        }
        const decades = gridlines.filter(gridline => gridline.multiple === 1);
        if (gridlines.length > 8 && decades.length >= 3) {
            const every = Math.ceil(decades.length / 8);
            gridlines = decades.filter(gridline => gridline.power % every === 0);
        }
        if (gridlines.length >= 3) return gridlines;
        // 区间上下界不取整，因此间隔取不超过四分之一区间的最大整数间隔，保证有足够的刻度。
        const bottom = 10 ** low;
        const top = 10 ** high;
        const rough = (top - bottom) / 4;
        const magnitude = 10 ** Math.floor(Math.log10(rough));
        const step = [5, 2, 1].map(multiple => multiple * magnitude).find(value => value <= rough);
        const digits = Math.max(0, -Math.floor(Math.log10(step * 100) + 1e-9));
        gridlines = [];
        for (let index = Math.ceil(bottom / step); index * step <= top; index++) {
            const value = Math.abs(index * step - 1) < step / 1e6 ? 0 : index * step - 1;
            gridlines.push({value, label: this.format(value, true, digits)});
        }
        return gridlines;
    }

    // 直接绘制 SVG，不依赖 CDN 或外部图表库；折线图带十字准线和提示框。两条以上的线默认显示图例，因子对比图改用图上方可开关的图例。
    chart(target, labels, series, options = {}) {
        const container = document.getElementById(target);
        if (container.dataset.drawn && !this.nearby.has(target) && !this.printing) {
            this.pending[target] = [labels, series, options];
            return;
        }
        delete this.pending[target];
        container.dataset.drawn = '1';
        const percent = options.percent !== false;
        // 对数坐标（options.log）按净值（1 + 累计收益）的常用对数确定纵向位置，等比例的涨跌等高；净值低于 1e-6（含恰为 0）时按 1e-6 画，避免出现 −Infinity。线性坐标不做变换。
        const scale = options.log ? value => Math.log10(Math.max(1 + value, 1e-6)) : value => value;
        let low = 0;
        let high = 0;
        let count = 0;
        for (const item of series) {
            for (const value of item.values) {
                if (!Number.isFinite(value)) continue;
                low = Math.min(low, scale(value));
                high = Math.max(high, scale(value));
                count++;
            }
        }
        if (!count) {
            container.innerHTML = `<p class="empty">${series.length ? '所选区间没有可用数据。' : '没有显示的因子，请点击上方图例选择因子。'}</p>`;
            delete this.charts[target];
            return;
        }
        const padding = (high - low || 0.01) * (options.bars ? 0.15 : 0.04);
        low -= padding;
        high += padding;
        // 纵轴刻度：线性坐标的间隔取 1、2、5 × 10^k，上下界落在整刻度上；对数坐标的刻度见 logTicks()。
        let gridlines = [];
        if (options.log) {
            gridlines = this.logTicks(low, high);
        } else {
            const rough = (high - low) / 4;
            const magnitude = 10 ** Math.floor(Math.log10(rough));
            const step = [1, 2, 5, 10].map(multiple => multiple * magnitude).find(value => value >= rough);
            const digits = Math.max(0, -Math.floor(Math.log10(step * (percent ? 100 : 1)) + 1e-9));
            low = Math.floor(low / step) * step;
            high = Math.ceil(high / step) * step;
            for (let index = 0; index <= Math.round((high - low) / step); index++) {
                const value = Math.abs(low + index * step) < step / 1e6 ? 0 : low + index * step;
                gridlines.push({value, label: this.format(value, percent, digits)});
            }
        }
        const left = 84, top = 14, width = 990, height = 270;
        const x = index => left + (options.bars ? (index + 0.5) / labels.length : index / Math.max(labels.length - 1, 1)) * width;
        const y = value => top + (high - value) / (high - low) * height;
        let legend = '';
        if (series.length > 1 && options.legend !== false) {
            legend = '<div class="legend">' + series.map(item => `<span>${this.key(item)}${this.escape(item.name)}</span>`).join('') + '</div>';
        }
        let svg = `<svg viewBox="0 0 1120 320" role="img" aria-label="${this.escape(container.dataset.label || '')}">`;
        for (const gridline of gridlines) {
            svg += `<line class="grid" x1="${left}" x2="${left + width}" y1="${y(scale(gridline.value))}" y2="${y(scale(gridline.value))}"/>`;
            svg += `<text x="${left - 12}" y="${y(scale(gridline.value)) + 4}" text-anchor="end">${gridline.label}</text>`;
        }
        svg += `<line class="zero" x1="${left}" x2="${left + width}" y1="${y(scale(0))}" y2="${y(scale(0))}"/>`;
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
                    path += `${move ? 'M' : 'L'}${x(index).toFixed(1)},${y(scale(value)).toFixed(1)}`;
                    move = false;
                });
                svg += `<path class="series" data-series="${item.index === undefined ? '' : item.index}" d="${path}" fill="none" stroke="${item.color}" stroke-width="2" stroke-linejoin="round"${item.dash ? ` stroke-dasharray="${item.dash}"` : ''}/>`;
            }
            svg += `<line class="crosshair" x1="0" x2="0" y1="${top}" y2="${top + height}" visibility="hidden"/>`;
        }
        container.innerHTML = legend + svg + '</svg>';
        this.charts[target] = {labels, series, percent, left, width, sort: options.sort};
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

    // 准线吸附到最近的评估日，提示框列出该日全部曲线的数值，因子对比图按数值从高到低排列；名称用 textContent 写入。
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
        const items = chart.series.slice();
        if (chart.sort) items.sort((a, b) => (Number.isFinite(b.values[index]) ? b.values[index] : -1e9) - (Number.isFinite(a.values[index]) ? a.values[index] : -1e9));
        for (const item of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.innerHTML = this.key(item);
            const value = document.createElement('strong');
            value.textContent = this.format(item.values[index], chart.percent);
            const name = document.createElement('span');
            name.textContent = item.name;
            row.append(key, value, name);
            this.tooltip.append(row);
        }
        this.tooltip.classList.toggle('wide', items.length > 10);
        this.tooltip.hidden = false;
        const size = this.tooltip.getBoundingClientRect();
        this.tooltip.style.left = (event.clientX + 16 + size.width > innerWidth ? event.clientX - 16 - size.width : event.clientX + 16) + 'px';
        this.tooltip.style.top = Math.max(8, Math.min(event.clientY + 16, innerHeight - size.height - 8)) + 'px';
    }

    // 对比表：每行一个因子，各列最优值加粗；点击表头按该列从优到劣排序（IC 类按绝对值），再次点击反向，无效值始终排在最后。
    compareTable() {
        // 每列依次为：列名、是否百分比、最优值规则（abs 绝对值最大、max 最大、min 最小、空为不比较）；表头只在空格和 <wbr> 处换行。
        const columns = [['IC 均值', false, 'abs'], ['RankIC 均值', false, 'abs'], ['ICIR', false, 'abs'], ['RankICIR', false, 'abs'], ['IC &gt; 0 占比', true, ''],
            ['多空<wbr>年化收益', true, 'max'], ['多空 Sharpe', false, 'max'], ['多空<wbr>最大回撤', true, 'max'], ['多头<wbr>年化收益', true, 'max'], ['多头 Sharpe', false, 'max'], ['多头<wbr>平均换手率', true, 'min']];
        const values = this.results.map(result => {
            const ic = this.icStats(result.ic, result.days);
            const rankIC = this.icStats(result.rankIC, result.days);
            const long = result.preferred === null ? [] : result.performances[result.preferred].metrics;
            const turnover = result.preferred === null ? NaN : this.mean(result.turnovers[result.preferred]);
            return [ic[1], rankIC[1], ic[3], rankIC[3], ic[5], result.longShort.metrics[2], result.longShort.metrics[5], result.longShort.metrics[4], long[2], long[5], turnover];
        });
        const score = (value, rule) => rule === 'abs' ? Math.abs(value) : rule === 'min' ? -value : value;
        const best = columns.map((column, index) => {
            let bestRow = -1;
            let bestScore = -Infinity;
            values.forEach((row, rowIndex) => {
                if (!column[2] || !Number.isFinite(row[index])) return;
                if (score(row[index], column[2]) > bestScore) {
                    bestScore = score(row[index], column[2]);
                    bestRow = rowIndex;
                }
            });
            return values.length > 1 ? bestRow : -1;
        });
        this.order = this.factors.map((_, index) => index);
        if (this.sort) {
            const column = this.sort.column;
            const sign = this.sort.descending ? 1 : -1;
            this.order.sort((a, b) => {
                const valid = [Number.isFinite(values[a][column]), Number.isFinite(values[b][column])];
                if (!valid[0] || !valid[1]) return valid[1] - valid[0];
                return sign * (score(values[b][column], columns[column][2]) - score(values[a][column], columns[column][2]));
            });
        }
        const head = (label, column) => {
            const sorted = this.sort !== null && this.sort.column === column;
            return `<th${sorted ? ` aria-sort="${this.sort.descending ? 'descending' : 'ascending'}"` : ''}><button type="button" class="sort" data-sort="${column}">${label}${sorted ? (this.sort.descending ? ' ▼' : ' ▲') : ''}</button></th>`;
        };
        let html = '<table class="metrics compare"><thead><tr>' + head('因子', -1) + '<th>完整<wbr>收益日</th><th>多头组</th>' + columns.map((column, index) => head(column[0], index)).join('') + '</tr></thead><tbody>';
        for (const rowIndex of this.order) {
            const factor = this.factors[rowIndex];
            const result = this.results[rowIndex];
            html += `<tr><td><button type="button" class="factor-link" data-factor="${rowIndex}" title="查看 ${this.escape(factor.name)} 的因子详情">${this.key(factor)}${this.escape(factor.name)}</button></td>`;
            html += `<td>${result.complete.filter(Boolean).length}</td><td>${result.preferred === null ? '—' : this.names[result.preferred]}</td>`;
            values[rowIndex].forEach((value, index) => {
                html += `<td${best[index] === rowIndex ? ' class="best"' : ''}>${this.format(value, columns[index][1])}</td>`;
            });
            html += '</tr>';
        }
        document.getElementById('compare-table').innerHTML = html + '</tbody></table>';
    }

    renderCompare(all) {
        this.compareTable();
        document.getElementById('group-heatmap').innerHTML = this.heatTable('因子', this.names, this.order.map(index => ({
            label: this.factors[index].name, values: this.results[index].performances.map(performance => performance.metrics[2]), long: this.results[index].preferred
        })), true);
        this.renderCurves();
        // 分年对比始终使用完整区间，各因子方向由完整区间 IC 均值确定；持有期和行顺序不变时不重算。
        const key = this.holding + '|' + this.order.join(',');
        if (!all && this.yearlyKey === key) return;
        this.yearlyKey = key;
        const results = this.order.map(index => this.fullResult(index, this.holding));
        const headers = this.years.map(year => year.year);
        const slice = (values, year) => values.slice(year.start, year.end + 1);
        document.getElementById('yearly-ls').innerHTML = this.heatTable('因子', headers, results.map((result, row) => ({
            label: this.factors[this.order[row]].name, values: this.years.map(year => this.performance(slice(result.spread, year), false).metrics[1])
        })), true);
        document.getElementById('yearly-ic').innerHTML = this.heatTable('因子', headers, results.map((result, row) => ({
            label: this.factors[this.order[row]].name, values: this.years.map(year => this.mean(slice(result.ic, year)))
        })), false);
    }

    // 累计收益和累积 IC 两组曲线图的图例同步，只在显示的因子变化时重画。
    renderLegend() {
        const legend = this.factors.map((factor, index) => `<button type="button" data-toggle="${index}" aria-pressed="${this.visible.has(index)}" title="显示或隐藏 ${this.escape(factor.name)}">${this.key(factor)}${this.escape(factor.name)}</button>`).join('');
        const actions = '<span class="legend-actions"><button type="button" data-legend="all">全部显示</button>' + (this.factors.length > MAX_VISIBLE ? `<button type="button" data-legend="top">|ICIR| 前 ${MAX_VISIBLE}</button>` : '') + '<button type="button" data-legend="none">全部隐藏</button></span>';
        for (const element of document.querySelectorAll('.legend-toggles')) {
            element.innerHTML = legend + actions;
        }
    }

    // 各因子累计收益和累积 IC 只画图例中显示的因子，颜色和线型不随显示的因子变化；累计收益的两张图按所选纵轴（对数或线性）同步绘制。
    renderCurves() {
        for (const button of document.querySelectorAll('[data-scale]')) {
            button.setAttribute('aria-pressed', String((button.dataset.scale === 'log') === this.log));
        }
        const labels = ['起点', ...this.results[0].dates];
        const shown = this.factors.map((_, index) => index).filter(index => this.visible.has(index));
        const series = values => shown.map(index => ({name: this.factors[index].name, color: this.factors[index].color, dash: this.factors[index].dash, index, values: values(this.results[index])}));
        this.chart('compare-ls', labels, series(result => result.longShort.cumulative), {legend: false, sort: true, log: this.log});
        this.chart('compare-long', labels, series(result => result.preferred === null ? [] : result.performances[result.preferred].cumulative), {legend: false, sort: true, log: this.log});
        this.chart('compare-ic', labels, series(result => this.cumulativeSum(result.ic)), {legend: false, sort: true, percent: false});
        this.chart('compare-rankic', labels, series(result => this.cumulativeSum(result.rankIC)), {legend: false, sort: true, percent: false});
    }

    renderDetail(result) {
        const factor = this.factors[this.current];
        const preferred = result.preferred;
        const short = result.short;
        const long = preferred === null ? [] : result.performances[preferred].metrics;
        const ic = this.icStats(result.ic, result.days);
        const rankIC = this.icStats(result.rankIC, result.days);
        [ic[1], rankIC[1], ic[3], result.longShort.metrics[2], long[2], long[5]].forEach((value, index) => {
            document.getElementById('card-' + index).textContent = this.format(value, index === 3 || index === 4);
        });
        const completeDays = result.complete.filter(Boolean).length;
        document.getElementById('detail-summary').textContent = `${result.dates.length} 个评估日 · ${completeDays} 个完整收益日 · 剔除 ${result.dates.length - completeDays} 个收益缺失日`;
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
        this.chart('ls-cumulative', labels, [{name: 'Long-short', color: factor.color, values: result.longShort.cumulative}]);
        this.chart('ls-drawdown', labels, [{name: 'Long-short', color: factor.color, values: result.longShort.drawdown}]);
        this.chart('detail-ic', labels, [{name: '累积 IC', color: '#2a78d6', values: this.cumulativeSum(result.ic)},
            {name: '累积 RankIC', color: '#eb6834', values: this.cumulativeSum(result.rankIC)}], {percent: false});
    }

    // 分年表现始终使用完整区间，多头组由完整区间 IC 均值确定，持有 1 天时与持仓清单一致；因子和持有期不变时不重画。
    renderYears(all) {
        const key = this.current + '-' + this.holding;
        if (!all && this.yearsKey === key) return;
        this.yearsKey = key;
        const result = this.fullResult(this.current, this.holding);
        const preferred = result.preferred;
        const slice = (values, year) => values.slice(year.start, year.end + 1);
        document.getElementById('yearly-long').textContent = preferred === null ? '无有效 IC' : `多头组 ${this.names[preferred]}`;
        document.getElementById('yearly-table').innerHTML = preferred === null ? '<p class="empty">没有有效 IC，无法确定多头组。</p>' : this.table(
            ['年份', '有效交易日', '日均股票数量', '平均换手率', '区间收益', '年化收益', '年化波动率', '最大回撤', 'Sharpe'],
            this.years.map(year => this.metricRow(year.year, slice(result.groups[preferred], year), slice(result.counts[preferred], year), slice(result.turnovers[preferred], year))));
        const container = document.getElementById('yearly-curves');
        container.replaceChildren();
        for (const year of this.years) {
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
    def plot_results_html(config):
        """用因子目录中已有评估结果的全部因子生成一份离线 HTML 报告，写入 visualization_output_dir/factor_evaluation_report.html；单因子评估和批量评估都生成这一份报告。"""
        results_dir = Path(config["output_dir"])
        # 报告包含因子目录中已有评估结果的因子，顺序与因子文件一致；每个因子只读与文件名同名的结果，残留的旧滚动均值列结果（如 Mom1m_5_m.csv）不进入报告。
        visualizers = []
        for factor_path in sorted(Path(config["factor_data_dir"]).glob("*.parquet")):
            if (results_dir / f"{factor_path.stem}.csv").is_file():
                visualizers.append(ResultsVisualizer(results_dir / f"{factor_path.stem}.csv", factor_path.stem, config["start_date"]))
        if not visualizers:
            raise ValueError(f"No evaluation results in {results_dir} for factors in {config['factor_data_dir']}.")
        factor_list = [visualizer.factor_name for visualizer in visualizers]
        correlation = FactorCorrelationAnalyzer(config, factor_list).correlation_matrix()
        groups = visualizers[0].GROUPS

        # 持有期取各因子结果的并集，由短到长排列。
        holding_periods = []
        for visualizer in visualizers:
            for holding_period in visualizer.holding_periods:
                if holding_period not in holding_periods:
                    holding_periods.append(holding_period)
        holding_periods.sort()
        holding_picker = ""
        for index, holding_period in enumerate(holding_periods):
            holding_picker += f'<button type="button" data-holding="{index}" aria-pressed="{str(index == 0).lower()}">{holding_period} 天</button>'

        # 各因子按评估日期的并集对齐，某个因子缺失的日期留空。
        dates = visualizers[0].factor_results.index
        for visualizer in visualizers[1:]:
            dates = dates.union(visualizer.factor_results.index)

        # 每个因子、每个持有期一个数据块，列顺序与持有 1 天的列名 columns 一致，缺失为 null。收益和 IC 保留 6 位小数（日收益精确到 0.0001%），股票数保留 2 位、换手率保留 4 位；
        # 数据块经 gzip 压缩、base64 编码后嵌入页面，页面首次用到某个持有期时才解压，控制多因子报告的文件大小和打开速度。
        blocks = ""
        for holding_index, holding_period in enumerate(holding_periods):
            suffix = "" if holding_period == 1 else f"_{holding_period}d"
            return_columns = [f"{group}{suffix}" for group in groups]
            count_columns = [f"{column}_count" for column in return_columns]
            turnover_columns = [f"{column}_turnover" for column in return_columns]
            for factor_index, visualizer in enumerate(visualizers):
                payload = visualizer.factor_results.reindex(index=dates, columns=return_columns + [f"IC{suffix}", f"rankIC{suffix}"] + count_columns + turnover_columns)
                payload[count_columns] = payload[count_columns].round(2)
                payload[turnover_columns] = payload[turnover_columns].round(4)
                payload = payload.T.to_json(orient="values", double_precision=6)
                blocks += f'<script id="data-{factor_index}-{holding_index}" type="application/octet-stream">{base64.b64encode(gzip.compress(payload.encode("utf-8"), mtime=0)).decode("ascii")}</script>\n'

        factors = []
        for index, visualizer in enumerate(visualizers):
            factors.append({"name": visualizer.factor_name, "color": ResultsVisualizer.FACTOR_COLORS[index % len(ResultsVisualizer.FACTOR_COLORS)], "dash": ResultsVisualizer.FACTOR_DASHES[index // len(ResultsVisualizer.FACTOR_COLORS) % len(ResultsVisualizer.FACTOR_DASHES)]})
        report_data = {"dates": list(dates.strftime("%Y-%m-%d")), "columns": groups + ["IC", "rankIC"] + [f"{group}_count" for group in groups] + [f"{group}_turnover" for group in groups], "factors": factors, "holdings": [{"days": holding_period, "label": f"持有 {holding_period} 天"} for holding_period in holding_periods], "periods": visualizers[0].periods_per_year, "riskFree": visualizers[0].risk_free_rate, "colors": visualizers[0].group_colors, "names": [f"G{group}" for group in range(1, len(groups) + 1)]}
        report_data = json.dumps(report_data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")

        # 相关性矩阵：正相关为红、负相关为蓝，颜色深浅与相关系数绝对值成正比；对角线恒为 1，不着色。
        correlation_header = "".join(f'<th class="vertical"><span>{escape(factor_name)}</span></th>' for factor_name in factor_list)
        correlation_rows = ""
        for row_name in factor_list:
            correlation_rows += f"<tr><th>{escape(row_name)}</th>"
            for column_name in factor_list:
                value = correlation.loc[row_name, column_name]
                alpha = 0 if row_name == column_name or pd.isna(value) else abs(value)
                color = "199, 72, 69" if value > 0 else "28, 92, 171"
                correlation_rows += f'<td style="background: rgba({color}, {alpha:.3f}); color: {"#ffffff" if alpha > 0.65 else "#172b4d"}" title="{escape(row_name)} × {escape(column_name)}">{ResultsVisualizer._format(value)}</td>'
            correlation_rows += "</tr>"

        cards = "".join(f'<div class="stat"><span>{label}</span><strong id="card-{index}">—</strong></div>'
                        for index, label in enumerate(["IC 均值", "RankIC 均值", "ICIR", "多空年化收益", "多头年化收益", "多头 Sharpe"]))
        holding_labels = "、".join(str(holding_period) for holding_period in holding_periods)
        last_group = f"G{len(groups)}"
        html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>因子评估报告</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif; color: #172b4d; background: #f2f5fa; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; background: #f2f5fa; }} main {{ max-width: 1280px; margin: auto; padding: 24px 28px; }} [hidden] {{ display: none !important; }}
header {{ padding: 30px 32px; background: linear-gradient(120deg, #172c52, #28578c); color: white; border-radius: 20px; }}
.eyebrow {{ color: #9edcfa; font-size: 12px; letter-spacing: 3px; }} h1 {{ font-size: 30px; margin: 14px 0 10px; overflow-wrap: anywhere; }} header p {{ margin: 0; color: #d2dfef; line-height: 1.8; }}
.toolbar {{ position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; padding: 14px 0; background: #f2f5fa; }}
.jumps {{ display: flex; gap: 8px; }} .jumps a {{ border: 1px solid #dce3ed; border-radius: 10px; padding: 9px 18px; background: white; color: #28578c; font-size: 14px; text-decoration: none; }} .jumps a:hover {{ background: #e8f0fb; }}
#toolbar-range {{ flex: 1; font-size: 13px; color: #6b7c94; }}
.picker, .factor-picker {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }} .picker > span {{ font-size: 13px; color: #6b7c94; margin-right: 2px; }}
button {{ font: inherit; cursor: pointer; }} button:focus-visible, input:focus-visible, a:focus-visible {{ outline: 3px solid #06a6a0; outline-offset: 3px; }}
.picker button, .factor-picker button {{ display: inline-flex; align-items: center; border: 1px solid #dce3ed; border-radius: 999px; padding: 6px 13px; background: white; color: #28578c; font-size: 13px; }}
.picker button[aria-pressed="true"], .factor-picker button[aria-pressed="true"] {{ color: white; background: #172c52; border-color: #172c52; }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 8px; }} .tabs button {{ border: 1px solid #dce3ed; border-radius: 10px; padding: 11px 20px; background: white; color: #28578c; }}
.tabs button[aria-selected="true"] {{ color: white; background: #28578c; border-color: #28578c; }}
svg.key {{ display: inline-block; flex: none; width: 22px; height: 6px; margin-right: 7px; vertical-align: middle; }}
section, .stat {{ background: white; border: 1px solid #e3e9f2; border-radius: 16px; box-shadow: 0 5px 20px #23395605; }}
section {{ padding: 24px 26px; margin-bottom: 20px; }} h2 {{ margin: 0; font-size: 20px; }} h3 {{ margin: 24px 0 4px; font-size: 15px; }}
section p, footer {{ font-size: 13px; color: #6b7c94; line-height: 1.9; }} .content, .chart {{ overflow-x: auto; }} .empty {{ text-align: center; padding: 40px; }}
.notice {{ padding: 14px 18px; border-radius: 12px; background: #fdecec; color: #9e3432; }}
.part {{ scroll-margin-top: 76px; }} .part-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; margin: 34px 0 16px; padding-bottom: 12px; border-bottom: 2px solid #dce3ed; }}
.part-head h2 {{ font-size: 24px; }} .part-head p {{ margin: 0; font-size: 13px; color: #6b7c94; }}
.section-head {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; }}
.ghost {{ border: 1px solid #dce3ed; border-radius: 8px; padding: 7px 14px; background: #f4f7fb; color: #28578c; font-size: 13px; }}
.range-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 28px; margin: 6px 0 4px; }}
.range-grid label {{ display: flex; justify-content: space-between; gap: 8px; font-size: 13px; color: #6b7c94; margin-bottom: 4px; }} output {{ color: #2563a4; font-weight: 600; }}
input[type="range"] {{ width: 100%; accent-color: #2563eb; cursor: ew-resize; }}
.selector {{ display: grid; gap: 18px; }} .selector > div > span {{ display: block; margin-bottom: 8px; font-size: 13px; color: #6b7c94; }}
.detail-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; margin: 4px 0 14px; }} .detail-head h2 {{ font-size: 22px; overflow-wrap: anywhere; }} .detail-head p {{ margin: 0; font-size: 13px; color: #6b7c94; }}
.stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 20px; }}
.stat {{ padding: 18px 20px; }} .stat span {{ display: block; color: #718096; font-size: 13px; }} .stat strong {{ display: block; margin-top: 10px; font-size: 24px; color: #2563a4; }}
.chart > svg {{ display: block; width: 100%; height: auto; min-width: 560px; }} .chart > svg text {{ font-family: inherit; font-size: 12px; fill: #64748b; }} .chart > svg text.value {{ fill: #172b4d; }}
.chart .grid {{ stroke: #e8edf5; }} .chart .zero {{ stroke: #94a3b8; stroke-dasharray: 4 4; }} .chart .crosshair {{ stroke: #475569; stroke-width: 1; }} .chart path.faded {{ opacity: 0.12; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 8px 0 2px; font-size: 12px; color: #52607a; }}
.legend-toggles {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 14px 0 4px; }}
.legend-toggles button {{ display: inline-flex; align-items: center; border: 1px solid #dce3ed; border-radius: 8px; padding: 4px 10px; background: white; color: #172b4d; font-size: 12px; }}
.legend-toggles button[aria-pressed="false"] {{ color: #94a3b8; background: #f7f9fc; border-style: dashed; }} .legend-toggles button[aria-pressed="false"] svg.key {{ opacity: 0.25; }}
.legend-actions {{ display: inline-flex; flex-wrap: wrap; gap: 6px; margin-left: auto; }} .legend-actions button {{ background: #f4f7fb; color: #28578c; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }} th {{ background: #f4f7fb; color: #61718b; font-weight: 500; }}
th, td {{ padding: 12px 13px; text-align: right; border-bottom: 1px solid #edf1f6; }} th:first-child, td:first-child {{ text-align: left; }} tbody tr:hover {{ background: #f4f8ff; }}
table.compare th, table.compare td {{ padding: 12px 8px; }} table.compare th {{ white-space: normal; word-break: keep-all; vertical-align: bottom; }} table.compare td.best {{ font-weight: 700; color: #1c5cab; background: #eef4fd; }}
th button.sort {{ border: 0; padding: 0; background: none; color: inherit; font-weight: 500; text-align: inherit; }} th button.sort:hover {{ color: #172b4d; text-decoration: underline; }} th[aria-sort] {{ color: #172b4d; }}
.factor-link {{ display: inline-flex; align-items: center; border: 0; padding: 0; background: none; color: #28578c; font-weight: 600; text-decoration: underline dotted #94a3b8; text-underline-offset: 4px; }}
table.heatmap th, table.heatmap td {{ padding: 9px 10px; }} table.heatmap td.heat {{ text-align: center; border: 1px solid white; min-width: 72px; }} table.heatmap td.long {{ box-shadow: inset 0 0 0 2px #172b4d; font-weight: 700; }}
table.correlation {{ width: auto; }} table.correlation td {{ min-width: 64px; text-align: center; border: 1px solid white; }}
th.vertical {{ vertical-align: bottom; text-align: center; }} th.vertical span {{ writing-mode: vertical-rl; transform: rotate(180deg); }}
#tooltip {{ position: fixed; z-index: 20; pointer-events: none; min-width: 150px; padding: 10px 12px; background: white; border: 1px solid #dce3ed; border-radius: 10px; box-shadow: 0 8px 24px #1729521f; font-size: 12px; color: #52607a; }}
#tooltip div {{ display: flex; align-items: center; gap: 6px; line-height: 1.7; break-inside: avoid; }} #tooltip strong {{ min-width: 64px; color: #172b4d; font-variant-numeric: tabular-nums; }} #tooltip svg.key {{ margin-right: 0; }}
#tooltip .tooltip-title {{ margin-bottom: 4px; color: #172b4d; font-weight: 600; column-span: all; }} #tooltip.wide {{ columns: 2; column-gap: 20px; }}
footer {{ padding: 4px 12px 20px; }}
@media(max-width: 1000px) {{ .stats {{ grid-template-columns: repeat(3, 1fr); }} .range-grid {{ grid-template-columns: 1fr; }} }}
@media(max-width: 700px) {{ main {{ padding: 12px 16px; }} header, section {{ padding: 18px; }} h1 {{ font-size: 24px; }} .stats {{ grid-template-columns: repeat(2, 1fr); gap: 10px; }} .stat strong {{ font-size: 21px; }} .toolbar {{ position: static; }} #toolbar-range {{ display: none; }} .legend-actions {{ margin-left: 0; }} .tabs button {{ flex: 1; padding: 11px 2px; }} }}
@media print {{ main {{ padding: 0; }} .toolbar, #range-controls, #tooltip, .legend-actions {{ display: none !important; }} [role="tabpanel"][hidden] {{ display: block !important; }} section {{ break-inside: avoid; }} .content, .chart {{ overflow: visible; }} .chart > svg {{ min-width: 0; }} }}
</style></head><body><main>
<header><div class="eyebrow">FACTOR RESEARCH / EVALUATION REPORT</div><h1>因子评估报告</h1>
<p>{dates.min():%Y-%m-%d} — {dates.max():%Y-%m-%d} · {len(dates):,} 个评估日 · {len(visualizers)} 个因子 · 每日分 {len(groups)} 组 · 持有期：{holding_labels} 天</p></header>
<nav class="toolbar" aria-label="报告导航">
<div class="jumps"><a href="#compare">因子对比</a><a href="#single">单因子表现</a></div><span id="toolbar-range"></span>
<div class="picker" id="holding-picker"><span>持有期</span>{holding_picker}</div>
</nav>
<noscript><p>请启用 JavaScript，以使用因子切换、时间滑块和交互收益图表。</p></noscript>
<p id="load-error" class="notice" role="alert" hidden></p>
<section id="range-controls"><div class="section-head"><h2>分析时间区间</h2><button type="button" id="range-reset" class="ghost">恢复完整区间</button></div>
<p id="range-summary" aria-live="polite"></p>
<div class="range-grid">
<div><label for="range-start"><span>开始日期</span><output id="start-date" for="range-start"></output></label><input id="range-start" type="range" min="0" value="0" step="1"></div>
<div><label for="range-end"><span>结束日期</span><output id="end-date" for="range-end"></output></label><input id="range-end" type="range" min="0" value="0" step="1"></div>
<div><label for="range-pan"><span>平移整个区间</span><span>评估日数不变</span></label><input id="range-pan" type="range" min="0" value="0" step="1"></div>
</div><p>拖动滑块后，因子对比中的指标对比、分组年化收益、累计收益与累积 IC 曲线，以及单因子的因子详情立即重算；分年对比、相关性矩阵和分年表现始终使用完整区间。</p></section>

<div id="compare" class="part">
<div class="part-head"><h2>因子对比</h2><p>{len(visualizers)} 个因子并列分析，各因子单独分组回测，指标随顶栏的持有期变化。</p></div>
<section><h2>因子指标对比 · <span class="holding-name"></span></h2><p>每行一个因子，全部指标按所选区间和持有期计算，各因子分别由区间 IC 均值确定多头组。蓝底加粗为该列最优：IC 类取绝对值最大，换手率取最小，其余取最大。点击表头按该列从优到劣排序，再次点击反向，点击「因子」恢复原顺序；下方分组年化收益和分年对比的行顺序与本表一致。点击因子名查看该因子详情。</p><div id="compare-table" class="content"></div></section>
<section><h2>各因子分组年化收益 · <span class="holding-name"></span></h2><p>每行一个因子，从左到右为 G1（因子值最低）到 {last_group}（因子值最高）。红色为正、蓝色为负，颜色越深绝对值越大；加框为该因子的多头组。用于比较各因子的分组单调性。</p><div id="group-heatmap" class="content"></div></section>
<section class="curves"><div class="section-head"><h2>各因子累计收益 · <span class="holding-name"></span></h2><div class="picker" role="group" aria-label="纵轴刻度"><span>纵轴</span><button type="button" data-scale="log" aria-pressed="true">对数</button><button type="button" data-scale="linear" aria-pressed="false">线性</button></div></div>
<p>按日复利累计，所选区间起点收益为 0；各因子使用各自的多空方向，某因子收益缺失的日期净值持平。纵轴默认为对数坐标：纵向位置按净值（1 + 累计收益）取对数，等比例的涨跌在图上等高，刻度和提示框仍显示累计收益；可切换为线性坐标，两张图同步。每个因子的颜色和线型固定，点击图例显示或隐藏因子（累计收益与累积 IC 两组图同步），悬停图例突出该因子；因子超过 8 个时默认只显示完整区间 |ICIR| 最大的 8 个。</p>
<div class="legend-toggles" role="group" aria-label="显示的因子"></div>
<h3>多空组合</h3><div id="compare-ls" class="chart" data-label="各因子多空累计收益"></div><h3>多头组合</h3><div id="compare-long" class="chart" data-label="各因子多头累计收益"></div></section>
<section class="curves"><h2>各因子累积 IC · <span class="holding-name"></span></h2><p>每日 IC / RankIC 的算术累加，所选区间起点为 0；缺失日期不累加。负向因子的曲线向下，斜率越陡越有效。持有期大于 1 天时为该持有期收益的 IC，相邻日期的收益区间重叠。</p>
<div class="legend-toggles" role="group" aria-label="显示的因子"></div>
<h3>累积 IC</h3><div id="compare-ic" class="chart" data-label="各因子累积 IC"></div><h3>累积 RankIC</h3><div id="compare-rankic" class="chart" data-label="各因子累积 RankIC"></div></section>
<section><h2>分年对比 · <span class="holding-name"></span></h2><p>使用完整区间，不随时间滑块变化；各因子的多空方向由完整区间 IC 均值确定。每行一个因子、每列一个年份；多空收益为该年实际覆盖日期的复利收益，首尾年份可能不完整。红色为正、蓝色为负，颜色深浅在每张表内单独标定。</p>
<h3>分年多空收益</h3><div id="yearly-ls" class="content"></div><h3>分年 IC 均值</h3><div id="yearly-ic" class="content"></div></section>
<section><h2>因子相关性矩阵</h2><p>全部因子两两的每日截面 Pearson 相关系数的时间均值，因子按因子文件顺序排列，不随持有期和时间区间变化，计算口径见页脚。红色为正相关、蓝色为负相关，颜色越深相关性越强；对角线恒为 1，不着色。</p>
<div class="content"><table class="correlation"><thead><tr><th></th>{correlation_header}</tr></thead><tbody>{correlation_rows}</tbody></table></div></section>
</div>

<div id="single" class="part">
<div class="part-head"><h2>单因子表现</h2><p>点击按钮切换因子；因子详情随持有期和所选区间变化，分年表现始终使用完整区间。</p></div>
<section class="selector"><div><span>因子</span><div class="factor-picker" id="factor-picker" role="group" aria-label="选择因子"></div></div>
<div class="tabs" role="tablist" aria-label="单因子页面"><button type="button" role="tab" data-page="detail" aria-controls="page-detail" aria-selected="true">因子详情</button><button type="button" role="tab" data-page="years" aria-controls="page-years" aria-selected="false">分年表现</button></div></section>

<div id="page-detail" role="tabpanel" aria-label="因子详情">
<div class="detail-head"><h2><span class="factor-name"></span> · <span class="holding-name"></span> · 因子详情</h2><p id="detail-summary"></p></div>
<div class="stats">{cards}</div>
<section><h2>分组绩效</h2><p id="direction-note"></p><p>G1 为因子值最低组，{last_group} 为最高组。各组与多空组合使用相同的完整收益日；股票数量与换手率按这些日期取均值，缺失值不参与均值。多空股票数量为两端之和，不展示多空换手率。</p><div id="group-table" class="content"></div></section>
<section><h2>各组累计收益和动态回撤</h2><p>按日复利累计；所选区间起点收益为 0、净值为 1。动态回撤 = 当前净值 / 区间内历史最高净值 − 1，包含初始净值。</p>
<h3>各组累计收益</h3><div id="group-cumulative" class="chart" data-label="各组累计收益"></div><h3>各组动态回撤</h3><div id="group-drawdown" class="chart" data-label="各组动态回撤"></div></section>
<section><h2>各组年化收益 · 单调性</h2><p>按因子值从低到高排列 G1 → {last_group}。年化收益按所选区间的有效收益日数折算；正向因子观察是否递增，负向因子观察是否递减。</p><div id="annual-returns" class="chart" data-label="各组年化收益"></div></section>
<section><h2>Long-short 累计收益和动态回撤</h2><p>每日多头收益减空头收益，再复利累计；累计收益与回撤均在所选区间起点重置。</p>
<h3>Long-short 累计收益</h3><div id="ls-cumulative" class="chart" data-label="Long-short 累计收益"></div><h3>Long-short 动态回撤</h3><div id="ls-drawdown" class="chart" data-label="Long-short 动态回撤"></div></section>
<section><h2>IC / RankIC</h2><p>所选区间的截面相关性统计，IR 保留方向符号；累积曲线为每日值的算术累加，缺失日期不累加。</p><div id="ic-table" class="content"></div>
<h3>累积 IC 与 RankIC</h3><div id="detail-ic" class="chart" data-label="累积 IC 与 RankIC"></div></section>
</div>

<div id="page-years" role="tabpanel" aria-label="分年表现" hidden>
<section><h2><span class="factor-name"></span> · <span class="holding-name"></span> · 分年绩效（<span id="yearly-long"></span>）</h2><p>使用完整区间，多头组由完整区间 IC 均值确定。区间收益为该年实际覆盖日期的复利收益；回撤每年重置。</p><div id="yearly-table" class="content"></div></section>
<section><h2><span class="factor-name"></span> · <span class="holding-name"></span> · 分年各组累计收益</h2><p>各年从 0 重新开始，净值从 1 按该年有效日收益复利累计，不继承上一年净值。</p><div id="yearly-curves"></div></section>
</div>
</div>
<footer><b>计算口径</b><br>
每个因子文件只评估与文件名同名的因子列；各因子单独分组回测，互不影响，报告中并列比较。<br>
持有期：因子在 t 日收盘计算，t+1 日 10 点按分组买入；持有 1 天在 t+2 日 10 点卖出，持有 N 天在 t+1+N 日 10 点卖出。持有 N 天时每个交易日买入一批、同时持有最近 N 批，各批等权，组合日收益为各批当日收益的均值（重叠持仓，结果不依赖调仓起始日）；起始阶段只平均已买入的批次。
每批只买入当日可交易（有收益）的股票，持有期内停牌的日收益记 0。持有 N 天的 IC / RankIC 为因子值与之后 N 天复利累计收益的截面相关系数，最后 N − 1 个评估日的收益尚未实现，记为空。
持有 N 天的换手率为当日买入批与当日卖出批（N 天前买入）的成分变动比例，不除以 N，前 N 个评估日为空；股票数量为各批的平均股票数。<br>
日收益使用小数；年化交易日数 {visualizers[0].periods_per_year:g}，年化无风险利率 {visualizers[0].risk_free_rate:.2%}。
年化收益 = ∏(1 + 日收益)^(年化交易日数 / 有效交易日数) − 1；年化波动率 = 日收益样本标准差 × √年化交易日数。
Sharpe = (平均日收益 − 等效日无风险利率) / 日收益样本标准差 × √年化交易日数。
ICIR / RankICIR = 均值 / 样本标准差；年化 IR 再乘 √(年化交易日数 / 持有天数)。标准差为零或样本不足时显示「—」。<br>
IC 均值 &gt; 0 时做多 {last_group}、做空 G1，否则做多 G1、做空 {last_group}。每个因子、每个持有期分别确定方向：指标对比、分组年化收益、累计曲线与因子详情按所选区间确定，分年对比与分年表现按完整区间确定，属于事后分析。
多空按多头 100%、空头 100% 的收益差计算，未除以 2；不计手续费、滑点和融券成本。<br>
缺失 IC / RankIC 各自剔除；任一组缺失收益的日期从该因子的全部收益统计中共同剔除，不填充为零。各因子按评估日期的并集对齐，收益图横轴为评估日，收益缺失日净值持平。
因子对比的累计收益图默认用对数纵轴：纵向位置按净值（1 + 累计收益）的常用对数确定，净值低于 0.000001（含恰为 0）时按 0.000001 画；刻度和提示框显示真实累计收益，其余图表均为线性纵轴。
平均换手率沿用原始日换手率，在有效收益日内求均值；拖动区间不重建持仓，首日沿用已有值。<br>
相关性矩阵：自 {escape(str(config["start_date"]))} 起读取各因子文件中与文件名同名的因子列，使用当日因子值，不做滞后；板块范围与单因子评估相同（{escape(str(config["stock_board"]))}），每日在各因子样本的并集中按上月末市值保留最小的 {config["stock_pool"]} 只股票。
各因子剔除无穷值后逐日截面标准化（减均值、除以样本标准差），未去极值；相关系数逐日按两两均有效的股票计算，再对日期取算术平均。<br>
日期沿用评估结果的收益起始日标签，跨年收益按该标签归属年份。所有数据与图表已嵌入（数据经 gzip 压缩），可离线查看。生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}。
</footer></main>
<div id="tooltip" role="tooltip" hidden></div>
<script id="report-data" type="application/json">{report_data}</script>
{blocks}<script>{ResultsVisualizer._report_script()}</script></body></html>'''
        output_dir = Path(config["visualization_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "factor_evaluation_report.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()


if __name__ == "__main__":
    # 不重新评估，直接用已有评估结果重新生成因子评估报告。
    from main import load_config
    config = load_config("portfolio_20260905/config/config_factor_evaluation.json")
    print(ResultsVisualizer.plot_results_html(config))
