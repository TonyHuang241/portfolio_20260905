"""Generate a self-contained HTML report from daily factor evaluation results."""

import argparse
import io
import json
import sys
from html import escape
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


class ResultsVisualizer:
    GROUPS = [f"group_{group}" for group in range(1, 6)]
    COLORS = ["#2563eb", "#06a6a0", "#e7a32e", "#9764d9", "#ed6976"]

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
        if data.index.has_duplicates or data.index.hasnans:
            raise ValueError("Evaluation dates must be unique and non-null.")
        count_columns = [f"{group}_count" for group in self.GROUPS]
        turnover_columns = [f"{group}_turnover" for group in self.GROUPS]
        data = data[self.GROUPS + ["IC", "rankIC"] + [column for column in count_columns + turnover_columns if column in data.columns]].astype(float)
        if start_date is not None:
            data = data.loc[data.index >= pd.to_datetime(str(start_date))]
        if data.empty:
            raise ValueError("No evaluation results in the selected date range.")
        if np.isinf(data.to_numpy()).any():
            raise ValueError("Evaluation results contain infinite values.")
        if (data[self.GROUPS] < -1).any().any():
            raise ValueError("Long-only group returns cannot be below -100%.")
        if (data[["IC", "rankIC"]].abs() > 1).any().any():
            raise ValueError("IC and RankIC must be in [-1, 1].")
        if data["IC"].notna().sum() == 0:
            raise ValueError("At least one valid IC value is required to select the long-short direction.")
        self.factor_results = data
        self.group_returns = data[self.GROUPS].dropna()
        self.group_counts = data.reindex(columns=count_columns)
        if self.group_returns.empty:
            raise ValueError("No dates with valid returns for all five groups.")
        self.ic_mean = data["IC"].mean()
        self.preferred_group = "group_5" if self.ic_mean > 0 else "group_1"
        self.short_group = "group_1" if self.ic_mean > 0 else "group_5"
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

    def _performance_table(self, rows, label, turnovers=None):
        columns = [label, "有效交易日", "日均股票数量", "区间收益", "年化收益", "年化波动率", "最大回撤", "Sharpe"]
        records = []
        for name, returns, counts in rows:
            metrics = self._performance(returns)
            average_count = counts.reindex(returns.dropna().index).mean()
            records.append([name, str(metrics[0]), "—" if pd.isna(average_count) else f"{average_count:.2f}"] + [self._format(value, percent=i < 5) for i, value in enumerate(metrics[1:], 1)])
        table = pd.DataFrame(records, columns=columns)
        if turnovers is not None:
            table.insert(3, "平均换手率", [self._format(value, percent=True) for value in turnovers])
        return table.to_html(index=False, border=0, classes="metrics", escape=True)

    def _ic_table(self):
        records = []
        for column, label in [("IC", "IC"), ("rankIC", "RankIC")]:
            values = self.factor_results[column].dropna()
            deviation = values.std(ddof=1)
            ratio = values.mean() / deviation if deviation > 0 else np.nan
            records.append([label, len(values), self._format(values.mean()), self._format(deviation),
                            self._format(ratio), self._format(ratio * np.sqrt(self.periods_per_year)),
                            self._format((values > 0).mean() if len(values) else np.nan, percent=True)])
        return pd.DataFrame(records, columns=["指标", "有效观测数", "均值", "标准差", "IR（均值 / 标准差）", "年化 IR", "大于零占比"]).to_html(index=False, border=0, classes="metrics")

    def _chart(self, data, ylabel, percent=False, zero=False):
        figure = Figure(figsize=(12, 3.8), layout="constrained", facecolor="white")
        axis = figure.subplots()
        for (label, values), color in zip(data.items(), self.COLORS):
            axis.plot(values.index, values, label=label, color=color, linewidth=1.3, alpha=0.9)
        if zero:
            axis.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
        axis.set_ylabel(ylabel, color="#64748b", fontsize=10)
        axis.grid(axis="y", color="#e8edf5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#dce3ed")
        axis.tick_params(colors="#64748b", labelsize=9)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        if percent:
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.legend(loc="upper left", frameon=False, ncol=len(data), fontsize=9)
        buffer = io.StringIO()
        FigureCanvasSVG(figure).print_svg(buffer)
        svg = buffer.getvalue()
        return svg[svg.index("<svg"):]

    def _annual_return_chart(self):
        annual_returns = [self._performance(self.group_returns[group])[2] for group in self.GROUPS]
        figure = Figure(figsize=(12, 3.8), layout="constrained", facecolor="white")
        axis = figure.subplots()
        bars = axis.bar([f"G{group[-1]}" for group in self.GROUPS], annual_returns, color=self.COLORS, width=0.55, zorder=3)
        axis.bar_label(bars, labels=[self._format(value, percent=True) for value in annual_returns], padding=5, fontsize=10, color="#172b4d")
        axis.axhline(0, color="#94a3b8", linewidth=0.8)
        axis.set_ylabel("Annualized return", color="#64748b", fontsize=10)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color="#e8edf5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#dce3ed")
        axis.tick_params(colors="#64748b", labelsize=10)
        axis.margins(y=0.18)
        buffer = io.StringIO()
        FigureCanvasSVG(figure).print_svg(buffer)
        svg = buffer.getvalue()
        return svg[svg.index("<svg"):]

    @staticmethod
    def _report_script():
        """Keep interactive calculations local so the exported report works offline."""
        return r"""
class FactorReport {
    constructor(data) {
        this.data = data;
        this.colors = ['#2563eb', '#06a6a0', '#e7a32e', '#9764d9', '#ed6976'];
        this.names = ['G1', 'G2', 'G3', 'G4', 'G5'];
        this.start = document.getElementById('range-start');
        this.end = document.getElementById('range-end');
        this.pan = document.getElementById('range-pan');
        this.start.max = this.end.max = data.rows.length - 1;
        this.end.value = data.rows.length - 1;
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
            this.end.value = data.rows.length - 1;
            this.schedule();
        });
        for (const button of document.querySelectorAll('[role="tab"]')) {
            button.addEventListener('click', () => {
                for (const tab of document.querySelectorAll('[role="tab"]')) {
                    const active = tab === button;
                    tab.setAttribute('aria-selected', String(active));
                    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
                }
                this.updateCards(button.id === 'tab-groups' ? this.selected : data.rows);
                document.getElementById('card-scope').textContent = button.id === 'tab-groups'
                    ? '指标范围：分组结果所选区间' : '指标范围：完整展示区间';
            });
        }
        this.renderGroups();
        this.renderYears();
    }

    schedule() {
        cancelAnimationFrame(this.frame);
        this.frame = requestAnimationFrame(() => this.renderGroups());
    }

    mean(values) {
        const valid = values.filter(value => value !== null && Number.isFinite(value));
        return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : NaN;
    }

    deviation(values) {
        if (values.length < 2) return NaN;
        const average = this.mean(values);
        return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1));
    }

    format(value, percent = false) {
        return value !== null && Number.isFinite(value) ? (value * (percent ? 100 : 1)).toFixed(percent ? 2 : 3) + (percent ? '%' : '') : '—';
    }

    complete(rows) {
        return rows.filter(row => this.names.every((_, index) => row['group_' + (index + 1)] !== null));
    }

    direction(rows) {
        const mean = this.mean(rows.map(row => row.IC));
        return Number.isFinite(mean) ? (mean > 0 ? 4 : 0) : null;
    }

    performance(returns) {
        if (!returns.length || returns.some(value => value < -1)) {
            return {metrics: [returns.length, NaN, NaN, NaN, NaN, NaN], cumulative: [], drawdown: []};
        }
        let wealth = 1;
        let peak = 1;
        let minimum = 0;
        const cumulative = [0];
        const drawdown = [0];
        for (const value of returns) {
            wealth *= 1 + value;
            peak = Math.max(peak, wealth);
            cumulative.push(wealth - 1);
            drawdown.push(wealth / peak - 1);
            minimum = Math.min(minimum, wealth / peak - 1);
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

    updateCards(rows) {
        const ic = rows.map(row => row.IC).filter(value => value !== null);
        const deviation = this.deviation(ic);
        const preferred = this.direction(rows);
        const sharpe = preferred === null ? NaN : this.performance(this.complete(rows).map(row => row['group_' + (preferred + 1)])).metrics[5];
        [this.mean(ic), this.mean(rows.map(row => row.rankIC)), deviation > 0 ? this.mean(ic) / deviation : NaN, sharpe]
            .forEach((value, index) => document.getElementById('card-' + index).textContent = this.format(value));
    }

    table(headers, rows) {
        return '<table class="metrics"><thead><tr>' + headers.map(value => `<th>${value}</th>`).join('')
            + '</tr></thead><tbody>' + rows.map(row => '<tr>' + row.map(value => `<td>${value}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
    }

    metricRow(label, returns, counts, turnover) {
        const metrics = this.performance(returns).metrics;
        const row = [label, metrics[0], this.format(this.mean(counts))];
        if (turnover !== undefined) row.push(this.format(this.mean(turnover), true));
        return row.concat(metrics.slice(1).map((value, index) => this.format(value, index < 4)));
    }

    // Draw SVG directly: no CDN, network requests, or external chart dependency.
    chart(target, labels, series, percent = true, bars = false) {
        const container = document.getElementById(target);
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
            return;
        }
        const padding = (high - low || 0.01) * 0.15;
        low -= padding;
        high += padding;
        const left = 100, top = 48, width = 990, height = 260;
        const x = index => left + (bars ? (index + 0.5) / labels.length : index / Math.max(labels.length - 1, 1)) * width;
        const y = value => top + (high - value) / (high - low) * height;
        let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 365" role="img" aria-label="' + container.dataset.label + '">';
        for (let index = 0; index <= 4; index++) {
            const value = low + (high - low) * index / 4;
            svg += `<line x1="${left}" x2="${left + width}" y1="${y(value)}" y2="${y(value)}" stroke="#e8edf5"/>`;
            svg += `<text x="${left - 12}" y="${y(value) + 4}" text-anchor="end">${this.format(value, percent)}</text>`;
        }
        svg += `<line x1="${left}" x2="${left + width}" y1="${y(0)}" y2="${y(0)}" stroke="#94a3b8" stroke-dasharray="4 4"/>`;
        const ticks = Math.min(labels.length, bars ? 5 : 7);
        for (let tick = 0; tick < ticks; tick++) {
            const index = ticks === 1 ? 0 : Math.round(tick * (labels.length - 1) / (ticks - 1));
            svg += `<text x="${x(index)}" y="338" text-anchor="middle">${labels[index]}</text>`;
        }
        series.forEach((item, seriesIndex) => {
            const color = this.colors[seriesIndex % this.colors.length];
            if (bars) {
                item.values.forEach((value, index) => {
                    if (!Number.isFinite(value)) return;
                    svg += `<rect x="${x(index) - 48}" y="${Math.min(y(0), y(value))}" width="96" height="${Math.abs(y(value) - y(0))}" fill="${this.colors[index]}"><title>${labels[index]}: ${this.format(value, percent)}</title></rect>`;
                    svg += `<text x="${x(index)}" y="${value >= 0 ? y(value) - 9 : y(value) + 18}" text-anchor="middle">${this.format(value, percent)}</text>`;
                });
            } else {
                let path = '';
                let move = true;
                item.values.forEach((value, index) => {
                    if (!Number.isFinite(value)) { move = true; return; }
                    path += `${move ? 'M' : 'L'}${x(index).toFixed(2)},${y(value).toFixed(2)} `;
                    move = false;
                });
                svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.8"><title>${item.name}</title></path>`;
                svg += `<line x1="${left + seriesIndex * 150}" x2="${left + 22 + seriesIndex * 150}" y1="22" y2="22" stroke="${color}" stroke-width="3"/>`;
                svg += `<text x="${left + 30 + seriesIndex * 150}" y="26">${item.name}</text>`;
            }
        });
        container.innerHTML = svg + '</svg>';
    }

    renderGroups() {
        const start = +this.start.value;
        const end = +this.end.value;
        this.selected = this.data.rows.slice(start, end + 1);
        const rows = this.complete(this.selected);
        const preferred = this.direction(this.selected);
        const short = preferred === null ? null : 4 - preferred;
        const groups = this.names.map((_, index) => rows.map(row => row['group_' + (index + 1)]));
        const performances = groups.map(returns => this.performance(returns));
        const spread = preferred === null ? [] : groups[preferred].map((value, index) => value - groups[short][index]);
        const invalidSpread = spread.some(value => value < -1);
        const longShort = this.performance(spread);
        this.pan.max = this.data.rows.length - (end - start + 1);
        this.pan.value = start;
        this.pan.disabled = +this.pan.max === 0;
        for (const [id, index] of [['start-date', start], ['end-date', end]]) {
            document.getElementById(id).textContent = this.data.rows[index].date;
        }
        document.getElementById('range-summary').textContent = `${this.data.rows[start].date} — ${this.data.rows[end].date} · ${this.selected.length} 个评估日 · ${rows.length} 个完整收益日 · 剔除 ${this.selected.length - rows.length} 个收益缺失日`;
        document.getElementById('direction-note').textContent = preferred === null
            ? '所选区间没有有效 IC，无法确定多头组和多空方向。'
            : `当前方向：${this.names[preferred]} − ${this.names[short]}，由所选区间 IC 均值确定。` + (invalidSpread ? '该区间多空单日亏损超过 100%，不计算多空复利指标和曲线。' : '');
        const records = groups.map((returns, index) => this.metricRow(this.names[index] + (index === preferred ? ' · 多头' : ''), returns,
            rows.map(row => row['group_' + (index + 1) + '_count']), rows.map(row => row['group_' + (index + 1) + '_turnover'])));
        records.push(this.metricRow('Long-short', spread, preferred === null ? [] : rows.map(row => {
            const longCount = row['group_' + (preferred + 1) + '_count'];
            const shortCount = row['group_' + (short + 1) + '_count'];
            return longCount === null || shortCount === null ? null : longCount + shortCount;
        }), []));
        document.getElementById('group-table').innerHTML = this.table(
            ['组合', '有效交易日', '日均股票数量', '平均换手率', '区间收益', '年化收益', '年化波动率', '最大回撤', 'Sharpe'], records);
        const labels = ['起点', ...rows.map(row => row.date)];
        for (const [target, key] of [['group-cumulative', 'cumulative'], ['group-drawdown', 'drawdown']]) {
            this.chart(target, labels, performances.map((result, index) => ({name: this.names[index], values: result[key]})));
        }
        this.chart('annual-returns', this.names, [{name: '年化收益', values: performances.map(result => result.metrics[2])}], true, true);
        this.chart('ls-cumulative', labels, [{name: 'Long-short', values: longShort.cumulative}]);
        this.chart('ls-drawdown', labels, [{name: 'Long-short', values: longShort.drawdown}]);
        this.updateCards(this.selected);
    }

    renderYears() {
        const byYear = new Map();
        for (const row of this.complete(this.data.rows)) {
            const year = row.date.slice(0, 4);
            if (!byYear.has(year)) byYear.set(year, []);
            byYear.get(year).push(row);
        }
        const container = document.getElementById('yearly-curves');
        for (const [year, rows] of byYear) {
            const heading = document.createElement('h3');
            heading.textContent = year + ' · 五组累计收益';
            const chart = document.createElement('div');
            chart.id = 'year-' + year;
            chart.dataset.label = heading.textContent;
            container.append(heading, chart);
            this.chart(chart.id, ['起点', ...rows.map(row => row.date)], this.names.map((name, index) => ({
                name, values: this.performance(rows.map(row => row['group_' + (index + 1)])).cumulative
            })));
        }
    }
}
new FactorReport(JSON.parse(document.getElementById('report-data').textContent));
"""

    def plot_results_html(self):
        """Write an offline HTML report with tabs and an interactive date range."""
        data = self.factor_results
        yearly_rows = [(str(year), returns, self.group_counts[f"{self.preferred_group}_count"]) for year, returns in self.group_returns[self.preferred_group].groupby(self.group_returns.index.year)]
        yearly_table = self._performance_table(yearly_rows, "年份")
        ic_chart = self._chart({"Cumulative IC": data["IC"].cumsum()}, "Cumulative IC", zero=True)
        rank_ic_chart = self._chart({"Cumulative RankIC": data["rankIC"].cumsum()}, "Cumulative RankIC", zero=True)
        columns = self.GROUPS + ["IC", "rankIC"] + [f"{group}_{suffix}" for suffix in ("count", "turnover") for group in self.GROUPS]
        payload = data.reindex(columns=columns).copy()
        payload.insert(0, "date", data.index.strftime("%Y-%m-%d"))
        report_data = json.dumps({"rows": json.loads(payload.to_json(orient="records", double_precision=15)),
                                  "periods": self.periods_per_year, "riskFree": self.risk_free_rate}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        cards = "".join(f'<div class="stat"><span>{label}</span><strong id="card-{index}">—</strong></div>'
                        for index, label in enumerate(["IC 均值", "RankIC 均值", "ICIR", "多头 Sharpe"]))
        title = escape(self.factor_name)
        html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 因子评估报告</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif; color: #172b4d; background: #f2f5fa; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; }} main {{ max-width: 1280px; margin: auto; padding: 24px 28px; }}
.tabs {{ position: sticky; top: 0; z-index: 10; display: flex; gap: 10px; padding: 14px 0; background: #f2f5fa; }}
button {{ border: 1px solid #dce3ed; border-radius: 10px; padding: 12px 20px; background: white; color: #28578c; cursor: pointer; font: inherit; }}
button[aria-selected="true"] {{ color: white; background: #28578c; border-color: #28578c; }}
button:focus-visible, input:focus-visible {{ outline: 3px solid #06a6a0; outline-offset: 3px; }}
[hidden] {{ display: none !important; }} header {{ padding: 32px; background: linear-gradient(120deg, #172c52, #28578c); color: white; border-radius: 20px; }}
.eyebrow {{ color: #9edcfa; font-size: 12px; letter-spacing: 3px; }} h1 {{ font-size: 30px; margin: 16px 0; overflow-wrap: anywhere; }}
header p {{ color: #d2dfef; line-height: 1.8; }} .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin: 14px 0 24px; }}
.stat, section {{ background: white; border: 1px solid #e3e9f2; border-radius: 16px; box-shadow: 0 5px 20px #23395605; }}
.stat {{ padding: 24px; }} .stat span {{ display: block; color: #718096; font-size: 13px; }} .stat strong {{ display: block; margin-top: 12px; font-size: 28px; color: #2563a4; }}
section {{ padding: 26px; margin-bottom: 22px; }} h2 {{ margin: 0; font-size: 20px; }} h3 {{ margin: 24px 0 6px; font-size: 16px; }}
section p, footer, #card-scope {{ font-size: 13px; color: #6b7c94; line-height: 1.9; }} .content {{ overflow-x: auto; }}
svg {{ display: block; width: 100%; height: auto; min-width: 560px; }} svg text {{ font-family: inherit; font-size: 12px; fill: #64748b; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }} th {{ background: #f4f7fb; color: #61718b; font-weight: 500; }}
th, td {{ padding: 15px 13px; text-align: right; border-bottom: 1px solid #edf1f6; }} th:first-child, td:first-child {{ text-align: left; }} tbody tr:hover {{ background: #f4f8ff; }}
.range-control {{ display: grid; grid-template-columns: 210px 1fr; align-items: center; gap: 18px; margin: 18px 0; font-size: 14px; }}
input[type="range"] {{ width: 100%; accent-color: #2563eb; cursor: ew-resize; }} output {{ color: #2563a4; }} .empty {{ text-align: center; padding: 40px; }}
footer {{ padding: 4px 12px 20px; }} @media(max-width: 700px) {{ main {{ padding: 12px; }} header, section {{ padding: 18px; }} .stats {{ grid-template-columns: repeat(2, 1fr); gap: 10px; }} .stat {{ padding: 18px; }} h1 {{ font-size: 24px; }} .stat strong {{ font-size: 23px; }} .range-control {{ grid-template-columns: 1fr; gap: 8px; }} .tabs button {{ flex: 1; padding: 12px 6px; }} }}
@media print {{ main {{ padding: 0; }} .tabs, #range-controls {{ display: none; }} [role="tabpanel"][hidden] {{ display: block !important; }} section {{ break-inside: avoid; }} .content {{ overflow: visible; }} svg {{ min-width: 0; }} }}
</style></head><body><main>
<nav class="tabs" role="tablist" aria-label="报告页面">
<button id="tab-groups" role="tab" aria-controls="page-groups" aria-selected="true">分组结果</button>
<button id="tab-ic" role="tab" aria-controls="page-ic" aria-selected="false">IC 分析</button>
<button id="tab-years" role="tab" aria-controls="page-years" aria-selected="false">分年收益</button>
</nav>
<header><div class="eyebrow">FACTOR RESEARCH / PERFORMANCE REPORT</div><h1>{title} · 因子评估报告</h1>
<p>{data.index.min():%Y-%m-%d} — {data.index.max():%Y-%m-%d} · {len(data):,} 个评估日 · {len(self.group_returns):,} 个完整收益日</p></header>
<p id="card-scope">指标范围：分组结果所选区间</p><div class="stats">{cards}</div>
<noscript><p>请启用 JavaScript，以使用页面切换、时间滑块和交互收益图表。</p></noscript>
<div id="page-groups" role="tabpanel" aria-labelledby="tab-groups">
<section id="range-controls"><h2>分析时间区间</h2><p id="range-summary" aria-live="polite"></p>
<div class="range-control"><label for="range-start">开始日期 <output id="start-date" for="range-start"></output></label><input id="range-start" type="range" min="0" value="0" step="1"></div>
<div class="range-control"><label for="range-end">结束日期 <output id="end-date" for="range-end"></output></label><input id="range-end" type="range" min="0" value="0" step="1"></div>
<div class="range-control"><label for="range-pan">平移整个时间窗口</label><input id="range-pan" type="range" min="0" value="0" step="1"></div>
<button id="range-reset">恢复完整区间</button><p>拖动滑块立即重算本页指标与曲线；平移窗口保持评估日数不变。IC 分析与分年收益页始终使用完整展示区间。</p></section>
<section><h2>分组绩效</h2><p id="direction-note"></p><p>G1 为因子值最低组，G5 为最高组。各组与多空组合使用相同的完整收益日；股票数量与换手率按这些日期取均值，缺失值不参与均值。多空股票数量为两端之和，不展示多空换手率。</p><div id="group-table" class="content"></div></section>
<section><h2>五组累计收益和动态回撤</h2><p>按日复利累计；所选区间起点收益为 0、净值为 1。动态回撤 = 当前净值 / 区间内历史最高净值 − 1，包含初始净值。</p>
<h3>五组累计收益</h3><div id="group-cumulative" class="content" data-label="五组累计收益"></div><h3>五组动态回撤</h3><div id="group-drawdown" class="content" data-label="五组动态回撤"></div></section>
<section><h2>五组年化收益 · 单调性</h2><p>按因子值从低到高排列 G1 → G5。年化收益按所选区间的有效收益日数折算；正向因子观察是否递增，负向因子观察是否递减。</p><div id="annual-returns" class="content" data-label="五组年化收益"></div></section>
<section><h2>Long-short 累计收益和动态回撤</h2><p>每日多头收益减空头收益，再复利累计；累计收益与回撤均在所选区间起点重置。</p><h3>Long-short 累计收益</h3><div id="ls-cumulative" class="content" data-label="Long-short 累计收益"></div><h3>Long-short 动态回撤</h3><div id="ls-drawdown" class="content" data-label="Long-short 动态回撤"></div></section>
</div>
<div id="page-ic" role="tabpanel" aria-labelledby="tab-ic" hidden>
<section><h2>IC / RankIC 统计</h2><p>完整展示区间的截面相关性统计，IR 保留方向符号。</p><div class="content">{self._ic_table()}</div></section>
<section><h2>累积 IC</h2><p>每日 Pearson IC 的算术累加；缺失日期留空，不参与累加。</p><div class="content">{ic_chart}</div></section>
<section><h2>累积 RankIC</h2><p>每日秩相关系数 RankIC 的算术累加；缺失日期留空，不参与累加。</p><div class="content">{rank_ic_chart}</div></section>
</div>
<div id="page-years" role="tabpanel" aria-labelledby="tab-years" hidden>
<section><h2>IC方向优选组 G{self.preferred_group[-1]} · 分年绩效</h2><p>多头组由完整展示区间的 IC 均值确定。区间收益为该年实际覆盖日期的复利收益，首尾年份可能不完整；回撤每年重置。</p><div class="content">{yearly_table}</div></section>
<section><h2>分年五组累计收益</h2><p>按年份分面展示五组收益。各年从 0 重新开始，净值从 1 按该年有效日收益复利累计，不继承上一年净值；横轴按有效交易日排列。</p><div id="yearly-curves" class="content"></div></section>
</div>
<footer><b>计算口径</b><br>
日收益使用小数；年化交易日数 {self.periods_per_year:g}，年化无风险利率 {self.risk_free_rate:.2%}。
年化收益 = ∏(1 + 日收益)^(年化交易日数 / 有效交易日数) − 1；年化波动率 = 日收益样本标准差 × √年化交易日数。
Sharpe = (平均日收益 − 等效日无风险利率) / 日收益样本标准差 × √年化交易日数。
ICIR / RankICIR = 均值 / 样本标准差；年化 IR 再乘 √年化交易日数。标准差为零或样本不足时显示「—」。<br>
IC 均值 &gt; 0 时做多 G5、做空 G1，否则做多 G1、做空 G5。第一页按所选区间确定方向，其他页按完整展示区间确定，属于事后分析。
多空按多头 100%、空头 100% 的收益差计算，未除以 2；不计手续费、滑点和融券成本。
缺失 IC / RankIC 各自剔除；任一组缺失收益的日期从全部收益统计中共同剔除，不填充为零。
平均换手率沿用原始日换手率，在有效收益日内求均值；拖动区间不重建持仓，首日沿用已有值。收益图横轴按有效交易日等距排列。<br>
日期沿用评估结果的收益起始日标签，跨年收益按该标签归属年份。分年收益以实际样本区间为准，年化指标按有效日数折算。
所有数据与图表已嵌入，可离线查看。生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}。
</footer></main><script id="report-data" type="application/json">{report_data}</script><script>{self._report_script()}</script></body></html>'''
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(character if character.isalnum() or character in "-_." else "_" for character in self.factor_name).strip(".") or "factor"
        output_path = self.output_dir / f"{safe_name}_report.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Evaluation CSV; defaults to the configured factor results.")
    parser.add_argument("--factor-name")
    parser.add_argument("--start-date")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[3] / "config/config_factor_evaluation.json")
    args = parser.parse_args()
    output_dir = args.output_dir
    if args.results is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from main import load_config

        config = load_config(args.config)
        factor_name = args.factor_name or config["specified_column"] or pd.read_parquet(config["factor_data_dir"]).columns[2]
        results = Path(config["output_dir"]) / f"{factor_name}.csv"
        start_date = args.start_date or config["start_date"]
        if output_dir is None:
            output_dir = config.get("visualization_output_dir")
    else:
        results = args.results
        factor_name = args.factor_name or results.stem
        start_date = args.start_date
    print(ResultsVisualizer(results, factor_name, start_date, output_dir).plot_results_html())


if __name__ == "__main__":
    main()
