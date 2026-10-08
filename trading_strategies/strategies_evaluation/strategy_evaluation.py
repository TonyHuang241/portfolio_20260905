import json
import os

import numpy as np
import pandas as pd

from factors.evaluation.single_factor_evaluation_v1 import SingleFactorEvaluation


class StrategyEvaluation:
    LABELS = {"return": "理论", "net_return": "扣费", "slippage_return": "扣费+滑点"}
    PERCENT_COLUMNS = ["区间收益", "年化收益", "年化波动", "最大回撤", "日胜率", "持仓日占比", "日均换手"]
    # 报告样式：颜色集中定义为 CSS 变量，深色模式另配一套；曲线颜色与 LABELS 的键同名。
    STYLE = """
:root { --bg: #f5f7fb; --card: #ffffff; --ink: #172b4d; --muted: #6b7c94; --line: #e3e9f2; --grid: #e8edf5; --head: #f4f7fb; --accent: #2563a4; --return: #2a78d6; --net_return: #eb6834; --slippage_return: #1baf7a; }
@media (prefers-color-scheme: dark) { :root { --bg: #111110; --card: #1a1a19; --ink: #f2f2ee; --muted: #a3a29a; --line: #2e2e2b; --grid: #2a2a28; --head: #222220; --accent: #86b6ef; --return: #3987e5; --net_return: #d95926; --slippage_return: #199e70; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.6 -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; font-variant-numeric: tabular-nums; }
main { max-width: 1200px; margin: 0 auto; padding: 28px 24px; }
header { margin-bottom: 22px; } .eyebrow { font-size: 12px; letter-spacing: .12em; color: var(--muted); } h1 { margin: 6px 0; font-size: 28px; } header p { margin: 0; color: var(--muted); }
.caption { margin-bottom: 8px; font-size: 13px; color: var(--muted); }
.stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 20px; }
.stat, section { background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 5px 20px #23395608; }
.stat { padding: 16px 18px; } .stat span { display: block; font-size: 13px; color: var(--muted); } .stat strong { display: block; margin-top: 8px; font-size: 24px; color: var(--accent); }
section { padding: 22px 24px; margin-bottom: 20px; } h2 { margin: 0; font-size: 18px; } section p { margin: 4px 0 0; font-size: 13px; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 12px 0 4px; font-size: 13px; color: var(--muted); }
.legend i, #tooltip i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.chart svg { display: block; } .chart text { font-size: 12px; fill: var(--muted); } .chart .grid { stroke: var(--grid); }
.chart path { fill: none; stroke-width: 2; stroke-linejoin: round; } .chart .crosshair { stroke: var(--muted); stroke-dasharray: 3 3; }
.content { overflow-x: auto; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; } th { background: var(--head); color: var(--muted); font-weight: 500; }
th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--line); } th:first-child, td:first-child, .yearly th:nth-child(2), .yearly td:nth-child(2) { text-align: left; } tbody tr:hover { background: var(--head); } .yearly tbody tr:nth-child(3n + 1) td { border-top: 2px solid var(--line); }
footer { padding: 0 4px 20px; font-size: 12px; line-height: 1.9; color: var(--muted); }
#tooltip { position: fixed; z-index: 10; pointer-events: none; min-width: 170px; padding: 10px 12px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 8px 24px #0000001f; font-size: 12px; color: var(--muted); }
#tooltip div { display: flex; align-items: center; } #tooltip strong { margin-left: auto; padding-left: 16px; color: var(--ink); } #tooltip .tooltip-title { margin-bottom: 4px; color: var(--ink); font-weight: 600; }
@media (max-width: 900px) { .stats { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 600px) { main { padding: 16px; } section { padding: 16px; } .stats { grid-template-columns: repeat(2, 1fr); } h1 { font-size: 22px; } }
"""
    # 报告脚本：按容器宽度绘制 SVG 折线图，悬停时显示十字线和各口径数值，窗口缩放时重绘。
    SCRIPT = """
const data = JSON.parse(document.getElementById("report-data").textContent);
const tooltip = document.getElementById("tooltip");
const margin = {left: 56, right: 16, top: 12, bottom: 28};

function drawChart(id, key, height, format) {
    const element = document.getElementById(id);
    const width = element.clientWidth, count = data.dates.length;
    const values = data.series.flatMap(item => item[key]);
    // 纵轴刻度取 1 / 2 / 2.5 / 5 × 10^n 的整齐步长；回撤图顶端固定为 0。
    const rough = (Math.max(...values) - Math.min(...values)) / 6, magnitude = 10 ** Math.floor(Math.log10(rough));
    const tick = [1, 2, 2.5, 5, 10].find(factor => factor * magnitude >= rough) * magnitude;
    const low = Math.floor(Math.min(...values) / tick) * tick, high = key === "drawdown" ? 0 : Math.ceil(Math.max(...values) / tick) * tick;
    const x = index => margin.left + index / (count - 1) * (width - margin.left - margin.right);
    const y = value => margin.top + (high - value) / (high - low) * (height - margin.top - margin.bottom);
    let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    for (let step = Math.round(low / tick); step <= Math.round(high / tick); step++) {
        const value = step * tick;
        svg += `<line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${y(value)}" y2="${y(value)}"/><text x="${margin.left - 8}" y="${y(value) + 4}" text-anchor="end">${format(value)}</text>`;
    }
    data.dates.forEach((date, index) => {
        if (index === 0 || date.slice(0, 4) !== data.dates[index - 1].slice(0, 4)) svg += `<line class="grid" x1="${x(index)}" x2="${x(index)}" y1="${margin.top}" y2="${height - margin.bottom}"/><text x="${x(index) + 4}" y="${height - 8}">${date.slice(0, 4)}</text>`;
    });
    data.series.forEach(item => {
        svg += `<path style="stroke: var(--${item.id})" d="M${item[key].map((value, index) => x(index).toFixed(1) + "," + y(value).toFixed(1)).join("L")}"/>`;
    });
    element.innerHTML = svg + `<line class="crosshair" y1="${margin.top}" y2="${height - margin.bottom}" visibility="hidden"/></svg>`;

    const crosshair = element.querySelector(".crosshair");
    element.onpointermove = event => {
        const index = Math.max(0, Math.min(count - 1, Math.round((event.clientX - element.getBoundingClientRect().left - margin.left) / (width - margin.left - margin.right) * (count - 1))));
        crosshair.setAttribute("x1", x(index));
        crosshair.setAttribute("x2", x(index));
        crosshair.setAttribute("visibility", "visible");
        tooltip.innerHTML = `<div class="tooltip-title">${data.dates[index].replace(/(\\d{4})(\\d{2})(\\d{2})/, "$1-$2-$3")}</div>` + data.series.map(item => `<div><i style="background: var(--${item.id})"></i>${item.label}<strong>${format(item[key][index])}</strong></div>`).join("");
        tooltip.hidden = false;
        tooltip.style.left = Math.min(event.clientX + 16, window.innerWidth - tooltip.offsetWidth - 8) + "px";
        tooltip.style.top = Math.min(event.clientY + 16, window.innerHeight - tooltip.offsetHeight - 8) + "px";
    };
    element.onpointerleave = () => {
        tooltip.hidden = true;
        crosshair.setAttribute("visibility", "hidden");
    };
}

function drawCharts() {
    drawChart("nav", "nav", 360, value => value.toFixed(2));
    drawChart("drawdown", "drawdown", 220, value => (value * 100).toFixed(1) + "%");
}
drawCharts();
window.addEventListener("resize", drawCharts);
"""

    def __init__(self, holdings, backtest_data_dir, limit_price_dir, output_dir, fee_rate, slippage_rate):
        self.holdings = holdings
        self.backtest_data_dir = backtest_data_dir
        self.limit_price_dir = limit_price_dir
        self.output_dir = output_dir
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    @staticmethod
    def _metrics(result, column):
        """计算区间收益、年化收益、年化波动、Sharpe、最大回撤、Calmar、持仓日胜率、持仓日占比和日均换手。"""
        nav = (1 + result[column]).cumprod()
        annual_return = nav.iloc[-1] ** (252 / len(nav)) - 1
        max_drawdown = (nav / nav.cummax().clip(lower=1) - 1).min()
        return {"区间收益": nav.iloc[-1] - 1, "年化收益": annual_return, "年化波动": result[column].std() * 252 ** 0.5, "Sharpe": result[column].mean() / result[column].std() * 252 ** 0.5,
                "最大回撤": max_drawdown, "Calmar": annual_return / -max_drawdown, "日胜率": result.loc[result["exposure"].gt(0), column].gt(0).mean(), "持仓日占比": result["exposure"].gt(0).mean(), "日均换手": result["turnover"].mean()}

    def evaluate(self):
        """按持仓清单在 trade_date 10 点等权买入、持有到下一交易日 10 点，换手（买卖双边）乘 fee_rate 扣费；尚无收益的下一交易日不参与评估。"""
        prices = SingleFactorEvaluation.load_prices(self.backtest_data_dir)
        returns = (prices.shift(-1) / prices - 1).iloc[:-1]
        returns = returns.loc[self.holdings["trade_date"].min():self.holdings["trade_date"].max()]
        held = self.holdings.loc[self.holdings["hold"]]
        weights = pd.crosstab(held["trade_date"], held["code"])
        weights = weights.reindex(index=returns.index, columns=returns.columns, fill_value=0)

        # 10 点价格（还原为未复权价）达到涨停价时不能买入或加仓，达到跌停价时不能卖出或减仓；unlimited 为 1 时无涨跌停限制。
        limits = pd.read_parquet(self.backtest_data_dir, columns=["code", "date", "close", "adj_factor"])
        limits = limits.merge(pd.read_parquet(os.path.join(self.limit_price_dir, "limit_price.parquet")), on=["code", "date"])
        limits["close"] = (limits["close"] / limits["adj_factor"]).round(2)
        limits["up_limit"] = limits["close"].ge(limits["high_limit"]) & limits["unlimited"].eq(0)
        limits["down_limit"] = limits["close"].le(limits["low_limit"]) & limits["unlimited"].eq(0)
        up_limit = limits.pivot(index="date", columns="code", values="up_limit").reindex(index=returns.index, columns=returns.columns).eq(True).to_numpy()
        down_limit = limits.pivot(index="date", columns="code", values="down_limit").reindex(index=returns.index, columns=returns.columns).eq(True).to_numpy()

        # 逐日调仓：跌停的原持仓保持原权重，其余资金等分给未被锁定的目标股票，涨停股票的权重不超过前一日；买不进的部分持有现金。
        target = weights.to_numpy() > 0
        actual = np.zeros(target.shape)
        previous = np.zeros(target.shape[1])
        for day in range(len(target)):
            locked = (previous > 0) & down_limit[day]
            actual[day] = np.where(target[day] & ~locked, (1 - previous[locked].sum()) / max((target[day] & ~locked).sum(), 1), 0)
            actual[day] = np.where(up_limit[day], np.minimum(actual[day], previous), actual[day])
            actual[day] = np.where(locked, previous, actual[day])
            previous = actual[day]
        weights = pd.DataFrame(actual, index=weights.index, columns=weights.columns)

        # 无法买入（无价格）或停牌的股票收益记 0，相当于该部分持有现金。
        result = pd.DataFrame(index=returns.index)
        result["exposure"] = weights.sum(axis=1)
        result["return"] = (weights * returns.fillna(0)).sum(axis=1)
        result["turnover"] = weights.diff().fillna(weights).abs().sum(axis=1)
        result["net_return"] = result["return"] - result["turnover"] * self.fee_rate
        # 每笔成交的滑点为成交权重 × slippage_rate，方向随机（+1 为成交价不利、-1 为有利），固定随机种子保证结果可复现。
        result["slippage_return"] = result["net_return"] - (weights.diff().fillna(weights).abs() * np.random.default_rng(0).choice([-1, 1], size=weights.shape)).sum(axis=1) * self.slippage_rate

        # 绩效表：全区间三种口径对比，分年按年份排列、同一年内依次为各口径。
        rows = []
        for column in self.LABELS:
            rows.append({"区间": "全区间", "口径": self.LABELS[column], **self._metrics(result, column)})
            for year, year_result in result.groupby(result.index.str[:4]):
                rows.append({"区间": year, "口径": self.LABELS[column], **self._metrics(year_result, column)})
        rows = pd.DataFrame(rows)
        formatters = {column: "{:.2%}".format for column in self.PERCENT_COLUMNS}
        full_table = rows.loc[rows["区间"].eq("全区间")].drop(columns="区间").to_html(index=False, border=0, float_format="{:.2f}".format, formatters=formatters)
        yearly_table = rows.loc[rows["区间"].ne("全区间")].sort_values("区间", kind="stable").to_html(index=False, border=0, classes="yearly", float_format="{:.2f}".format, formatters=formatters)

        # 顶部指标卡片取最贴近实盘的扣费+滑点口径；曲线数据嵌入页面，由脚本绘制可悬停的累积净值和动态回撤（当前净值 / 历史最高净值 − 1，含初始净值 1）。
        summary = self._metrics(result, "slippage_return")
        cards = "".join(f'<div class="stat"><span>{name}</span><strong>{summary[name]:.2%}</strong></div>' if name in self.PERCENT_COLUMNS else f'<div class="stat"><span>{name}</span><strong>{summary[name]:.2f}</strong></div>' for name in ["年化收益", "Sharpe", "最大回撤", "Calmar", "持仓日占比", "日均换手"])
        series = []
        for column in self.LABELS:
            result[f"{column}_nav"] = (1 + result[column]).cumprod()
            series.append({"id": column, "label": self.LABELS[column], "nav": result[f"{column}_nav"].round(5).tolist(), "drawdown": (result[f"{column}_nav"] / result[f"{column}_nav"].cummax().clip(lower=1) - 1).round(5).tolist()})
        legend = "".join(f'<span><i style="background:var(--{column})"></i>{self.LABELS[column]}</span>' for column in self.LABELS)
        strategy_name = os.path.basename(self.output_dir)

        html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{strategy_name} 策略回测</title><style>{self.STYLE}</style></head><body><main>
<header><div class="eyebrow">TRADING STRATEGY / BACKTEST REPORT</div><h1>{strategy_name} · 策略回测报告</h1>
<p>{pd.Timestamp(result.index[0]):%Y-%m-%d} — {pd.Timestamp(result.index[-1]):%Y-%m-%d} · {len(result):,} 个交易日 · 单边费率 {self.fee_rate:.2%} · 滑点 ±{self.slippage_rate:.2%}（随机方向）</p></header>
<div class="caption">扣费+滑点口径 · 全区间</div><div class="stats">{cards}</div>
<section><h2>累积净值</h2><p>按日复利累计，起点净值为 1。</p><div class="legend">{legend}</div><div id="nav" class="chart"></div></section>
<section><h2>动态回撤</h2><p>当前净值 / 历史最高净值 − 1，包含初始净值 1。</p><div class="legend">{legend}</div><div id="drawdown" class="chart"></div></section>
<section><h2>全区间绩效对比</h2><div class="content">{full_table}</div></section>
<section><h2>分年绩效</h2><p>每年单独计算，回撤每年重置；首尾年份可能不完整。</p><div class="content">{yearly_table}</div></section>
<footer><b>计算口径</b><br>信号日收盘后确定持仓，下一交易日 10 点等权调仓，持有到再下一交易日 10 点；尚无收益的下一交易日不参与评估。
10 点价格达到涨停价不买入、达到跌停价不卖出，买不进的部分持有现金；停牌或无价格的股票当日收益记 0。<br>
扣费 = 买卖双边换手 × 单边费率；滑点在扣费基础上，每笔成交按成交权重 × 滑点率随机加减（方向随机、固定随机种子）。
年化收益 = 净值^(252 / 交易日数) − 1；Sharpe 不扣无风险利率；日胜率只统计有持仓的交易日；日均换手为买卖双边。生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}。</footer>
</main><div id="tooltip" hidden></div>
<script id="report-data" type="application/json">{json.dumps({"dates": result.index.tolist(), "series": series}, ensure_ascii=False)}</script><script>{self.SCRIPT}</script></body></html>'''
        with open(os.path.join(self.output_dir, f"{strategy_name}Evaluation.html"), "w", encoding="utf-8") as file:
            file.write(html)
        return result
