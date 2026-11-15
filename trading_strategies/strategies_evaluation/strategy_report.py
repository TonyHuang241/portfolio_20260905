import json
import os

import pandas as pd


class StrategyReport:
    LABELS = {"return": "理论", "net_return": "扣费", "slippage_return": "扣费+滑点"}
    PERCENT_COLUMNS = ["区间收益", "年化收益", "年化波动", "最大回撤", "日胜率", "月胜率", "持仓日占比", "日均换手", "最大单日收益", "最大单日亏损"]
    DETAIL_COLUMNS = ["code", "name", "previous_shares", "shares", "buy_price", "sell_price", "stock_return", "pnl", "fee", "slippage", "target", "up_limit", "down_limit", "suspended"]
    # 报告样式：颜色集中定义为 CSS 变量，深色模式另配一套；曲线颜色与 LABELS 的键同名；月度收益按 A 股习惯红涨绿跌。
    STYLE = """
:root { --bg: #f5f7fb; --card: #ffffff; --ink: #172b4d; --muted: #6b7c94; --line: #e3e9f2; --grid: #e8edf5; --head: #f4f7fb; --accent: #2563a4; --return: #2a78d6; --net_return: #eb6834; --slippage_return: #1baf7a; --up: #d03b32; --down: #148a52; }
@media (prefers-color-scheme: dark) { :root { --bg: #111110; --card: #1a1a19; --ink: #f2f2ee; --muted: #a3a29a; --line: #2e2e2b; --grid: #2a2a28; --head: #222220; --accent: #86b6ef; --return: #3987e5; --net_return: #d95926; --slippage_return: #199e70; --up: #f0726a; --down: #4cc38a; } }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.6 -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; font-variant-numeric: tabular-nums; }
main { max-width: 1200px; margin: 0 auto; padding: 28px 24px; }
header { margin-bottom: 18px; } .eyebrow { font-size: 12px; letter-spacing: .12em; color: var(--muted); } h1 { margin: 6px 0; font-size: 28px; } header p { margin: 0; color: var(--muted); }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.tabs button { padding: 7px 18px; font: inherit; color: var(--muted); background: var(--card); border: 1px solid var(--line); border-radius: 999px; cursor: pointer; } .tabs button.active { color: var(--card); background: var(--accent); border-color: var(--accent); }
.caption { margin-bottom: 8px; font-size: 13px; color: var(--muted); }
.stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 20px; }
.stat, section { background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 5px 20px #23395608; }
.stat { padding: 16px 18px; } .stat span { display: block; font-size: 13px; color: var(--muted); } .stat strong { display: block; margin-top: 8px; font-size: 24px; color: var(--accent); }
section .stats { margin: 14px 0 0; } section .stat { box-shadow: none; padding: 12px 14px; } section .stat strong { font-size: 18px; }
section { padding: 22px 24px; margin-bottom: 20px; } h2 { margin: 0; font-size: 18px; } section p { margin: 4px 0 0; font-size: 13px; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 12px 0 4px; font-size: 13px; color: var(--muted); }
.legend i, #tooltip i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.chart { cursor: pointer; } .chart svg { display: block; } .chart text { font-size: 12px; fill: var(--muted); } .chart .grid { stroke: var(--grid); }
.chart path { fill: none; stroke-width: 2; stroke-linejoin: round; } .chart .crosshair { stroke: var(--muted); stroke-dasharray: 3 3; } .chart .selected { stroke: var(--accent); stroke-width: 1.5; }
.brush { margin-top: 4px; cursor: pointer; touch-action: none; user-select: none; } .brush svg { display: block; } .brush path { fill: none; stroke: var(--muted); stroke-width: 1.2; } .brush + .caption { margin: 6px 0 0; }
.brush .mask { fill: var(--bg); opacity: .8; } .brush .window { fill: var(--accent); fill-opacity: .08; stroke: var(--accent); cursor: grab; } .brush .handle { fill: var(--accent); cursor: ew-resize; }
.content { overflow-x: auto; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; } th { background: var(--head); color: var(--muted); font-weight: 500; }
th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--line); } th:first-child, td:first-child, .yearly th:nth-child(2), .yearly td:nth-child(2), .names th:nth-child(2), .names td:nth-child(2) { text-align: left; } tbody tr:hover { background: var(--head); } .yearly tbody tr:nth-child(3n + 1) td { border-top: 2px solid var(--line); }
.up { color: var(--up); } .down { color: var(--down); } .muted { color: var(--muted); }
footer { padding: 0 4px 20px; font-size: 12px; line-height: 1.9; color: var(--muted); }
#tooltip { position: fixed; z-index: 10; pointer-events: none; min-width: 170px; padding: 10px 12px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 8px 24px #0000001f; font-size: 12px; color: var(--muted); }
#tooltip div { display: flex; align-items: center; } #tooltip strong { margin-left: auto; padding-left: 16px; color: var(--ink); } #tooltip .tooltip-title { margin-bottom: 4px; color: var(--ink); font-weight: 600; }
@media (max-width: 900px) { .stats { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 600px) { main { padding: 16px; } section { padding: 16px; } .stats { grid-template-columns: repeat(2, 1fr); } h1 { font-size: 22px; } }
"""
    # 报告脚本：标签页切换；按容器宽度绘制 SVG 折线图，悬停时显示十字线和各口径数值，点击某一天显示当日逐股明细，窗口缩放时重绘；累计盈亏图下方的滑块选择时间窗口，顶部指标卡片和两张图都按窗口重新计算。
    SCRIPT = """
const data = JSON.parse(document.getElementById("report-data").textContent);
const tooltip = document.getElementById("tooltip");
const margin = {left: 68, right: 16, top: 12, bottom: 28};
let selected = data.dates.length - 1;
// range 为滑块选中的时间窗口（起止日期的下标），drag 为滑块正在进行的拖动。
let range = {start: 0, end: data.dates.length - 1};
let drag = null;
const formatDate = date => date.slice(0, 4) + "-" + date.slice(4, 6) + "-" + date.slice(6);
const percent = (value, digits = 2) => value === null ? "—" : (value * 100).toFixed(digits) + "%";
const signed = (value, digits = 2) => value === null ? "—" : `<span class="${value > 0 ? "up" : value < 0 ? "down" : ""}">${percent(value, digits)}</span>`;
const amount = (value, digits = 0) => value === null ? "—" : value.toLocaleString("zh-CN", {maximumFractionDigits: digits});
const signedAmount = value => value === null ? "—" : `<span class="${value > 0 ? "up" : value < 0 ? "down" : ""}">${amount(value)}</span>`;

// 时间窗口内的曲线：第一个点是窗口首日调仓前的起点（累计盈亏和回撤都为 0，start 为起点资产），之后每个交易日一个点；回撤按窗口内（含起点）的最高资产重新计算。
function windowSeries() {
    return data.series.map(item => {
        const base = range.start > 0 ? item.pnl[range.start - 1] : 0;
        const pnl = [0], drawdown = [0];
        let peak = data.initial_cash + base;
        for (let index = range.start; index <= range.end; index++) {
            peak = Math.max(peak, data.initial_cash + item.pnl[index]);
            pnl.push(item.pnl[index] - base);
            drawdown.push((data.initial_cash + item.pnl[index]) / peak - 1);
        }
        return {id: item.id, label: item.label, start: data.initial_cash + base, pnl: pnl, drawdown: drawdown};
    });
}

// 顶部指标卡片：按时间窗口内扣费+滑点口径的日收益率（当日盈亏 / 前一日资产）重新计算，算法与全区间绩效表一致。
function drawSummary(series) {
    const item = series[series.length - 1], count = range.end - range.start + 1;
    let mean = 0, variance = 0, held = 0, turnover = 0;
    const returns = [];
    for (let point = 1; point <= count; point++) {
        returns.push((item.pnl[point] - item.pnl[point - 1]) / (item.start + item.pnl[point - 1]));
        mean += returns[point - 1] / count;
        held += data.daily.exposure[range.start + point - 1] > 0 ? 1 : 0;
        turnover += data.daily.turnover[range.start + point - 1];
    }
    for (let point = 0; point < count; point++) variance += (returns[point] - mean) ** 2 / (count - 1);
    const annual = (1 + item.pnl[count] / item.start) ** (252 / count) - 1, worst = Math.min(...item.drawdown);
    const cards = [["年化收益", percent(annual)], ["Sharpe", variance > 0 ? (mean / Math.sqrt(variance) * Math.sqrt(252)).toFixed(2) : "—"], ["最大回撤", percent(worst)],
        ["Calmar", worst < 0 ? (annual / -worst).toFixed(2) : "—"], ["持仓日占比", percent(held / count)], ["日均换手", percent(turnover / count)]];
    document.getElementById("summary").innerHTML = cards.map(card => `<div class="stat"><span>${card[0]}</span><strong>${card[1]}</strong></div>`).join("");
    document.getElementById("summary-caption").textContent = `扣费+滑点口径 · ${formatDate(data.dates[range.start])} — ${formatDate(data.dates[range.end])} · ${count} 个交易日 · 区间盈亏 ${amount(item.pnl[count])} 元`;
}

// 画时间窗口内的曲线，index 为窗口内交易日的序号（起点为 -1），对应 data.dates 的下标 range.start + index 和曲线上的第 index + 1 个点。
function drawChart(id, key, height, format, series) {
    const element = document.getElementById(id);
    const width = element.clientWidth, count = range.end - range.start + 1;
    const values = series.flatMap(item => item[key]);
    // 纵轴刻度取 1 / 2 / 2.5 / 5 × 10^n 的整齐步长；回撤图顶端固定为 0；窗口内曲线全为 0 时用默认量程。
    const rough = (Math.max(...values) - Math.min(...values) || (key === "drawdown" ? 0.06 : 6000)) / 6, magnitude = 10 ** Math.floor(Math.log10(rough));
    const tick = [1, 2, 2.5, 5, 10].find(factor => factor * magnitude >= rough) * magnitude;
    let low = Math.floor(Math.min(...values) / tick) * tick, high = key === "drawdown" ? 0 : Math.ceil(Math.max(...values) / tick) * tick;
    if (low === high && key === "drawdown") low = -tick;
    if (low === high) high = low + tick;
    const x = index => margin.left + (index + 1) / count * (width - margin.left - margin.right);
    const y = value => margin.top + (high - value) / (high - low) * (height - margin.top - margin.bottom);
    const indexAt = event => Math.max(0, Math.min(count - 1, Math.round((event.clientX - element.getBoundingClientRect().left - margin.left) / (width - margin.left - margin.right) * count) - 1));
    let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    for (let step = Math.round(low / tick); step <= Math.round(high / tick); step++) {
        const value = step * tick;
        svg += `<line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${y(value)}" y2="${y(value)}"/><text x="${margin.left - 8}" y="${y(value) + 4}" text-anchor="end">${format(value)}</text>`;
    }
    // 横轴刻度：窗口超过 750 个交易日按年、超过 45 个按月、否则按日标注（unit 为日期前缀的长度）；与上一个标签重叠或超出右边界的跳过。
    const unit = count > 750 ? 4 : count > 45 ? 6 : 8, labelWidth = {4: 34, 6: 54, 8: 74}[unit];
    let labelled = -Infinity;
    for (let index = 0; index < count; index++) {
        const date = data.dates[range.start + index];
        if (index > 0 && date.slice(0, unit) === data.dates[range.start + index - 1].slice(0, unit)) continue;
        if (x(index) - labelled < labelWidth + 12 || x(index) + labelWidth > width) continue;
        labelled = x(index);
        svg += `<line class="grid" x1="${x(index)}" x2="${x(index)}" y1="${margin.top}" y2="${height - margin.bottom}"/><text x="${x(index) + 4}" y="${height - 8}">${unit === 8 ? formatDate(date) : formatDate(date).slice(0, unit === 4 ? 4 : 7)}</text>`;
    }
    series.forEach(item => {
        svg += `<path style="stroke: var(--${item.id})" d="M${item[key].map((value, point) => x(point - 1).toFixed(1) + "," + y(value).toFixed(1)).join("L")}"/>`;
    });
    if (selected >= range.start && selected <= range.end) svg += `<line class="selected" x1="${x(selected - range.start)}" x2="${x(selected - range.start)}" y1="${margin.top}" y2="${height - margin.bottom}"/>`;
    element.innerHTML = svg + `<line class="crosshair" y1="${margin.top}" y2="${height - margin.bottom}" visibility="hidden"/></svg>`;

    const crosshair = element.querySelector(".crosshair");
    element.onpointermove = event => {
        const index = indexAt(event);
        crosshair.setAttribute("x1", x(index));
        crosshair.setAttribute("x2", x(index));
        crosshair.setAttribute("visibility", "visible");
        tooltip.innerHTML = `<div class="tooltip-title">${formatDate(data.dates[range.start + index])}</div>` + series.map(item => `<div><i style="background: var(--${item.id})"></i>${item.label}<strong>${format(item[key][index + 1])}</strong></div>`).join("");
        tooltip.hidden = false;
        tooltip.style.left = Math.min(event.clientX + 16, window.innerWidth - tooltip.offsetWidth - 8) + "px";
        tooltip.style.top = Math.min(event.clientY + 16, window.innerHeight - tooltip.offsetHeight - 8) + "px";
    };
    element.onpointerleave = () => {
        tooltip.hidden = true;
        crosshair.setAttribute("visibility", "hidden");
    };
    element.onclick = event => showDay(range.start + indexAt(event));
}

// 时间窗口滑块：缩略图为全区间扣费+滑点口径的累计盈亏；拖动两端的手柄调整窗口的开头和结尾，拖动中间平移窗口，双击恢复全区间；窗口至少保留 5 个交易日。
function drawBrush() {
    const element = document.getElementById("brush");
    const width = element.clientWidth, height = 56, count = data.dates.length, minimum = Math.min(5, count - 1);
    const values = data.series[data.series.length - 1].pnl;
    const low = Math.min(...values), high = Math.max(...values);
    const x = index => margin.left + index / (count - 1) * (width - margin.left - margin.right);
    const y = value => 4 + (high - value) / (high - low || 1) * (height - 8);
    const indexAt = event => Math.max(0, Math.min(count - 1, Math.round((event.clientX - element.getBoundingClientRect().left - margin.left) / (width - margin.left - margin.right) * (count - 1))));
    let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><path d="M${values.map((value, index) => x(index).toFixed(1) + "," + y(value).toFixed(1)).join("L")}"/>`;
    svg += `<rect class="mask" x="${margin.left}" y="0" width="${x(range.start) - margin.left}" height="${height}"/><rect class="mask" x="${x(range.end)}" y="0" width="${width - margin.right - x(range.end)}" height="${height}"/>`;
    svg += `<rect class="window" x="${x(range.start)}" y="1" width="${x(range.end) - x(range.start)}" height="${height - 2}"/>`;
    svg += `<rect class="handle" x="${x(range.start) - 4}" y="${height / 2 - 12}" width="8" height="24" rx="3"/><rect class="handle" x="${x(range.end) - 4}" y="${height / 2 - 12}" width="8" height="24" rx="3"/>`;
    element.innerHTML = svg + "</svg>";
    document.getElementById("range-label").textContent = `时间窗口 ${formatDate(data.dates[range.start])} — ${formatDate(data.dates[range.end])} · ${range.end - range.start + 1} 个交易日 · 窗口内扣费+滑点盈亏 ${amount(values[range.end] - (range.start > 0 ? values[range.start - 1] : 0))} 元`;

    element.onpointerdown = event => {
        const position = event.clientX - element.getBoundingClientRect().left, index = indexAt(event);
        const startDistance = Math.abs(position - x(range.start)), endDistance = Math.abs(position - x(range.end));
        // 在窗口内部（离两个手柄都超过 10 像素）按下时平移窗口，否则拖动较近的手柄。
        if (Math.min(startDistance, endDistance) > 10 && index > range.start && index < range.end) drag = {mode: "move", offset: index - range.start};
        else drag = {mode: endDistance < startDistance || (endDistance === startDistance && position >= x(range.end)) ? "end" : "start"};
        element.setPointerCapture(event.pointerId);
        element.onpointermove(event);
    };
    element.onpointermove = event => {
        if (!drag) return;
        const index = indexAt(event), span = range.end - range.start;
        if (drag.mode === "start") range.start = Math.max(0, Math.min(index, range.end - minimum));
        if (drag.mode === "end") range.end = Math.min(count - 1, Math.max(index, range.start + minimum));
        if (drag.mode === "move") {
            range.start = Math.max(0, Math.min(index - drag.offset, count - 1 - span));
            range.end = range.start + span;
        }
        drawCharts();
    };
    element.onpointerup = () => drag = null;
    element.onpointercancel = () => drag = null;
    element.ondblclick = () => {
        range = {start: 0, end: count - 1};
        drawCharts();
    };
}

function drawCharts() {
    // 总览页隐藏时容器宽度为 0，切回总览时再绘制。
    if (document.getElementById("overview").hidden) return;
    const series = windowSeries();
    drawSummary(series);
    // 窗口内盈亏的绝对值都不到 10 万元时纵轴以元为单位，否则以万元为单位。
    const largest = Math.max(...series.flatMap(item => item.pnl).map(Math.abs));
    drawChart("pnl", "pnl", 360, largest < 100000 ? value => amount(value) : value => (value / 10000).toFixed(2) + "万", series);
    drawChart("drawdown", "drawdown", 220, value => (value * 100).toFixed(2) + "%", series);
    drawBrush();
}

// 当日明细：每行依次为 代码、名称、调仓前股数、股数、买入价、卖出价、个股收益、盈亏、费用、滑点、是否目标、涨停、跌停、停牌；盈亏、费用、滑点单位为元。
function showDay(index) {
    selected = index;
    const date = data.dates[index], rows = data.details[date] || [];
    const cards = [["理论盈亏（元）", signedAmount(data.daily.return_pnl[index])], ["扣费盈亏（元）", signedAmount(data.daily.net_return_pnl[index])], ["扣费+滑点盈亏（元）", signedAmount(data.daily.slippage_return_pnl[index])],
        ["换手", percent(data.daily.turnover[index])], ["仓位", percent(data.daily.exposure[index])], ["持仓数", rows.filter(row => row[3] > 0).length + " 只"]];
    document.getElementById("day-title").textContent = formatDate(date) + " 当日明细";
    document.getElementById("day-summary").innerHTML = cards.map(card => `<div class="stat"><span>${card[0]}</span><strong>${card[1]}</strong></div>`).join("");
    const body = rows.map(row => {
        const flags = [[row[11], "涨停"], [row[12], "跌停"], [row[13], "停牌"], [!row[10], "非目标"]].filter(flag => flag[0]).map(flag => flag[1]).join(" ");
        const price = value => value === null ? "—" : value.toFixed(2);
        return `<tr><td>${row[0]}</td><td>${row[1] || ""}</td><td>${amount(row[2], 2)}</td><td>${amount(row[3], 2)}</td><td>${price(row[4])}</td><td>${price(row[5])}</td><td>${signed(row[6])}</td><td>${signedAmount(row[7])}</td><td>${amount(row[8], 2)}</td><td>${amount(row[9], 2)}</td><td>${flags}</td></tr>`;
    }).join("");
    document.getElementById("day-table").innerHTML = rows.length ? `<table class="names"><thead><tr><th>代码</th><th>名称</th><th>调仓前股数</th><th>股数</th><th>买入价</th><th>卖出价</th><th>个股收益</th><th>盈亏</th><th>费用</th><th>滑点</th><th>标记</th></tr></thead><tbody>${body}</tbody></table>` : "<p>当日没有持仓，也没有目标股票。</p>";
    drawCharts();
}

document.querySelectorAll(".tabs button").forEach(button => {
    button.onclick = () => {
        document.querySelectorAll(".tabs button").forEach(item => item.classList.toggle("active", item === button));
        document.querySelectorAll(".tab").forEach(tab => tab.hidden = tab.id !== button.dataset.tab);
        drawCharts();
    };
});
showDay(selected);
window.addEventListener("resize", drawCharts);
"""

    def __init__(self, output_dir, strategy_name, fee_rate, min_fee, initial_cash):
        self.strategy_name = strategy_name
        self.output_dir = os.path.join(output_dir, strategy_name)
        self.fee_rate = fee_rate
        self.min_fee = min_fee
        self.initial_cash = initial_cash

    def run(self):
        """读取持仓清单、曲线、指标和明细四个 CSV，生成包含总览、分年和下一交易日持仓三个标签页的 HTML 报告。"""
        holdings = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Holdings.csv"), dtype={"code": str, "date": str, "trade_date": str})
        curve = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Curve.csv"), dtype={"date": str})
        metrics = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Metrics.csv"), dtype={"区间": str, "最大回撤开始": str, "最大回撤结束": str})
        details = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Details.csv"), dtype={"code": str, "date": str})

        # 绩效表：全区间三种口径对比，分年按年份排列、同一年内依次为各口径。
        formatters = {column: "{:.2%}".format for column in self.PERCENT_COLUMNS}
        formatters["区间盈亏"] = "{:,.0f}".format
        full_table = metrics.loc[metrics["区间"].eq("全区间")].drop(columns="区间").to_html(index=False, border=0, float_format="{:.2f}".format, formatters=formatters)
        yearly_table = metrics.loc[metrics["区间"].ne("全区间")].sort_values("区间", kind="stable").to_html(index=False, border=0, classes="yearly", float_format="{:.2f}".format, formatters=formatters)

        # 月度收益矩阵：扣费+滑点口径，年 × 月，正收益红色、负收益绿色。
        monthly = ((1 + curve["slippage_return"]).groupby(curve["date"].str[:6]).prod() - 1).to_frame("return")
        monthly["year"] = monthly.index.str[:4]
        monthly["month"] = monthly.index.str[4:].astype(int)
        monthly = monthly.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))
        monthly_rows = ""
        for year, row in monthly.iterrows():
            cells = ""
            for value in row:
                cells += '<td class="muted">—</td>' if pd.isna(value) else f'<td class="{"up" if value > 0 else "down" if value < 0 else ""}">{value:.2%}</td>'
            monthly_rows += f"<tr><td>{year}</td>{cells}</tr>"
        monthly_table = "<table><thead><tr><th>年份</th>" + "".join(f"<th>{month} 月</th>" for month in range(1, 13)) + f"</tr></thead><tbody>{monthly_rows}</tbody></table>"

        # 下一交易日持仓：清单中尚无收益的行（trade_date 晚于曲线最后一天，或最新信号日的交易日尚未确定）；按信号日逐批与上一批持仓比较，标出买入 / 保持 / 卖出，第一批与曲线最后一天的实际持仓比较。
        upcoming = holdings.loc[holdings["trade_date"].isna() | holdings["trade_date"].gt(curve["date"].iloc[-1])]
        factor_columns = [column for column in holdings.columns if column not in ["strategy", "date", "trade_date", "code", "name", "hold"]]
        previous = details.loc[details["date"].eq(curve["date"].iloc[-1]) & details["shares"].gt(0), ["code", "name"]]
        upcoming_table = "" if len(upcoming) else "<section><h2>下一交易日持仓</h2><p>持仓清单中没有尚无收益的交易日。</p></section>"
        for date, group in upcoming.groupby("date"):
            group = group.copy()
            group["动作"] = "不买入"
            group.loc[group["hold"], "动作"] = "买入"
            group.loc[group["hold"] & group["code"].isin(previous["code"]), "动作"] = "保持"
            group = pd.concat([group, previous.loc[~previous["code"].isin(group.loc[group["hold"], "code"])].assign(动作="卖出")], ignore_index=True)
            trade_date = f"{pd.Timestamp(group['trade_date'].iloc[0]):%Y-%m-%d}" if pd.notna(group["trade_date"].iloc[0]) else "下一交易日"
            note = "信号为空仓，目标股票均不买入。" if not group["hold"].eq(True).any() else f"持有 {int(group['hold'].eq(True).sum())} 只，10 点等权买入整 100 股。"
            table = group[["code", "name", *factor_columns, "hold", "动作"]].rename(columns={"code": "代码", "name": "名称", "hold": "持有"}).to_html(index=False, border=0, classes="names", na_rep="", float_format="{:.4g}".format)
            upcoming_table += f'<section><h2>交易日 {trade_date} · 信号日 {pd.Timestamp(date):%Y-%m-%d}</h2><p>{note}动作与上一批持仓比较；卖出行为上一批持有、本批不再持有的股票。</p><div class="content">{table}</div></section>'
            previous = group.loc[group["hold"].eq(True), ["code", "name"]]

        # 曲线、每日盈亏和逐股明细嵌入页面，由脚本绘制图表和当日明细，并按滑块选择的时间窗口计算顶部指标卡片（取最贴近实盘的扣费+滑点口径）。
        series = []
        for column in self.LABELS:
            series.append({"id": column, "label": self.LABELS[column], "pnl": (curve[f"{column}_asset"] - self.initial_cash).round(2).tolist()})
        daily = {column: curve[column].round(6).tolist() for column in ["return_pnl", "net_return_pnl", "slippage_return_pnl", "turnover", "exposure"]}
        details = details.round(6).astype(object)
        details = details.where(details.notna(), None)
        day_rows = {}
        for date, rows in details.groupby("date"):
            day_rows[date] = rows[self.DETAIL_COLUMNS].values.tolist()
        legend = "".join(f'<span><i style="background:var(--{column})"></i>{self.LABELS[column]}</span>' for column in self.LABELS)

        html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{self.strategy_name} 策略回测</title><style>{self.STYLE}</style></head><body><main>
<header><div class="eyebrow">TRADING STRATEGY / BACKTEST REPORT</div><h1>{self.strategy_name} · 策略回测报告</h1>
<p>{pd.Timestamp(curve["date"].iloc[0]):%Y-%m-%d} — {pd.Timestamp(curve["date"].iloc[-1]):%Y-%m-%d} · {len(curve):,} 个交易日 · 初始现金 {self.initial_cash:,.0f} 元 · 单边费率 {self.fee_rate:.2%}、每笔最低 {self.min_fee:g} 元 · 滑点：买入按 10 点最高价、卖出按 10 点最低价</p></header>
<nav class="tabs"><button class="active" data-tab="overview">总览</button><button data-tab="yearly">分年</button><button data-tab="upcoming">下一交易日持仓</button></nav>
<div id="overview" class="tab">
<div id="summary-caption" class="caption"></div><div id="summary" class="stats"></div>
<section><h2>累计盈亏</h2><p>从所选时间窗口首日调仓前的 0 开始累计，顶部指标也按该窗口重新计算；单位为元（达到 10 万元时以万元显示）；拖动下方滑块两端调整窗口的开头和结尾，拖动中间平移，双击恢复全区间；点击曲线查看当日逐股明细。</p><div class="legend">{legend}</div><div id="pnl" class="chart"></div><div id="brush" class="brush"></div><div id="range-label" class="caption"></div></section>
<section><h2>动态回撤</h2><p>当前资产 / 时间窗口内（含起点）的最高资产 − 1，随上方滑块选择的窗口重新计算；点击曲线查看当日逐股明细。</p><div class="legend">{legend}</div><div id="drawdown" class="chart"></div></section>
<section><h2 id="day-title">当日明细</h2><p>日期为交易日：当日 10 点按买入价调仓，持有到下一交易日 10 点按卖出价计算盈亏；价格为未复权的 10 点价格，卖出价已按复权因子折算到当日股数口径。盈亏 = 股数 × (卖出价 − 买入价)，费用 = 成交金额 × 单边费率（每笔不足最低佣金时按最低佣金收），滑点 = 成交价与 10 点价格的差额，单位为元。</p><div id="day-summary" class="stats"></div><div id="day-table" class="content"></div></section>
<section><h2>全区间绩效对比</h2><div class="content">{full_table}</div></section>
</div>
<div id="yearly" class="tab" hidden>
<section><h2>分年绩效</h2><p>每年单独计算，回撤每年重置；首尾年份可能不完整。</p><div class="content">{yearly_table}</div></section>
<section><h2>月度收益</h2><p>扣费+滑点口径，按月复利累计；红色为正收益、绿色为负收益。</p><div class="content">{monthly_table}</div></section>
</div>
<div id="upcoming" class="tab" hidden>
{upcoming_table}
</div>
<footer><b>计算口径</b><br>信号日收盘后确定持仓，下一交易日 10 点按未复权价调仓，持有到再下一交易日 10 点；尚无收益的下一交易日不参与评估。
调仓时把现金与可卖持仓的变现金额（按 10 点最低价、扣费后）等分给目标股票，每只按 10 点最高价加费用向下取整到 100 股，剩余部分持有现金，现金不会透支；持仓数量每日按复权因子的变化调整。
10 点价格达到涨停价不买入、达到跌停价不卖出，停牌或无行情的股票不能买卖。<br>
三种口径是同一组成交下的盈亏：理论按 10 点价格成交且不收费；扣费再减去费用，每只股票每天的买入或卖出算一笔，费用 = 成交金额 × 单边费率、不足最低佣金时按最低佣金收；扣费+滑点再减去买入按 10 点分钟线最高价、卖出按最低价成交相对 10 点价格的差额，即账户的真实盈亏。日收益率 = 当日盈亏 / 前一日资产。<br>
年化收益 = 净值^(252 / 交易日数) − 1；Sharpe、Sortino 不扣无风险利率，Sortino 的下行波动 = √(mean(min(日收益, 0)²))；日胜率只统计有持仓的交易日，月胜率只统计有持仓的月份；最长连亏天数为日收益连续为负的交易日数；日均换手为买卖双边。生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}。</footer>
</main><div id="tooltip" hidden></div>
<script id="report-data" type="application/json">{json.dumps({"dates": curve["date"].tolist(), "initial_cash": self.initial_cash, "series": series, "daily": daily, "details": day_rows}, ensure_ascii=False)}</script><script>{self.SCRIPT}</script></body></html>'''
        with open(os.path.join(self.output_dir, f"{self.strategy_name}Evaluation.html"), "w", encoding="utf-8") as file:
            file.write(html)
