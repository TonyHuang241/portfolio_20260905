import json
import os

import pandas as pd


class StrategyReport:
    LABELS = {"return": "理论", "net_return": "扣费", "slippage_return": "扣费+滑点"}
    PERCENT_COLUMNS = ["区间收益", "年化收益", "年化波动", "最大回撤", "日胜率", "月胜率", "持仓日占比", "日均换手", "最大单日收益", "最大单日亏损"]
    DETAIL_COLUMNS = ["code", "name", "previous_weight", "weight", "buy_price", "sell_price", "stock_return", "contribution", "fee", "slippage", "target", "up_limit", "down_limit", "suspended"]
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
    # 报告脚本：标签页切换；按容器宽度绘制 SVG 折线图，悬停时显示十字线和各口径数值，点击某一天显示当日逐股明细，窗口缩放时重绘。
    SCRIPT = """
const data = JSON.parse(document.getElementById("report-data").textContent);
const tooltip = document.getElementById("tooltip");
const margin = {left: 56, right: 16, top: 12, bottom: 28};
let selected = data.dates.length - 1;
const formatDate = date => date.slice(0, 4) + "-" + date.slice(4, 6) + "-" + date.slice(6);
const percent = (value, digits = 2) => value === null ? "—" : (value * 100).toFixed(digits) + "%";
const signed = (value, digits = 2) => value === null ? "—" : `<span class="${value > 0 ? "up" : value < 0 ? "down" : ""}">${percent(value, digits)}</span>`;

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
    const indexAt = event => Math.max(0, Math.min(count - 1, Math.round((event.clientX - element.getBoundingClientRect().left - margin.left) / (width - margin.left - margin.right) * (count - 1))));
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
    svg += `<line class="selected" x1="${x(selected)}" x2="${x(selected)}" y1="${margin.top}" y2="${height - margin.bottom}"/>`;
    element.innerHTML = svg + `<line class="crosshair" y1="${margin.top}" y2="${height - margin.bottom}" visibility="hidden"/></svg>`;

    const crosshair = element.querySelector(".crosshair");
    element.onpointermove = event => {
        const index = indexAt(event);
        crosshair.setAttribute("x1", x(index));
        crosshair.setAttribute("x2", x(index));
        crosshair.setAttribute("visibility", "visible");
        tooltip.innerHTML = `<div class="tooltip-title">${formatDate(data.dates[index])}</div>` + data.series.map(item => `<div><i style="background: var(--${item.id})"></i>${item.label}<strong>${format(item[key][index])}</strong></div>`).join("");
        tooltip.hidden = false;
        tooltip.style.left = Math.min(event.clientX + 16, window.innerWidth - tooltip.offsetWidth - 8) + "px";
        tooltip.style.top = Math.min(event.clientY + 16, window.innerHeight - tooltip.offsetHeight - 8) + "px";
    };
    element.onpointerleave = () => {
        tooltip.hidden = true;
        crosshair.setAttribute("visibility", "hidden");
    };
    element.onclick = event => showDay(indexAt(event));
}

function drawCharts() {
    // 总览页隐藏时容器宽度为 0，切回总览时再绘制。
    if (document.getElementById("overview").hidden) return;
    drawChart("nav", "nav", 360, value => value.toFixed(2));
    drawChart("drawdown", "drawdown", 220, value => (value * 100).toFixed(1) + "%");
}

// 当日明细：每行依次为 代码、名称、前日权重、权重、买入价、卖出价、个股收益、贡献、费用、滑点、是否目标、涨停、跌停、停牌。
function showDay(index) {
    selected = index;
    const date = data.dates[index], rows = data.details[date] || [];
    const cards = [["理论收益", signed(data.daily.return[index])], ["扣费收益", signed(data.daily.net_return[index])], ["扣费+滑点收益", signed(data.daily.slippage_return[index])],
        ["换手", percent(data.daily.turnover[index])], ["仓位", percent(data.daily.exposure[index])], ["持仓数", rows.filter(row => row[3] > 0).length + " 只"]];
    document.getElementById("day-title").textContent = formatDate(date) + " 当日明细";
    document.getElementById("day-summary").innerHTML = cards.map(card => `<div class="stat"><span>${card[0]}</span><strong>${card[1]}</strong></div>`).join("");
    const body = rows.map(row => {
        const flags = [[row[11], "涨停"], [row[12], "跌停"], [row[13], "停牌"], [!row[10], "非目标"]].filter(flag => flag[0]).map(flag => flag[1]).join(" ");
        const price = value => value === null ? "—" : value.toFixed(2);
        return `<tr><td>${row[0]}</td><td>${row[1] || ""}</td><td>${percent(row[2])}</td><td>${percent(row[3])}</td><td>${price(row[4])}</td><td>${price(row[5])}</td><td>${signed(row[6])}</td><td>${signed(row[7], 3)}</td><td>${percent(row[8], 3)}</td><td>${percent(row[9], 3)}</td><td>${flags}</td></tr>`;
    }).join("");
    document.getElementById("day-table").innerHTML = rows.length ? `<table class="names"><thead><tr><th>代码</th><th>名称</th><th>前日权重</th><th>权重</th><th>买入价</th><th>卖出价</th><th>个股收益</th><th>贡献</th><th>费用</th><th>滑点</th><th>标记</th></tr></thead><tbody>${body}</tbody></table>` : "<p>当日没有持仓，也没有目标股票。</p>";
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

    def __init__(self, output_dir, strategy_name, fee_rate):
        self.strategy_name = strategy_name
        self.output_dir = os.path.join(output_dir, strategy_name)
        self.fee_rate = fee_rate

    def run(self):
        """读取持仓清单、曲线、指标和明细四个 CSV，生成包含总览、分年和下一交易日持仓三个标签页的 HTML 报告。"""
        holdings = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Holdings.csv"), dtype={"code": str, "date": str, "trade_date": str})
        curve = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Curve.csv"), dtype={"date": str})
        metrics = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Metrics.csv"), dtype={"区间": str, "最大回撤开始": str, "最大回撤结束": str})
        details = pd.read_csv(os.path.join(self.output_dir, f"{self.strategy_name}Details.csv"), dtype={"code": str, "date": str})

        # 绩效表：全区间三种口径对比，分年按年份排列、同一年内依次为各口径。
        formatters = {column: "{:.2%}".format for column in self.PERCENT_COLUMNS}
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
        previous = details.loc[details["date"].eq(curve["date"].iloc[-1]) & details["weight"].gt(0), ["code", "name"]]
        upcoming_table = "" if len(upcoming) else "<section><h2>下一交易日持仓</h2><p>持仓清单中没有尚无收益的交易日。</p></section>"
        for date, group in upcoming.groupby("date"):
            group = group.copy()
            group["动作"] = "不买入"
            group.loc[group["hold"], "动作"] = "买入"
            group.loc[group["hold"] & group["code"].isin(previous["code"]), "动作"] = "保持"
            group = pd.concat([group, previous.loc[~previous["code"].isin(group.loc[group["hold"], "code"])].assign(动作="卖出")], ignore_index=True)
            trade_date = f"{pd.Timestamp(group['trade_date'].iloc[0]):%Y-%m-%d}" if pd.notna(group["trade_date"].iloc[0]) else "下一交易日"
            note = "信号为空仓，目标股票均不买入。" if not group["hold"].eq(True).any() else f"持有 {int(group['hold'].eq(True).sum())} 只，10 点等权买入。"
            table = group[["code", "name", *factor_columns, "hold", "动作"]].rename(columns={"code": "代码", "name": "名称", "hold": "持有"}).to_html(index=False, border=0, classes="names", na_rep="", float_format="{:.4g}".format)
            upcoming_table += f'<section><h2>交易日 {trade_date} · 信号日 {pd.Timestamp(date):%Y-%m-%d}</h2><p>{note}动作与上一批持仓比较；卖出行为上一批持有、本批不再持有的股票。</p><div class="content">{table}</div></section>'
            previous = group.loc[group["hold"].eq(True), ["code", "name"]]

        # 顶部指标卡片取最贴近实盘的扣费+滑点口径；曲线、每日收益和逐股明细嵌入页面，由脚本绘制图表和当日明细。
        summary = metrics.loc[metrics["区间"].eq("全区间") & metrics["口径"].eq(self.LABELS["slippage_return"])].iloc[0]
        cards = "".join(f'<div class="stat"><span>{name}</span><strong>{summary[name]:.2%}</strong></div>' if name in self.PERCENT_COLUMNS else f'<div class="stat"><span>{name}</span><strong>{summary[name]:.2f}</strong></div>' for name in ["年化收益", "Sharpe", "最大回撤", "Calmar", "持仓日占比", "日均换手"])
        series = []
        for column in self.LABELS:
            series.append({"id": column, "label": self.LABELS[column], "nav": curve[f"{column}_nav"].round(5).tolist(), "drawdown": curve[f"{column}_drawdown"].round(5).tolist()})
        daily = {column: curve[column].round(6).tolist() for column in ["return", "net_return", "slippage_return", "turnover", "exposure"]}
        details = details.round(6).astype(object)
        details = details.where(details.notna(), None)
        day_rows = {}
        for date, rows in details.groupby("date"):
            day_rows[date] = rows[self.DETAIL_COLUMNS].values.tolist()
        legend = "".join(f'<span><i style="background:var(--{column})"></i>{self.LABELS[column]}</span>' for column in self.LABELS)

        html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{self.strategy_name} 策略回测</title><style>{self.STYLE}</style></head><body><main>
<header><div class="eyebrow">TRADING STRATEGY / BACKTEST REPORT</div><h1>{self.strategy_name} · 策略回测报告</h1>
<p>{pd.Timestamp(curve["date"].iloc[0]):%Y-%m-%d} — {pd.Timestamp(curve["date"].iloc[-1]):%Y-%m-%d} · {len(curve):,} 个交易日 · 单边费率 {self.fee_rate:.2%} · 滑点：买入按 10 点最高价、卖出按 10 点最低价</p></header>
<nav class="tabs"><button class="active" data-tab="overview">总览</button><button data-tab="yearly">分年</button><button data-tab="upcoming">下一交易日持仓</button></nav>
<div id="overview" class="tab">
<div class="caption">扣费+滑点口径 · 全区间</div><div class="stats">{cards}</div>
<section><h2>累积净值</h2><p>按日复利累计，起点净值为 1；点击曲线查看当日逐股明细。</p><div class="legend">{legend}</div><div id="nav" class="chart"></div></section>
<section><h2>动态回撤</h2><p>当前净值 / 历史最高净值 − 1，包含初始净值 1；点击曲线查看当日逐股明细。</p><div class="legend">{legend}</div><div id="drawdown" class="chart"></div></section>
<section><h2 id="day-title">当日明细</h2><p>日期为交易日：当日 10 点按买入价调仓，持有到下一交易日 10 点按卖出价计算收益；价格为后复权价。贡献 = 权重 × 个股收益，费用 = 权重变化 × 单边费率。</p><div id="day-summary" class="stats"></div><div id="day-table" class="content"></div></section>
<section><h2>全区间绩效对比</h2><div class="content">{full_table}</div></section>
</div>
<div id="yearly" class="tab" hidden>
<section><h2>分年绩效</h2><p>每年单独计算，回撤每年重置；首尾年份可能不完整。</p><div class="content">{yearly_table}</div></section>
<section><h2>月度收益</h2><p>扣费+滑点口径，按月复利累计；红色为正收益、绿色为负收益。</p><div class="content">{monthly_table}</div></section>
</div>
<div id="upcoming" class="tab" hidden>
{upcoming_table}
</div>
<footer><b>计算口径</b><br>信号日收盘后确定持仓，下一交易日 10 点等权调仓，持有到再下一交易日 10 点；尚无收益的下一交易日不参与评估。
10 点价格达到涨停价不买入、达到跌停价不卖出，买不进的部分持有现金；停牌或无价格的股票当日收益记 0。<br>
扣费 = 买卖双边换手 × 单边费率；滑点在扣费基础上，买入按 10 点分钟线最高价、卖出按最低价成交，扣除买入权重 × (1 − 收盘价 / 最高价) 与卖出权重 × (1 − 最低价 / 收盘价)。<br>
年化收益 = 净值^(252 / 交易日数) − 1；Sharpe、Sortino 不扣无风险利率，Sortino 的下行波动 = √(mean(min(日收益, 0)²))；日胜率只统计有持仓的交易日，月胜率只统计有持仓的月份；最长连亏天数为日收益连续为负的交易日数；日均换手为买卖双边。生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}。</footer>
</main><div id="tooltip" hidden></div>
<script id="report-data" type="application/json">{json.dumps({"dates": curve["date"].tolist(), "series": series, "daily": daily, "details": day_rows}, ensure_ascii=False)}</script><script>{self.SCRIPT}</script></body></html>'''
        with open(os.path.join(self.output_dir, f"{self.strategy_name}Evaluation.html"), "w", encoding="utf-8") as file:
            file.write(html)
