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

    def plot_results_html(self):
        """Write an offline HTML report and return its absolute Path."""
        data = self.factor_results
        direction = f"G{self.preferred_group[-1]} − G{self.short_group[-1]}"
        group_rows = [(f"G{group[-1]}" + (" · IC方向优选组" if group == self.preferred_group else ""), self.group_returns[group], self.group_counts[f"{group}_count"]) for group in self.GROUPS]
        group_rows.append((f"Long-short（{direction}）", self.long_short, self.group_counts[f"{self.preferred_group}_count"] + self.group_counts[f"{self.short_group}_count"]))
        average_turnovers = data.reindex(index=self.group_returns.index, columns=[f"{group}_turnover" for group in self.GROUPS]).mean().tolist() + [np.nan]
        yearly_rows = [(str(year), returns, self.group_counts[f"{self.preferred_group}_count"]) for year, returns in self.group_returns[self.preferred_group].groupby(self.group_returns.index.year)]
        group_curves = {f"G{group[-1]}": (1 + self.group_returns[group]).cumprod() - 1 for group in self.GROUPS}
        sections = [
            ("01", "IC / RankIC 统计", "衡量因子值与下一期收益的相关性，IR 保留方向符号。", self._ic_table()),
            ("02", "五组累计收益", "G1 为因子值最低组，G5 为因子值最高组；各组收益按日复利累计。", self._chart(group_curves, "Cumulative return", percent=True, zero=True)),
            ("03", "五组年化收益 · 单调性", "按因子值从低到高排列 G1 → G5，颜色与累计收益曲线一致。年化收益口径与分组绩效表一致；正向因子观察是否递增，负向因子观察是否递减。", self._annual_return_chart()),
            ("04", "Long-short 累计收益", f"当前方向：{direction}。每日多头收益减空头收益，再复利累计。", self._chart({"Long-short": (1 + self.long_short).cumprod() - 1}, "Cumulative return", percent=True, zero=True)),
            ("05", "IC 时间序列", "每日截面 Pearson 相关系数。", self._chart({"IC": data["IC"]}, "IC", zero=True)),
            ("06", "RankIC 时间序列", "每日截面秩相关系数。", self._chart({"RankIC": data["rankIC"]}, "RankIC", zero=True)),
            ("07", "分组绩效", "各组与多空组合使用相同的有效交易日，最大回撤包含初始净值 1。日均股票数量为对应有效收益日内参与收益计算的股票数量均值，多空组合为多头与空头数量之和；缺失数量不参与均值计算，全部缺失时显示「—」。平均换手率为有效收益日内单边换手率的均值：按当日因子分组等权目标持仓与前一交易日目标持仓的权重差绝对值之和除以 2，不计价格漂移；首日及缺失值不参与均值，多空组合不展示。", self._performance_table(group_rows, "组合", average_turnovers)),
            ("08", f"IC方向优选组 G{self.preferred_group[-1]} · 分年绩效", "按 IC 方向选组，未按事后收益排名选组。区间收益为该年实际覆盖日期的复利收益，首尾年份可能不完整；回撤每年重置。日均股票数量按该年有效收益日统计。", self._performance_table(yearly_rows, "年份")),
        ]
        cards = [("IC 均值", self._format(self.ic_mean)), ("RankIC 均值", self._format(data["rankIC"].mean())),
                 ("ICIR", self._format(self.ic_mean / data["IC"].std() if data["IC"].std() > 0 else np.nan)),
                 ("多空方向", direction)]
        card_html = "".join(f'<div class="stat"><span>{label}</span><strong>{value}</strong></div>' for label, value in cards)
        section_html = "".join(f'<section id="section-{number}"><h2><span>{number}</span>{title}</h2><p>{description}</p><div class="content">{body}</div></section>' for number, title, description, body in sections)
        skipped = len(data) - len(self.group_returns)
        title = escape(self.factor_name)
        html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 因子评估报告</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif; color: #172b4d; background: #f2f5fa; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; }} main {{ max-width: 1280px; margin: auto; padding: 40px 28px; }}
header {{ padding: 36px; background: linear-gradient(120deg, #172c52, #28578c); color: white; border-radius: 20px; }}
.eyebrow {{ color: #9edcfa; font-size: 12px; letter-spacing: 3px; }} h1 {{ font-size: 32px; margin: 16px 0; overflow-wrap: anywhere; }}
header p {{ color: #d2dfef; line-height: 1.8; }} .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin: 24px 0; }}
.stat, section {{ background: white; border: 1px solid #e3e9f2; border-radius: 16px; box-shadow: 0 5px 20px #23395605; }}
.stat {{ padding: 24px; }} .stat span {{ display: block; color: #718096; font-size: 13px; }} .stat strong {{ display: block; margin-top: 12px; font-size: 28px; color: #2563a4; }}
section {{ padding: 26px; margin-bottom: 22px; }} h2 {{ margin: 0; font-size: 20px; }} h2 span {{ color: #94a9c6; margin-right: 14px; font-size: 14px; }}
section p, footer {{ font-size: 13px; color: #6b7c94; line-height: 1.9; }} .content {{ overflow-x: auto; }} svg {{ display: block; width: 100%; height: auto; min-width: 560px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }} th {{ background: #f4f7fb; color: #61718b; font-weight: 500; }}
th, td {{ padding: 15px 13px; text-align: right; border-bottom: 1px solid #edf1f6; }} th:first-child, td:first-child {{ text-align: left; }} tbody tr:hover {{ background: #f4f8ff; }}
footer {{ padding: 4px 12px 20px; }} @media(max-width: 700px) {{ main {{ padding: 16px; }} header, section {{ padding: 20px; }} .stats {{ grid-template-columns: repeat(2, 1fr); gap: 10px; }} .stat {{ padding: 18px; }} h1 {{ font-size: 24px; }} .stat strong {{ font-size: 23px; }} }}
@media print {{ main {{ padding: 0; }} section {{ break-inside: avoid; }} .content {{ overflow: visible; }} svg {{ min-width: 0; }} }}
</style></head><body><main>
<header><div class="eyebrow">FACTOR RESEARCH / PERFORMANCE REPORT</div><h1>{title} · 因子评估报告</h1>
<p>{data.index.min():%Y-%m-%d} — {data.index.max():%Y-%m-%d} · {len(data):,} 个评估日 · {len(self.group_returns):,} 个完整收益日</p></header>
<div class="stats">{card_html}</div>{section_html}
<footer><b>计算口径</b><br>
日收益使用小数；年化交易日数 {self.periods_per_year:g}，年化无风险利率 {self.risk_free_rate:.2%}。
年化收益 = ∏(1 + 日收益)^(年化交易日数 / 有效交易日数) − 1；年化波动率 = 日收益样本标准差 × √年化交易日数。
Sharpe = (平均日收益 − 等效日无风险利率) / 日收益样本标准差 × √年化交易日数。
ICIR / RankICIR = 均值 / 样本标准差；年化 IR 再乘 √年化交易日数。标准差为零或样本不足时显示「—」。<br>
IC 均值 &gt; 0 时做多 G5、做空 G1，否则做多 G1、做空 G5。方向由整个展示区间确定，属于事后分析。
多空按多头 100%、空头 100% 的收益差计算，未除以 2；不计手续费、滑点和融券成本。
缺失 IC / RankIC 各自剔除；任一组缺失收益的日期从全部收益统计中共同剔除（共 {skipped} 日），不填充为零。<br>
日期沿用评估结果的收益起始日标签，跨年收益按该标签归属年份。分年收益以实际样本区间为准，年化指标按有效日数折算。
图表已嵌入，可离线查看。生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}。
</footer></main></body></html>'''
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
