import io
from html import escape
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from factors.evaluation.results_visualization.results_visualizer import ResultsVisualizer


class FactorCorrelationAnalyzer:
    """多因子截面分析：计算全部原始因子的相关性矩阵，以及基础因子对其他因子（含滚动均值列）的每日解释度 R²，与批量评估指标一起输出 HTML 报告。"""
    # R² 折线颜色：原始因子为灰色，滚动均值列按窗口由短到长从浅到深。
    LINE_COLORS = ["#94a3b8", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

    def __init__(self, config):
        self.factor_dir = Path(config["factor_data_dir"])
        self.monthly_mkcap_path = Path(config["stock_minutes_dir"]).parent / "fundamentals" / "mkcap_monthly.parquet"
        self.output_dir = Path(config["visualization_output_dir"]) / "multi_factor_evaluation"
        self.start_date = config["start_date"]
        self.stock_pool = config["stock_pool"]
        self.stock_board = config["stock_board"]
        self.base_factor_list = config["base_factor_list"]

        # 因子文件名即原始因子列名；相关性矩阵只用原始值，基础因子排在最前面。
        missing_factors = [factor_name for factor_name in self.base_factor_list if not (self.factor_dir / f"{factor_name}.parquet").is_file()]
        if missing_factors:
            raise FileNotFoundError(f"Base factor files not found in {self.factor_dir}: {missing_factors}")
        self.explained_factor_list = [path.stem for path in sorted(self.factor_dir.glob("*.parquet")) if path.stem not in self.base_factor_list]
        self.factor_list = self.base_factor_list + self.explained_factor_list

        # R² 对非基础因子文件中的全部列计算（原始值和滚动均值列），按因子记录列名，作图时每个因子一张图。
        self.explained_columns = {}
        self.explained_column_list = []
        for factor_name in self.explained_factor_list:
            self.explained_columns[factor_name] = [column for column in pq.read_schema(self.factor_dir / f"{factor_name}.parquet").names if column not in ("code", "date")]
            self.explained_column_list += self.explained_columns[factor_name]

    def _load_factors(self):
        """读取基础因子原始值和其他因子的全部列，按单因子评估的板块和小市值股票池筛选，再逐日截面标准化。"""
        factor = pd.concat([pd.read_parquet(self.factor_dir / f"{factor_name}.parquet", columns=["code", "date"] + self.explained_columns.get(factor_name, [factor_name]), filters=[("date", ">=", self.start_date)]).set_index(["code", "date"]) for factor_name in self.factor_list], axis=1)
        factor = factor.reset_index()
        if self.stock_board == "main board":
            factor = factor.loc[factor["code"].str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))]

        # 与单因子评估一致：按因子日期所在月份取上月末市值，每日保留市值最小的 stock_pool 只股票。
        monthly_mkcap = pd.read_parquet(self.monthly_mkcap_path, columns=["code", "date", "mkcap"])
        monthly_mkcap = monthly_mkcap.rename(columns={"date": "month"})
        factor["month"] = factor["date"].str[:6] + "01"
        factor = factor.merge(monthly_mkcap, on=["code", "month"])
        factor = factor.dropna(subset=["mkcap"])
        factor = factor.sort_values("mkcap", kind="stable")
        factor = factor.groupby("date", sort=False).head(self.stock_pool)
        factor = factor.sort_values(["date", "code"]).reset_index(drop=True)

        # 无穷值视为缺失；逐日减截面均值、除以截面标准差，标准差为 0 时结果为空。
        columns = self.base_factor_list + self.explained_column_list
        factor[columns] = factor[columns].replace([np.inf, -np.inf], np.nan)
        factor[columns] = (factor[columns] - factor.groupby("date")[columns].transform("mean")) / factor.groupby("date")[columns].transform("std").replace(0, np.nan)
        return factor

    def _correlation_matrix(self, factor):
        """逐日计算两两因子的截面 Pearson 相关系数，再对全部日期取均值。"""
        correlation = factor.groupby("date")[self.factor_list].corr()
        return correlation.groupby(level=1).mean().reindex(index=self.factor_list, columns=self.factor_list)

    def _regression_r2(self, factor):
        """每日将每个非基础因子列（含滚动均值列）对基础因子原始值做带截距的截面回归，返回日期 × 因子列的 R²。"""
        r2 = pd.DataFrame(index=pd.Index(sorted(factor["date"].unique()), name="date"), columns=self.explained_column_list, dtype=float)
        for date, daily in tqdm(factor.groupby("date"), desc="Explaining factors", unit="day"):
            for factor_name in self.explained_column_list:
                sample = daily[self.base_factor_list + [factor_name]].dropna()
                # 有效股票数需多于回归参数个数（基础因子 + 截距），否则当日 R² 为空。
                if len(sample) <= len(self.base_factor_list) + 1:
                    continue
                x = np.column_stack([np.ones(len(sample)), sample[self.base_factor_list]])
                sample["residual"] = sample[factor_name] - x @ np.linalg.lstsq(x, sample[factor_name], rcond=None)[0]
                r2.loc[date, factor_name] = 1 - (sample["residual"] ** 2).sum() / ((sample[factor_name] - sample[factor_name].mean()) ** 2).sum()
        return r2

    @staticmethod
    def _chart(values):
        """在一张图中绘制单个因子及其滚动均值列的每日 R² 折线，首列为原始因子，图例附全区间平均 R²。"""
        figure = Figure(figsize=(12, 3.6), layout="constrained", facecolor="white")
        axis = figure.subplots()
        values = values.set_axis(pd.to_datetime(values.index, format="%Y%m%d"))
        for column, color in zip(values.columns, FactorCorrelationAnalyzer.LINE_COLORS):
            label = "raw" if column == values.columns[0] else column.removeprefix(f"{values.columns[0]}_")
            axis.plot(values.index, values[column], label=f"{label} {ResultsVisualizer._format(values[column].mean())}", color=color, linewidth=1.2, alpha=0.9)
        axis.legend(loc="lower left", bbox_to_anchor=(0, 1), frameon=False, ncol=len(values.columns), fontsize=9)
        axis.set_ylim(bottom=0)
        axis.set_ylabel("R²", color="#64748b", fontsize=10)
        axis.grid(axis="y", color="#e8edf5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#dce3ed")
        axis.tick_params(colors="#64748b", labelsize=9)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        buffer = io.StringIO()
        FigureCanvasSVG(figure).print_svg(buffer)
        svg = buffer.getvalue()
        return svg[svg.index("<svg"):]

    def plot_results_html(self, summary):
        """计算相关性矩阵和每日 R²，与批量评估指标表一起写入 visualization_output_dir/multi_factor_evaluation。"""
        factor = self._load_factors()
        correlation = self._correlation_matrix(factor)
        r2 = self._regression_r2(factor)

        # 指标表沿用 factor_comparison.csv 的内容，追加平均 R²；基础因子的滚动均值列不参与回归，显示为「—」。
        table = summary.copy()
        table["平均 R²"] = table["因子"].map(r2.mean())
        percent_columns = ["多空年化收益", "多头年化收益", "多头换手率"]
        for column in ["IC", "RankIC", "ICIR", "平均 R²"] + percent_columns:
            table[column] = [ResultsVisualizer._format(value, column in percent_columns) for value in table[column]]
        table.loc[table["因子"].isin(self.base_factor_list), "平均 R²"] = "基础因子"

        # 相关性矩阵：正相关为红、负相关为蓝，颜色深浅与相关系数绝对值成正比；对角线恒为 1，不着色。
        correlation_header = "".join(f'<th class="vertical"><span>{escape(factor_name)}</span></th>' for factor_name in self.factor_list)
        correlation_rows = ""
        for row_name in self.factor_list:
            correlation_rows += f"<tr><th>{escape(row_name)}</th>"
            for column_name in self.factor_list:
                value = correlation.loc[row_name, column_name]
                alpha = 0 if row_name == column_name or pd.isna(value) else abs(value)
                color = "199, 72, 69" if value > 0 else "28, 92, 171"
                correlation_rows += f'<td style="background: rgba({color}, {alpha:.3f}); color: {"#ffffff" if alpha > 0.65 else "#172b4d"}" title="{escape(row_name)} × {escape(column_name)}">{ResultsVisualizer._format(value)}</td>'
            correlation_rows += "</tr>"

        r2_charts = ""
        for factor_name in self.explained_factor_list:
            chart = self._chart(r2[self.explained_columns[factor_name]]) if r2[self.explained_columns[factor_name]].notna().any().any() else '<p class="empty">没有有效的 R²。</p>'
            r2_charts += f'<h3>{escape(factor_name)} · 平均 R² {ResultsVisualizer._format(r2[factor_name].mean())}</h3><div class="content">{chart}</div>'

        base_factors = escape("、".join(self.base_factor_list))
        html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>多因子评估报告</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif; color: #172b4d; background: #f2f5fa; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; background: #f2f5fa; }} main {{ max-width: 1280px; margin: auto; padding: 24px 28px; }}
header {{ padding: 32px; margin-bottom: 22px; background: linear-gradient(120deg, #172c52, #28578c); color: white; border-radius: 20px; }}
.eyebrow {{ color: #9edcfa; font-size: 12px; letter-spacing: 3px; }} h1 {{ font-size: 30px; margin: 16px 0; }} header p {{ color: #d2dfef; line-height: 1.8; }}
section {{ padding: 26px; margin-bottom: 22px; background: white; border: 1px solid #e3e9f2; border-radius: 16px; box-shadow: 0 5px 20px #23395605; }}
h2 {{ margin: 0; font-size: 20px; }} h3 {{ margin: 24px 0 6px; font-size: 16px; }}
section p, footer {{ font-size: 13px; color: #6b7c94; line-height: 1.9; }} .content {{ overflow-x: auto; }} .empty {{ text-align: center; padding: 40px; }}
svg {{ display: block; width: 100%; height: auto; min-width: 560px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }} th {{ background: #f4f7fb; color: #61718b; font-weight: 500; }}
th, td {{ padding: 12px 13px; text-align: right; border-bottom: 1px solid #edf1f6; }} th:first-child, td:first-child {{ text-align: left; }} tbody tr:hover {{ background: #f4f8ff; }}
table.correlation {{ width: auto; }} table.correlation td {{ min-width: 64px; text-align: center; border: 1px solid white; }}
th.vertical {{ vertical-align: bottom; text-align: center; }} th.vertical span {{ writing-mode: vertical-rl; transform: rotate(180deg); }}
footer {{ padding: 4px 12px 20px; }} @media(max-width: 700px) {{ main {{ padding: 12px; }} header, section {{ padding: 18px; }} h1 {{ font-size: 24px; }} }}
</style></head><body><main>
<header><div class="eyebrow">FACTOR RESEARCH / MULTI-FACTOR REPORT</div><h1>多因子评估报告</h1>
<p>{r2.index[0]} — {r2.index[-1]} · {len(r2):,} 个交易日 · {len(self.factor_list)} 个原始因子 · 基础因子：{base_factors}</p></header>
<section><h2>因子评估指标对比</h2><p>IC 等指标与 factor_comparison.csv 一致，收益和换手率口径见各因子报告。平均 R² 为基础因子（{base_factors}）对该因子每日截面解释度的均值，原始因子和滚动均值列分别计算；基础因子的滚动均值列显示为「—」。</p><div class="content">{table.to_html(index=False, border=0, classes="metrics")}</div></section>
<section><h2>因子相关性矩阵</h2><p>全部原始因子的每日截面 Pearson 相关系数的时间均值，不含滚动均值列；前 {len(self.base_factor_list)} 个为基础因子。红色为正相关、蓝色为负相关，颜色越深相关性越强。</p><div class="content"><table class="correlation"><thead><tr><th></th>{correlation_header}</tr></thead><tbody>{correlation_rows}</tbody></table></div></section>
<section><h2>基础因子解释度 · 每日 R²</h2><p>每日将因子及其滚动均值列分别对 {base_factors} 做带截距的截面回归得到的 R²，每个因子一张图：灰线为原始因子，蓝线为滚动均值列，窗口越长颜色越深，图例数字为全区间平均 R²。</p>{r2_charts}</section>
<footer><b>计算口径</b><br>
样本为因子目录中全部因子文件的原始因子列，自 {self.start_date} 起；板块范围和股票池与单因子评估相同（{escape(str(self.stock_board))}，每日按上月末市值保留最小的 {self.stock_pool} 只股票），使用当日因子值，不做滞后。<br>
各因子剔除无穷值后逐日截面标准化（减均值、除以样本标准差），未去极值。相关系数逐日按两两均有效的股票计算，再对日期取算术平均。<br>
R² 每日以该因子列和全部基础因子均有效的股票为样本做 OLS，滚动均值列同样对基础因子的原始值回归，有效股票数不多于参数个数的日期记为空；平均 R² 为每日 R² 的算术平均。<br>
生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}。
</footer></main></body></html>'''
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "multi_factor_evaluation_report.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()
