import io
from html import escape
from multiprocessing import Pool
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
        self.output_dir = Path(config["visualization_output_dir"])
        self.start_date = config["start_date"]
        self.stock_pool = config["stock_pool"]
        self.stock_board = config["stock_board"]
        self.base_factor_list = config["base_factor_list"]
        self.processes = config["processes"]

        # 因子文件名即原始因子列名；相关性矩阵只用原始值，基础因子排在最前面。
        missing_factors = [factor_name for factor_name in self.base_factor_list if not (self.factor_dir / f"{factor_name}.parquet").is_file()]
        if missing_factors:
            raise FileNotFoundError(f"Base factor files not found in {self.factor_dir}: {missing_factors}")
        self.explained_factor_list = [path.stem for path in sorted(self.factor_dir.glob("*.parquet")) if path.stem not in self.base_factor_list]
        self.factor_list = self.base_factor_list + self.explained_factor_list

        # 按因子记录文件中的全部列（原始值和滚动均值列），报告中每个因子一个页面；R² 对非基础因子的全部列计算。
        self.factor_columns = {}
        self.base_column_list = []
        self.explained_column_list = []
        for factor_name in self.factor_list:
            self.factor_columns[factor_name] = [column for column in pq.read_schema(self.factor_dir / f"{factor_name}.parquet").names if column not in ("code", "date")]
            if factor_name in self.explained_factor_list:
                self.explained_column_list += self.factor_columns[factor_name]
            else:
                self.base_column_list += self.factor_columns[factor_name]

        # 统一平均窗口：原始值用基础因子原始值解释，滚动均值列（如 _5_m）用基础因子相同窗口的滚动均值列解释。
        self.base_columns = {}
        for factor_name in self.explained_factor_list:
            for column in self.factor_columns[factor_name]:
                self.base_columns[column] = [base_factor + column.removeprefix(factor_name) for base_factor in self.base_factor_list]

    def _load_factors(self):
        """读取全部因子的原始值和滚动均值列，按单因子评估的板块和小市值股票池筛选，再逐日截面标准化。"""
        factor = pd.concat([pd.read_parquet(self.factor_dir / f"{factor_name}.parquet", columns=["code", "date"] + self.factor_columns[factor_name], filters=[("date", ">=", self.start_date)]).set_index(["code", "date"]) for factor_name in self.factor_list], axis=1)
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
        columns = self.base_column_list + self.explained_column_list
        factor[columns] = factor[columns].replace([np.inf, -np.inf], np.nan)
        factor[columns] = (factor[columns] - factor.groupby("date")[columns].transform("mean")) / factor.groupby("date")[columns].transform("std").replace(0, np.nan)
        return factor

    def _correlation_matrix(self, factor):
        """逐日计算两两因子的截面 Pearson 相关系数，再对全部日期取均值。"""
        correlation = factor.groupby("date")[self.factor_list].corr()
        return correlation.groupby(level=1).mean().reindex(index=self.factor_list, columns=self.factor_list)

    @staticmethod
    def _regression_r2_daily(task):
        """在工作进程中将单个交易日的每个非基础因子列对相同窗口的基础因子列做截面回归，返回该日各列的 R²。"""
        date, daily, base_columns = task
        r2 = pd.Series(index=list(base_columns), dtype=float)
        for factor_name in base_columns:
            sample = daily[base_columns[factor_name] + [factor_name]].dropna()
            # 有效股票数需多于回归参数个数（基础因子 + 截距），否则当日 R² 为空。
            if len(sample) <= len(base_columns[factor_name]) + 1:
                continue
            x = np.column_stack([np.ones(len(sample)), sample[base_columns[factor_name]]])
            sample["residual"] = sample[factor_name] - x @ np.linalg.lstsq(x, sample[factor_name], rcond=None)[0]
            r2[factor_name] = 1 - (sample["residual"] ** 2).sum() / ((sample[factor_name] - sample[factor_name].mean()) ** 2).sum()
        return date, r2

    def _regression_r2(self, factor):
        """每日将每个非基础因子列（含滚动均值列）对相同窗口的基础因子列做带截距的截面回归，返回日期 × 因子列的 R²。"""
        r2 = pd.DataFrame(index=pd.Index(sorted(factor["date"].unique()), name="date"), columns=self.explained_column_list, dtype=float)
        # 按交易日并行，每个任务只传当日回归用到的列。
        tasks = ((date, daily[self.base_column_list + self.explained_column_list], self.base_columns) for date, daily in factor.groupby("date"))
        with Pool(processes=self.processes) as pool:
            for date, daily_r2 in tqdm(pool.imap(FactorCorrelationAnalyzer._regression_r2_daily, tasks), total=len(r2), desc="Explaining factors", unit="day"):
                r2.loc[date] = daily_r2
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
        """计算相关性矩阵和每日 R²，与批量评估指标表一起写入 visualization_output_dir/multi_factor_evaluation_report.html。"""
        factor = self._load_factors()
        correlation = self._correlation_matrix(factor)
        r2 = self._regression_r2(factor)

        # 指标表沿用 factor_comparison.csv 的内容，追加平均 R²；基础因子的各列不参与回归，显示为「基础因子」。
        table = summary.copy()
        table["平均 R²"] = table["因子"].map(r2.mean())
        percent_columns = ["多空年化收益", "多头年化收益", "多头换手率"]
        for column in ["IC", "RankIC", "ICIR", "平均 R²"] + percent_columns:
            table[column] = [ResultsVisualizer._format(value, column in percent_columns) for value in table[column]]
        table.loc[table["因子"].isin(self.base_column_list), "平均 R²"] = "基础因子"

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

        # 每个因子一个页面，按钮分基础因子和其他因子两组：表格只列该因子的原始值和滚动均值列；其他因子另附每日 R² 折线图，基础因子只放表格。
        base_factors = escape("、".join(self.base_factor_list))
        top_link = '<a class="top" href="#factor-nav">↑ 返回按钮</a>'
        base_buttons = ""
        explained_buttons = ""
        factor_panels = ""
        for factor_name in self.factor_list:
            button = f'<button type="button" data-target="factor-{escape(factor_name)}">{escape(factor_name)}</button>'
            factor_table = table.loc[table["因子"].isin(self.factor_columns[factor_name])].to_html(index=False, border=0, classes="metrics")
            # 单因子报告在同级的 single_factor_evaluation 文件夹中，按相对路径嵌入页面底部，点击因子按钮时才加载。
            report = f'<h3>单因子评估报告</h3><p>嵌入 {escape(factor_name)}_report.html，可在其中切换窗口对比、持有期对比、窗口详情和分年表现，并选择窗口、持有期（1 / 5 / 10 / 20 天等）和时间区间。</p><iframe class="factor-report" data-src="single_factor_evaluation/{escape(factor_name)}_report.html#embed" title="{escape(factor_name)} 单因子评估报告"></iframe>'
            if factor_name in self.base_factor_list:
                base_buttons += button
                factor_panels += f'<section class="panel" id="factor-{escape(factor_name)}" hidden><h2>{escape(factor_name)} · 基础因子{top_link}</h2><p>基础因子不参与解释回归，表格列出原始值和各滚动均值列持有 1 天的评估指标，各持有期的结果见下方单因子报告。</p><div class="content">{factor_table}</div>{report}</section>'
            else:
                explained_buttons += button
                chart = self._chart(r2[self.factor_columns[factor_name]]) if r2[self.factor_columns[factor_name]].notna().any().any() else '<p class="empty">没有有效的 R²。</p>'
                factor_panels += f'<section class="panel" id="factor-{escape(factor_name)}" hidden><h2>{escape(factor_name)} · 原始值平均 R² {ResultsVisualizer._format(r2[factor_name].mean())}{top_link}</h2><p>表格列出原始值和各滚动均值列持有 1 天的评估指标，平均 R² 为基础因子对该列每日截面解释度的均值；各持有期的结果见下方单因子报告。</p><div class="content">{factor_table}</div><h3>基础因子解释度 · 每日 R²</h3><p>原始值对 {base_factors} 的原始值回归，滚动均值列对基础因子相同窗口的滚动均值列回归（如 _5_m 对各基础因子的 _5_m）。灰线为原始值，蓝线为滚动均值列，窗口越长颜色越深，图例数字为全区间平均 R²。</p><div class="content">{chart}</div>{report}</section>'

        html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>多因子评估报告</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif; color: #172b4d; background: #f2f5fa; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; background: #f2f5fa; }} main {{ max-width: 1280px; margin: auto; padding: 24px 28px; }}
header {{ padding: 32px; margin-bottom: 22px; background: linear-gradient(120deg, #172c52, #28578c); color: white; border-radius: 20px; }}
.eyebrow {{ color: #9edcfa; font-size: 12px; letter-spacing: 3px; }} h1 {{ font-size: 30px; margin: 16px 0; }} header p {{ color: #d2dfef; line-height: 1.8; }}
section {{ padding: 26px; margin-bottom: 22px; background: white; border: 1px solid #e3e9f2; border-radius: 16px; box-shadow: 0 5px 20px #23395605; }}
section[hidden] {{ display: none; }} h2 {{ margin: 0; font-size: 20px; }} h3 {{ margin: 28px 0 6px; font-size: 16px; }}
section p, footer {{ font-size: 13px; color: #6b7c94; line-height: 1.9; }} .content {{ overflow-x: auto; }} .empty {{ text-align: center; padding: 40px; }}
nav .group {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 8px 0; }} nav .group + .group {{ border-top: 1px solid #edf1f6; }}
nav .label {{ width: 72px; font-size: 13px; color: #6b7c94; }}
nav button {{ padding: 7px 13px; font: inherit; font-size: 13px; color: #28578c; background: #f4f7fb; border: 1px solid #dce3ed; border-radius: 8px; cursor: pointer; }}
nav button:hover {{ background: #e8f0fb; }} nav button.active {{ color: white; background: #28578c; border-color: #28578c; }}
a.top {{ float: right; font-size: 13px; font-weight: 400; color: #28578c; text-decoration: none; }}
iframe.factor-report {{ display: block; width: 100%; height: calc(100vh - 40px); min-height: 640px; margin-top: 8px; border: 1px solid #e3e9f2; border-radius: 12px; background: #f2f5fa; }}
svg {{ display: block; width: 100%; height: auto; min-width: 560px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }} th {{ background: #f4f7fb; color: #61718b; font-weight: 500; }}
th, td {{ padding: 12px 13px; text-align: right; border-bottom: 1px solid #edf1f6; }} th:first-child, td:first-child {{ text-align: left; }} tbody tr:hover {{ background: #f4f8ff; }}
table.correlation {{ width: auto; }} table.correlation td {{ min-width: 64px; text-align: center; border: 1px solid white; }}
th.vertical {{ vertical-align: bottom; text-align: center; }} th.vertical span {{ writing-mode: vertical-rl; transform: rotate(180deg); }}
footer {{ padding: 4px 12px 20px; }} @media(max-width: 700px) {{ main {{ padding: 12px; }} header, section {{ padding: 18px; }} h1 {{ font-size: 24px; }} nav .label {{ width: 100%; }} }}
</style></head><body><main>
<header><div class="eyebrow">FACTOR RESEARCH / MULTI-FACTOR REPORT</div><h1>多因子评估报告</h1>
<p>{r2.index[0]} — {r2.index[-1]} · {len(r2):,} 个交易日 · {len(self.factor_list)} 个原始因子 · 基础因子：{base_factors}</p></header>
<section id="factor-nav"><nav>
<div class="group"><span class="label">总览</span><button type="button" class="active" data-target="overview">指标对比与相关性</button></div>
<div class="group"><span class="label">基础因子</span>{base_buttons}</div>
<div class="group"><span class="label">其他因子</span>{explained_buttons}</div>
</nav></section>
<section class="panel" id="overview"><h2>因子评估指标对比</h2><p>IC 等指标与 factor_comparison.csv 一致，收益和换手率口径见各因子报告。平均 R² 为基础因子（{base_factors}）对该列每日截面解释度的均值，滚动均值列用基础因子相同窗口的滚动均值列解释；基础因子的各列显示为「基础因子」。</p><div class="content">{table.to_html(index=False, border=0, classes="metrics")}</div>
<h3>因子相关性矩阵</h3><p>全部原始因子的每日截面 Pearson 相关系数的时间均值，不含滚动均值列；前 {len(self.base_factor_list)} 个为基础因子。红色为正相关、蓝色为负相关，颜色越深相关性越强。</p><div class="content"><table class="correlation"><thead><tr><th></th>{correlation_header}</tr></thead><tbody>{correlation_rows}</tbody></table></div></section>
{factor_panels}
<footer><b>计算口径</b><br>
样本为因子目录中全部因子文件的原始值和滚动均值列，自 {self.start_date} 起；板块范围和股票池与单因子评估相同（{escape(str(self.stock_board))}，每日按上月末市值保留最小的 {self.stock_pool} 只股票），使用当日因子值，不做滞后。<br>
各因子剔除无穷值后逐日截面标准化（减均值、除以样本标准差），未去极值。相关系数逐日按两两均有效的股票计算，再对日期取算术平均。<br>
R² 每日以该因子列和对应窗口的全部基础因子列均有效的股票为样本做 OLS：原始值对基础因子原始值回归，滚动均值列对基础因子相同窗口的滚动均值列回归；有效股票数不多于参数个数的日期记为空，平均 R² 为每日 R² 的算术平均。<br>
各因子页面底部嵌入的单因子报告读取同级 single_factor_evaluation 文件夹中的 &lt;因子&gt;_report.html，移动本报告时需连同该文件夹一起移动。<br>
生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}。
</footer></main>
<script>
// 点击按钮只显示对应页面，并跳转到该页面顶部；页面中的单因子报告在首次打开时才加载。
const buttons = document.querySelectorAll("nav button");
for (const button of buttons) {{
    button.addEventListener("click", () => {{
        for (const item of buttons) item.classList.toggle("active", item === button);
        for (const panel of document.querySelectorAll(".panel")) panel.hidden = panel.id !== button.dataset.target;
        for (const frame of document.getElementById(button.dataset.target).querySelectorAll("iframe[data-src]")) {{
            if (!frame.getAttribute("src")) frame.setAttribute("src", frame.dataset.src);
        }}
        document.getElementById(button.dataset.target).scrollIntoView({{ behavior: "smooth" }});
    }});
}}
</script></body></html>'''
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "multi_factor_evaluation_report.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()
