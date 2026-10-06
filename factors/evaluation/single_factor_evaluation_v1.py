from pathlib import Path

import pandas as pd
import numpy as np

import os
import shutil
from factors.evaluation.results_visualization.results_visualizer import ResultsVisualizer


def _clear_existing_results(output_dir, visualization_output_dir):
    for directory in (output_dir, visualization_output_dir):
        if directory:
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)


class SingleFactorEvaluation:
    def __init__(self, config, factor_data=None):
        self.backtest_data_dir = config["backtest_data_dir"]
        self.start_date = config["start_date"]
        self.update_all = config["update_all"]
        self.stock_pool = config["stock_pool"]
        self.stock_board = config["stock_board"]
        self.group_number = config.get("group_number", 10)
        # 持有期（交易日数）：1 天为默认口径，持仓清单和批量指标都基于它，始终参与评估。
        self.holding_periods = [1] + [holding_period for holding_period in config.get("holding_periods", [1]) if holding_period != 1]
        self.stock_info_path = Path(config["stock_minutes_dir"]).parent / "__daily_data_update" / "stock_info.csv"
        self._clear_outputs = factor_data is None

        specified_column = config["specified_column"]
        self.factor_path = Path(config["factor_data_dir"]) / f"{specified_column.split('_')[0]}.parquet"
        if factor_data is None:
            factor = pd.read_parquet(self.factor_path)
        else:
            factor = factor_data

        factor = factor[["code", "date", specified_column]]
        # 因子构建不区分板块，按 stock_board 决定评估范围：main board 只保留沪深主板，all 不筛选。
        if self.stock_board == "main board":
            factor = factor.loc[factor["code"].str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))]
        elif self.stock_board != "all":
            raise ValueError(f"stock_board must be 'main board' or 'all', got {self.stock_board!r}")

        self.factor_name = specified_column
        self.factor = factor.rename(columns={self.factor_name: "__factor"})
        # 保留最新收盘因子，供下一交易日选股；历史持仓仍使用滞后一期的因子。
        self.next_factor = self.factor.loc[self.factor["date"].eq(self.factor["date"].max())].copy()
        self.factor["__factor"] = self.factor.groupby(["code"])["__factor"].shift(1)

        # 用上月末市值作为当月排序依据，在每日有效因子样本中选取小市值股票。
        monthly_mkcap = pd.read_parquet(Path(config["stock_minutes_dir"]).parent / "fundamentals" / "mkcap_monthly.parquet", columns=["code", "date", "mkcap"])
        monthly_mkcap = monthly_mkcap.rename(columns={"date": "month"})
        self.factor = self._select_stock_pool(self.factor, monthly_mkcap)
        self.next_factor = self._select_stock_pool(self.next_factor, monthly_mkcap)
        self.factor = self._assign_groups(self.factor)
        self.next_factor = self._assign_groups(self.next_factor)
        self.trade_dates = sorted(date for date in self.factor["date"].unique() if date >= self.start_date)

        self.output_dir = config["output_dir"]
        self.visualization_output_dir = config.get("visualization_output_dir")
        os.makedirs(self.output_dir, exist_ok=True)

    def _select_stock_pool(self, factor, monthly_mkcap):
        """按因子所标日期的月份，在有效样本中筛选小市值股票。"""
        factor = factor.dropna(subset=["__factor"])
        factor["month"] = factor["date"].str[:6] + "01"
        factor = factor.merge(monthly_mkcap, on=["code", "month"])
        factor = factor.dropna(subset=["mkcap"])
        factor = factor.sort_values("mkcap", kind="stable")
        factor = factor.groupby("date", sort=False).head(self.stock_pool)
        return factor.drop(columns=["month", "mkcap"])

    def _assign_groups(self, factor):
        """每日按因子值从小到大等分为 group_number 组；同值时保持原有行顺序，与稳定排序的结果一致。"""
        factor = factor.copy()
        factor["__count"] = factor.groupby("date")["__factor"].transform("size")
        factor["__position"] = factor.groupby("date")["__factor"].rank(method="first") - 1
        factor["group"] = 1
        for group in range(1, self.group_number):
            factor["group"] += (factor["__position"] >= (1 + (factor["__count"] - 1) * group / self.group_number).astype(int)).astype(int)
        return factor.drop(columns=["__count", "__position"])

    def _save_stock_list(self, preferred_group):
        """保存历史（含最新交易日）多头持仓，以及最新收盘信号的下一日目标。"""
        stock_info = pd.read_csv(self.stock_info_path, dtype=str)
        stock_info["code"] = stock_info["code"].str.split(".", n=1).str[0].str.zfill(6)
        output_dir = Path(self.output_dir).parent / "stock_list"
        for factor, directory, date_column in (
            (self.factor, output_dir, "date"),
            (self.next_factor, output_dir / "next_day", "signal_date"),
        ):
            stock_list = factor.loc[(factor["date"] >= self.start_date) & factor["group"].eq(int(preferred_group.split("_")[1]))]
            stock_list = stock_list.sort_values(["date", "code"])
            stock_list = pd.DataFrame({date_column: stock_list["date"], "code": stock_list["code"], "group": preferred_group, "factor_value": stock_list["__factor"]})
            directory.mkdir(parents=True, exist_ok=True)
            stock_list = stock_list.merge(stock_info, on="code", how="left", validate="many_to_one")
            stock_list.to_csv(directory / f"{self.factor_name}.csv", index=False, encoding="utf-8-sig")

    @staticmethod
    def load_prices(backtest_data_dir):
        """读取并清洗价格，整理为日期 × 股票代码的收盘价宽表，供单因子或批量评估复用。"""
        data = pd.read_parquet(backtest_data_dir, columns=["trade_time", "code", "close"])
        data["close"] = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        # 只对不重复的时间点格式化日期再映射回各行，避免对全部行逐行 strftime。
        trade_times = pd.Series(data["trade_time"].unique())
        data["date"] = data["trade_time"].map(dict(zip(trade_times, pd.to_datetime(trade_times).dt.strftime("%Y%m%d"))))
        return data.pivot(index="date", columns="code", values="close")

    def evaluate(self, prices_by_date=None, plot_report=True):
        """保存结果和报告并返回汇总指标；未传入价格宽表时才读取价格。批量评估传入 plot_report=False，待全部列完成后按因子统一生成报告。"""
        if self.update_all == 1 and self._clear_outputs:
            _clear_existing_results(self.output_dir, self.visualization_output_dir)

        output_path = Path(self.output_dir) / f"{self.factor_name}.csv"
        # 持有 1 天的列沿用原列名，其他持有期的列名加 _{持有期}d 后缀，如 group_1_5d、IC_5d、group_1_5d_count、group_1_5d_turnover。
        columns = []
        for holding_period in self.holding_periods:
            suffix = "" if holding_period == 1 else f"_{holding_period}d"
            return_columns = [f"group_{group}{suffix}" for group in range(1, self.group_number + 1)]
            columns += return_columns + [f"IC{suffix}", f"rankIC{suffix}"] + [f"{column}_count" for column in return_columns] + [f"{column}_turnover" for column in return_columns]

        # 读取现有的evaluation数据
        if self.update_all == 0 and output_path.is_file():
            existing_evaluation = pd.read_csv(output_path, index_col="date")
            existing_evaluation.index = existing_evaluation.index.map(str)
            existing_evaluation = existing_evaluation.loc[~np.isinf(existing_evaluation).any(axis=1)]
        else:
            existing_evaluation = pd.DataFrame(columns=columns, dtype=float)
            existing_evaluation.index.name = "date"

        evaluation = pd.DataFrame(index=pd.Index(self.trade_dates[:-1], name="date"), columns=columns, dtype=float)

        if prices_by_date is None:
            prices_by_date = self.load_prices(self.backtest_data_dir)

        # 在日期 × 股票的宽表上计算：当日分组、滞后一期的因子值，以及当日 10 点买入、持有到下一交易日 10 点的日收益。
        groups = self.factor.pivot(index="date", columns="code", values="group").reindex(evaluation.index)
        factor_values = self.factor.pivot(index="date", columns="code", values="__factor").reindex(evaluation.index)
        # 当日有价、下一交易日无价（如停牌）时日收益记为 0；当日无价时为空，当日不能买入。
        current_prices = prices_by_date.reindex(self.trade_dates)
        daily_returns = current_prices.shift(-1) / current_prices - 1
        daily_returns = daily_returns.where(daily_returns.notna() | current_prices.isna(), 0)
        daily_returns = daily_returns.reindex(index=evaluation.index, columns=groups.columns)
        # 买入后的持有日日收益为空（如停牌、退市）时记 0，股票仍留在持仓中。
        held_returns = daily_returns.fillna(0)

        # 每天按当日分组买入一批，只买当日有收益（可交易）的股票；lag 天前买入的一批在当日的收益为批内股票当日收益的等权均值。
        cohort_returns = {}
        cohort_counts = {}
        for group in range(1, self.group_number + 1):
            members = groups.eq(group) & daily_returns.notna()
            for lag in range(max(self.holding_periods)):
                held = members.shift(lag, fill_value=False)
                cohort_counts[(group, lag)] = held.sum(axis=1)
                cohort_returns[(group, lag)] = held_returns.where(held, 0).sum(axis=1) / cohort_counts[(group, lag)]

        for holding_period in self.holding_periods:
            suffix = "" if holding_period == 1 else f"_{holding_period}d"
            # 持有 holding_period 天：同时持有最近 holding_period 天买入的各批，各批等权，组合日收益为各批当日收益的均值；起始阶段只平均已买入的批次。
            for group in range(1, self.group_number + 1):
                returns = pd.DataFrame(index=evaluation.index)
                counts = pd.DataFrame(index=evaluation.index)
                for lag in range(holding_period):
                    returns[lag] = cohort_returns[(group, lag)]
                    counts[lag] = cohort_counts[(group, lag)]
                evaluation[f"group_{group}{suffix}"] = returns.mean(axis=1)
                # 股票数量为各批的平均股票数，尚未买入的批次不计入。
                evaluation[f"group_{group}{suffix}_count"] = counts.replace(0, np.nan).mean(axis=1).fillna(0)

            # 持有期收益：买入后 holding_period 天的日收益按复利累计；买入日无收益的股票不参与，末尾不足 holding_period 天的日期为空。
            holding_returns = 1
            for lag in range(holding_period):
                holding_returns = holding_returns * (1 + held_returns.shift(-lag))
            holding_returns = (holding_returns - 1).where(daily_returns.notna())

            # IC 与 rankIC：按日在因子值和持有期收益都有效的股票上计算 Pearson 相关系数，rankIC 先在当日内取排名。
            factor_sample = factor_values.where(holding_returns.notna())
            return_sample = holding_returns.where(factor_values.notna())
            for column, x, y in ((f"IC{suffix}", factor_sample, return_sample), (f"rankIC{suffix}", factor_sample.rank(axis=1), return_sample.rank(axis=1))):
                x = x.sub(x.mean(axis=1), axis=0)
                y = y.sub(y.mean(axis=1), axis=0)
                evaluation[column] = ((x * y).sum(axis=1) / np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum(axis=1))).clip(-1, 1)

            # 换手率：当日买入的一批与当日卖出的一批（holding_period 天前买入）相比的变动比例，不除以 holding_period；前 holding_period 个交易日没有卖出的批次，记为空值。
            previous_groups = groups.shift(holding_period)
            for group in range(1, self.group_number + 1):
                turnover = pd.DataFrame(index=evaluation.index)
                turnover["current"] = groups.eq(group).sum(axis=1)
                turnover["previous"] = previous_groups.eq(group).sum(axis=1)
                turnover["overlap"] = (groups.eq(group) & previous_groups.eq(group)).sum(axis=1)
                turnover["turnover"] = 1 - turnover["overlap"] / turnover[["current", "previous"]].max(axis=1)
                # 任一侧为空组时：两侧都空记 0，只有一侧为空记 0.5。
                turnover["turnover"] = turnover["turnover"].where(turnover["current"].gt(0) & turnover["previous"].gt(0), (turnover["current"].gt(0).astype(int) + turnover["previous"].gt(0).astype(int)) / 2)
                evaluation[f"group_{group}{suffix}_turnover"] = turnover["turnover"]
            evaluation.loc[evaluation.index[:holding_period], [f"group_{group}{suffix}_turnover" for group in range(1, self.group_number + 1)]] = np.nan

        # 已有完整结果的日期沿用旧值，只补充其余日期；持有期大于 1 天时末尾的 IC 尚未实现（为空），这些日期之后会重新计算。
        check_columns = [column for column in columns if column.startswith(("IC", "rankIC")) or column.endswith(("_count", "_turnover"))]
        existing_dates = existing_evaluation.reindex(columns=check_columns).dropna().index
        evaluation = evaluation.loc[~evaluation.index.isin(existing_dates)]
        evaluation = evaluation.dropna(subset=[f"group_{group}" for group in range(1, self.group_number + 1)] + ["IC", "rankIC"], how="all")
        evaluation = pd.concat([existing_evaluation, evaluation])
        evaluation = evaluation[~evaluation.index.duplicated(keep="last")].sort_index()
        evaluation.to_csv(output_path)
        visualizer = ResultsVisualizer(evaluation, self.factor_name, self.start_date, self.visualization_output_dir)
        self._save_stock_list(visualizer.preferred_group)
        # 报告按因子合并原始值和各滚动均值列，包含输出目录中该因子已有的全部列结果。
        if plot_report:
            ResultsVisualizer.plot_results_html(self.factor_path, self.output_dir, self.start_date, self.visualization_output_dir)
        return visualizer.summary()
