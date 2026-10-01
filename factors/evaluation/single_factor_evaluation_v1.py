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
        self.group_number = config.get("group_number", 10)
        self.stock_info_path = Path(config["stock_minutes_dir"]).parent / "__daily_data_update" / "stock_info.csv"
        self._clear_outputs = factor_data is None

        specified_column = config["specified_column"]
        if factor_data is None:
            factor_path = Path(config["factor_data_dir"]) / f"{specified_column.split('_')[0]}.parquet"
            factor = pd.read_parquet(factor_path)
        else:
            factor = factor_data

        factor = factor[["code", "date", specified_column]]

        self.factor_name = factor.columns[-1]
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
        factor = factor.dropna(subset=["__factor"]).copy()
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

    def evaluate(self, prices_by_date=None):
        """保存结果和报告并返回汇总指标；未传入价格宽表时才读取价格。"""
        if self.update_all == 1 and self._clear_outputs:
            _clear_existing_results(self.output_dir, self.visualization_output_dir)

        output_path = Path(self.output_dir) / f"{self.factor_name}.csv"
        return_columns = [f"group_{group}" for group in range(1, self.group_number + 1)]
        count_columns = [f"{column}_count" for column in return_columns]
        turnover_columns = [f"{column}_turnover" for column in return_columns]
        columns = return_columns + ["IC", "rankIC"] + count_columns + turnover_columns

        # 读取现有的evaluation数据
        if self.update_all == 1 and output_path.is_file():
            existing_evaluation = pd.read_csv(output_path, index_col="date")
            existing_evaluation.index = existing_evaluation.index.map(str)
            existing_evaluation = existing_evaluation.loc[~np.isinf(existing_evaluation).any(axis=1)]
        else:
            existing_evaluation = pd.DataFrame(columns=columns, dtype=float)
            existing_evaluation.index.name = "date"

        evaluation = pd.DataFrame(
            index=pd.Index([str(date) for date in self.trade_dates[:-1]], name="date"),
            columns=columns,
            dtype=float,
        )

        if prices_by_date is None:
            prices_by_date = self.load_prices(self.backtest_data_dir)

        # 持有到下一交易日的收益；当日有价、下一交易日无价（如停牌）时收益记为 0。
        current_prices = prices_by_date.reindex(self.trade_dates)
        next_returns = current_prices.shift(-1) / current_prices - 1
        next_returns = next_returns.where(next_returns.notna() | current_prices.isna(), 0)
        next_returns = next_returns.stack().rename("next_return").reset_index()
        factor = self.factor.loc[self.factor["date"].isin(evaluation.index)]
        factor = factor.merge(next_returns, on=["date", "code"], how="left")

        evaluation[return_columns] = factor.groupby(["date", "group"])["next_return"].mean().unstack().reindex(index=evaluation.index, columns=range(1, self.group_number + 1)).to_numpy()
        evaluation[count_columns] = factor.groupby(["date", "group"])["next_return"].count().unstack().reindex(index=evaluation.index, columns=range(1, self.group_number + 1)).fillna(0).to_numpy()

        # IC 与 rankIC：按日在有效收益样本上计算 Pearson 相关系数，rankIC 先在当日内取排名。
        factor = factor.dropna(subset=["next_return"])
        factor["factor_rank"] = factor.groupby("date")["__factor"].rank()
        factor["return_rank"] = factor.groupby("date")["next_return"].rank()
        for column, x, y in (("IC", "__factor", "next_return"), ("rankIC", "factor_rank", "return_rank")):
            factor["x"] = factor[x] - factor.groupby("date")[x].transform("mean")
            factor["y"] = factor[y] - factor.groupby("date")[y].transform("mean")
            factor["xy"] = factor["x"] * factor["y"]
            factor["xx"] = factor["x"] ** 2
            factor["yy"] = factor["y"] ** 2
            sums = factor.groupby("date")[["xy", "xx", "yy"]].sum()
            evaluation[column] = (sums["xy"] / np.sqrt(sums["xx"] * sums["yy"])).clip(-1, 1)

        # 换手率：当日各组与上一交易日同组股票相比的变动比例；首个交易日没有上一期分组，记为空值。
        groups = self.factor.pivot(index="date", columns="code", values="group").reindex(evaluation.index)
        previous_groups = groups.shift(1)
        for group in range(1, self.group_number + 1):
            turnover = pd.DataFrame(index=evaluation.index)
            turnover["current"] = groups.eq(group).sum(axis=1)
            turnover["previous"] = previous_groups.eq(group).sum(axis=1)
            turnover["overlap"] = (groups.eq(group) & previous_groups.eq(group)).sum(axis=1)
            turnover["turnover"] = 1 - turnover["overlap"] / turnover[["current", "previous"]].max(axis=1)
            # 任一侧为空组时：两侧都空记 0，只有一侧为空记 0.5。
            turnover["turnover"] = turnover["turnover"].where(turnover["current"].gt(0) & turnover["previous"].gt(0), (turnover["current"].gt(0).astype(int) + turnover["previous"].gt(0).astype(int)) / 2)
            evaluation[f"group_{group}_turnover"] = turnover["turnover"]
        evaluation.loc[evaluation.index[:1], turnover_columns] = np.nan

        # 已有完整结果的日期沿用旧值，只补充其余日期。
        existing_dates = existing_evaluation.reindex(columns=count_columns + turnover_columns).dropna().index
        evaluation = evaluation.loc[~evaluation.index.isin(existing_dates)]
        evaluation = evaluation.dropna(subset=return_columns + ["IC", "rankIC"], how="all")
        evaluation = pd.concat([existing_evaluation, evaluation])
        evaluation = evaluation[~evaluation.index.duplicated(keep="last")].sort_index()
        evaluation.to_csv(output_path)
        visualizer = ResultsVisualizer(evaluation, self.factor_name, self.start_date, self.visualization_output_dir)
        self._save_stock_list(visualizer.preferred_group)
        visualizer.plot_results_html()
        return visualizer.summary()
