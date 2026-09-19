from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
import numpy as np

import os
from factors.evaluation.results_visualization.results_visualizer import ResultsVisualizer

class SingleFactorEvaluation:
    def __init__(self, config, factor_data=None):
        self.backtest_data_dir = config["backtest_data_dir"]
        self.start_date = config["start_date"]
        self.update_all = config["update_all"]
        self.processes = config.get("processes", 4)

        specified_column = config["specified_column"]
        if factor_data is None:
            factor_path = Path(config["factor_data_dir"]) / f"{specified_column.split('_')[0]}.parquet"
            factor = pd.read_parquet(factor_path)
        else:
            factor = factor_data

        factor = factor[["code", "date", specified_column]]

        self.factor_name = factor.columns[-1]
        self.factor = factor.rename(columns={self.factor_name: "__factor"})
        self.factor["__factor"] = self.factor.groupby(["code"])["__factor"].shift(1)

        self.factor = self.factor.dropna(subset=["__factor"])
        self.trade_dates = sorted(date for date in self.factor["date"].unique() if date >= self.start_date)

        self.output_dir = config["output_dir"]
        self.visualization_output_dir = config.get("visualization_output_dir")
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def load_prices(backtest_data_dir):
        """读取并清洗价格，供单因子或批量评估复用。"""
        data = pd.read_parquet(backtest_data_dir, columns=["trade_time", "code", "close"])
        data["close"] = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        data["date"] = pd.to_datetime(data["trade_time"]).dt.strftime("%Y%m%d")
        return data.set_index("code").groupby("date")["close"]

    @staticmethod
    def _evaluate_date(task):
        """计算单日分组及指标；已有结果的日期只返回分组，供下一日计算换手率。"""
        evaluation_date, daily_factor, current_prices, next_prices = task
        factor_values = daily_factor["__factor"].to_numpy(dtype=float)
        sorted_indices = np.argsort(factor_values, kind="stable")
        split_points = (1 + (len(factor_values) - 1) * np.arange(1, 5) / 5).astype(int)
        groups = [set(group) for group in np.split(daily_factor["code"].to_numpy()[sorted_indices], split_points)]
        if current_prices is None:
            return evaluation_date, groups, None

        current_values = current_prices.reindex(daily_factor["code"]).to_numpy()
        next_returns = next_prices.reindex(daily_factor["code"]).to_numpy() / current_values - 1
        sorted_returns = np.split(next_returns[sorted_indices], split_points)
        valid = ~np.isnan(next_returns)
        ic = np.corrcoef(factor_values[valid], next_returns[valid])[0, 1]
        rank_ic = np.corrcoef(pd.Series(factor_values[valid]).rank(), pd.Series(next_returns[valid]).rank())[0, 1]
        values = [np.nanmean(group) for group in sorted_returns] + [ic, rank_ic]
        values += [np.count_nonzero(~np.isnan(group)) for group in sorted_returns]
        return evaluation_date, groups, values

    def evaluate(self, prices_by_date=None):
        """保存结果和报告并返回汇总指标；未传入价格分组时才读取价格。"""
        output_path = Path(self.output_dir) / f"{self.factor_name}.csv"
        columns = [f"group_{group}" for group in range(1, 6)] + ["IC", "rankIC"]
        columns += [f"group_{group}_count" for group in range(1, 6)]
        columns += [f"group_{group}_turnover" for group in range(1, 6)]

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
        factor_by_date = self.factor.dropna(subset=["__factor"]).groupby("date")

        if prices_by_date is None:
            prices_by_date = self.load_prices(self.backtest_data_dir)

        existing_dates = set(existing_evaluation.reindex(columns=columns[7:]).dropna().index)
        previous_groups = None
        tasks = (
            (
                str(trade_date), factor_by_date.get_group(trade_date),
                prices_by_date.get_group(str(trade_date)) if str(trade_date) not in existing_dates else None,
                prices_by_date.get_group(str(next_trade_date)) if str(trade_date) not in existing_dates else None,
            )
            for trade_date, next_trade_date in zip(self.trade_dates[:-1], self.trade_dates[1:])
        )
        with Pool(processes=self.processes) as pool:
            results = pool.imap(self._evaluate_date, tasks)
            for evaluation_date, groups, values in tqdm(results, total=len(evaluation), desc="Evaluating factor", unit="trading day"):
                # imap 按日期顺序返回，换手率仍使用上一交易日分组。
                turnover = [
                    1 - len(current & previous) / max(len(current), len(previous)) if current and previous else (bool(current) + bool(previous)) / 2
                    for current, previous in zip(groups, previous_groups)
                ] if previous_groups is not None else [np.nan] * 5
                previous_groups = groups
                if values is not None:
                    evaluation.loc[evaluation_date] = values + turnover

        evaluation = evaluation.dropna(subset=columns[:7], how="all")
        evaluation = pd.concat([existing_evaluation, evaluation])
        evaluation = evaluation[~evaluation.index.duplicated(keep="last")].sort_index()
        evaluation.to_csv(output_path)
        visualizer = ResultsVisualizer(evaluation, self.factor_name, self.start_date, self.visualization_output_dir)
        visualizer.plot_results_html()
        return visualizer.summary()


    
