from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from factors.evaluation.single_factor_evaluation_v1 import SingleFactorEvaluation, _clear_existing_results
from factors.evaluation.results_visualization.results_visualizer import ResultsVisualizer


class MultiFactorEvaluation:
    """遍历因子目录中的 Parquet 文件，每个文件评估与文件名同名的因子列，再生成全部因子的评估报告。"""
    def __init__(self, config):
        self.config = config
        self.factor_dir = Path(config["factor_data_dir"])
        self.processes = config["processes"]

    @staticmethod
    def _evaluate_column(task):
        """在工作进程中只读取单个因子列并完成评估。"""
        config, factor_path, factor_name, prices_by_date = task
        factor_data = pd.read_parquet(factor_path, columns=["code", "date", factor_name])
        evaluator = SingleFactorEvaluation({**config, "specified_column": factor_name}, factor_data)
        return evaluator.evaluate(prices_by_date=prices_by_date, plot_report=False)

    def evaluate(self):
        factor_files = sorted(self.factor_dir.glob("*.parquet"))

        prices_by_date = SingleFactorEvaluation.load_prices(self.config["backtest_data_dir"])
        if self.config.get("update_all") == 1:
            _clear_existing_results(self.config["output_dir"], self.config.get("visualization_output_dir"))

        # 每个因子文件一个任务，只评估与文件名同名的列，文件中的其他列（如旧格式的滚动均值列）一律忽略；imap 保持结果顺序与因子文件顺序一致。
        tasks = [(self.config, factor_path, factor_path.stem, prices_by_date) for factor_path in factor_files]
        with Pool(processes=self.processes) as pool:
            records = list(tqdm(pool.imap(MultiFactorEvaluation._evaluate_column, tasks), total=len(tasks), desc="Evaluating factors", unit="factor"))

        summary = pd.DataFrame(records)
        summary.to_csv(Path(self.config["output_dir"]) / "factor_comparison.csv", index=False, encoding="utf-8-sig")
        ResultsVisualizer.plot_results_html(self.config)
        return summary
