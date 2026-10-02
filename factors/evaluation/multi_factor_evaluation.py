from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from factors.evaluation.single_factor_evaluation_v1 import SingleFactorEvaluation, _clear_existing_results


class MultiFactorEvaluation:
    """遍历因子目录中的 Parquet 文件，评估所有原始及衍生因子列。"""
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
        return {"因子文件": factor_path.name, **evaluator.evaluate(prices_by_date=prices_by_date)}

    def evaluate(self):
        factor_files = sorted(self.factor_dir.glob("*.parquet"))

        prices_by_date = SingleFactorEvaluation.load_prices(self.config["backtest_data_dir"])
        if self.config.get("update_all") == 1:
            _clear_existing_results(self.config["output_dir"], self.config.get("visualization_output_dir"))

        # 按因子列并行，imap 保持结果顺序与因子文件、列的顺序一致。
        tasks = []
        for factor_path in factor_files:
            for factor_name in pq.read_schema(factor_path).names:
                if factor_name not in ("code", "date"):
                    tasks.append((self.config, factor_path, factor_name, prices_by_date))
        with Pool(processes=self.processes) as pool:
            records = list(tqdm(pool.imap(MultiFactorEvaluation._evaluate_column, tasks), total=len(tasks), desc="Evaluating factors", unit="factor"))

        summary = pd.DataFrame(records)
        summary.to_csv(Path(self.config["output_dir"]) / "factor_comparison.csv", index=False, encoding="utf-8-sig")
        return summary
