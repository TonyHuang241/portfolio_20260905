from pathlib import Path

import pandas as pd
from tqdm import tqdm

from factors.evaluation.single_factor_evaluation import SingleFactorEvaluation, _clear_existing_results


class MultiFactorEvaluation:
    """遍历因子目录中的 Parquet 文件，评估所有原始及衍生因子列。"""
    def __init__(self, config):
        self.config = config
        self.factor_dir = Path(config["factor_data_dir"])

    def evaluate(self):
        factor_files = sorted(self.factor_dir.glob("*.parquet"))

        prices_by_date = SingleFactorEvaluation.load_prices(self.config["backtest_data_dir"])
        records = []
        if self.config.get("update_all") == 1:
            _clear_existing_results(self.config["output_dir"], self.config.get("visualization_output_dir"))

        for factor_path in tqdm(factor_files, desc="Evaluating factor files", unit="file"):
            factor_data = pd.read_parquet(factor_path)
            for factor_name in factor_data.columns.drop(["code", "date"]):
                config = {**self.config, "specified_column": factor_name}
                evaluator = SingleFactorEvaluation(config, factor_data)
                records.append({"因子文件": factor_path.name, **evaluator.evaluate(prices_by_date=prices_by_date)})

        summary = pd.DataFrame(records)
        summary.to_csv(Path(self.config["output_dir"]) / "factor_comparison.csv", index=False, encoding="utf-8-sig")
        return summary
