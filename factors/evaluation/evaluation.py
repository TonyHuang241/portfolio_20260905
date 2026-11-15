from factors.evaluation.single_factor_evaluation_v1 import SingleFactorEvaluation
from factors.evaluation.multi_factor_evaluation import MultiFactorEvaluation


def run(config, evaluation_mode):
    """main.py 调用的唯一入口：evaluation_mode 取 single（只评估 specified_column 指定的因子）或 multi（评估因子目录下的全部因子），按 config 评估因子；两种模式都用已有结果生成同一份 HTML 报告。"""
    evaluator_class = {"single": SingleFactorEvaluation, "multi": MultiFactorEvaluation}[evaluation_mode]
    evaluator_class(config).evaluate()


if __name__ == "__main__":
    from main import load_config
    config = load_config("portfolio_20260905/config/config_factor_evaluation.json")
    run(config, config["evaluation_mode"])
