import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

CONFIG_FILE = "portfolio_20260905/config/config_factor_construction.json"
CONFIG_LOCAL_FILE = Path(__file__).resolve().parent / "config_local.json"

# test of git repository

def load_config(config_file=CONFIG_FILE):
    """根据 config_local.json 中的根目录，将配置的 *_dir / *_path 转为本地路径。"""
    config_local = json.loads(CONFIG_LOCAL_FILE.read_text(encoding="utf-8"))
    root_dir = Path(config_local["root_dir"]).expanduser()
    config_path = root_dir / config_file
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for key, value in config.items():
        if key.endswith(("_dir", "_path")):
            config[key] = str(root_dir / value)
    return config


if __name__ == "__main__":
    from update_data.update_data import DataUpdater
    from factors.construction.high_freq_factor_construction import HighFreqFactorConstructor
    from factors.evaluation.single_factor_evaluation import SingleFactorEvaluation
    from factors.evaluation.multi_factor_evaluation import MultiFactorEvaluation

    config = load_config()

    config_name = Path(CONFIG_FILE).name

    if config_name == "config_data_update.json":
        data_updater = DataUpdater(config)
        data_updater.integrate_data()
        data_updater.generate_backtest_data()

    if config_name == "config_factor_construction.json":
        data_constructor = HighFreqFactorConstructor(config)
        data_constructor.update_factors()

    if config_name == "config_factor_evaluation.json":
        evaluator_class = MultiFactorEvaluation if config.get("evaluation_mode", "single") == "multi" else SingleFactorEvaluation
        data_evaluator = evaluator_class(config)
        data_evaluator.evaluate()
