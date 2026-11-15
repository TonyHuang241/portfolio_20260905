import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from update_data import update_data
from factors.construction import factors
from factors.evaluation import evaluation as factor_evaluation
from trading_strategies import evaluation as strategy_evaluation

CONFIG_FILE = "portfolio_20260905/config/config_factor_evaluation.json"
CONFIG_LOCAL_FILE = Path(__file__).resolve().parent / "config_local.json"

def load_config(config_file=CONFIG_FILE):
    """根据 config_local.json 中的根目录，将配置的 *_dir / *_path 转为本地路径。"""
    config_local = json.loads(CONFIG_LOCAL_FILE.read_text(encoding="utf-8"))
    root_dir = Path(config_local["root_dir"]).expanduser()
    config_path = root_dir / config_file
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for key, value in config.items():
        if key.endswith(("_dir", "_path")) and value:
            config[key] = str(root_dir / value)
    return config


if __name__ == "__main__":
    config = load_config()

    config_name = Path(CONFIG_FILE).name

    if config_name == "config_data_update.json":
        update_data.run(config)

    if config_name == "config_factor_construction.json":
        factors.run(config, config["construction_mode"])

    if config_name == "config_factor_evaluation.json":
        factor_evaluation.run(config, config["evaluation_mode"])

    if config_name == "config_trading_strategies.json":
        strategy_evaluation.run(config)
