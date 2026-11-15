from factors.construction.high_freq_factor_construction import HighFreqFactorConstructor
from factors.construction.middle_freq_factor_construction import MiddleFreqFactorConstructor


def run(config, construction_mode):
    """main.py 调用的唯一入口：construction_mode 取 high（高频）或 middle（中频），按 config 构建对应因子。"""
    constructor_class = {"high": HighFreqFactorConstructor, "middle": MiddleFreqFactorConstructor}[construction_mode]
    constructor_class(config).update_factors()


if __name__ == "__main__":
    from main import load_config
    config = load_config("portfolio_20260905/config/config_factor_construction.json")
    run(config, config["construction_mode"])
