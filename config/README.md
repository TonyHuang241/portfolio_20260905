# 本地路径配置

首次使用时，将项目根目录的 `config_local.example.json` 复制为同目录的 `config_local.json`，将 `root_dir` 改为两个文件夹共同父目录的绝对路径。`config_local.json` 已加入 `.gitignore`，示例文件可提交到 Git。

例如，项目位于 `/mnt/f/portfolio_20260905`，行情数据位于 `/mnt/f/a_share_market_data` 时，只需配置：

```json
{
    "root_dir": "/mnt/f"
}
```

其他配置中，`_dir` 或 `_path` 结尾的字段使用相对于 `root_dir` 的路径，包含 `portfolio_20260905` 或 `a_share_market_data` 文件夹名称，并统一使用 `/` 分隔。例如，`a_share_market_data/stock_minutes/` 会拼接为 `/mnt/f/a_share_market_data/stock_minutes`，`portfolio_20260905/factors/construction/factor_calculation/` 会拼接为项目中的因子计算目录。

在 `main.py` 的 `CONFIG_FILE` 中选择要运行的配置，例如 `portfolio_20260905/config/config_data_update.json`。运行时，`load_config()` 会先读取与 `main.py` 同目录的 `config_local.json`，再加载并转换所选配置；日期、计算模式等非路径参数保持原值。相对的配置文件路径也以 `root_dir` 为基准，不依赖启动命令所在目录。

独立运行可视化脚本时也会使用相同的配置加载函数。更换机器后，保持两个文件夹在同一父目录下，只需修改 `config_local.json` 中的 `root_dir`。
