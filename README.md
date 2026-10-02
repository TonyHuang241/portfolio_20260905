# A 股高频因子研究

基于股票分钟行情的因子研究项目，支持数据整理、高频因子构建、单因子及批量因子评估，并生成 HTML 可视化报告。

## 项目结构

```text
main.py                         # 运行入口与配置加载
config/                         # 数据更新、因子构建、因子评估配置
update_data/                    # 整理行情并生成回测数据
construct_asset_pool/           # 股票基础信息与交易日历下载
factors/construction/           # 因子构建与具体计算逻辑
factors/evaluation/             # 因子评估与报告生成
factor_list.csv                 # 因子清单
config_local.example.json       # 本地路径配置示例
```

## 运行方式

在 `main.py` 中修改 `CONFIG_FILE`，选择要执行的任务，然后运行：

```bash
python main.py
```

| 任务 | `CONFIG_FILE` |
| --- | --- |
| 数据更新 | `portfolio_20260905/config/config_data_update.json` |
| 因子构建 | `portfolio_20260905/config/config_factor_construction.json` |
| 因子评估（默认） | `portfolio_20260905/config/config_factor_evaluation.json` |

首次使用按数据更新、因子构建、因子评估的顺序执行；已有对应数据时可直接运行后续步骤。行情数据需自行准备，不包含在仓库中。

- **数据更新**：从 `new_data_dir` 读取 `daily_minutes.csv`、`stock_exrights.csv`、`limit_price.csv`、`stock_st_status.csv` 和 `mkcap.csv`，整理数据并生成 `stock_daily/daily_stock_data_10am.parquet`（10 点行情）和 `stock_daily/daily_stock_data.parquet`（日频 OHLC，并按 `code`、`date` 合并当日总市值列 `mkcap`，单位为元，无市值时为空）。
- **因子构建**：`construction_mode` 选择 `high`（高频）或 `middle`（中频），省略时默认高频。高频通过 `factor_list` 选择因子，中频通过 `middle_factor_list` 选择，空列表表示对应目录全部因子。可设置输出日期范围，高频支持 `processes` 并行进程数。结果保存到配置的 `output_dir`。
- **中频计算**：一次读取 `stock_daily/daily_stock_data.parquet`，将最早待更新日期之前 252 个市场交易日再加因子回看期（`Mom1m` 为 20 日，共 272 日）起至数据最新日期的日线交给因子类；历史不足时从最早可用日期开始。先按股票计算 5、20、60、120、250 日完整窗口滚动均值（列名如 `Mom1m_250_m`），再筛选待更新日期和股票池；有效历史不足时均值为空。增量更新按已有日期跳过计算，并保留已有均值列。`Mom1m` 为 `close / close_20日前 - 1`（上涨为正，下跌为负），直接使用输入 `close`，不额外复权；端点价格缺失或非正时结果为空。
- **因子评估**：`evaluation_mode` 为 `multi` 时遍历因子目录中的全部因子列；为 `single` 时需将 `specified_column` 设为待评估的因子列名。`group_number` 设置每日按因子值等分的组数（默认 10）。输出包含分组收益、IC、Rank IC 等指标及 HTML 报告，批量模式额外生成 `factor_comparison.csv`。

评估同时在 `output_dir` 的同级目录 `stock_list/` 导出多头股票清单，单因子与批量模式均适用，每次覆盖本次评估策略的文件：

| 文件 | 内容 | 字段 |
| --- | --- | --- |
| `stock_list/<因子名>.csv` | 从 `start_date` 起的历史每日持仓，包含收益尚未兑现的最新交易日 | `date, code, group, factor_value` |
| `stock_list/next_day/<因子名>.csv` | 最新交易日收盘后计算的下一交易日目标持仓，仅保存最新一期 | `signal_date, code, group, factor_value` |

每行一只股票，按日期、股票代码排序；`date` 是持仓交易日，`signal_date` 是收盘信号日期，`factor_value` 是该次分组实际使用的因子值。历史持仓使用滞后一期的因子，下一日目标使用最新收盘因子。最新一期无有效样本时，目标文件仅保留表头，不回退到旧信号。默认输出位置为 `a_share_market_data/factors_evaluation_output/stock_list/`。

两类清单均按六位股票代码匹配 `a_share_market_data/__daily_data_update/stock_info.csv`，追加 `name`、`industry`、`csrc_industry`、`region`、`concept` 等信息；未匹配到的持仓保留，信息列留空。历史持仓匹配的是该文件中的最新信息，不是历史时点的股票资料。

多头方向与完整区间报告一致：IC 均值大于 0 取因子值最高组（10 组时为 G10），否则取 G1；历史清单是按当前报告方向还原的模型持仓，并非实盘成交记录。下一日目标按信号日所属月份的市值规则筛选股票，不依赖未来行情或交易日历；若下一交易日跨月，需按新月份股票池重新确认。CSV 日期采用 `YYYYMMDD`，股票代码按文本读取以保留前导零。

因子构建中的 `update_all` 设为 `1` 会重新计算并覆盖所选因子文件，保留其他因子文件；因子评估设为 `1` 会清空相应评估输出目录。设为 `0` 则执行增量更新。运行前请确认配置的输入、输出路径和日期范围。
