from datetime import datetime
from pathlib import Path
from time import sleep

import pandas as pd
import requests
from tqdm import tqdm


class AssetInfoDownloader:
    def __init__(self, config: dict):
        self.output_path = Path(config["asset_info_output_dir"])
        self.start_date = config.get("start_date", "20100101")

        self.output_path.mkdir(parents=True, exist_ok=True)

    def _fetch_data(self, data_type: str, **params) -> pd.DataFrame:
        headers = {"X-API-Key": "tsr_1FjRkziz3M7m0aLcTk0ZgnK03__xO3EYq0ZdwQqdwSE"}
        sleep(0.35)
        response = requests.get(f"https://pcd.mobcvb.cn/tushare/pro/{data_type}", headers=headers, params=params, timeout=30)
        if not response.ok:
            raise RuntimeError(f"{data_type} 参数={params}，HTTP={response.status_code}，响应={response.text}")
        body = response.json()
        if body.get("ok") is False or body.get("code") != 0:
            raise RuntimeError(f"{data_type} 请求失败：{body.get('message') or body.get('msg') or str(body)}")
        data = body["data"]
        return pd.DataFrame(data["items"], columns=data["fields"])

    def download(self):
        start_date = datetime.strptime(str(self.start_date), "%Y%m%d").strftime("%Y%m%d")
        end_date = datetime.today().strftime("%Y%m%d")

        with tqdm(total=2, desc="数据下载", unit="步") as progress:
            stocks = self._fetch_data("stock_basic", list_status="L")
            stocks.drop_duplicates("ts_code").sort_values("ts_code").to_csv(self.output_path / "asset_info.csv", index=False, encoding="utf-8-sig")
            progress.update(1)
            tqdm.write("股票基础资料已保存：asset_info.csv")

            # companies = pd.concat([self._fetch_data("stock_company", exchange=exchange) for exchange in ("SSE", "SZSE", "BSE")], ignore_index=True)
            # companies.drop_duplicates("ts_code").sort_values("ts_code").to_csv(self.output_path / "stock_company.csv", index=False, encoding="utf-8-sig")
            # progress.update(1)
            # tqdm.write("上市公司资料已保存：stock_company.csv")

            calendar = pd.concat([self._fetch_data("trade_cal", exchange=exchange, start_date=start_date, end_date=end_date) for exchange in ("SSE", "SZSE")], ignore_index=True)
            calendar = calendar.drop_duplicates(["exchange", "cal_date"]).sort_values(["cal_date", "exchange"])
            calendar.to_csv(self.output_path / "trade_cal.csv", index=False, encoding="utf-8-sig")
            progress.update(1)
            tqdm.write("交易日历已保存：trade_cal.csv")
