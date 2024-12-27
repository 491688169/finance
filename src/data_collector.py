"""
Author: Kim
Date: 2024-12-27 11:48:02
LastEditTime: 2024-12-27 15:09:56
LastEditors: Kim
Description: 
FilePath: /finance/src/data_collector.py
"""

"""
Author: Kim
Date: 2024-12-27 11:48:02
LastEditTime: 2024-12-27 11:49:43
LastEditors: Kim
Description: 
FilePath: /finance/src/data_collector.py
"""

import baostock as bs
import pandas as pd
from datetime import datetime, timedelta


class StockDataCollector:
    def __init__(self):
        self.bs = bs
        self.bs.login()

    def __del__(self):
        self.bs.logout()

    def get_stock_data(self, stock_code, start_date, end_date, period="5"):
        """获取股票数据"""
        try:
            print(f"\n正在获取{period}分钟数据:")
            print(f"股票代码: {stock_code}")
            print(f"开始日期: {start_date}")
            print(f"结束日期: {end_date}")

            # 根据不同的时间周期设置频率
            freq_map = {"5": "5", "15": "15", "30": "30", "60": "60"}

            if period not in freq_map:
                raise ValueError(f"不支持的时间周期: {period}")

            # 获取数据
            rs = bs.query_history_k_data_plus(
                stock_code,
                "date,time,code,open,high,low,close,volume,amount,adjustflag",
                start_date=start_date,
                end_date=end_date,
                frequency=freq_map[period],
                adjustflag="2",  # 使用前复权
            )

            df = rs.get_data()

            if not df.empty:
                # 转换时间格式
                df["datetime"] = pd.to_datetime(
                    df["date"].astype(str)
                    + " "
                    + df["time"].astype(str).str[8:10]
                    + ":"
                    + df["time"].astype(str).str[10:12]
                    + ":"
                    + df["time"].astype(str).str[12:14]
                )

                # 确保数据按时间排序
                df = df.sort_values("datetime")

                print(f"\n数据时间范围:")
                print(f"开始时间: {df['datetime'].min()}")
                print(f"结束时间: {df['datetime'].max()}")
                print(f"获取到的{period}数据形状: {df.shape}")
                print(f"数据列: {df.columns.tolist()}")

                # 打印数据示例
                print(f"\n{period}分钟数据示例（前5行）:")
                print(df[["datetime", "close"]].head())

            return df

        except Exception as e:
            print(f"获取股票数据时发生错误: {e}")
            return pd.DataFrame()

    def get_fundamental_data(self, code, year, quarter):
        """获取基本面数据"""
        rs = self.bs.query_profit_data(code=code, year=year, quarter=quarter)

        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)
        return df
