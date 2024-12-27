"""
Author: Kim
Date: 2024-12-27 11:48:12
LastEditTime: 2024-12-27 16:20:33
LastEditors: Kim
Description: 
FilePath: /finance/src/feature_engineering.py
"""

import pandas as pd
import numpy as np
from talib import RSI, MACD, MA


class FeatureEngineering:
    @staticmethod
    def add_technical_indicators(df):
        """添加技术指标"""
        # 确保数据类型正确
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 添加移动平均线
        df["MA5"] = MA(df["close"], timeperiod=5)
        df["MA20"] = MA(df["close"], timeperiod=20)

        # 添加RSI指标
        df["RSI"] = RSI(df["close"], timeperiod=14)

        # 添加MACD指标
        macd, signal, hist = MACD(df["close"])
        df["MACD"] = macd
        df["MACD_signal"] = signal
        df["MACD_hist"] = hist

        return df

    @staticmethod
    def create_labels(df, price_column="close", threshold=0.0005):
        """
        创建标签：计算未来收益率
        """
        print("\n标签创建详情:")
        print(f"使用的价格列: {price_column}")

        # 确保数据按时间排序
        df = df.sort_index()

        # 打印更详细的价格信息
        print("\n价格数据示例:")
        print(df[price_column].head(10))
        print("\n相邻价格点的差异:")
        print(df[price_column].diff().head(10))

        # 计算下一个时间点的收益率
        df["target"] = df[price_column].pct_change().shift(-1)

        # 检查是否有异常的收益率值
        print("\n收益率分布:")
        print(df["target"].describe())

        # 应用阈值过滤，但保留原始值用于对比
        df["target_raw"] = df["target"]
        df["target"] = df["target"].apply(lambda x: x if abs(x) > threshold else 0)

        # 打印被过滤掉的收益率信息
        filtered_returns = df[df["target_raw"] != df["target"]]
        print(f"\n被过滤掉的收益率数量: {len(filtered_returns)}")
        print("被过滤掉的收益率示例:")
        print(filtered_returns[["target_raw", "target"]].head())

        # 打印标签统计信息
        print("\n标签统计:")
        print(f"最大收益率: {df['target'].max():.4f}")
        print(f"最小收益率: {df['target'].min():.4f}")
        print(f"平均收益率: {df['target'].mean():.4f}")
        print(f"收益率标准差: {df['target'].std():.4f}")
        print(f"非零收益率数量: {(df['target'] != 0).sum()}")
        print(f"总样本数: {len(df)}")

        # 删除辅助列
        df = df.drop("target_raw", axis=1)

        return df
