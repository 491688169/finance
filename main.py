"""
Author: Kim
Date: 2024-12-27 11:48:15
LastEditTime: 2024-12-27 14:12:12
LastEditors: Kim
Description: 
FilePath: /finance/main.py
"""

from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineering
from src.model import StockPredictor
import pandas as pd
import os
import numpy as np


def main():
    # 初始化数据收集器
    collector = StockDataCollector()
    stock_code = "sh.600000"
    end_date = "2024-12-25"

    # 收集不同时间周期的数据
    timeframes = {
        "5m": {"period": "5", "days": 30},  # 一个月的5分钟数据，约 720 个数据点
        "15m": {"period": "15", "days": 45},  # 1.5个月的15分钟数据，约 360 个数据点
        "30m": {"period": "30", "days": 60},  # 2个月的30分钟数据，约 240 个数据点
        "60m": {"period": "60", "days": 90},  # 3个月的60分钟数据，约 180 个数据点
    }

    dfs = {}
    for tf, config in timeframes.items():
        # 计算开始日期
        start_date = (
            pd.to_datetime(end_date) - pd.Timedelta(days=config["days"])
        ).strftime("%Y-%m-%d")

        # 获取数据
        df = collector.get_stock_data(
            stock_code, start_date, end_date, period=config["period"]
        )
        print(f"获取到的{config['period']}数据形状:", df.shape)
        print(f"数据列:", df.columns.tolist())

        if df.empty:
            print(f"获取{config['period']}周期数据失败")
            return

        # 添加技术指标
        df = FeatureEngineering.add_technical_indicators(df)
        dfs[tf] = df

    # 合并不同时间周期的特征
    combined_features = combine_timeframe_features(dfs)

    # 在合并特征后，限制数据点数量
    combined_features = combined_features.tail(
        100
    )  # 增加到100个数据点以便进行训练测试分割

    # 创建标签
    combined_features = FeatureEngineering.create_labels(
        combined_features, price_column="close_60m"
    )
    combined_features = combined_features.dropna()

    # 定义特征列
    feature_columns = [
        f"{indicator}_{tf}"
        for tf in timeframes.keys()
        for indicator in [
            "close",
            "MA5",
            "MA20",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_hist",
        ]
    ]

    # 准备数据
    X = combined_features[feature_columns]
    y = combined_features["target"]

    # 定义回测参数
    TRAIN_SIZE = 60  # 训练窗口大小
    TEST_SIZE = 20  # 测试窗口大小
    TRANSACTION_COST = 0.0003  # 交易成本 0.03%

    # 初始化结果存储
    all_predictions = []
    all_actual_returns = []
    portfolio_value = 1.0  # 初始投资金额设为1
    positions = []  # 记录持仓状态

    # 在主函数中添加调试信息
    print(f"合并后的数据点数量: {len(combined_features)}")
    print(f"训练窗口大小: {TRAIN_SIZE}")
    print(f"测试窗口大小: {TEST_SIZE}")

    # 滚动预测和回测
    for i in range(TRAIN_SIZE, len(X) - TEST_SIZE + 1):
        # 获取训练数据
        X_train = X.iloc[i - TRAIN_SIZE : i]
        y_train = y.iloc[i - TRAIN_SIZE : i]

        # 获取测试数据
        X_test = X.iloc[i : i + 1]  # 只预测下一个时间点
        y_test = y.iloc[i : i + 1]

        # 训练模型（移除形状打印）
        predictor = StockPredictor()
        predictor.train(X_train, y_train)

        # 预测
        pred = predictor.predict(X_test)[0]
        actual = y_test.iloc[0]

        # 存储预测和实际值
        all_predictions.append(pred)
        all_actual_returns.append(actual)

        # 交易决策
        if pred > 0.001:  # 设置一阈值，避免小幅波动导致过度交易
            position = 1  # 做多
        elif pred < -0.001:
            position = -1  # 做空
        else:
            position = 0  # 不交易

        # 如果持仓发生变化，计算交易成本
        if len(positions) > 0 and position != positions[-1]:
            portfolio_value *= 1 - TRANSACTION_COST

        positions.append(position)

    # 计算回测指标
    def calculate_backtest_metrics(predictions, actual_returns, positions):
        # 添加数据检查
        if len(predictions) == 0 or len(actual_returns) == 0 or len(positions) == 0:
            print("警告：没有足够的数据进行回测")
            return {
                "prediction_accuracy": 0,
                "strategy_returns": 0,
                "market_returns": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
            }

        # 转换为numpy数组
        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)
        positions = np.array(positions)

        # 确保有足够的数据进行计算
        if len(positions) <= 1 or len(actual_returns) <= 1:
            print("警告：数据点太少，无法计算回测指标")
            return {
                "prediction_accuracy": 0,
                "strategy_returns": 0,
                "market_returns": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
            }

        # 计算策略收益（考虑持仓方向）
        strategy_returns = positions[:-1] * actual_returns[1:]  # 使用前一个持仓信号

        # 计算累积收益
        cumulative_strategy = (1 + strategy_returns).cumprod()[-1] - 1
        cumulative_market = (1 + actual_returns).cumprod()[-1] - 1

        # 添加详细的准确率计算
        pred_direction = predictions > 0
        actual_direction = actual_returns > 0
        correct_predictions = pred_direction == actual_direction
        accuracy = np.mean(correct_predictions)

        # 添加调试信息
        print("\n预测方向分布:")
        print(f"上涨预测次数: {np.sum(pred_direction)}")
        print(f"下跌预测次数: {np.sum(~pred_direction)}")
        print("\n实际方向分布:")
        print(f"实际上涨次数: {np.sum(actual_direction)}")
        print(f"实际下跌次数: {np.sum(~actual_direction)}")
        print(f"正确预测次数: {np.sum(correct_predictions)}")
        print(f"总预测次数: {len(predictions)}")

        # 计算夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        excess_returns = strategy_returns - risk_free_rate / 252  # 假设252个交易日
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

        # 计算最大回撤
        def calculate_max_drawdown(returns):
            cumulative = (1 + pd.Series(returns)).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()

        max_drawdown = calculate_max_drawdown(strategy_returns)

        return {
            "prediction_accuracy": accuracy,
            "strategy_returns": cumulative_strategy,
            "market_returns": cumulative_market,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    # 计算并打印回测结果
    metrics = calculate_backtest_metrics(all_predictions, all_actual_returns, positions)

    print("\n====== 回测结果 ======")
    print(f"预测准确率: {metrics['prediction_accuracy']:.2%}")
    print(f"策略累积收益: {metrics['strategy_returns']:.2%}")
    print(f"市场累积收益: {metrics['market_returns']:.2%}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"考虑交易成本后的最终收益: {(portfolio_value - 1):.2%}")

    # 打印最后20个交易点的预测结果
    print("\n最后20个交易点的预测结果：")
    for i in range(-20, 0):
        print(
            f"时间点 {len(all_predictions)+i+1}: "
            f"预测={all_predictions[i]*100:.2f}%, "
            f"实际={all_actual_returns[i]*100:.2f}%, "
            f"持仓={'多头' if positions[i]==1 else '空头' if positions[i]==-1 else '空仓'} "
            f"{'✓' if (all_predictions[i] > 0) == (all_actual_returns[i] > 0) else '✗'}"
        )

    # ... 后续预测和展示代码 ...


def combine_timeframe_features(dfs):
    """
    合并不同时周期的特征
    """
    # 首先处理5分钟数据作为基准
    base_df = None
    for tf, df in sorted(dfs.items(), key=lambda x: int(x[0][:-1])):  # 按时间周期排序
        print(f"\n处理 {tf} 数据:")
        print(f"原始数据形状: {df.shape}")

        # 设置时间索引
        df.set_index("datetime", inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"去重后数据形状: {df.shape}")

        # 选择需要的列并重命名
        needed_columns = {
            "close": f"close_{tf}",
            "MA5": f"MA5_{tf}",
            "MA20": f"MA20_{tf}",
            "RSI": f"RSI_{tf}",
            "MACD": f"MACD_{tf}",
            "MACD_signal": f"MACD_signal_{tf}",
            "MACD_hist": f"MACD_hist_{tf}",
        }
        df_selected = df[needed_columns.keys()].rename(columns=needed_columns)

        # 如果是第一个时间周期（5分钟），直接作为基准
        if base_df is None:
            base_df = df_selected
            print("\n5分钟数据示例:")
            print(base_df.head())
            continue

        # 对于其他时间周期，保持原始时间间隔
        period_minutes = int(tf[:-1])
        print(f"\n{tf} 原始数据示例:")
        print(df_selected.head())

        # 合并到基准数据框
        base_df = pd.merge(
            base_df, df_selected, left_index=True, right_index=True, how="left"
        )

        # 向前填充，但只填充到下一个实际数据点
        for col in df_selected.columns:
            # 使用 ffill() 替代 fillna(method='ffill')
            base_df[col] = base_df[col].ffill(limit=period_minutes // 5 - 1)

    # 删除任何含有缺失值的行
    base_df = base_df.dropna()
    print(f"\n最终数据形状: {base_df.shape}")

    # 添加调试信息
    print("\n数据合并详情:")
    print(f"合并后的数据点数���: {len(base_df)}")
    print("\n各时间周期的收盘价示例:")

    # 修改采样逻辑
    for col in sorted(
        [c for c in base_df.columns if c.startswith("close_")],
        key=lambda x: int(x.split("_")[1][:-1]),
    ):
        period = int(col.split("_")[1][:-1])  # 提取时间周期（分钟）
        print(f"\n{col} 数据示例 (每{period}分钟一个点):")

        # 使用 resample 进行规范的时间序列重采样
        sample_data = base_df[col].resample(f"{period}T").last()
        # 只显示交易时段的数据（9:30-11:30, 13:00-15:00）
        sample_data = sample_data[
            (
                sample_data.index.hour.isin([9, 10, 11, 13, 14])
                & ((sample_data.index.hour != 9) | (sample_data.index.minute >= 30))
                & ((sample_data.index.hour != 11) | (sample_data.index.minute <= 30))
                & ((sample_data.index.hour != 15) | (sample_data.index.minute <= 0))
            )
        ]
        print(sample_data.head())

        print(f"{col} 的实际变化次数: {(base_df[col].diff() != 0).sum()}")
        print(f"{col} 的唯一值数量: {base_df[col].nunique()}")

    return base_df


if __name__ == "__main__":
    main()
