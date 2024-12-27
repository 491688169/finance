"""
Author: Kim
Date: 2024-12-27 11:48:12
LastEditTime: 2024-12-27 14:24:17
LastEditors: Kim
Description: 
FilePath: /finance/src/model.py
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib


class StockPredictor:
    def __init__(self, model_type="rf"):
        self.model_type = model_type
        self.model = None

    def create_model(self):
        """创建模型"""
        if self.model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif self.model_type == "xgb":
            self.model = xgb.XGBRegressor(
                max_depth=10, learning_rate=0.1, n_estimators=100
            )
        elif self.model_type == "lgb":
            self.model = lgb.LGBMRegressor(
                max_depth=10, learning_rate=0.1, n_estimators=100
            )

    def train(self, X, y):
        """训练模型"""
        if self.model is None:
            self.create_model()
        self.model.fit(X, y)

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def save_model(self, path):
        """保存模型"""
        joblib.dump(self.model, path)

    def load_model(self, path):
        """加载模型"""
        self.model = joblib.load(path)
