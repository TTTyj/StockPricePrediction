# Stock Price Prediction and Trading Strategy

This project implements machine learning models to predict stock price movements and develop trading strategies, using Xiaomi Group (1810.HK) as an example.

## Project Structure

```
.
├── README.md
├── data.py              # Data processing and visualization
├── ml_model.py          # Random Forest model implementation
└── logistic_model.py    # Logistic Regression model implementation
```

## Features

- Data Processing:
  - Download historical stock data
  - Calculate technical indicators (MA, RSI, MACD, etc.)
  - Data preprocessing and feature engineering

- Machine Learning Models:
  - Random Forest Classifier
  - Logistic Regression
  - Feature importance analysis
  - Model performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)

- Trading Strategy:
  - Probability-based trading signals
  - Strategy backtesting
  - Performance metrics (Returns, Maximum Drawdown)
  - Strategy visualization

## Generated Files

- `roc_curve.png`: ROC curve showing model classification performance
- `logistic_feature_importance.png`: Feature importance analysis for logistic regression
- `logistic_strategy_performance.png`: Trading strategy performance visualization

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - yfinance

## Usage
0. clone code：
```bash
git clone https://github.com/TTTyj/StockPricePrediction.git
```
1. Activate virtual environment:
```bash
source venv/bin/activate && which python
```

2. Run data processing and visualization:
```bash
python data.py
```

3. Run Random Forest model:
```bash
python ml_model.py
```

4. Run Logistic Regression model:
```bash
python logistic_model.py
```

5. Deactivate virtual environment:
```bash
deactivate
```

# 股票价格预测与交易策略

本项目使用机器学习模型对股票价格走势进行预测并开发交易策略，以小米集团（1810.HK）为例。

## 项目结构

```
.
├── README.md
├── data.py              # 数据处理与可视化
├── ml_model.py          # 随机森林模型实现
└── logistic_model.py    # 逻辑回归模型实现
```

## 功能特点

- 数据处理：
  - 下载历史股票数据
  - 计算技术指标（均线、RSI、MACD等）
  - 数据预处理与特征工程

- 机器学习模型：
  - 随机森林分类器
  - 逻辑回归
  - 特征重要性分析
  - 模型性能评估（准确率、精确率、召回率、F1值、ROC-AUC）

- 交易策略：
  - 基于概率的交易信号
  - 策略回测
  - 性能指标（收益率、最大回撤）
  - 策略可视化

## 生成文件

- `roc_curve.png`：显示模型分类性能的ROC曲线
- `logistic_feature_importance.png`：逻辑回归的特征重要性分析
- `logistic_strategy_performance.png`：交易策略表现可视化

## 环境要求

- Python 3.x
- 依赖包：
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - yfinance

## 使用方法
1. 克隆项目：
git clone https://github.com/TTTyj/StockPricePrediction.git

2. 激活虚拟环境：
source venv/bin/activate && which python

3. 运行数据处理与可视化：
```bash
python data.py
```

4. 运行随机森林模型：
```bash
python ml_model.py
```

5. 运行逻辑回归模型：
```bash
python logistic_model.py
``` 
6. 退出虚拟环境：
```bash
deactivate
``` 