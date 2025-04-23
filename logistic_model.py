import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from data import download_data, add_indicators

def preprocess_data(data):
    """预处理数据并创建分类标签"""
    # 移除缺失值
    data = data.dropna()
    
    # 创建分类标签：1表示上涨（收益率>0），0表示下跌（收益率<=0）
    data['Target'] = (data['Return'] > 0).astype(int)
    
    return data

def prepare_features(X_train, X_test):
    """特征标准化"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """训练逻辑回归模型"""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """评估分类模型性能"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
    
    # 计算基础指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return accuracy, precision, recall, f1, y_pred, roc_auc

def plot_feature_importance(model, features):
    """绘制特征重要性图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取特征系数的绝对值作为重要性指标
    importance = np.abs(model.coef_[0])
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title('特征重要性分析 (系数绝对值)', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('重要性', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('logistic_feature_importance.png')
    plt.close()
    
    return feature_importance

def backtest_strategy(model, X_test, y_test, returns, threshold=0.5):
    """回测策略"""
    initial_balance = 10000  # 初始资金
    balance = initial_balance
    positions = []  # 记录每次交易后的资金状况
    
    # 获取预测概率
    y_prob = model.predict_proba(X_test)[:, 1]
    
    for i in range(len(y_prob)):
        if y_prob[i] > threshold:  # 只有当模型预测上涨的概率大于阈值时才做多
            balance *= (1 + returns.iloc[i])
        positions.append(balance)
    
    # 计算策略收益率
    strategy_returns = (balance - initial_balance) / initial_balance
    
    # 计算最大回撤
    cummax = pd.Series(positions).cummax()
    drawdown = (cummax - positions) / cummax
    max_drawdown = drawdown.max()
    
    # 绘制资金曲线
    plt.figure(figsize=(12, 6))
    plt.plot(positions, label='策略资金曲线')
    plt.title('逻辑回归策略资金曲线')
    plt.xlabel('交易日')
    plt.ylabel('资金')
    plt.legend()
    plt.tight_layout()
    plt.savefig('logistic_strategy_performance.png')
    plt.close()
    
    return strategy_returns, max_drawdown, positions

def main():
    # 下载和处理数据
    start_date = '2018-07-09'  # 小米集团上市日期
    end_date = '2024-03-19'    # 今天
    data = download_data('1810.HK', start_date, end_date)
    data = add_indicators(data)  # 添加技术指标到原始数据
    data = preprocess_data(data)  # 预处理数据
    
    # 准备特征和目标变量
    features = ['Close', 'MA_50', 'MA_200', 'Volatility', 'RSI', 'MACD', 'Signal_Line']
    X = data[features]
    y = data['Target']  # 使用分类标签
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征标准化
    X_train_scaled, X_test_scaled, scaler = prepare_features(X_train, X_test)
    
    # 训练模型
    model = train_model(X_train_scaled, y_train)
    
    # 评估模型
    accuracy, precision, recall, f1, y_pred, roc_auc = evaluate_model(model, X_test_scaled, y_test)
    
    print("\n=== 逻辑回归模型评估结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # 分析特征重要性
    feature_importance = plot_feature_importance(model, features)
    print("\n=== 特征重要性 ===")
    print(feature_importance)
    
    # 回测策略
    returns = data.loc[X_test.index, 'Return']  # 获取测试集对应的收益率
    strategy_returns, max_drawdown, positions = backtest_strategy(model, X_test_scaled, y_test, returns, threshold=0.6)
    print(f"\n=== 策略回测结果 ===")
    print(f"策略总收益率: {strategy_returns:.4f}")
    print(f"最大回撤: {max_drawdown:.4f}")

if __name__ == "__main__":
    main() 