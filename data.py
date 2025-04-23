import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

np.random.seed(42)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    # 重置列名，移除多重索引
    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    return data

def add_indicators(data):
    """添加技术指标"""
    # Calculate daily returns
    data['Return'] = data['Close'].pct_change()

    # Create additional features
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Return'].rolling(window=50).std()
    
    # 添加更多技术指标
    # RSI - 相对强弱指标
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD - 移动平均收敛散度
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

def plot_technical_analysis(data):
    """绘制技术分析图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 设置图表大小
    plt.rcParams['figure.figsize'] = (15, 10)

    # 1. 股价走势图（包含移动平均线）
    plt.figure(figsize=(15, 10))
    plt.plot(data.index, data['Close'], label='收盘价', alpha=0.8)
    plt.plot(data.index, data['MA_50'], label='50日移动平均线', alpha=0.8)
    plt.plot(data.index, data['MA_200'], label='200日移动平均线', alpha=0.8)
    plt.title('小米集团 (1810.HK) 股价走势图', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('股价 (港币)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_price.png')
    plt.close()

    # 2. 成交量图
    plt.figure(figsize=(15, 5))
    plt.bar(data.index, data['Volume'], alpha=0.8, color='blue')
    plt.title('小米集团 (1810.HK) 成交量', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('成交量', fontsize=12)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('volume.png')
    plt.close()

    # 3. 收益率分布图
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Return'].dropna(), bins=100, kde=True)
    plt.title('小米集团 (1810.HK) 日收益率分布', fontsize=14)
    plt.xlabel('日收益率', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('returns_distribution.png')
    plt.close()

    # 4. 波动率走势图
    plt.figure(figsize=(15, 5))
    plt.plot(data.index, data['Volatility'], color='red', alpha=0.8)
    plt.title('小米集团 (1810.HK) 50日波动率走势', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('波动率', fontsize=12)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('volatility.png')
    plt.close()

    # 5. RSI指标图
    plt.figure(figsize=(15, 5))
    plt.plot(data.index, data['RSI'], color='purple', alpha=0.8)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.title('小米集团 (1810.HK) RSI指标', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('RSI', fontsize=12)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rsi.png')
    plt.close()

    # 6. MACD指标图
    plt.figure(figsize=(15, 5))
    plt.plot(data.index, data['MACD'], label='MACD', color='blue', alpha=0.8)
    plt.plot(data.index, data['Signal_Line'], label='Signal Line', color='orange', alpha=0.8)
    plt.title('小米集团 (1810.HK) MACD指标', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('MACD', fontsize=12)
    plt.legend(fontsize=10)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('macd.png')
    plt.close()

def main():
    # 小米集团的股票代码是 1810.HK
    start_date = '2018-07-09'  # 小米集团上市日期
    end_date = '2024-03-19'    # 今天
    data = download_data('1810.HK', start_date, end_date)
    data = add_indicators(data)  # 添加技术指标到原始数据

    # 打印数据信息
    print("\n=== 小米集团(1810.HK)数据信息 ===")
    print(data.info())
    print("\n=== 数据前5行 ===")
    print(data.head())
    print("\n=== 数据统计信息 ===")
    print(data.describe())

    # 绘制技术分析图表
    plot_technical_analysis(data)
    
    print("\n=== 图表生成完成 ===")
    print("已生成以下图表文件：")
    print("1. stock_price.png - 股价走势图（含移动平均线）")
    print("2. volume.png - 成交量图")
    print("3. returns_distribution.png - 收益率分布图")
    print("4. volatility.png - 波动率走势图")
    print("5. rsi.png - RSI指标图")
    print("6. macd.png - MACD指标图")

if __name__ == "__main__":
    main()


