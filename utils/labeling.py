import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_future_price(df, delta=30):
    df.loc[:, [f'Price_t_plus_{delta}']] = df['Close'].shift(-delta)
    return df

def calculate_threshold(df, delta=30):
    """
    Return: delta, threshold
    """
    df = get_future_price(df, delta)

    # Tính tỷ lệ thay đổi giá
    df.loc[:, ['Return']] = df[f'Price_t_plus_{delta}'] / df['Close'] - 1

    # Loại bỏ giá trị NaN (do shift)
    df = df.dropna()

    # Tách return thành tăng (dương) và giảm (âm)
    positive_returns = df['Return'][df['Return'] > 0]  # Tăng giá

    std_positive = positive_returns.std()

    print(f'Delta = {delta}')
    
    # Đề xuất threshold 
    threshold = std_positive * 0.45 # 1.5 lần độ lệch chuẩn cho BUY
    
    print(f"Threshold gợi ý: {threshold:.4%}")
    
    # Tính giá trị tiền dựa trên giá trung bình trong df
    avg_price = df['Close'].mean()
    print(f"\nGiá trung bình trong dữ liệu: ${avg_price:.2f}")

    # Tính giá trị tiền
    threshold_in_money = avg_price * threshold

    print(f"Threshold ({threshold}) thành tiền: ${threshold_in_money:.2f}")
    
    return delta, threshold
    
# Simple labeling
def simple_labeling(df, delta, threshold):
    """
    Return: labelled dataframe
    """
    df.loc[:, [f'Price_t_plus_{delta}']] = df['Close'].shift(-delta)

    # Tính tỷ lệ thay đổi giá
    df.loc[:, ['Return']] = df[f'Price_t_plus_{delta}'] / df['Close'] - 1

    # Loại bỏ giá trị NaN (do shift)
    df = df.dropna()
    
    # Kiểm tra phân bố nhãn với threshold riêng
    df.loc[:, ['Label']] = np.where(df[f'Price_t_plus_{delta}'] > df['Close'] * (1 + threshold), 'BUY',
                        np.where(df[f'Price_t_plus_{delta}'] < df['Close'] * (1 - threshold), 'SELL', 'HOLD'))

    label_distribution = df['Label'].value_counts(normalize=True)
    print("\nPhân bố nhãn với threshold riêng:")
    print(label_distribution)
    
    return df

def dynamic_labeling(df, delta, threshold):
    """
    Return: labelled dataframe
    """
    df = df.copy()
    
    # Tính sẵn min/max trong cửa sổ kế tiếp delta dòng => Cực trị
    # Dịch ngược DataFrame, tính rolling min/max, rồi dịch đúng thứ tự
    reversed_close = df['Close'].iloc[::-1]
    rolling_min = reversed_close.rolling(window=delta, min_periods=1).min().iloc[::-1].shift(1)
    rolling_max = reversed_close.rolling(window=delta, min_periods=1).max().iloc[::-1].shift(1)
    
    # Bước gán nhãn ban đầu
    df = simple_labeling(df, delta, threshold)

    # Áp dụng min/max tương ứng từng hàng
    df['future_min'] = rolling_min
    df['future_max'] = rolling_max

    # Cập nhật nhãn theo giá cực trị
    df.loc[(df['Label'] == 'BUY') & (df['future_min'] < df['Close'] * (1 - threshold)), 'Label'] = 'SELL'
    df.loc[(df['Label'] == 'SELL') & (df['future_max'] > df['Close'] * (1 + threshold)), 'Label'] = 'BUY'

    label_distribution = df['Label'].value_counts(normalize=True)
    print("\nPhân bố nhãn sau cập nhật:")
    print(label_distribution)

    return df
