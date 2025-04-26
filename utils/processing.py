import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator

from tqdm import tqdm


label_mapping = {"BUY": 0, "SELL": 1, "HOLD": 2}


def map_label(x):
    return label_mapping[x] if x in label_mapping else x


# 1. Hàm load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Xử lý datetime
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("Datetime").drop(["Date", "Time"], axis=1)
    return df


# 2. Hàm thêm basic features
def add_basic_features(df):
    df["Price_Spread"] = df["High"] - df["Low"]
    df["Price_Change"] = df["Close"] - df["Open"]
    df["Body_Ratio"] = (df["Close"] - df["Open"]) / (df["Price_Spread"] + 1e-8)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["Cumulative_Return_5D"] = (
        np.exp(df["Log_Return"].rolling(5).sum()) - 1
    )  # Lũy kế 5 ngày

    return df


# 3. Hàm thêm technical indicators
def add_technical_indicators(df):
    # # Sử dụng thư viện 'ta'
    # df = add_all_ta_features(
    #     df, open="Open", high="High", low="Low", close="Close", volume="Volume"
    # )

    # Volume-Weighted Momentum
    df["VW_Momentum"] = (
        (df["Volume"] * (df["Close"] - df["Close"].shift(1))).rolling(5).sum()
    )

    # On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

    # Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["Stoch_%K"] = stoch.stoch()
    df["Stoch_%D"] = stoch.stoch_signal()

    # Williams %R
    williams = WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14
    )
    df["Williams_%R"] = williams.williams_r()

    df = df.copy()
    # Average True Range (ATR)
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ATR"] = atr.average_true_range()

    indic_cols = {}
    # Thêm indicators custom
    rsi = RSIIndicator(close=df["Close"], window=14)
    indic_cols["RSI_14"] = rsi.rsi()

    macd = MACD(close=df["Close"])
    indic_cols["MACD"] = macd.macd()
    indic_cols["MACD_Signal"] = macd.macd_signal()

    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    indic_cols["BB_Upper"] = bb.bollinger_hband()
    indic_cols["BB_Lower"] = bb.bollinger_lband()

    df = pd.concat([df, pd.DataFrame(indic_cols)], axis=1)
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # Kết hợp RSI và MACD
    df["RSI_MACD_Interaction"] = df["RSI_14"] * df["MACD_Histogram"]

    # Tương quan Volume-Giá
    df["Volume_Price_Correlation"] = df["Volume"].rolling(10).corr(df["Close"])

    # ADX để xác định thị trường có trend hay range
    adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ADX"] = adx.adx()
    df["Market_Trend"] = (df["ADX"] > 25).astype(int)  # >25: có xu hướng

    return df


# 4. Hàm thêm statistical features
def add_statistical_features(df, window=20):
    new_cols = {
        "Rolling_Mean": df["Close"].rolling(window).mean(),
        "Rolling_Std": df["Close"].rolling(window).std(),
        "Rolling_Max": df["High"].rolling(window).max(),
        "Rolling_Min": df["Low"].rolling(window).min(),
    }
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # Biến động "nén" trong ngắn hạn
    df["Volatility_Clustering"] = df["Rolling_Std"] / df["Rolling_Std"].shift(5)

    df["Rolling_Skew"] = df["Close"].rolling(window).skew()
    df["Rolling_Kurtosis"] = df["Close"].rolling(window).kurt()
    df["Rolling_Q80"] = df["Close"].rolling(window).quantile(0.8)
    df["Rolling_Q20"] = df["Close"].rolling(window).quantile(0.2)
    df["Quantile_Spread"] = df["Rolling_Q80"] - df["Rolling_Q20"]  # Độ phân tán phân vị

    return df


# 5. Hàm thêm time-based features
def add_time_features(df):
    new_cols = {
        "Hour": df["Datetime"].dt.hour,
        "DayOfWeek": df["Datetime"].dt.dayofweek,  # 0=Monday
    }

    # Cyclical encoding cho giờ và ngày
    new_cols["Hour_sin"] = np.sin(2 * np.pi * new_cols["Hour"] / 24)
    new_cols["Hour_cos"] = np.cos(2 * np.pi * new_cols["Hour"] / 24)
    new_cols["Day_sin"] = np.sin(2 * np.pi * new_cols["DayOfWeek"] / 7)
    new_cols["Day_cos"] = np.cos(2 * np.pi * new_cols["DayOfWeek"] / 7)

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df


def drop_na_cols(df, threshold=0.01):
    cnt = 0
    for col in df.columns:
        na = df[[col]].isna().sum()
        if na.values > len(df) * threshold:
            df.drop(col, axis=1, inplace=True)
            cnt += 1
    print(f"Deleted {cnt} cols")
    return df


# 6. Hàm xử lý missing values
def handle_missing_data(df, threshold=0.01):
    df = drop_na_cols(df, threshold)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Xóa các hàng có NaN sinh ra bởi indicators
    # print(df.isna().sum())
    df = df.dropna()
    # # Forward fill cho các features nhất định
    # df.loc[:, ['Volume', 'Open']] = df[['Volume', 'Open']].ffill()
    return df


# 7. Hàm chuẩn hóa dữ liệu (Áp dụng riêng cho từng tập)
def scale_features(
    train_df,
    val_df,
    test_df,
    excepts=["Label", "Datetime"],
    scaler_path="data/scaler.pkl",
):
    # Chọn các cột cần chuẩn hóa (bỏ các cột không phải số)
    feature_columns = [col for col in train_df.columns if col not in excepts]

    # Chuẩn hóa theo train
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_columns])
    val_scaled = scaler.transform(val_df[feature_columns])
    test_scaled = scaler.transform(test_df[feature_columns])

    # Tạo DataFrame mới đã scaled
    scaled_train_df = pd.DataFrame(
        train_scaled, columns=feature_columns, index=train_df.index
    )
    scaled_val_df = pd.DataFrame(
        val_scaled, columns=feature_columns, index=val_df.index
    )
    scaled_test_df = pd.DataFrame(
        test_scaled, columns=feature_columns, index=test_df.index
    )

    # Thêm lại các cột không phải feature
    for scaled_df, df in zip(
        [scaled_train_df, scaled_val_df, scaled_test_df], [train_df, val_df, test_df]
    ):
        for col in excepts:
            scaled_df[col] = df[col].values

    pickle.dump(scaler, open(scaler_path, "wb"))

    return scaled_train_df, scaled_val_df, scaled_test_df


def create_sequences_sequential(
    X, y, sequence_length, save_path, idx_file, target="Label", stride=1
):
    """
    Tạo sequences từng sample một, dùng memmap để lưu đúng shape
    """
    try:
        n_samples = int(np.ceil((len(X) - sequence_length) / stride))
        n_features = X.shape[1]
        if n_samples <= 0:
            raise ValueError("Input array too short for given sequence_length")

        # Chuẩn bị file memmap
        os.makedirs(save_path, exist_ok=True)
        sequences_file = f"{save_path}/sequences.dat"
        labels_file = f"{save_path}/labels.dat"
        shape_file = f"{save_path}/shape.txt"

        # Đọc index khởi đầu
        start_idx = 0
        if os.path.exists(idx_file):
            with open(idx_file, "r") as f:
                start_idx = int(f.read().strip() or 0)
        else:
            if os.path.exists(sequences_file):
                os.remove(sequences_file)
            if os.path.exists(labels_file):
                os.remove(labels_file)
            if os.path.exists(shape_file):
                os.remove(shape_file)

        # Tạo memmap với shape đầy đủ
        sequences = np.memmap(
            sequences_file,
            dtype=np.float32,
            mode="w+",
            shape=(n_samples, sequence_length, n_features),
        )
        labels = np.memmap(labels_file, dtype=np.int64, mode="w+", shape=(n_samples,))

        # Ghi dữ liệu từ start_idx
        for i in tqdm(range(start_idx, (len(X) - sequence_length), stride)):
            sequences[i // stride] = X[i : i + sequence_length]
            labels[i // stride] = y[target].values[i + sequence_length]

            # Ghi index hiện tại
            with open(idx_file, "w") as f:
                f.write(str(i + 1))

        # Lưu shape vào file
        with open(shape_file, "w") as f:
            f.write(f"{n_samples}\n{sequence_length}\n{n_features}")

        # Flush để đảm bảo dữ liệu được ghi
        sequences.flush()
        labels.flush()
        print(f"Sequences saved to {save_path}, shape: {sequences.shape}")
        return n_samples, n_features
    except Exception as e:
        print(e)
        start_idx = 0
        if os.path.exists(idx_file):
            with open(idx_file, "r") as f:
                start_idx = int(f.read().strip() or 0)
        return start_idx, n_features
