import os
import sys

# 关闭 TensorFlow 普通提示
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 项目根目录
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# 加入 Python 搜索路径
sys.path.append(BASE_DIR)

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from utils.data_loader import load_pv_data


# =========================
# 1. 加载光伏数据
# =========================

df = load_pv_data(
    os.path.join(BASE_DIR, "data", "pv_data.csv")
)

features = [
    "light",
    "diffuse",
    "direct",
    "temperature",
    "power"
]

data = df[features].values


# =========================
# 2. 先按时间顺序划分原始数据
# 比例：6 : 2 : 2
# 注意：
# 先划分，再归一化，避免测试集信息泄漏
# =========================

total_len = len(data)

train_size = int(total_len * 0.6)
val_size = int(total_len * 0.2)

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]


# =========================
# 3. 数据归一化
# 只使用训练集 fit
# 验证集和测试集只 transform
# =========================

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)


# =========================
# 4. 保存归一化器
# =========================

scaler_path = os.path.join(
    BASE_DIR,
    "models",
    "lstm_scaler.pkl"
)

joblib.dump(scaler, scaler_path)


# =========================
# 5. 构建时序样本函数
# =========================

def create_sequences(dataset, time_step=3):
    X = []
    y = []

    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i])
        y.append(dataset[i][4])

    return np.array(X), np.array(y)


time_step = 3

X_train, y_train = create_sequences(train_scaled, time_step)
X_val, y_val = create_sequences(val_scaled, time_step)
X_test, y_test = create_sequences(test_scaled, time_step)


print("数据集划分完成")
print("训练集样本数：", len(X_train))
print("验证集样本数：", len(X_val))
print("测试集样本数：", len(X_test))


# =========================
# 6. 创建 LSTM 模型
# =========================

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# =========================
# 7. 训练模型
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val)
)


# =========================
# 8. 使用测试集评估模型
# =========================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("LSTM测试集评估结果：")
print("MAE：", round(mae, 4))
print("MSE：", round(mse, 4))
print("R²：", round(r2, 4))


# =========================
# 9. 保存模型
# =========================

model_path = os.path.join(
    BASE_DIR,
    "models",
    "lstm_model.h5"
)

model.save(model_path)


# =========================
# 10. 保存训练损失和验证损失
# =========================

loss_df = pd.DataFrame({
    "epoch": list(range(1, len(history.history["loss"]) + 1)),
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
})

loss_path = os.path.join(
    BASE_DIR,
    "data",
    "lstm_loss.csv"
)

loss_df.to_csv(
    loss_path,
    index=False
)


# =========================
# 11. 保存测试集评估结果
# =========================

eval_df = pd.DataFrame({
    "metric": ["MAE", "MSE", "R2"],
    "value": [
        round(mae, 4),
        round(mse, 4),
        round(r2, 4)
    ]
})

eval_path = os.path.join(
    BASE_DIR,
    "data",
    "lstm_evaluate.csv"
)

eval_df.to_csv(
    eval_path,
    index=False
)


print("LSTM模型训练完成")
print("模型已保存：", model_path)
print("归一化器已保存：", scaler_path)
print("训练损失已保存：", loss_path)
print("测试集评估结果已保存：", eval_path)