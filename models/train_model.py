import os
import sys
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 项目根目录
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# 让 Python 能找到 utils 文件夹
sys.path.append(BASE_DIR)

from utils.data_loader import load_pv_data


# =========================
# 1. 读取光伏数据
# =========================

df = load_pv_data(
    os.path.join(BASE_DIR, "data", "pv_data.csv")
)

X = df[["light", "diffuse", "direct", "temperature"]]
y = df["power"]


# =========================
# 2. 按时间顺序划分训练集、验证集、测试集
# 比例：6 : 2 : 2
# =========================

total_len = len(X)

train_size = int(total_len * 0.6)
val_size = int(total_len * 0.2)

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:train_size + val_size]
y_val = y.iloc[train_size:train_size + val_size]

X_test = X.iloc[train_size + val_size:]
y_test = y.iloc[train_size + val_size:]


print("数据集划分完成")
print("训练集样本数：", len(X_train))
print("验证集样本数：", len(X_val))
print("测试集样本数：", len(X_test))


# =========================
# 3. 训练线性回归模型
# =========================

model = LinearRegression()
model.fit(X_train, y_train)


# =========================
# 4. 验证集评估
# 用于观察模型效果
# =========================

val_pred = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_pred)
val_mse = mean_squared_error(y_val, val_pred)
val_r2 = r2_score(y_val, val_pred)


# =========================
# 5. 测试集评估
# 用于最终模型精度评估
# =========================

test_pred = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_pred)
test_mse = mean_squared_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)


print("验证集评估结果：")
print("MAE：", round(val_mae, 4))
print("MSE：", round(val_mse, 4))
print("R²：", round(val_r2, 4))

print("测试集评估结果：")
print("MAE：", round(test_mae, 4))
print("MSE：", round(test_mse, 4))
print("R²：", round(test_r2, 4))


# =========================
# 6. 保存模型
# =========================

model_path = os.path.join(
    BASE_DIR,
    "models",
    "pv_model.pkl"
)

joblib.dump(model, model_path)


# =========================
# 7. 保存评估结果
# =========================

evaluate_df = pd.DataFrame({
    "dataset": [
        "validation",
        "validation",
        "validation",
        "test",
        "test",
        "test"
    ],
    "metric": [
        "MAE",
        "MSE",
        "R2",
        "MAE",
        "MSE",
        "R2"
    ],
    "value": [
        round(val_mae, 4),
        round(val_mse, 4),
        round(val_r2, 4),
        round(test_mae, 4),
        round(test_mse, 4),
        round(test_r2, 4)
    ]
})

evaluate_path = os.path.join(
    BASE_DIR,
    "data",
    "model_evaluate.csv"
)

evaluate_df.to_csv(
    evaluate_path,
    index=False
)

print("线性回归模型训练完成")
print("模型已保存：", model_path)
print("评估结果已保存：", evaluate_path)