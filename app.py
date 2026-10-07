import os

# 关闭 TensorFlow 普通提示，必须放在 tensorflow 导入之前
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sqlite3
import joblib
import markdown
import requests
import numpy as np
import pandas as pd
import subprocess
from functools import wraps
from flask import Flask, render_template, request, redirect, session

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

from utils.data_loader import load_pv_data
from utils.logger import init_logger

# =========================
# Flask基础配置
# =========================

app = Flask(__name__)
# 建议通过环境变量 SECRET_KEY 设置会话密钥；未设置时使用内置默认值（本地单机使用）
app.secret_key = os.environ.get("SECRET_KEY", "pv_forecast_secret")

logger = init_logger()


# =========================
# 公共工具函数
# =========================

def login_required(func):
    """登录验证装饰器：未登录用户自动跳转到登录页"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect("/login")
        return func(*args, **kwargs)

    return wrapper


def get_db_connection():
    """获取SQLite数据库连接"""
    return sqlite3.connect("database.db")


def init_database():
    """系统启动时自动初始化数据库，并兼容旧数据库字段"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # =========================
    # 用户信息表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        register_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # =========================
    # 预测记录表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        light REAL,
        diffuse REAL,
        direct REAL,
        temperature REAL,
        predicted_power REAL,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # =========================
    # 光伏数据表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pv_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        light REAL,
        diffuse REAL,
        direct REAL,
        temperature REAL,
        power REAL
    )
    """)

    # =========================
    # 模型信息表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        model_type TEXT,
        model_path TEXT,
        model_status TEXT DEFAULT '正常',
        file_size REAL DEFAULT 0,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 兼容旧数据库：如果 model_info 旧表缺字段，就自动补字段
    cursor.execute("PRAGMA table_info(model_info)")
    model_info_columns = [item[1] for item in cursor.fetchall()]

    if "model_status" not in model_info_columns:
        cursor.execute("""
        ALTER TABLE model_info ADD COLUMN model_status TEXT DEFAULT '正常'
        """)

    if "file_size" not in model_info_columns:
        cursor.execute("""
        ALTER TABLE model_info ADD COLUMN file_size REAL DEFAULT 0
        """)

    if "create_time" not in model_info_columns:
        cursor.execute("""
        ALTER TABLE model_info ADD COLUMN create_time TEXT
        """)

    # =========================
    # 模型评估表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_evaluate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        mae REAL,
        mse REAL,
        r2_score REAL,
        evaluate_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # =========================
    # LSTM预测结果表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lstm_predict (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predict_time TEXT,
        predict_power REAL,
        actual_power REAL,
        avg_power REAL,
        max_power REAL,
        min_power REAL,
        trend_text TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # =========================
    # LSTM训练损失表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lstm_loss (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch INTEGER,
        loss REAL,
        val_loss REAL,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # =========================
    # AI问答记录表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        question TEXT,
        answer TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # =========================
    # AI分析报告表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        avg_power REAL,
        max_power REAL,
        min_power REAL,
        avg_temp REAL,
        avg_light REAL,
        report TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # =========================
    # 系统日志表
    # =========================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_name TEXT,
        log_level TEXT,
        log_content TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

    logger.info("数据库初始化完成，所有系统表已创建或已兼容旧字段")


def save_system_log_to_db(log_name, log_level, log_content):
    """保存系统日志到数据库"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO system_logs (
            log_name,
            log_level,
            log_content
        ) VALUES (?, ?, ?)
        """, (
            log_name,
            log_level,
            log_content
        ))

        conn.commit()
        conn.close()

    except Exception as e:
        print("系统日志写入数据库失败：", e)


def get_system_status():
    """检测系统运行状态"""
    status = {
        "db_status": "正常",
        "lstm_status": "正常",
        "ai_status": "正常",
        "log_status": "正常"
    }

    try:
        conn = get_db_connection()
        conn.close()
    except Exception:
        status["db_status"] = "异常"

    if not os.path.exists("models/lstm_model.h5"):
        status["lstm_status"] = "异常"

    if not os.path.exists("logs"):
        status["log_status"] = "异常"

    try:
        requests.get("http://127.0.0.1:11434", timeout=2)
    except Exception:
        status["ai_status"] = "离线"

    return status


def get_dashboard_data():
    """读取光伏数据并计算首页统计信息"""
    df = load_pv_data("data/pv_data.csv")

    dashboard_data = {
        "total_power": round(df["power"].sum(), 2),
        "avg_power": round(df["power"].mean(), 2),
        "max_power": round(df["power"].max(), 2),
        "avg_temp": round(df["temperature"].mean(), 2),
        "avg_light": round(df["light"].mean(), 2),
        "times": df["time"].tolist(),
        "powers": df["power"].tolist()
    }

    return dashboard_data


import sys
import subprocess


def run_training_script(script_path):
    """
    使用当前项目虚拟环境Python执行训练脚本
    避免网页训练和命令行训练使用不同解释器
    """

    logger.info(
        "开始执行训练脚本：%s",
        script_path
    )

    # 当前 Flask 所使用的解释器
    python_exe = sys.executable

    logger.info(
        "当前Python解释器:%s",
        python_exe
    )

    result = subprocess.run(

        [
            python_exe,
            script_path
        ],

        capture_output=True,

        text=True,

        encoding="utf-8",

        errors="ignore"
    )

    if result.stdout:
        logger.info(
            "训练输出:\n%s",
            result.stdout
        )

    if result.stderr:
        logger.error(
            "训练错误:\n%s",
            result.stderr
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"""
训练失败：

{script_path}

错误：

{result.stderr}
"""
        )

    logger.info(
        "训练完成:%s",
        script_path
    )

    return True


def save_model_info_to_db():
    """保存模型文件信息到数据库"""
    model_files = [
        {
            "name": "线性回归模型",
            "type": "机器学习模型",
            "path": "models/pv_model.pkl"
        },
        {
            "name": "LSTM时序预测模型",
            "type": "深度学习模型",
            "path": "models/lstm_model.h5"
        },
        {
            "name": "LSTM归一化器",
            "type": "数据预处理文件",
            "path": "models/lstm_scaler.pkl"
        }
    ]

    conn = get_db_connection()
    cursor = conn.cursor()

    for item in model_files:
        file_path = item["path"]

        if os.path.exists(file_path):
            status = "正常"
            file_size = round(os.path.getsize(file_path) / 1024, 2)
        else:
            status = "缺失"
            file_size = 0

        cursor.execute("""
        INSERT INTO model_info (
            model_name,
            model_type,
            model_path,
            model_status,
            file_size
        ) VALUES (?, ?, ?, ?, ?)
        """, (
            item["name"],
            item["type"],
            file_path,
            status,
            file_size
        ))

    conn.commit()
    conn.close()

    logger.info("模型信息已写入数据库 model_info 表")


def save_evaluate_to_db():
    """保存模型评估结果到数据库"""
    evaluate_path = "data/model_evaluate.csv"

    if not os.path.exists(evaluate_path):
        logger.warning("模型评估文件不存在，无法写入 model_evaluate 表")
        return

    evaluate_df = pd.read_csv(evaluate_path)

    metric_dict = {}

    for _, row in evaluate_df.iterrows():
        metric_dict[row["metric"]] = row["value"]

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM model_evaluate")

    cursor.execute("""
    INSERT INTO model_evaluate (
        model_name,
        mae,
        mse,
        r2_score
    ) VALUES (?, ?, ?, ?)
    """, (
        "线性回归模型",
        metric_dict.get("MAE"),
        metric_dict.get("MSE"),
        metric_dict.get("R2")
    ))

    conn.commit()
    conn.close()

    logger.info("模型评估结果已写入数据库 model_evaluate 表")


# =========================
# 用户登录注册模块
# =========================

@app.route("/register", methods=["GET", "POST"])
def register():
    """用户注册"""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO user (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()
            return redirect("/login")

        except Exception as e:
            logger.error("用户注册失败：%s", str(e))
            return "用户名已存在，请重新注册"

        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """用户登录"""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM user WHERE username=? AND password=?",
            (username, password)
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            session["username"] = username
            return redirect("/")
        else:
            return "用户名或密码错误"

    return render_template("login.html")


@app.route("/logout")
def logout():
    """退出登录"""
    session.clear()
    return redirect("/login")


# =========================
# 首页与监控大屏模块
# =========================

@app.route("/")
@login_required
def index():
    """首页监控大屏"""
    try:
        dashboard_data = get_dashboard_data()
        system_status = get_system_status()

        return render_template(
            "index.html",
            **dashboard_data,
            **system_status
        )

    except Exception as e:
        logger.error("首页加载失败：%s", str(e))
        return "首页加载失败，请检查数据文件"


@app.route("/api/dashboard")
@login_required
def api_dashboard():
    """首页AJAX自动刷新接口"""
    try:
        dashboard_data = get_dashboard_data()
        system_status = get_system_status()

        return {
            **dashboard_data,
            **system_status
        }

    except Exception as e:
        logger.error("首页数据接口错误：%s", str(e))
        return {
            "error": "首页数据加载失败"
        }


# =========================
# 数据管理与可视化模块
# =========================

@app.route("/data", methods=["GET", "POST"])
@login_required
def data():
    """数据上传、写入数据库、数据表格展示"""
    try:
        if request.method == "POST":
            file = request.files["file"]

            if file:
                save_path = os.path.join("data", "pv_data.csv")
                file.save(save_path)

                logger.info("用户上传数据集成功：%s", save_path)

                # 读取并标准化CSV数据
                df = load_pv_data(save_path)

                # 写入SQLite数据库 pv_data 表
                conn = get_db_connection()
                cursor = conn.cursor()

                # 清空旧数据，避免重复导入
                cursor.execute("DELETE FROM pv_data")

                for _, row in df.iterrows():
                    cursor.execute("""
                    INSERT INTO pv_data (
                        time,
                        light,
                        diffuse,
                        direct,
                        temperature,
                        power
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        str(row["time"]),
                        float(row["light"]),
                        float(row["diffuse"]),
                        float(row["direct"]),
                        float(row["temperature"]),
                        float(row["power"])
                    ))

                conn.commit()
                conn.close()

                logger.info("光伏数据已成功写入数据库 pv_data 表，共 %s 条", len(df))

        # 页面展示仍然读取标准化后的CSV，保证页面不出错
        df = load_pv_data("data/pv_data.csv")
        pv_data = df.values.tolist()

        return render_template("data.html", pv_data=pv_data)

    except Exception as e:
        logger.error("数据管理模块错误：%s", str(e))
        return "数据读取或写入数据库失败，请检查CSV文件格式"


@app.route("/chart")
@login_required
def chart():
    """发电功率趋势图"""
    try:
        df = load_pv_data("data/pv_data.csv")

        times = df["time"].tolist()
        powers = df["power"].tolist()

        return render_template(
            "chart.html",
            times=times,
            powers=powers
        )

    except Exception as e:
        logger.error("数据可视化模块错误：%s", str(e))
        return "图表加载失败，请检查数据文件"


# =========================
# 模型训练模块
# =========================

@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    """模型训练页面：GET显示已有训练状态，POST执行模型训练"""

    def get_train_page_data(trained=False, error_message=None):
        """读取训练状态、数据集数量和评估结果"""

        train_count = "-"
        val_count = "-"
        test_count = "-"

        val_mae = "-"
        val_mse = "-"
        val_r2 = "-"
        test_mae = "-"
        test_mse = "-"
        test_r2 = "-"

        try:
            df = load_pv_data("data/pv_data.csv")

            total = len(df)
            train_count = int(total * 0.6)
            val_count = int(total * 0.2)
            test_count = total - train_count - val_count

        except Exception as e:
            logger.error("训练页面读取数据集数量失败：%s", str(e))

        try:
            evaluate_path = "data/lstm_evaluate.csv"

            if os.path.exists(evaluate_path):
                evaluate_df = pd.read_csv(evaluate_path)

                test_mae = evaluate_df[evaluate_df["metric"] == "MAE"]["value"].values[0]
                test_mse = evaluate_df[evaluate_df["metric"] == "MSE"]["value"].values[0]
                test_r2 = evaluate_df[evaluate_df["metric"] == "R2"]["value"].values[0]

        except Exception as e:
            logger.error("训练页面读取评估结果失败：%s", str(e))

        return render_template(
            "train.html",
            trained=trained,
            error_message=error_message,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            val_mae=val_mae,
            val_mse=val_mse,
            val_r2=val_r2,
            test_mae=test_mae,
            test_mse=test_mse,
            test_r2=test_r2
        )

    # =========================
    # 1. 打开训练页面时，检查模型是否已经存在
    # =========================
    if request.method == "GET":
        model_exists = os.path.exists("models/pv_model.pkl")
        lstm_exists = os.path.exists("models/lstm_model.h5")
        scaler_exists = os.path.exists("models/lstm_scaler.pkl")
        evaluate_exists = os.path.exists("data/lstm_evaluate.csv")

        trained = (
                model_exists
                and lstm_exists
                and scaler_exists
                and evaluate_exists
        )

        return get_train_page_data(trained=trained)

    # =========================
    # 2. 点击按钮后，重新执行训练
    # =========================
    try:
        logger.info("用户点击开始训练，系统开始执行模型训练")
        save_system_log_to_db(
            "模型训练",
            "INFO",
            "用户点击开始训练，系统开始执行模型训练"
        )
        run_training_script("models/train_model.py")
        run_training_script("models/train_lstm.py")

        save_model_info_to_db()
        save_evaluate_to_db()

        logger.info("模型训练全部完成，模型信息和评估结果已写入数据库")

        return get_train_page_data(trained=True)

    except Exception as e:
        logger.error("模型训练失败：%s", str(e))
        save_system_log_to_db(
            "模型训练",
            "ERROR",
            str(e)
        )
        return get_train_page_data(
            trained=False,
            error_message=str(e)
        )


# =========================
# 传统机器学习预测模块
# =========================

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """普通机器学习发电功率预测：支持预测值与实际值比较"""

    next_power = None
    actual_power = None
    error_value = None
    error_rate = None
    error_message = None

    model_path = "models/pv_model.pkl"
    model_exists = os.path.exists(model_path)

    if request.method == "POST":
        try:
            if not model_exists:
                raise FileNotFoundError("线性回归模型文件不存在，请先进入【模型训练】页面完成训练")

            light = float(request.form["light"])
            diffuse = float(request.form["diffuse"])
            direct = float(request.form["direct"])
            temperature = float(request.form["temperature"])

            actual_input = request.form.get("actual_power")

            if actual_input:
                actual_power = float(actual_input)

            model = joblib.load(model_path)

            future_data = [[light, diffuse, direct, temperature]]
            prediction = model.predict(future_data)

            next_power = round(prediction[0], 2)

            if actual_power is not None:
                error_value = round(abs(next_power - actual_power), 2)

                if actual_power != 0:
                    error_rate = round(error_value / actual_power * 100, 2)
                else:
                    error_rate = 0

            # 保存预测记录到数据库
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO prediction_log (
                light,
                diffuse,
                direct,
                temperature,
                predicted_power
            ) VALUES (?, ?, ?, ?, ?)
            """, (
                light,
                diffuse,
                direct,
                temperature,
                next_power
            ))

            conn.commit()
            conn.close()

            logger.info(
                "普通预测完成并写入数据库：预测值=%s，实际值=%s，误差=%s，误差率=%s",
                next_power,
                actual_power,
                error_value,
                error_rate
            )

        except Exception as e:
            logger.error("普通预测模块发生错误：%s", str(e))
            error_message = str(e)

    return render_template(
        "predict.html",
        next_power=next_power,
        actual_power=actual_power,
        error_value=error_value,
        error_rate=error_rate,
        error_message=error_message,
        model_exists=model_exists
    )


@app.route("/evaluate")
@login_required
def evaluate():
    """模型结果评估：同时展示线性回归模型和LSTM模型评估结果"""
    try:
        # =========================
        # 1. 线性回归模型评估
        # =========================

        df = load_pv_data("data/pv_data.csv")

        X = df[["light", "diffuse", "direct", "temperature"]]
        y = df["power"]

        total_len = len(X)

        train_size = int(total_len * 0.6)
        val_size = int(total_len * 0.2)

        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]

        lr_model = joblib.load("models/pv_model.pkl")
        lr_pred = lr_model.predict(X_test)

        lr_mae = round(mean_absolute_error(y_test, lr_pred), 4)
        lr_mse = round(mean_squared_error(y_test, lr_pred), 4)
        lr_r2 = round(r2_score(y_test, lr_pred), 4)

        # =========================
        # 2. LSTM模型评估结果读取
        # =========================

        lstm_mae = "未生成"
        lstm_mse = "未生成"
        lstm_r2 = "未生成"

        lstm_eval_path = "data/lstm_evaluate.csv"

        if os.path.exists(lstm_eval_path):
            lstm_eval_df = pd.read_csv(lstm_eval_path)

            lstm_mae = lstm_eval_df[lstm_eval_df["metric"] == "MAE"]["value"].values[0]
            lstm_mse = lstm_eval_df[lstm_eval_df["metric"] == "MSE"]["value"].values[0]
            lstm_r2 = lstm_eval_df[lstm_eval_df["metric"] == "R2"]["value"].values[0]

        return render_template(
            "evaluate.html",

            lr_mae=lr_mae,
            lr_mse=lr_mse,
            lr_r2=lr_r2,

            lstm_mae=lstm_mae,
            lstm_mse=lstm_mse,
            lstm_r2=lstm_r2,

            test_count=len(X_test)
        )

    except Exception as e:
        logger.error("模型评估失败：%s", str(e))
        return "模型评估失败，请先训练模型"


# =========================
# LSTM深度学习预测模块
# =========================

@app.route("/lstm_predict")
@login_required
def lstm_predict():
    """LSTM预测：显示0-15分钟每个点的预测值与实际值对比"""

    model_path = "models/lstm_model.h5"
    scaler_path = "models/lstm_scaler.pkl"

    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)

    error_message = None

    future_predictions = []
    actual_values = []
    future_times = []

    avg_future_power = "-"
    max_future_power = "-"
    min_future_power = "-"
    trend_text = "暂无预测结果"

    try:
        if not model_exists:
            raise FileNotFoundError("LSTM模型文件不存在，请先进入【模型训练】页面完成训练")

        if not scaler_exists:
            raise FileNotFoundError("LSTM归一化器文件不存在，请先进入【模型训练】页面完成训练")

        df = load_pv_data("data/pv_data.csv")

        features = [
            "light",
            "diffuse",
            "direct",
            "temperature",
            "power"
        ]

        data = df[features].values

        # 需要3条历史数据作为输入，再拿后面16个点做0-15分钟对比
        if len(data) < 19:
            raise ValueError("数据量不足，至少需要19条数据才能进行0-15分钟预测对比")

        scaler = joblib.load(scaler_path)
        model = load_model(model_path, compile=False)

        # 取最后16个点作为实际值对比区间
        compare_data = data[-16:]

        # 取对比区间前面的3个点作为LSTM输入窗口
        input_data = data[-19:-16]

        current_input = scaler.transform(input_data)

        for i in range(16):

            X = np.array([current_input], dtype=np.float32)

            pred_scaled = model.predict(X, verbose=0)

            fake_row = np.zeros((1, 5))
            fake_row[0, 4] = pred_scaled[0][0]

            pred_power = scaler.inverse_transform(fake_row)[0][4]

            # 光伏发电功率不能为负数
            pred_power = max(0, pred_power)

            pred_power = round(pred_power, 2)

            actual_power = round(compare_data[i][4], 2)

            future_predictions.append(pred_power)
            actual_values.append(actual_power)

            if i == 0:
                future_times.append("0分钟")
            else:
                future_times.append(f"{i}分钟后")

            # 使用真实环境变量 + 预测功率，组成下一步输入
            next_row_raw = compare_data[i].copy()
            next_row_raw[4] = pred_power

            next_row_scaled = scaler.transform([next_row_raw])[0]

            current_input = np.vstack([
                current_input[1:],
                next_row_scaled
            ])

        max_future_power = round(max(future_predictions), 2)
        min_future_power = round(min(future_predictions), 2)
        avg_future_power = round(
            sum(future_predictions) / len(future_predictions),
            2
        )

        if future_predictions[-1] > future_predictions[0]:
            trend_text = "0-15分钟内预测发电功率整体呈上升趋势"
        elif future_predictions[-1] < future_predictions[0]:
            trend_text = "0-15分钟内预测发电功率整体呈下降趋势"
        else:
            trend_text = "0-15分钟内预测发电功率整体较为平稳"

        # 保存LSTM预测结果到数据库
        conn = get_db_connection()
        cursor = conn.cursor()

        for i in range(len(future_predictions)):
            cursor.execute("""
            INSERT INTO lstm_predict (
                predict_time,
                predict_power,
                actual_power,
                avg_power,
                max_power,
                min_power,
                trend_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                future_times[i],
                future_predictions[i],
                actual_values[i],
                avg_future_power,
                max_future_power,
                min_future_power,
                trend_text
            ))

        conn.commit()
        conn.close()

        logger.info(
            "LSTM预测完成并写入数据库：预测值=%s，实际值=%s",
            future_predictions,
            actual_values
        )

    except Exception as e:
        logger.error("LSTM预测模块发生错误：%s", str(e))
        error_message = str(e)

    return render_template(
        "lstm_predict.html",

        model_exists=model_exists,
        scaler_exists=scaler_exists,
        error_message=error_message,

        future_predictions=future_predictions,
        actual_values=actual_values,
        future_times=future_times,

        avg_future_power=avg_future_power,
        max_future_power=max_future_power,
        min_future_power=min_future_power,
        trend_text=trend_text
    )


@app.route("/lstm_loss")
@login_required
def lstm_loss():
    """LSTM训练损失曲线：同时展示训练集Loss和验证集Val_Loss"""
    try:
        df = pd.read_csv("data/lstm_loss.csv")

        epochs = df["epoch"].tolist()
        losses = df["loss"].round(6).tolist()
        val_losses = df["val_loss"].round(6).tolist()

        return render_template(
            "lstm_loss.html",
            epochs=epochs,
            losses=losses,
            val_losses=val_losses
        )

    except Exception as e:
        logger.error("LSTM损失曲线加载失败：%s", str(e))
        return "LSTM损失曲线加载失败，请先训练LSTM模型"


# =========================
# AI智能助手模块
# =========================

@app.route("/ai_agent", methods=["GET", "POST"])
@login_required
def ai_agent():
    """AI智能助手"""
    answer = None

    if request.method == "POST":
        question = request.form["question"]

        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "qwen2.5:1.5b",
                    "prompt": question,
                    "stream": False
                },
                timeout=60
            )

            result = response.json()
            answer = result.get("response", "AI暂无回答")

            # 保存AI问答记录到数据库
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO ai_chat (
                user_id,
                question,
                answer
            ) VALUES (?, ?, ?)
            """, (
                None,
                question,
                answer
            ))

            conn.commit()
            conn.close()

            logger.info("AI问答记录已写入数据库 ai_chat 表")

        except Exception as e:
            logger.error("AI智能助手调用失败：%s", str(e))
            answer = "AI服务调用失败，请检查Ollama是否正常运行。"

    return render_template("ai_agent.html", answer=answer)
# @app.route("/ai_agent", methods=["GET", "POST"])
# @login_required
# def ai_agent():
#     answer = None
#     # 定义可用模型列表（便于前端渲染）
#     available_models = [
#         {"value": "qwen2.5:1.5b", "label": "千问 1.5B"},
#         {"value": "deepseek-r1:7b", "label": "DeepSeek R1 7B"},
#     ]
#     # 默认选中第一个
#     current_model = request.form.get("model", available_models[0]["value"])
#
#     if request.method == "POST":
#         question = request.form.get("question", "").strip()
#         if not question:
#             answer = "请输入问题"
#         else:
#             try:
#                 response = requests.post(
#                     "http://127.0.0.1:11434/api/generate",
#                     json={
#                         "model": current_model,   # 使用用户选择的模型
#                         "prompt": question,
#                         "stream": False
#                     },
#                     timeout=60
#                 )
#                 result = response.json()
#                 answer = result.get("response", "AI暂无回答")
#
#                 # 保存到数据库（可选，建议同时记录模型名称）
#                 conn = get_db_connection()
#                 cursor = conn.cursor()
#                 cursor.execute("""
#                     INSERT INTO ai_chat (
#                         user_id,
#                         question,
#                         answer,
#                         model          -- 如果你的表没有这一列，需要先 ALTER TABLE 添加
#                     ) VALUES (?, ?, ?, ?)
#                 """, (None, question, answer, current_model))
#                 conn.commit()
#                 conn.close()
#
#                 logger.info("AI问答记录已写入，使用模型：%s", current_model)
#
#             except Exception as e:
#                 logger.error("AI智能助手调用失败：%s", str(e))
#                 answer = "AI服务调用失败，请检查Ollama是否正常运行或模型是否已下载。"
#
#     # 将模型列表和当前模型传给模板
#     return render_template(
#         "ai_agent.html",
#         answer=answer,
#         available_models=available_models,
#         current_model=current_model
#     )
#

# # 从环境变量读取 DeepSeek API Key（务必设置）
# DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
# OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
#
# @app.route("/ai_agent", methods=["GET", "POST"])
# @login_required
# def ai_agent():
#     answer = None
#     # 可用模型列表（展示给用户）
#     available_models = [
#         {"value": "qwen", "label": "Qwen 千问 (Ollama)"},
#         {"value": "deepseek", "label": "DeepSeek (官方API)"}
#     ]
#     # 默认选中的模型（从表单获取，或默认 qwen）
#     current_model = request.form.get("model", "qwen")
#
#     if request.method == "POST":
#         question = request.form.get("question", "").strip()
#         if not question:
#             answer = "请输入问题"
#         else:
#             try:
#                 if current_model == "deepseek":
#                     # --- 调用 DeepSeek 官方 API ---
#                     if not DEEPSEEK_API_KEY:
#                         raise ValueError("DeepSeek API Key 未配置，请检查环境变量 DEEPSEEK_API_KEY")
#                     headers = {
#                         "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
#                         "Content-Type": "application/json"
#                     }
#                     payload = {
#                         "model": "deepseek-chat",      # 或 "deepseek-reasoner" 等
#                         "messages": [
#                             {"role": "user", "content": question}
#                         ],
#                         "stream": False
#                     }
#                     resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
#                     resp.raise_for_status()
#                     result = resp.json()
#                     # DeepSeek 返回格式与 OpenAI 兼容
#                     answer = result["choices"][0]["message"]["content"]
#
#                 else:  # qwen (Ollama)
#                     response = requests.post(
#                         OLLAMA_URL,
#                         json={
#                             "model": "qwen2.5:1.5b",   # 可根据需要调整
#                             "prompt": question,
#                             "stream": False
#                         },
#                         timeout=60
#                     )
#                     response.raise_for_status()
#                     result = response.json()
#                     answer = result.get("response", "AI暂无回答")
#
#                 # 保存记录到数据库（建议增加 model 字段）
#                 conn = get_db_connection()
#                 cursor = conn.cursor()
#                 # 如果表结构没有 model 列，请先执行 ALTER TABLE ai_chat ADD COLUMN model TEXT;
#                 cursor.execute("""
#                     INSERT INTO ai_chat (user_id, question, answer, model)
#                     VALUES (?, ?, ?, ?)
#                 """, (None, question, answer, current_model))
#                 conn.commit()
#                 conn.close()
#                 logger.info("AI问答记录已写入，模型: %s", current_model)
#
#             except Exception as e:
#                 logger.error("AI调用失败: %s", str(e))
#                 answer = f"AI服务调用失败：{str(e)}"
#
#     # 渲染模板，传递模型列表和当前选中模型
#     return render_template(
#         "ai_agent.html",
#         answer=answer,
#         available_models=available_models,
#         current_model=current_model,
#         # 可选：传递当前模型显示名称
#         current_model_label=dict(available_models).get(current_model, "未知")
#     )
@app.route("/ai_analysis")
@login_required
def ai_analysis():
    """AI数据分析：生成更完整的光伏发电运行分析报告"""
    try:
        df = load_pv_data("data/pv_data.csv")

        avg_power = round(df["power"].mean(), 2)
        max_power = round(df["power"].max(), 2)
        min_power = round(df["power"].min(), 2)
        avg_temp = round(df["temperature"].mean(), 2)
        avg_light = round(df["light"].mean(), 2)

        row_count = len(df)
        zero_power_count = int((df["power"] <= 0).sum())
        zero_power_rate = round(zero_power_count / row_count * 100, 2)

        power_std = round(df["power"].std(), 2)
        light_std = round(df["light"].std(), 2)

        max_power_row = df.loc[df["power"].idxmax()]
        max_power_time = max_power_row["time"]

        if df["light"].std() != 0:
            light_power_corr = round(df["light"].corr(df["power"]), 4)
        else:
            light_power_corr = 0

        if df["temperature"].std() != 0:
            temp_power_corr = round(df["temperature"].corr(df["power"]), 4)
        else:
            temp_power_corr = 0

        daytime_df = df[df["light"] > 0]

        if len(daytime_df) > 0:
            daytime_avg_power = round(daytime_df["power"].mean(), 2)
            daytime_max_power = round(daytime_df["power"].max(), 2)
            daytime_avg_light = round(daytime_df["light"].mean(), 2)
        else:
            daytime_avg_power = 0
            daytime_max_power = 0
            daytime_avg_light = 0

        if light_power_corr >= 0.8:
            light_analysis = "光照强度与发电功率呈现较强正相关关系，说明光照是当前数据中影响发电功率的主要因素。"
        elif light_power_corr >= 0.5:
            light_analysis = "光照强度与发电功率存在一定正相关关系，但发电功率还可能受到温度、天气变化或设备状态等因素影响。"
        else:
            light_analysis = "光照强度与发电功率的相关性不明显，建议进一步检查数据质量或分析其他影响因素。"

        if temp_power_corr > 0.5:
            temp_analysis = "温度与发电功率呈正相关，说明在当前数据中温度升高通常伴随较好的光照条件。"
        elif temp_power_corr < -0.3:
            temp_analysis = "温度与发电功率存在一定负相关，可能说明高温会对光伏组件效率产生一定影响。"
        else:
            temp_analysis = "温度与发电功率相关性较弱，说明温度不是当前数据中最主要的影响因素。"

        if zero_power_rate > 40:
            zero_analysis = "数据中零功率样本占比较高，可能包含大量夜间或无光照时段数据，这会使模型更容易学习到“光照为0时功率为0”的规律。"
        elif zero_power_rate > 15:
            zero_analysis = "数据中存在一定比例的零功率样本，主要可能来自夜间、阴天或低光照时段。"
        else:
            zero_analysis = "零功率样本占比较低，数据整体更集中在有效发电时段。"

        if power_std > avg_power:
            stable_analysis = "发电功率波动较大，说明该数据集中的发电输出受光照变化影响明显，模型需要具备较好的时序学习能力。"
        else:
            stable_analysis = "发电功率波动相对平稳，说明该时间段内光伏系统运行状态较稳定。"

        report = f"""
当前光伏发电数据智能分析报告：

一、数据整体概况
本次系统共读取 {row_count} 条光伏运行数据。数据中平均发电功率为 {avg_power} kW，最大发电功率为 {max_power} kW，最小发电功率为 {min_power} kW。整体来看，该数据能够反映光伏电站在不同光照和温度条件下的发电变化情况。

二、发电功率变化分析
当前数据的发电功率标准差为 {power_std}，说明发电功率存在一定波动。{stable_analysis}
最大发电功率出现在 {max_power_time}，该时刻可能对应光照条件较强、天气较好或系统输出效率较高的时间段。

三、光照因素影响分析
当前平均光照强度为 {avg_light} W/m²，光照强度标准差为 {light_std}。
光照强度与发电功率的相关系数为 {light_power_corr}。{light_analysis}
在有效光照时段内，平均发电功率为 {daytime_avg_power} kW，最高发电功率为 {daytime_max_power} kW，平均光照强度为 {daytime_avg_light} W/m²。

四、温度因素影响分析
当前平均温度为 {avg_temp} ℃。温度与发电功率的相关系数为 {temp_power_corr}。{temp_analysis}
从光伏发电原理来看，温度升高有时会降低组件转换效率，但在实际数据中，温度往往也会与白天强光照同时出现，因此需要结合光照强度一起分析。

五、低功率与零功率样本分析
当前零功率样本数量为 {zero_power_count} 条，占总数据的 {zero_power_rate}%。
{zero_analysis}
如果零功率样本过多，线性回归模型可能会得到较高的R²，因为模型很容易学习到“夜间无光照则无发电”的规律。

六、模型预测结果说明
从数据特征来看，发电功率与光照强度之间存在明显关系，因此线性回归模型可能会取得较高的拟合效果。
不过，光伏发电不仅受当前光照影响，还受时间变化、天气波动、温度变化和历史发电状态影响。因此，LSTM时序模型更适合用于短期发电趋势预测。

七、运行优化建议
1. 建议在模型训练前区分白天和夜间数据，避免大量零功率数据影响模型评价结果。
2. 建议重点关注光照强度、温度和历史功率变化，因为这些因素会直接影响短期发电趋势。
3. 如果线性回归模型R²接近1，可能说明数据集中Target与GHI存在较强线性关系，应在论文或答辩中说明原因。
4. 后续可以增加天气类型、湿度、风速等特征，从而提高模型对复杂天气条件下发电变化的预测能力。
"""

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO ai_analysis (
            avg_power,
            max_power,
            min_power,
            avg_temp,
            avg_light,
            report
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            avg_power,
            max_power,
            min_power,
            avg_temp,
            avg_light,
            report
        ))

        conn.commit()
        conn.close()

        logger.info("AI数据分析报告已写入数据库 ai_analysis 表")

        return render_template(
            "ai_analysis.html",
            avg_power=avg_power,
            max_power=max_power,
            min_power=min_power,
            report=report
        )

    except Exception as e:
        logger.error("AI数据分析失败：%s", str(e))
        return "AI数据分析失败，请检查数据文件"
# @app.route("/ai_analysis")
# @login_required
# def ai_analysis():
#     """AI数据分析：调用本地 Ollama 生成光伏发电分析报告"""
#     try:
#         df = load_pv_data("data/pv_data.csv")
#
#         avg_power = round(df["power"].mean(), 2)
#         max_power = round(df["power"].max(), 2)
#         min_power = round(df["power"].min(), 2)
#         avg_temp = round(df["temperature"].mean(), 2)
#         avg_light = round(df["light"].mean(), 2)
#
#         row_count = len(df)
#         zero_power_count = int((df["power"] <= 0).sum())
#         zero_power_rate = round(zero_power_count / row_count * 100, 2)
#
#         light_power_corr = round(df["light"].corr(df["power"]), 4)
#         temp_power_corr = round(df["temperature"].corr(df["power"]), 4)
#
#         prompt = f"""
# 你是一个光伏发电预测系统的数据分析助手。
# 请根据下面的光伏运行统计数据，生成一份适合本科毕业设计系统展示的中文分析报告。
#
# 要求：
# 1. 不要编造数据，只能基于我提供的统计值分析。
# 2. 内容包括：数据概况、发电功率情况、光照影响、温度影响、零功率样本原因、模型预测建议、系统优化建议。
# 3. 语言要自然、清晰，适合放在“AI数据分析”页面。
# 4. 不要使用太复杂的专业术语。
#
# 统计数据如下：
# - 数据总行数：{row_count}
# - 平均发电功率：{avg_power} kW
# - 最大发电功率：{max_power} kW
# - 最小发电功率：{min_power} kW
# - 平均温度：{avg_temp} ℃
# - 平均光照强度：{avg_light} W/m²
# - 零功率样本数量：{zero_power_count}
# - 零功率样本占比：{zero_power_rate}%
# - 光照强度与发电功率相关系数：{light_power_corr}
# - 温度与发电功率相关系数：{temp_power_corr}
# """
#
#         response = requests.post(
#             "http://127.0.0.1:11434/api/generate",
#             json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             },
#             timeout=120
#         )
#
#         result = response.json()
#         report = result.get("response", "AI暂无分析结果")
#
#         if not report or report.strip() == "":
#             report = "AI暂无分析结果，请检查Ollama模型是否正常运行。"
#
#         conn = get_db_connection()
#         cursor = conn.cursor()
#
#         cursor.execute("""
#         INSERT INTO ai_analysis (
#             avg_power,
#             max_power,
#             min_power,
#             avg_temp,
#             avg_light,
#             report
#         ) VALUES (?, ?, ?, ?, ?, ?)
#         """, (
#             avg_power,
#             max_power,
#             min_power,
#             avg_temp,
#             avg_light,
#             report
#         ))
#
#         conn.commit()
#         conn.close()
#
#         logger.info("Ollama AI数据分析报告已写入数据库 ai_analysis 表")
#
#         return render_template(
#             "ai_analysis.html",
#             avg_power=avg_power,
#             max_power=max_power,
#             min_power=min_power,
#             report=report
#         )
#
#     except Exception as e:
#         logger.error("AI数据分析失败：%s", str(e))
#         return f"AI数据分析失败：{str(e)}"
# =========================
# 预测记录模块
# =========================


@app.route("/clear_logs")
@login_required
def clear_logs():
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("DELETE FROM prediction_log")

        conn.commit()
        conn.close()

        logger.info("预测记录已清空")

        return redirect("/logs")

    except Exception as e:
        logger.error("清空预测记录失败：%s", str(e))
        return "清空失败"


@app.route("/data_info")
@login_required
def data_info():
    """数据集信息与数据质量检测"""
    try:
        raw_df = pd.read_csv("data/pv_data.csv")
        clean_df = load_pv_data("data/pv_data.csv")

        raw_row_count = len(raw_df)
        clean_row_count = len(clean_df)
        col_count = len(clean_df.columns)

        start_time = clean_df["time"].iloc[0]
        end_time = clean_df["time"].iloc[-1]

        avg_power = round(clean_df["power"].mean(), 2)
        max_power = round(clean_df["power"].max(), 2)
        min_power = round(clean_df["power"].min(), 2)

        avg_temp = round(clean_df["temperature"].mean(), 2)
        avg_light = round(clean_df["light"].mean(), 2)

        raw_missing_count = int(raw_df.isnull().sum().sum())
        raw_duplicate_count = int(raw_df.duplicated().sum())

        clean_missing_count = int(clean_df.isnull().sum().sum())
        clean_duplicate_count = int(clean_df.duplicated().sum())

        removed_count = raw_row_count - clean_row_count

        total = clean_row_count
        train_count = int(total * 0.6)
        val_count = int(total * 0.2)
        test_count = total - train_count - val_count

        quality_score = 100

        if raw_missing_count > 0:
            quality_score -= 20

        if raw_duplicate_count > 0:
            quality_score -= 15

        if clean_row_count < 50:
            quality_score -= 20

        if clean_missing_count > 0:
            quality_score -= 30

        if clean_duplicate_count > 0:
            quality_score -= 20

        if quality_score < 0:
            quality_score = 0

        if quality_score >= 90:
            quality_status = "优秀"
        elif quality_score >= 75:
            quality_status = "良好"
        elif quality_score >= 60:
            quality_status = "一般"
        else:
            quality_status = "需检查"

        logger.info(
            "数据集信息检测完成：原始行数=%s，清洗后行数=%s，质量评分=%s",
            raw_row_count,
            clean_row_count,
            quality_score
        )

        return render_template(
            "data_info.html",

            raw_row_count=raw_row_count,
            clean_row_count=clean_row_count,
            col_count=col_count,

            start_time=start_time,
            end_time=end_time,

            avg_power=avg_power,
            max_power=max_power,
            min_power=min_power,
            avg_temp=avg_temp,
            avg_light=avg_light,

            raw_missing_count=raw_missing_count,
            raw_duplicate_count=raw_duplicate_count,

            clean_missing_count=clean_missing_count,
            clean_duplicate_count=clean_duplicate_count,

            removed_count=removed_count,

            train_count=train_count,
            val_count=val_count,
            test_count=test_count,

            quality_status=quality_status,
            quality_score=quality_score
        )

    except Exception as e:
        logger.error("数据集信息模块错误：%s", str(e))
        return "数据集信息读取失败，请检查CSV文件"


@app.route("/model_info")
@login_required
def model_info():
    """模型文件与训练结果文件完整性检查"""
    try:
        model_files = [
            {
                "name": "线性回归模型",
                "path": "models/pv_model.pkl",
                "desc": "用于普通机器学习发电功率预测"
            },
            {
                "name": "线性回归评估文件",
                "path": "data/model_evaluate.csv",
                "desc": "保存线性回归模型在验证集和测试集上的评估结果"
            },
            {
                "name": "LSTM模型",
                "path": "models/lstm_model.h5",
                "desc": "用于深度学习时序发电功率预测"
            },
            {
                "name": "LSTM归一化器",
                "path": "models/lstm_scaler.pkl",
                "desc": "用于LSTM输入数据归一化处理"
            },
            {
                "name": "LSTM训练损失文件",
                "path": "data/lstm_loss.csv",
                "desc": "保存LSTM训练过程中的Loss和Val_Loss"
            },
            {
                "name": "LSTM评估文件",
                "path": "data/lstm_evaluate.csv",
                "desc": "保存LSTM模型在测试集上的评估结果"
            }
        ]

        normal_count = 0
        missing_count = 0

        for item in model_files:
            if os.path.exists(item["path"]):
                item["status"] = "正常"
                item["size"] = round(os.path.getsize(item["path"]) / 1024, 2)
                normal_count += 1
            else:
                item["status"] = "缺失"
                item["size"] = 0
                missing_count += 1

        if missing_count == 0:
            overall_status = "完整"
        else:
            overall_status = "不完整"

        logger.info(
            "模型信息检测完成：正常文件=%s，缺失文件=%s",
            normal_count,
            missing_count
        )

        return render_template(
            "model_info.html",
            model_files=model_files,
            normal_count=normal_count,
            missing_count=missing_count,
            overall_status=overall_status
        )

    except Exception as e:
        logger.error("模型信息模块错误：%s", str(e))
        return "模型信息读取失败"


@app.route("/system_logs")
@login_required
def system_logs():
    """系统运行日志查看：支持选择不同日志文件"""
    try:
        log_dir = "logs"

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        files = os.listdir(log_dir)
        files = [
            file for file in files
            if file.endswith(".log")
        ]

        files.sort(reverse=True)

        selected_log = request.args.get("file")

        if selected_log not in files:
            selected_log = files[0] if files else ""

        log_content = ""

        if selected_log:
            log_path = os.path.join(log_dir, selected_log)

            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()

        logger.info("用户查看系统日志文件：%s", selected_log)

        return render_template(
            "system_logs.html",
            log_files=files,
            current_log=selected_log,
            log_content=log_content
        )

    except Exception as e:
        logger.error("系统日志模块错误：%s", str(e))
        return "日志读取失败"


# =========================
# 程序入口
# =========================

if __name__ == "__main__":
    init_database()
    app.run(debug=True, use_reloader=False)
