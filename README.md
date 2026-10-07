# 基于多变量的光伏发电预测系统

基于 **Flask + Scikit-learn + TensorFlow (LSTM)** 的多变量光伏发电预测 Web 系统。系统读取光照、散射辐射、直射辐射、温度等多变量历史数据，提供**线性回归预测**与**LSTM 短时（0-15 分钟）趋势预测**，并内置监控大屏、数据管理、模型训练、结果评估、AI 智能问答等完整功能。

> 本系统在 GitHub 上有两个内容完全一致的仓库，任选其一克隆即可：
> - [pv-power-forecast-system](https://github.com/kulat55/pv-power-forecast-system)（主仓库）
> - [pv-forecast-system](https://github.com/kulat55/pv-forecast-system)（短名仓库）
> 注：GitHub 不支持中文仓库名，中文字符会被自动替换为连字符，故未使用中文仓库名。

## 功能特性

- **监控大屏**：发电总量、平均功率、最大功率、平均温度、平均光照实时统计
- **数据管理**：CSV 数据上传入库、数据集信息与质量检测（自动清洗、缺失值处理）
- **图表展示**：发电功率趋势图、LSTM 训练损失曲线
- **模型训练**：网页一键触发训练，线性回归 + LSTM 自动完成、评估并入库
- **功率预测**
  - 线性回归：基于光照 / 散射 / 直射 / 温度预测发电功率，支持与实际值对比误差
  - LSTM：基于历史时序数据预测未来 0-15 分钟发电功率，含趋势判断
- **模型评估**：MAE / MSE / R² 指标对比（线性回归 + LSTM）
- **AI 智能助手**：基于本地 Ollama 的问答助手（默认模型 qwen2.5:1.5b）
- **AI 数据分析**：自动生成光伏发电运行分析报告
- **用户体系**：注册 / 登录 / 会话管理
- **系统日志**：运行日志查看、预测记录管理

## 技术栈

| 类别 | 技术 |
|---|---|
| 后端框架 | Flask (Python) |
| 数据库 | SQLite |
| 机器学习 | Scikit-learn（线性回归） |
| 深度学习 | TensorFlow / Keras（LSTM） |
| 数据科学 | pandas / numpy / joblib |
| AI 助手 | Ollama 本地大模型（qwen2.5:1.5b） |
| 前端 | Jinja2 模板 + ECharts |

## 快速开始（下载即用）

### 环境要求

- Windows / macOS / Linux
- Python **3.9 ~ 3.12**（推荐 3.10）
- （可选）[Ollama](https://ollama.com/) —— 仅使用「AI 智能助手」功能时需要

### 1. 克隆仓库

```bash
git clone https://github.com/kulat55/pv-power-forecast-system.git
cd pv-power-forecast-system
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> TensorFlow 安装包较大（约 200-500MB），请耐心等待。若安装缓慢可换用国内镜像：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. 启动系统

```bash
python app.py
```

启动成功后，浏览器打开 **http://127.0.0.1:5000** ，注册账号即可使用。

> 系统首次启动会自动创建 `database.db` 数据库与全部数据表，无需手动初始化。
> 仓库已附带预训练模型（`models/pv_model.pkl`、`models/lstm_model.h5`），克隆后即可直接预测；如需重新训练，进入「模型训练」页面点击按钮即可。

### 4. （可选）启用 AI 智能助手

AI 助手调用本地 Ollama 服务（默认地址 `http://127.0.0.1:11434`），需要先安装并启动 Ollama，然后拉取模型：

```bash
ollama pull qwen2.5:1.5b
```

未安装 Ollama 时，系统其余功能不受影响，「AI 助手」页面会提示服务调用失败。

## 目录结构

```
pv_forecast_system/
├── app.py                  # Flask 主程序（路由、业务逻辑）
├── init_db.py              # 数据库初始化脚本
├── requirements.txt        # Python 依赖清单
├── utils/
│   ├── data_loader.py      # 数据读取与清洗（支持 3 种 CSV 格式）
│   └── logger.py           # 日志初始化
├── models/
│   ├── train_model.py      # 线性回归训练脚本
│   ├── train_lstm.py       # LSTM 训练脚本
│   ├── pv_model.pkl        # 预训练线性回归模型
│   ├── lstm_model.h5       # 预训练 LSTM 模型
│   └── lstm_scaler.pkl     # LSTM 归一化器
├── data/
│   ├── pv_data.csv         # 示例光伏数据（默认数据集）
│   ├── 01-pov.csv          # 备用数据集 1
│   └── 02-pov.csv          # 备用数据集 2
└── templates/              # 前端页面模板（15 个页面）
```

## 支持的 CSV 数据格式

系统支持以下三种表头格式，上传时自动识别并清洗（去重、缺失值填充、按时间排序）：

| 格式 | 字段 |
|---|---|
| PVOD 格式 | `Time, lmd_totalirrad, lmd_diffuseirrad, lmd_temperature, power` |
| 气象格式 | `date, T, DHI, DNI, GHI, Target` |
| 系统标准格式 | `time, light, diffuse, direct, temperature, power` |

## 安全说明

- 会话密钥默认使用内置值，生产环境请通过环境变量设置：`set SECRET_KEY=你的随机字符串`（Windows）或 `export SECRET_KEY=你的随机字符串`（Linux/macOS）
- 演示项目默认以明文存储用户密码，仅建议本地或教学环境使用

## 许可证

本项目基于 MIT License 开源，详情见 [LICENSE](LICENSE)。
