import pandas as pd


def clean_pv_data(df):
    """
    清洗光伏数据：
    1. 去除重复行
    2. 将数值字段转换为数字类型
    3. 缺失值使用前后填充
    4. 如果仍有缺失值，则使用0填充
    5. 按时间顺序排序
    """

    required_cols = [
        "time",
        "light",
        "diffuse",
        "direct",
        "temperature",
        "power"
    ]

    df = df[required_cols].copy()

    # 去除重复数据
    df = df.drop_duplicates()

    # 数值字段转换
    numeric_cols = [
        "light",
        "diffuse",
        "direct",
        "temperature",
        "power"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 时间字段转换
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    # 删除时间为空的数据
    df = df.dropna(
        subset=["time"]
    )

    # 按时间排序
    df = df.sort_values(
        by="time"
    )

    # 缺失值填充
    df[numeric_cols] = df[numeric_cols].ffill()
    df[numeric_cols] = df[numeric_cols].bfill()
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 时间转回字符串，方便页面显示
    df["time"] = df["time"].astype(str)

    # 重置索引
    df = df.reset_index(drop=True)

    return df


def load_pv_data(file_path="data/pv_data.csv"):
    """
    读取光伏CSV数据，并统一转换为系统标准字段：
    time, light, diffuse, direct, temperature, power
    """

    df = pd.read_csv(file_path)

    # 去除字段名前后空格
    df.columns = df.columns.str.strip()

    # =========================
    # 数据集1：PVOD格式
    # =========================
    if "Time" in df.columns and "lmd_totalirrad" in df.columns:
        new_df = pd.DataFrame()

        new_df["time"] = df["Time"]
        new_df["light"] = df["lmd_totalirrad"]
        new_df["diffuse"] = df["lmd_diffuseirrad"]

        # PVOD数据中如果没有直射辐照度，则统一补0
        new_df["direct"] = 0

        new_df["temperature"] = df["lmd_temperature"]
        new_df["power"] = df["power"]

        return clean_pv_data(new_df)

    # =========================
    # 数据集2：宁夏/气象格式
    # =========================
    if "date" in df.columns and "GHI" in df.columns:
        new_df = pd.DataFrame()

        new_df["time"] = df["date"]
        new_df["light"] = df["GHI"]
        new_df["diffuse"] = df["DHI"]
        new_df["direct"] = df["DNI"]
        new_df["temperature"] = df["T"]
        new_df["power"] = df["Target"]

        return clean_pv_data(new_df)

    # =========================
    # 数据集3：系统标准格式
    # =========================
    required_cols = [
        "time",
        "light",
        "diffuse",
        "direct",
        "temperature",
        "power"
    ]

    if all(col in df.columns for col in required_cols):
        return clean_pv_data(df)

    # =========================
    # 字段不匹配时提示错误
    # =========================
    raise ValueError(
        "数据集字段不符合系统要求，请检查CSV表头。"
        "系统支持字段格式："
        "1）Time,lmd_totalirrad,lmd_diffuseirrad,lmd_temperature,power；"
        "2）date,T,DHI,DNI,GHI,Target；"
        "3）time,light,diffuse,direct,temperature,power。"
    )