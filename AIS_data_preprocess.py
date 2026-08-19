import pandas as pd
import numpy as np

STATIC_THRESHOLD_M = 10  # 船舶首尾总位移小于该值判定为静止船舶，整船剔除
JUMP_SPEED_MPS = 20  # 瞬时速度超过该值判定经纬度跳变，删除该异常点
MAX_TIME_GAP_SEC = 3600  # 两点时间间隔超过该阈值，不做跳变判定（断联保护）
MAX_VALID_SOG = 40  # SOG对地航速最大有效值(节)，超过视为传感器故障脏点
KNOT_TO_MPS = 0.5144  # 节 转 m/s 换算系数


df = pd.read_csv("C:/Users/13103/Desktop/aisdk-2026-06-10.csv")

# ===================== 基础预处理 =====================
# 时间转换、排序
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
df = df.sort_values(by=["MMSI", "Timestamp"]).reset_index(drop=True)

# 去重：同一船舶同一时间重复报文删除
df = df.drop_duplicates(subset=["MMSI", "Timestamp"], keep="first").reset_index(drop=True)

# 过滤合法经纬度
df = df[(df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180))].reset_index(drop=True)

# 过滤SOG不合理脏数据
df = df[df["SOG"] <= MAX_VALID_SOG].reset_index(drop=True)

# ===================== 保留指定航行状态=====================
keep_status = {
    "Under way using engine",
    "Under way sailing",
    "Constrained by her draught",
    "Restricted manoeuvrability",
    "Engaged in fishing",
    "Power‑driven vessel pushing ahead or towing alongside",
    "Power‑driven vessel towing astern",
    "Not under command"
}
df = df[df["Navigational status"].isin(keep_status)].reset_index(drop=True)


# ===================== 工具函数：Haversine 球面距离计算(米) =====================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c


# =====================  按MMSI逐船清洗：剔除静止船、轨迹跳变点 =====================
clean_list = []
remove_static_ship = 0
remove_jump_point = 0

for mmsi, group in df.groupby("MMSI"):
    group = group.reset_index(drop=True)
    if len(group) < 2:
        continue

    # 判断整船是否静止：首尾点位位移
    total_move = haversine(
        group["Latitude"].iloc[0], group["Longitude"].iloc[0],
        group["Latitude"].iloc[-1], group["Longitude"].iloc[-1]
    )
    if total_move < STATIC_THRESHOLD_M:
        remove_static_ship += 1
        continue

    # 计算相邻点时间差、位移、瞬时速度
    group["dt_sec"] = group["Timestamp"].diff().dt.total_seconds()
    group["dist_m"] = haversine(
        group["Latitude"].shift(1), group["Longitude"].shift(1),
        group["Latitude"], group["Longitude"]
    )
    group["inst_speed_mps"] = group["dist_m"] / group["dt_sec"]

    # 判定有效点位：间隔过大保留；速度不超标保留；首行强制保留
    valid_mask = (group["dt_sec"] > MAX_TIME_GAP_SEC) | (group["inst_speed_mps"] <= JUMP_SPEED_MPS)
    valid_mask.iloc[0] = True

    remove_jump_point += len(valid_mask) - valid_mask.sum()
    clean_grp = group[valid_mask].drop(columns=["dt_sec", "dist_m", "inst_speed_mps"])
    clean_list.append(clean_grp)

# =====================  合并结果 & 保存文件 =====================
df_clean = pd.concat(clean_list, ignore_index=True)

print("===== 清洗汇总 =====")
print(f"原始数据行数: {len(df)}")
print(f"清洗后有效行数: {len(df_clean)}")
print(f"剔除静止船舶总数: {remove_static_ship}")
print(f"剔除轨迹跳变异常点总数: {remove_jump_point}")
print(f"剩余有效船舶数量: {df_clean['MMSI'].nunique()}")

df_clean.to_csv("ais_preprocessed_result.csv", index=False)
print("清洗完成，文件输出：ais_preprocessed_result.csv")