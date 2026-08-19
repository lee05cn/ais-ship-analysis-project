# -*- coding: utf-8 -*-
"""
encounter_detect_cluster.py
功能：
1. 按1min时间窗口切分AIS数据
2. DBSCAN时空聚类，筛选潜在会遇候选船舶对（减少两两计算量）
3. 对簇内船舶计算DCPA / TCPA，输出会遇风险
输入：ais_preprocessed_result.csv
输出：encounter_risk.csv 会遇风险结果
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a),0,1))
    return c * 6371.0 / 1.852

def calculate_dcpa_tcpa(shipA, shipB):
    """
    输入两行AIS数据（Series）
    shipA,shipB: 必须包含 Latitude,Longitude,SOG,COG
    return dcpa(nm), tcpa(minute)
    SOG单位：节；COG单位：度
    """
    deg2rad = np.pi / 180.0
    # 船A
    lat_a = shipA["Latitude"] * deg2rad
    lon_a = shipA["Longitude"] * deg2rad
    sog_a = shipA["SOG"]
    cog_a = shipA["COG"] * deg2rad
    # 船B
    lat_b = shipB["Latitude"] * deg2rad
    lon_b = shipB["Longitude"] * deg2rad
    sog_b = shipB["SOG"]
    cog_b = shipB["COG"] * deg2rad

    # 位置差，海里
    R = 6371.0 / 1.852
    dx = R * np.cos(lat_b) * (lon_b - lon_a)
    dy = R * (lat_b - lat_a)

    # 速度分量
    vax = sog_a * np.sin(cog_a)
    vay = sog_a * np.cos(cog_a)
    vbx = sog_b * np.sin(cog_b)
    vby = sog_b * np.cos(cog_b)

    rel_vx = vbx - vax
    rel_vy = vby - vay
    rel_speed_sq = rel_vx**2 + rel_vy**2

    if rel_speed_sq < 1e-6:
        dcpa = np.sqrt(dx**2 + dy**2)
        tcpa = np.inf
        return dcpa, tcpa

    tcpa = -(dx * rel_vx + dy * rel_vy) / rel_speed_sq
    dcpa_sq = (dx**2 + dy**2) - ((dx*rel_vx + dy*rel_vy)**2) / rel_speed_sq
    dcpa_sq = max(dcpa_sq, 0.0)
    dcpa = np.sqrt(dcpa_sq)

    tcpa_min = tcpa if tcpa>0 else np.inf
    return dcpa, tcpa_min

def run_encounter_detect(csv_path="ais_preprocessed_result.csv",
                          dcpa_thresh_nm=0.5,
                          tcpa_thresh_min=10.0,
                          time_window_freq="1min"):
    """
    dcpa_thresh_nm：判定会遇风险DCPA阈值(海里)
    tcpa_thresh_min：TCPA预警阈值(分钟)
    """
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["MMSI","Latitude","Longitude","SOG","COG"])
    df = df[(df["SOG"]>0.1)]
    df["time_window"] = df["Timestamp"].dt.floor(time_window_freq)

    risk_result = []

    for win_time, win_df in df.groupby("time_window"):
        win_df = win_df.reset_index(drop=True)
        if len(win_df) < 2:
            continue

        feat_arr = win_df[["Longitude","Latitude","SOG","COG"]].to_numpy()
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_arr)

        # DBSCAN做时空聚类，找出空间状态接近的候选船
        db = DBSCAN(eps=0.4, min_samples=2)
        win_df["cluster"] = db.fit_predict(feat_scaled)

        for c_label, c_df in win_df.groupby("cluster"):
            if c_label == -1:
                continue
            c_df = c_df.reset_index(drop=True)
            n = len(c_df)
            # 簇内两两计算DCPA/TCPA
            for i in range(n):
                for j in range(i+1, n):
                    s1 = c_df.iloc[i]
                    s2 = c_df.iloc[j]
                    dcpa, tcpa = calculate_dcpa_tcpa(s1, s2)
                    is_risk = False
                    if np.isfinite(tcpa) and (dcpa <= dcpa_thresh_nm) and (tcpa <= tcpa_thresh_min):
                        is_risk = True
                    risk_result.append({
                        "time_window": win_time,
                        "mmsi_1": int(s1["MMSI"]),
                        "mmsi_2": int(s2["MMSI"]),
                        "dcpa_nm": round(dcpa,3),
                        "tcpa_min": round(tcpa,3) if np.isfinite(tcpa) else np.inf,
                        "risk_flag": is_risk
                    })
    out_df = pd.DataFrame(risk_result)
    out_df.to_csv("encounter_risk.csv", index=False, encoding="utf‑8‑sig")
    print(f"会遇检测完成，共处理候选对：{len(out_df)}")
    risk_cnt = out_df["risk_flag"].sum()
    print(f"高风险会遇对数：{risk_cnt}")
    print("输出文件：encounter_risk.csv")
    return out_df

if __name__ == "__main__":
    run_encounter_detect()