# -*- coding: utf-8 -*-
"""
trajectory_cluster_route.py
Function: AIS trajectory HDBSCAN clustering, mine main sea routes, output visualization image
Input: ais_preprocessed_result.csv
Output: route_cluster.png , trajectory_cluster_result.csv
Note: Only use trajectory center point(lat/lon) for spatial clustering.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan


def haversine(lat1, lon1, lat2, lon2):
    """Calculate spherical distance, unit: nautical miles nm"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    return c * 6371.0 / 1.852


def group_trajectory_feature(group_df: pd.DataFrame):
    """Extract trajectory statistical features + trajectory center point"""
    group_df = group_df.sort_values("Timestamp").reset_index(drop=True)
    lat_arr = group_df["Latitude"].values
    lon_arr = group_df["Longitude"].values

    dist_list = []
    for i in range(1, len(group_df)):
        d = haversine(lat_arr[i-1], lon_arr[i-1], lat_arr[i], lon_arr[i])
        dist_list.append(d)
    total_dist = np.sum(dist_list)

    avg_speed = group_df["SOG"].mean()
    cog_diff = np.abs(np.diff(group_df["COG"]))
    total_cog_change = np.sum(cog_diff) if len(cog_diff) > 0 else 0.0
    max_rot = group_df["ROT"].max()

    # 轨迹中心点，用于空间聚类
    mean_lat = group_df["Latitude"].mean()
    mean_lon = group_df["Longitude"].mean()

    return pd.Series({
        "total_dist_nm": total_dist,
        "avg_sog": avg_speed,
        "total_cog_change": total_cog_change,
        "max_rot": max_rot,
        "point_cnt": len(group_df),
        "mean_lat": mean_lat,
        "mean_lon": mean_lon
    })


def resample_single_trajectory(group_df: pd.DataFrame, resample_freq="3min"):
    """Resample single ship trajectory to uniform time interval"""
    g = group_df.copy().sort_values("Timestamp").set_index("Timestamp")
    numeric_cols = g.select_dtypes(include=[np.number]).columns.tolist()
    g_res = g.resample(resample_freq).first()
    g_res[numeric_cols] = g_res[numeric_cols].interpolate(method="linear")
    return g_res.reset_index()


def run_trajectory_cluster(csv_path: str = "ais_preprocessed_result.csv"):
    df_raw = pd.read_csv(csv_path)
    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], errors="coerce")
    df_raw = df_raw.dropna(subset=["MMSI","Latitude","Longitude","SOG","COG"])

    print(f"[Trajectory Cluster] Raw data rows {len(df_raw)}, vessel count {df_raw['MMSI'].nunique()}")

    resample_list = []
    for mmsi, g in df_raw.groupby("MMSI"):
        if len(g) < 5:
            continue
        rg = resample_single_trajectory(g, resample_freq="3min")
        rg["MMSI"] = mmsi
        resample_list.append(rg)
    if not resample_list:
        print("No data after resampling, exit")
        return
    df_res = pd.concat(resample_list, ignore_index=True)

    feat_df = df_res.groupby("MMSI").apply(group_trajectory_feature, include_groups=False).reset_index()

    # ========== 只使用【中心点经纬度】做空间聚类 ==========
    feat_cols = ["mean_lat", "mean_lon"]
    X = feat_df[feat_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 空间聚类参数，如簇太少/太多可以调这两个数字
    cluster = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=4)
    feat_df["cluster_id"] = cluster.fit_predict(X_scaled)

    print("\n==== Cluster Statistics ====")
    print(feat_df["cluster_id"].value_counts().sort_index())

    plt.figure(figsize=(12,7), dpi=120)
    noise_flag = True
    for cid in sorted(feat_df["cluster_id"].unique()):
        sub_mmsi = feat_df.loc[feat_df["cluster_id"]==cid, "MMSI"].tolist()
        sub_df = df_res[df_res["MMSI"].isin(sub_mmsi)]
        if cid == -1:
            if noise_flag:
                plt.scatter(sub_df["Longitude"], sub_df["Latitude"], s=2, c="gray", alpha=0.3, label="Noise / abnormal trajectory")
                noise_flag=False
        else:
            plt.scatter(sub_df["Longitude"], sub_df["Latitude"], s=2, alpha=0.7, label=f"Route_cluster_{cid}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AIS Trajectory HDBSCAN Clustering - Main Route Mining")
    plt.legend(markerscale=4)
    plt.grid(True, alpha=0.2)
    plt.savefig("route_cluster.png", bbox_inches="tight")
    plt.close()
    print("Image saved: route_cluster.png")

    feat_df.to_csv("trajectory_cluster_result.csv", index=False, encoding="utf-8-sig")
    print("Cluster result table saved: trajectory_cluster_result.csv")
    return feat_df


if __name__ == "__main__":
    run_trajectory_cluster()