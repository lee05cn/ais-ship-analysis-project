# main_run.py
"""
前置条件：已完成数据清洗，目录存在 ais_preprocessed_result.csv
执行顺序：
1. 轨迹重采样 + HDBSCAN 航道聚类 trajectory_cluster_route.py
2. DCPA-TCPA 船只会遇风险检测 encounter_detect_cluster.py
"""
import os
import sys
import time

def run_script(script_name: str, desc: str):
    print("=" * 70)
    print(f"【运行模块】{desc}")
    print(f"脚本文件：{script_name}")
    print("=" * 70)
    ret_code = os.system(f"python {script_name}")
    if ret_code != 0:
        print(f"\n❌ {script_name} 执行异常，流水线终止！")
        sys.exit(1)
    print(f"\n✅ {desc} 执行完成\n")
    time.sleep(1)


if __name__ == "__main__":
    print("===== AIS船舶轨迹分析 流水线启动（不含数据清洗） =====")
    print("📌 前置校验：请确认 ais_preprocessed_result.csv 已生成\n")

    # 第一步：轨迹重采样 + HDBSCAN航线聚类
    run_script("trajectory_cluster_route.py", "轨迹等时间重采样 & HDBSCAN航道聚类")

    # 第二步：DCPA/TCPA 会遇风险计算
    run_script("encounter_detect_cluster.py", "DCPA-TCPA 船舶会遇风险检测")

    print("全部模块运行结束！")