#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 Q 值 - 简化直接运行版本
=====================================
公式: Q = q_fixed * (1 - p_fixed) * (l / v_f) + R - R_CV

直接运行即可,所有参数已在代码中设置
"""
import pandas as pd
import numpy as np
import os

# ==================== 配置参数 ====================
# 输入输出文件(请根据实际情况修改)
INPUT_CSV = "dynamic_params_per_cycle.csv"  # 输入文件名
OUTPUT_CSV = "eor_Q_section31_fixed.csv"    # 输出文件名

# 固定参数
P_FIXED = 0.4           # CV渗透率
Q_PER_HOUR = 415.0      # 流量(车辆/小时)
LINK_LENGTH = 1000.0    # 路段长度(米)
FREE_FLOW_SPEED = 13.89 # 自由流速度(米/秒)

# 必需的CSV列名
REQUIRED_COLUMNS = ["EoR_s", "cycle_id", "R", "R_CV"]
# ==================================================


def main():
    print("=" * 60)
    print("交通流量 Q 值计算工具 - Section 3.1")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误: 找不到输入文件 '{INPUT_CSV}'")
        print(f"   当前目录: {os.getcwd()}")
        print(f"   请确保 {INPUT_CSV} 文件在当前目录下")
        return
    
    # 读取CSV文件
    print(f"\n📂 正在读取文件: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"✓ 成功读取 {len(df)} 行数据")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 检查必需列
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"❌ 错误: 缺少必需的列: {missing_cols}")
        print(f"   文件中的列: {list(df.columns)}")
        return
    print(f"✓ 列检查通过")
    
    # 数据预处理
    print(f"\n🔄 正在处理数据...")
    for col in ("R", "R_CV", "EoR_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    original_rows = len(df)
    df = df.dropna(subset=["R", "R_CV"]).copy()
    dropped_rows = original_rows - len(df)
    
    if dropped_rows > 0:
        print(f"⚠️  删除了 {dropped_rows} 行含空值的数据")
    print(f"✓ 有效数据行数: {len(df)}")
    
    # 计算参数
    q_per_sec = Q_PER_HOUR / 3600.0  # 转换为 车辆/秒
    travel_time = LINK_LENGTH / max(FREE_FLOW_SPEED, 1e-9)  # 行程时间(秒)
    
    print(f"\n📊 计算参数:")
    print(f"   - CV渗透率 (p):        {P_FIXED}")
    print(f"   - 流量 (q):            {Q_PER_HOUR} 车辆/小时 = {q_per_sec:.6f} 车辆/秒")
    print(f"   - 路段长度 (l):        {LINK_LENGTH} 米")
    print(f"   - 自由流速度 (v_f):    {FREE_FLOW_SPEED} 米/秒")
    print(f"   - 行程时间 (t_ff):     {travel_time:.3f} 秒")
    
    # 计算 Q 值
    print(f"\n⚙️  正在计算 Q 值...")
    R = df["R"].to_numpy(float)
    R_CV = df["R_CV"].to_numpy(float)
    
    # 核心公式: Q = q * (1 - p) * t_ff + (R - R_CV)
    Q = q_per_sec * (1.0 - P_FIXED) * travel_time + (R - R_CV)
    
    # 准备输出数据
    output_df = df[["cycle_id", "EoR_s", "R", "R_CV"]].copy()
    output_df["t_ff"] = travel_time
    output_df["Q_sec31_fixed"] = Q
    
    # 保存结果
    print(f"\n💾 正在保存结果...")
    try:
        output_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✓ 结果已保存到: {OUTPUT_CSV}")
        print(f"✓ 输出行数: {len(output_df)}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return
    
    # 显示统计信息
    print(f"\n📈 Q 值统计:")
    print(f"   - 最小值: {Q.min():.4f}")
    print(f"   - 最大值: {Q.max():.4f}")
    print(f"   - 平均值: {Q.mean():.4f}")
    print(f"   - 标准差: {Q.std():.4f}")
    
    print(f"\n" + "=" * 60)
    print("✅ 计算完成!")
    print("=" * 60)
    
    # 显示前几行结果
    print(f"\n预览前5行结果:")
    print(output_df.head().to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")