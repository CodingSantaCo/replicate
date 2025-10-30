#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算B段(Type B segments)的非联网车辆分布
=====================================
实现论文 Section 3.2 Proposition 2 Part B (Equations 5-11)

Type B段特征: 至少一端的车辆在移动 (Vi ≠ 0 ∪ Vi+1 ≠ 0)

主要步骤:
1. 计算每个B段的最大容量 Q̃_B_i (Eq. 7)
2. 按比例分配剩余车辆到各B段 (Eq. 6, 8)
3. 计算可行插入空间 [L_i^(l), L_i^(u)] (Eq. 9, 10)
4. 均匀分布车辆位置 (Eq. 5)
5. 计算车辆速度 (Eq. 11)
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path

# ==================== 配置参数 ====================
# 输入文件
SEGMENTS_FILE = "segments_AB_resegmented_v05_entr.csv"  # AB段划分
Q_FILE = "eor_Q_section31_fixed.csv"                    # 总车辆数Q
A_COUNTS_FILE = "A_counts_improved.csv"                 # A段车辆数

# 输出文件
OUT_COUNTS = "B_counts_improved.csv"                    # B段车辆数量
OUT_POSITIONS = "B_positions_improved.csv"              # B段车辆位置和速度

# 物理参数
L_E = 6.44        # 平均有效车辆长度 (m)
DELTA_T = 1.59    # 最小安全时间车头间距 (s) - 论文 Section 5.1
LANE_LEN = 1000.0 # 车道长度/停止线坐标 (m)

# 诊断参数
VERBOSE = True    # 是否输出详细信息
# ==================================================


def round_nearest(x: float) -> int:
    """
    四舍五入到最近的整数
    对应论文中的 ⌊·⌉ 符号
    """
    return int(round(x))


def compute_Q_tilde_B_i(Li: float, Li_plus_1: float, 
                        Vi: float, Vi_plus_1: float, 
                        is_boundary: bool) -> int:
    """
    计算B段中可以插入的最大车辆数 Q̃_B_i
    实现论文 Eq. (7)
    
    推导基于:
    k = [2(Li - Li+1) - 2ΔtVi+1] / [Δt(Vi + Vi+1)]
    
    参数:
        Li: 下游位置 (m)
        Li_plus_1: 上游位置 (m)
        Vi: 下游速度 (m/s)
        Vi_plus_1: 上游速度 (m/s)
        is_boundary: 是否是边界段 (i=0 或 i=m)
    
    返回:
        Q̃_B_i: 最大可插入车辆数
    """
    # 如果两端都停止,这不是B段
    if Vi == 0 and Vi_plus_1 == 0:
        return 0
    
    # 计算 k 值 (Eq. 7中的中间变量)
    numerator = 2 * (Li - Li_plus_1) - 2 * DELTA_T * Vi_plus_1
    denominator = DELTA_T * (Vi + Vi_plus_1)
    
    if denominator <= 0:
        return 0
    
    k = numerator / denominator
    
    # 根据是否是边界段确定 Q̃_B_i
    if is_boundary:
        # i ∈ {0, m}: 可以在边界插入车辆
        Q_tilde = round_nearest(k + 1)
    else:
        # i ∈ (0, m): 两端都被CV封闭
        Q_tilde = round_nearest(k)
    
    return max(0, Q_tilde)


def distribute_vehicles_to_segments(Q_remaining: int, 
                                   Q_tilde_values: list) -> list:
    """
    将剩余车辆按比例分配到各个B段
    实现论文 Eq. (6) 和 Eq. (8)
    
    核心思想:
    1. 按 Q̃_B_i 作为权重进行比例分配
    2. 处理舍入误差产生的excess
    
    参数:
        Q_remaining: 分配给B段的剩余车辆总数
        Q_tilde_values: 各B段的最大容量列表
    
    返回:
        Q_B_values: 各B段实际分配的车辆数列表
    """
    sum_Q_tilde = sum(Q_tilde_values)
    Q_B_values = []
    excess = 0  # 累积的多余车辆 (ei in Eq. 8)
    
    for i, Q_tilde in enumerate(Q_tilde_values):
        if sum_Q_tilde > 0:
            # 缩放因子 ρ = Q_remaining / Σ Q̃_B_i
            # Q_B_i = ⌊ρ * Q̃_B_i⌉ + e_{i-1}
            scaled = (Q_remaining * Q_tilde) / sum_Q_tilde
            Q_B_i = round_nearest(scaled) + excess
        else:
            Q_B_i = excess
        
        # 确保 Q_B_i 不超过该段的最大容量 Q̃_B_i
        if Q_B_i > Q_tilde:
            excess = Q_B_i - Q_tilde  # 超出部分留给下一段
            Q_B_i = Q_tilde
        else:
            excess = 0
        
        Q_B_values.append(Q_B_i)
    
    return Q_B_values


def compute_vehicle_speeds(Vi: float, Vi_plus_1: float, 
                          Q_B_i: int, segment_type: str) -> list:
    """
    计算插入车辆的速度
    实现论文 Eq. (11)
    
    假设速度线性变化
    
    参数:
        Vi: 下游CV速度 (m/s)
        Vi_plus_1: 上游CV速度 (m/s)
        Q_B_i: 该段插入的车辆数
        segment_type: "first" (i=0), "middle" (0<i<m), "last" (i=m)
    
    返回:
        speeds: 各车辆速度列表
    """
    if Q_B_i == 0:
        return []
    
    speeds = []
    
    if segment_type == "first":  # i = 0
        # V_j^0 = V1 + j * (V0 - V1) / Q_B_0
        delta_V = (Vi - Vi_plus_1) / Q_B_i
        for j in range(1, Q_B_i + 1):
            V_j = Vi_plus_1 + j * delta_V
            speeds.append(V_j)
    
    elif segment_type == "last":  # i = m
        # V_j^m = V_{m+1} + (j-1) * (V_m - V_{m+1}) / Q_B_m
        delta_V = (Vi - Vi_plus_1) / Q_B_i
        for j in range(1, Q_B_i + 1):
            V_j = Vi_plus_1 + (j - 1) * delta_V
            speeds.append(V_j)
    
    else:  # middle: i ∈ (0, m)
        # V_j^i = V_{i+1} + j * (V_i - V_{i+1}) / (Q_B_i + 1)
        delta_V = (Vi - Vi_plus_1) / (Q_B_i + 1)
        for j in range(1, Q_B_i + 1):
            V_j = Vi_plus_1 + j * delta_V
            speeds.append(V_j)
    
    return speeds


def compute_feasible_space(Li: float, Li_plus_1: float, 
                          Vi: float, Vi_plus_1: float,
                          Q_B_i: int, segment_type: str,
                          speeds: list) -> tuple:
    """
    计算可行的车辆插入空间 [L_i^(l), L_i^(u)]
    实现论文 Eq. (9) 和 Eq. (10)
    
    参数:
        Li, Li_plus_1: 下游和上游位置
        Vi, Vi_plus_1: 下游和上游速度
        Q_B_i: 插入车辆数
        segment_type: 段类型
        speeds: 车辆速度列表
    
    返回:
        (L_lower, L_upper): 下界和上界
    """
    if Q_B_i == 0:
        return None, None
    
    # 计算下界 L_i^(l) - Eq. (9)
    if segment_type == "last":  # i = m (上游入口)
        L_lower = 0.0
    else:
        # 保持与上游车辆的安全距离
        L_lower = Li_plus_1 + max(Vi_plus_1 * DELTA_T, L_E)
    
    # 计算上界 L_i^(u) - Eq. (10)
    if segment_type == "first":  # i = 0 (停止线)
        L_upper = LANE_LEN
    else:
        # 保持与下游车辆的安全距离
        # 需要最后插入车辆的速度 V_{Q_B_i}^i
        if Q_B_i > 0 and len(speeds) >= Q_B_i:
            V_Q_B_i = speeds[-1]  # 最后一辆车的速度
        else:
            V_Q_B_i = Vi_plus_1
        
        L_upper = Li - max(V_Q_B_i * DELTA_T, L_E)
    
    return L_lower, L_upper


def distribute_positions_uniformly(Q_B_i: int, L_lower: float, 
                                  L_upper: float) -> list:
    """
    在可行空间内均匀分布车辆位置
    实现论文 Eq. (5)
    
    如果 Q_B_i = 1: 放在中心
    如果 Q_B_i > 1: 均匀分布在 [L_lower, L_upper]
    
    参数:
        Q_B_i: 车辆数
        L_lower, L_upper: 可行空间边界
    
    返回:
        positions: 车辆位置列表
    """
    if Q_B_i == 0:
        return []
    
    if Q_B_i == 1:
        # 单个车辆放在中心
        pos = 0.5 * (L_upper + L_lower)
        return [pos]
    else:
        # 多个车辆均匀分布
        positions = []
        for j in range(1, Q_B_i + 1):
            pos = L_lower + (j - 1) * (L_upper - L_lower) / (Q_B_i - 1)
            positions.append(pos)
        return positions


def main():
    print("=" * 70)
    print("B段车辆分布计算工具 (基于论文 Section 3.2 - Type B)")
    print("=" * 70)
    
    # 检查输入文件
    required_files = {
        "分段数据": SEGMENTS_FILE,
        "总车辆数": Q_FILE,
        "A段车辆数": A_COUNTS_FILE
    }
    
    for desc, filepath in required_files.items():
        if not Path(filepath).exists():
            print(f"❌ 错误: 找不到{desc}文件 '{filepath}'")
            input("\n按回车键退出...")
            return
    
    # 读取数据
    print(f"\n📂 正在读取数据文件...")
    try:
        segments = pd.read_csv(SEGMENTS_FILE)
        Q_data = pd.read_csv(Q_FILE)
        A_counts = pd.read_csv(A_COUNTS_FILE)
        print(f"✓ 成功读取所有文件")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        input("\n按回车键退出...")
        return
    
    # 筛选B段
    B_segs = segments[segments["seg_type"].astype(str).str.upper() == "B"].copy()
    
    if B_segs.empty:
        print("⚠️  警告: 没有找到B段数据")
        pd.DataFrame().to_csv(OUT_COUNTS, index=False)
        pd.DataFrame().to_csv(OUT_POSITIONS, index=False)
        print(f"✓ 已保存空文件")
        input("\n按回车键退出...")
        return
    
    print(f"✓ 找到 {len(B_segs)} 个B段记录")
    
    # 数据类型转换
    for col in ["EoR_s", "up_m", "down_m", "up_speed", "down_speed"]:
        if col in B_segs.columns:
            B_segs[col] = pd.to_numeric(B_segs[col], errors="coerce")
    
    count_rows = []
    position_rows = []
    
    processed_times = 0
    total_B_vehicles = 0
    
    # 按时间处理
    print(f"\n⚙️  正在处理每个时间点的B段...")
    
    for t in sorted(B_segs["EoR_s"].dropna().unique()):
        processed_times += 1
        
        # 获取该时刻的总车辆数 Q
        q_row = Q_data[np.isclose(Q_data["EoR_s"], t, atol=0.01)]
        if q_row.empty:
            if VERBOSE:
                print(f"  ⚠️  时间 {t:.2f}s: 找不到总车辆数Q,跳过")
            continue
        
        Q_total = float(q_row.iloc[0]["Q_sec31_fixed"])
        
        # 获取该时刻A段已分配的车辆总数
        a_sum_row = A_counts[np.isclose(A_counts["EoR_s"], t, atol=0.01)]
        Q_A_total = int(a_sum_row["Q_Ai"].sum()) if not a_sum_row.empty else 0
        
        # 剩余需要分配给B段的车辆数
        Q_remaining = max(0, int(Q_total - Q_A_total))
        
        if VERBOSE:
            print(f"\n  时间 {t:.2f}s:")
            print(f"    总车辆数Q = {Q_total:.1f}")
            print(f"    A段车辆数 = {Q_A_total}")
            print(f"    B段剩余车辆数 = {Q_remaining}")
        
        # 获取该时刻的B段,按down_m降序排列(下游→上游)
        b_t = B_segs[np.isclose(B_segs["EoR_s"], t, atol=0.01)].copy()
        b_t = b_t.sort_values("down_m", ascending=False).reset_index(drop=True)
        
        if len(b_t) == 0:
            continue
        
        # ===== 步骤1: 计算各B段的最大容量 Q̃_B_i (Eq. 7) =====
        Q_tilde_values = []
        
        for i, row in b_t.iterrows():
            Li = float(row["down_m"])
            Li_plus_1 = float(row["up_m"])
            Vi = float(row["down_speed"])
            Vi_plus_1 = float(row["up_speed"])
            
            is_boundary = (i == 0) or (i == len(b_t) - 1)
            Q_tilde = compute_Q_tilde_B_i(Li, Li_plus_1, Vi, Vi_plus_1, is_boundary)
            Q_tilde_values.append(Q_tilde)
        
        sum_Q_tilde = sum(Q_tilde_values)
        
        if VERBOSE:
            print(f"    B段总容量 Σ Q̃_B_i = {sum_Q_tilde}")
        
        # ===== 步骤2: 按比例分配车辆到各B段 (Eq. 6, 8) =====
        Q_B_values = distribute_vehicles_to_segments(Q_remaining, Q_tilde_values)
        
        # ===== 步骤3&4: 计算每个B段的车辆位置和速度 =====
        for i, row in b_t.iterrows():
            Q_B_i = Q_B_values[i]
            Q_tilde_i = Q_tilde_values[i]
            
            Li = float(row["down_m"])
            Li_plus_1 = float(row["up_m"])
            Vi = float(row["down_speed"])
            Vi_plus_1 = float(row["up_speed"])
            
            # 确定段类型
            if i == 0:
                segment_type = "first"
            elif i == len(b_t) - 1:
                segment_type = "last"
            else:
                segment_type = "middle"
            
            # 记录车辆数量
            count_rows.append({
                "EoR_s": t,
                "lane": row.get("lane", "lane7_0"),
                "seg_idx_B": i,
                "Q_Bi": Q_B_i,
                "Q_tilde_Bi": Q_tilde_i,
                "down_m": Li,
                "up_m": Li_plus_1,
                "down_speed": Vi,
                "up_speed": Vi_plus_1,
                "segment_type": segment_type
            })
            
            total_B_vehicles += Q_B_i
            
            if Q_B_i == 0:
                continue
            
            # 计算车辆速度 (Eq. 11)
            speeds = compute_vehicle_speeds(Vi, Vi_plus_1, Q_B_i, segment_type)
            
            # 计算可行空间 (Eq. 9, 10)
            L_lower, L_upper = compute_feasible_space(
                Li, Li_plus_1, Vi, Vi_plus_1, Q_B_i, segment_type, speeds
            )
            
            if L_lower is None or L_upper is None or L_upper < L_lower:
                if VERBOSE:
                    print(f"    ⚠️  B段 {i}: 可行空间无效,跳过位置计算")
                continue
            
            # 均匀分布位置 (Eq. 5)
            positions = distribute_positions_uniformly(Q_B_i, L_lower, L_upper)
            
            # 保存位置和速度
            for j, (pos, speed) in enumerate(zip(positions, speeds), start=1):
                position_rows.append({
                    "EoR_s": t,
                    "lane": row.get("lane", "lane7_0"),
                    "seg_idx_B": i,
                    "j_in_seg": j,
                    "est_pos_m": float(pos),
                    "est_speed_m_s": float(speed),
                    "L_lower": float(L_lower),
                    "L_upper": float(L_upper)
                })
    
    # 保存结果
    print(f"\n💾 正在保存结果...")
    
    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(OUT_COUNTS, index=False)
    print(f"✓ 车辆数量结果已保存: {OUT_COUNTS}")
    print(f"   - 总行数: {len(df_counts)}")
    
    df_positions = pd.DataFrame(position_rows)
    df_positions.to_csv(OUT_POSITIONS, index=False)
    print(f"✓ 车辆位置结果已保存: {OUT_POSITIONS}")
    print(f"   - 总行数: {len(df_positions)}")
    
    # 统计摘要
    print(f"\n📊 统计摘要:")
    print(f"   - 处理的时间点: {processed_times}")
    print(f"   - B段总数: {len(df_counts)}")
    print(f"   - B段车辆总数: {total_B_vehicles}")
    
    if len(df_counts) > 0:
        print(f"   - 平均每段车辆数: {df_counts['Q_Bi'].mean():.2f}")
        print(f"   - 最大单段车辆数: {df_counts['Q_Bi'].max()}")
    
    if len(df_positions) > 0:
        print(f"   - 平均车辆速度: {df_positions['est_speed_m_s'].mean():.2f} m/s")
    
    print(f"\n" + "=" * 70)
    print("✅ 计算完成!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")