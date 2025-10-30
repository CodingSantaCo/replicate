#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算A段(Type A segments)的车辆数量和位置分布
=====================================
基于论文 Section 3.2 (Proposition 2 - Type A segments)

公式:
- 第0个A段 (stopline到L1): Q = ⌊(l - L₁)/lₑ⌉
- 第i个A段 (Lᵢ到Lᵢ₊₁):    Q = ⌊(Lᵢ - lₑ - Lᵢ₊₁)/lₑ⌉

位置分布 (均匀分布):
- 第0个A段: 从L₁开始,间隔为 (l - L₁)/Q
- 第i个A段: 从Lᵢ₊₁开始,间隔为 (Lᵢ - lₑ - Lᵢ₊₁)/Q
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==================== 配置参数 ====================
IN_FILE = "segments_AB_resegmented_v05_entr.csv"  # 输入:AB段划分结果
OUT_CNT = "A_counts_improved.csv"                  # 输出:A段车辆数量
OUT_POS = "A_positions_improved.csv"               # 输出:A段车辆位置

STOP_POS = 1000.0  # 停止线坐标 (m)
L_E = 6.44         # 平均有效车辆长度 (m)
STOP_TH = 0.5      # 停止速度阈值 (m/s) - 仅用于诊断
# ==================================================


def round_nearest(x: float) -> int:
    """
    四舍五入到最近的整数
    对应论文中的 ⌊·⌉ 符号
    """
    return int(round(x))


def compute_type_a_vehicles(up_m: float, down_m: float, 
                           segment_index: int, stopline_pos: float, 
                           effective_length: float) -> tuple:
    """
    计算Type A段中的车辆数量
    
    参数:
        up_m: 上游边界位置 (米)
        down_m: 下游边界位置 (米)
        segment_index: 段索引 (0表示第一个,从stopline开始)
        stopline_pos: 停止线位置 (米)
        effective_length: 有效车辆长度 (米)
    
    返回:
        (车辆数量, 可用空间, 使用的公式描述)
    """
    if segment_index == 0:
        # 第0个A段: 从stopline到第一个停止的CV
        # Q = ⌊(l - L₁)/lₑ⌉
        available_space = stopline_pos - up_m
        Q = round_nearest(max(0.0, available_space / effective_length))
        formula = f"i=0: round(({stopline_pos:.2f} - {up_m:.2f})/{effective_length:.2f})"
    else:
        # 第i个A段: 两个停止CV之间
        # Q = ⌊(Lᵢ - lₑ - Lᵢ₊₁)/lₑ⌉
        available_space = down_m - effective_length - up_m
        Q = round_nearest(max(0.0, available_space / effective_length))
        formula = f"i={segment_index}: round(({down_m:.2f} - {effective_length:.2f} - {up_m:.2f})/{effective_length:.2f})"
    
    return Q, available_space, formula


def distribute_vehicles_uniformly(Q: int, up_m: float, down_m: float, 
                                 segment_index: int, stopline_pos: float) -> list:
    """
    在A段内均匀分布车辆位置
    
    根据论文 Eq.(3):
    - 第0个A段: Lⱼ = L₁ + j*(l - L₁)/Q
    - 第i个A段: Lⱼ = Lᵢ₊₁ + j*(Lᵢ - lₑ - Lᵢ₊₁)/Q
    
    参数:
        Q: 车辆数量
        up_m: 上游边界位置
        down_m: 下游边界位置
        segment_index: 段索引
        stopline_pos: 停止线位置
    
    返回:
        车辆位置列表
    """
    if Q <= 0:
        return []
    
    positions = []
    
    if segment_index == 0:
        # 第0个A段: 从up_m到stopline均匀分布
        step = (stopline_pos - up_m) / Q
        for j in range(1, Q + 1):
            pos = up_m + j * step
            positions.append(pos)
    else:
        # 第i个A段: 从up_m到(down_m - L_E)均匀分布
        step = (down_m - L_E - up_m) / Q
        for j in range(1, Q + 1):
            pos = up_m + j * step
            positions.append(pos)
    
    return positions


def main():
    print("=" * 70)
    print("A段车辆数量和位置计算工具 (基于论文 Section 3.2)")
    print("=" * 70)
    
    # 检查输入文件
    input_path = Path(IN_FILE)
    if not input_path.exists():
        print(f"❌ 错误: 找不到输入文件 '{IN_FILE}'")
        print(f"   请先运行 segments_AB_resegmented_v05_entr.py 生成分段数据")
        input("\n按回车键退出...")
        return
    
    # 读取分段数据
    print(f"\n📂 正在读取文件: {IN_FILE}")
    try:
        segs = pd.read_csv(input_path)
        print(f"✓ 成功读取 {len(segs)} 行分段数据")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        input("\n按回车键退出...")
        return
    
    # 数据类型转换
    for col in ["EoR_s", "up_m", "down_m", "up_speed", "down_speed"]:
        if col in segs.columns:
            segs[col] = pd.to_numeric(segs[col], errors="coerce")
    
    # 筛选A段
    A_segments = segs[segs["seg_type"].astype(str).str.upper() == "A"].copy()
    
    if A_segments.empty:
        print("⚠️  警告: 没有找到A段数据")
        # 创建空输出
        pd.DataFrame(columns=[
            "EoR_s", "lane", "seg_idx_A", "Q_Ai", "up_m", "down_m", 
            "up_anchor", "down_anchor", "available_space_m", "formula", "note"
        ]).to_csv(OUT_CNT, index=False)
        
        pd.DataFrame(columns=[
            "EoR_s", "lane", "seg_idx_A", "j_in_seg", "est_pos_m"
        ]).to_csv(OUT_POS, index=False)
        
        print(f"✓ 已保存空文件: {OUT_CNT} 和 {OUT_POS}")
        input("\n按回车键退出...")
        return
    
    print(f"✓ 找到 {len(A_segments)} 个A段")
    
    # 确定分组键
    group_keys = ["EoR_s"]
    if "lane" in A_segments.columns:
        group_keys.append("lane")
    
    # 结果存储
    rows_count = []    # 车辆数量结果
    rows_position = [] # 车辆位置结果
    
    total_vehicles = 0
    processed_groups = 0
    
    print(f"\n⚙️  正在处理每个时间点和车道的A段...")
    
    # 按时间和车道分组处理
    for key, group in A_segments.groupby(group_keys, as_index=False):
        processed_groups += 1
        
        # 提取分组信息
        if isinstance(key, tuple):
            time_t = float(key[0])
            lane_id = key[1] if len(key) > 1 else "lane7_0"
        else:
            time_t = float(key)
            lane_id = group["lane"].iloc[0] if "lane" in group.columns else "lane7_0"
        
        # 按down_m降序排序 (从下游到上游)
        group = group.sort_values("down_m", ascending=False).reset_index(drop=True)
        
        # 提取位置和速度
        L_down = group["down_m"].to_numpy(float)
        L_up = group["up_m"].to_numpy(float)
        
        # 确保 down_m >= up_m
        for i in range(len(group)):
            if L_down[i] < L_up[i]:
                L_down[i], L_up[i] = L_up[i], L_down[i]
        
        # 处理每个A段
        for i in range(len(group)):
            seg_info = group.iloc[i]
            up_m = float(L_up[i])
            down_m = float(L_down[i])
            
            up_anchor = seg_info.get("up_anchor", "")
            down_anchor = seg_info.get("down_anchor", "")
            up_speed = float(seg_info.get("up_speed", 0.0))
            down_speed = float(seg_info.get("down_speed", 0.0))
            
            # ===== 核心计算 =====
            Q, available_space, formula = compute_type_a_vehicles(
                up_m=up_m,
                down_m=down_m,
                segment_index=i,
                stopline_pos=STOP_POS,
                effective_length=L_E
            )
            
            total_vehicles += Q
            
            # 生成车辆位置
            if Q > 0:
                positions = distribute_vehicles_uniformly(
                    Q=Q,
                    up_m=up_m,
                    down_m=down_m,
                    segment_index=i,
                    stopline_pos=STOP_POS
                )
                
                for j, pos in enumerate(positions, start=1):
                    rows_position.append({
                        "EoR_s": time_t,
                        "lane": lane_id,
                        "seg_idx_A": i,
                        "j_in_seg": j,
                        "est_pos_m": float(pos)
                    })
            
            # 诊断信息
            note = ""
            if str(down_anchor).lower() != "stopline" and down_speed >= STOP_TH:
                note += "down_anchor not fully stopped; "
            if up_speed >= STOP_TH:
                note += "up_anchor not fully stopped; "
            
            # 保存数量信息
            rows_count.append({
                "EoR_s": time_t,
                "lane": lane_id,
                "seg_idx_A": i,
                "Q_Ai": int(Q),
                "up_m": up_m,
                "down_m": down_m,
                "up_anchor": up_anchor,
                "down_anchor": down_anchor,
                "available_space_m": float(available_space),
                "formula": formula,
                "note": note.strip()
            })
    
    # 保存结果
    print(f"\n💾 正在保存结果...")
    
    df_count = pd.DataFrame(rows_count)
    df_count.to_csv(OUT_CNT, index=False)
    print(f"✓ 车辆数量结果已保存: {OUT_CNT}")
    print(f"   - 总行数: {len(df_count)}")
    print(f"   - A段总数: {len(df_count)}")
    
    df_position = pd.DataFrame(rows_position)
    df_position.to_csv(OUT_POS, index=False)
    print(f"✓ 车辆位置结果已保存: {OUT_POS}")
    print(f"   - 总行数: {len(df_position)}")
    print(f"   - 估计车辆总数: {total_vehicles}")
    
    # 统计摘要
    print(f"\n📊 统计摘要:")
    print(f"   - 处理的时间点/车道组合: {processed_groups}")
    print(f"   - A段总数: {len(df_count)}")
    print(f"   - 估计车辆总数: {total_vehicles}")
    
    if len(df_count) > 0:
        print(f"   - 平均每段车辆数: {df_count['Q_Ai'].mean():.2f}")
        print(f"   - 最大单段车辆数: {df_count['Q_Ai'].max()}")
        print(f"   - 最小单段车辆数: {df_count['Q_Ai'].min()}")
    
    print(f"\n" + "=" * 70)
    print("✅ 计算完成!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")