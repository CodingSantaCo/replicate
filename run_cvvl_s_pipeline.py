#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Pipeline for CVVL-S Baseline Reproduction - WITH WARMUP PERIOD
==================================================
修改:
1. 添加热身期支持 (前30个cycle)
2. 评估和可视化只使用热身期之后的数据
3. 添加详细的调试信息
4. 改进错误处理
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


class BaselinePipeline:
    """
    Main pipeline class for CVVL-S baseline reproduction with warmup period
    """
    
    def __init__(self, working_dir: str = "./", warmup_cycles: int = 30):
        """
        Initialize pipeline
        
        Args:
            working_dir: Working directory path
            warmup_cycles: Number of cycles to skip for warmup (default: 30)
        """
        self.working_dir = Path(working_dir)
        self.output_dir = Path("./")
        self.output_dir.mkdir(exist_ok=True)
        
        # File paths
        self.files = {
            'fcd': self.working_dir / 'fcd_output.xml',
            'eor': self.working_dir / 'holding_EoR_fixed.csv',
            'segments': self.working_dir / 'segments_AB_resegmented_v05_entr.csv',
            'dynamic_params': self.working_dir / 'dynamic_params_per_cycle.csv',
            'Q': self.working_dir / 'eor_Q_section31_fixed.csv',
            'A_counts': self.working_dir / 'A_counts_improved.csv',
            'A_positions': self.working_dir / 'A_positions_improved.csv',
            'B_counts': self.working_dir / 'B_counts_improved.csv',
            'B_positions': self.working_dir / 'B_positions_improved.csv'
        }
        
        # Baseline parameters (from Table 1)
        self.params = {
            'r': 30,           # Red period (s)
            'V/C': 0.5,        # Volume/Capacity ratio
            'p': 0.4,          # CV penetration rate
            'ToI': 'EoR',      # Time of Interest: End of Red
            'L_E': 6.44,       # Effective vehicle length (m)
            'STOP_POS': 1000.0,  # Stop line position (m)
            'DELTA_T': 1.59,   # Minimum safe time headway (s)
            'STOP_TH': 0.5,    # Stop speed threshold (m/s)
            'CYCLE': 60,       # Cycle length (s)
            'WARMUP_CYCLES': warmup_cycles,  # Number of warmup cycles to skip
        }
        
        # Calculate warmup time
        self.warmup_time = self.params['WARMUP_CYCLES'] * self.params['CYCLE']
        
        print("=" * 70)
        print("CVVL-S Baseline Pipeline Initialized (WITH WARMUP)")
        print("=" * 70)
        print(f"Working Directory: {self.working_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"\nBaseline Parameters:")
        for key, val in self.params.items():
            print(f"  {key}: {val}")
        print(f"\n⏰ Warmup Period: {self.params['WARMUP_CYCLES']} cycles ({self.warmup_time}s)")
        print(f"   → Evaluation starts from t = {self.warmup_time}s")
        print("=" * 70)
    
    
    def check_prerequisites(self) -> bool:
        """
        Check if required input files exist
        
        Returns:
            True if all required files exist
        """
        print("\n📋 Checking Prerequisites...")
        required_files = ['fcd', 'eor']
        
        all_exist = True
        for file_key in required_files:
            file_path = self.files[file_key]
            exists = file_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_key}: {file_path}")
            if not exists:
                all_exist = False
        
        return all_exist
    
    
    def run_script(self, script_name: str, description: str) -> bool:
        """
        Run a Python script in the pipeline
        
        Args:
            script_name: Name of the script file
            description: Description of what the script does
            
        Returns:
            True if script ran successfully
        """
        print(f"\n⚙️  Running: {description}")
        print(f"   Script: {script_name}")
        
        script_path = self.working_dir / script_name
        
        if not script_path.exists():
            print(f"   ✗ Script not found: {script_path}")
            return False
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"   ✓ Success")
                # Print last few lines of output
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:
                        print(f"     {line}")
                return True
            else:
                print(f"   ✗ Failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:500]}")
                if result.stdout:
                    print(f"   Output: {result.stdout[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ✗ Timeout after 5 minutes")
            return False
        except Exception as e:
            print(f"   ✗ Exception: {e}")
            return False
    
    
    def step1_segment_classification(self) -> bool:
        """
        Step 1: AB Segment Classification
        Implements paper Section 3.2 - segment identification
        """
        return self.run_script(
            'segments_AB_resegmented_v05_entr.py',
            'Step 1: AB Segment Classification (Section 3.2)'
        )
    
    
    def step2_estimate_dynamic_params(self) -> bool:
        """
        Step 2: Estimate Dynamic Parameters (R, R_CV)
        Required for Section 3.1
        """
        print(f"\n⚙️  Running: Step 2: Estimate Dynamic Parameters")
        print(f"   Note: This step requires estimate_dynamic_params_improved.py")
        
        # Check if file exists
        if not self.files['dynamic_params'].exists():
            print(f"   ⚠️  Warning: {self.files['dynamic_params']} not found")
            print(f"   Please run estimate_dynamic_params_improved.py first")
            return False
        
        print(f"   ✓ Dynamic parameters file exists")
        return True
    
    
    def step3_compute_Q(self) -> bool:
        """
        Step 3: Compute Q (total vehicles)
        Implements Equation (1) from Section 3.1
        """
        return self.run_script(
            'compute_Q.py',
            'Step 3: Compute Q - Total Vehicles (Eq. 1, Section 3.1)'
        )
    
    
    def step4_compute_A_segments(self) -> bool:
        """
        Step 4: Compute A segment vehicles
        Implements Proposition 2 Part A (Type A segments)
        """
        return self.run_script(
            'compute_A_eq34_ROUND.py',
            'Step 4: Compute Type A Segments (Proposition 2-A, Eqs. 3-4)'
        )
    
    
    def step5_compute_B_segments(self) -> bool:
        """
        Step 5: Compute B segment vehicles
        Implements Proposition 2 Part B (Type B segments)
        """
        return self.run_script(
            'compute_B_distribution.py',
            'Step 5: Compute Type B Segments (Proposition 2-B, Eqs. 5-11)'
        )
    
    
    def load_ground_truth(self) -> pd.DataFrame:
        """
        Load ground truth vehicle positions from FCD data
        
        Returns:
            DataFrame with ground truth positions
        """
        print(f"\n📊 Loading Ground Truth Data...")
        
        try:
            # Parse FCD XML file (simplified - you may need to adjust)
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(str(self.files['fcd']))
            root = tree.getroot()
            
            data = []
            for timestep in root.findall('timestep'):
                time = float(timestep.get('time'))
                
                for vehicle in timestep.findall('vehicle'):
                    data.append({
                        'time': time,
                        'id': vehicle.get('id'),
                        'pos': float(vehicle.get('pos', 0)),
                        'speed': float(vehicle.get('speed', 0)),
                        'lane': vehicle.get('lane', ''),
                        'type': vehicle.get('type', '')
                    })
            
            df = pd.DataFrame(data)
            print(f"   ✓ Loaded {len(df)} vehicle observations")
            return df
            
        except Exception as e:
            print(f"   ✗ Failed to load ground truth: {e}")
            return pd.DataFrame()
    
    
    def compute_metrics(self, estimated_positions: List[float], 
                       ground_truth_positions: List[float],
                       threshold: float = 10.0) -> Dict[str, float]:
        """
        Compute evaluation metrics: Precision, Recall, F1
        
        Args:
            estimated_positions: List of estimated vehicle positions
            ground_truth_positions: List of ground truth vehicle positions  
            threshold: Distance threshold for matching (meters)
            
        Returns:
            Dictionary with precision, recall, F1, TP, FP, FN
        """
        # Handle edge cases - always return all keys
        if len(estimated_positions) == 0 and len(ground_truth_positions) == 0:
            return {
                'precision': 1.0, 
                'recall': 1.0, 
                'f1': 1.0,
                'TP': 0,
                'FP': 0,
                'FN': 0
            }
        
        if len(estimated_positions) == 0:
            return {
                'precision': 0.0, 
                'recall': 0.0, 
                'f1': 0.0,
                'TP': 0,
                'FP': 0,
                'FN': len(ground_truth_positions)
            }
        
        if len(ground_truth_positions) == 0:
            return {
                'precision': 0.0, 
                'recall': 0.0, 
                'f1': 0.0,
                'TP': 0,
                'FP': len(estimated_positions),
                'FN': 0
            }
        
        # Convert to numpy arrays
        est = np.array(sorted(estimated_positions))
        gt = np.array(sorted(ground_truth_positions))
        
        # Compute True Positives
        TP = 0
        matched_gt = set()
        
        for e_pos in est:
            # Find closest ground truth position
            distances = np.abs(gt - e_pos)
            min_idx = np.argmin(distances)
            
            if distances[min_idx] <= threshold and min_idx not in matched_gt:
                TP += 1
                matched_gt.add(min_idx)
        
        # False Positives and False Negatives
        FP = len(est) - TP
        FN = len(gt) - TP
        
        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }
    
    
    def evaluate_baseline(self) -> pd.DataFrame:
        """
        Evaluate baseline performance (EXCLUDING WARMUP PERIOD)
        
        Returns:
            DataFrame with evaluation metrics
        """
        print(f"\n📈 Evaluating Baseline Performance...")
        print(f"   ⏰ Skipping warmup period: first {self.params['WARMUP_CYCLES']} cycles (< {self.warmup_time}s)")
        
        try:
            # Load estimated positions
            A_pos = pd.read_csv(self.files['A_positions'])
            B_pos = pd.read_csv(self.files['B_positions'])
            
            print(f"   📊 Loaded A positions: {len(A_pos)} rows")
            print(f"   📊 Loaded B positions: {len(B_pos)} rows")
            
            # Load EoR times
            eor_df = pd.read_csv(self.files['eor'])
            print(f"   📊 Total EoR times: {len(eor_df)}")
            
            # ⏰ FILTER OUT WARMUP PERIOD
            eor_df_filtered = eor_df[eor_df['EoR_s'] > self.warmup_time].copy()
            print(f"   ✂️  After warmup filter: {len(eor_df_filtered)} EoR times")
            
            if len(eor_df_filtered) == 0:
                print(f"   ⚠️  No EoR times after warmup period!")
                print(f"   → Check if simulation is long enough (need > {self.warmup_time}s)")
                return pd.DataFrame()
            
            # Load ground truth
            gt_df = self.load_ground_truth()
            
            if gt_df.empty:
                print("   ⚠️  No ground truth data available")
                return pd.DataFrame()
            
            # Filter for target lane (lane7_0 for source lane)
            target_lane = 'lane7_0'
            gt_df = gt_df[gt_df['lane'] == target_lane].copy()
            print(f"   📊 Ground truth vehicles in {target_lane}: {len(gt_df)}")
            
            results = []
            
            # Evaluate each EoR time point (AFTER WARMUP)
            for _, row in eor_df_filtered.iterrows():
                eor_time = row['EoR_s']
                
                # Get estimated positions at this time
                est_A = A_pos[np.isclose(A_pos['EoR_s'], eor_time, atol=0.1)]
                est_B = B_pos[np.isclose(B_pos['EoR_s'], eor_time, atol=0.1)]
                
                estimated_positions = []
                if not est_A.empty:
                    estimated_positions.extend(est_A['est_pos_m'].tolist())
                if not est_B.empty:
                    estimated_positions.extend(est_B['est_pos_m'].tolist())
                
                # Get ground truth positions at this time
                gt_time = gt_df[np.isclose(gt_df['time'], eor_time, atol=0.5)]
                
                # Separate CVs and NCs
                gt_nc = gt_time[~gt_time['type'].str.upper().str.contains('CV', na=False)]
                gt_positions = gt_nc['pos'].tolist()
                
                # Compute metrics
                metrics = self.compute_metrics(estimated_positions, gt_positions)
                
                results.append({
                    'EoR_s': eor_time,
                    'cycle': int((eor_time / self.params['CYCLE'])) + 1,
                    'precision': metrics['precision'] * 100,  # Convert to percentage
                    'recall': metrics['recall'] * 100,
                    'f1': metrics['f1'] * 100,
                    'n_estimated': len(estimated_positions),
                    'n_ground_truth': len(gt_positions),
                    'TP': metrics['TP'],
                    'FP': metrics['FP'],
                    'FN': metrics['FN']
                })
            
            results_df = pd.DataFrame(results)
            
            if len(results_df) == 0:
                print(f"   ⚠️  No evaluation results generated")
                return pd.DataFrame()
            
            # Calculate average metrics
            avg_precision = results_df['precision'].mean()
            avg_recall = results_df['recall'].mean()
            avg_f1 = results_df['f1'].mean()
            
            print(f"\n   ✓ Baseline Metrics (Table 1 Format):")
            print(f"   {'Metric':<15} {'Value':<10}")
            print(f"   {'-'*25}")
            print(f"   {'Precision':<15} {avg_precision:>6.1f}%")
            print(f"   {'Recall':<15} {avg_recall:>6.1f}%")
            print(f"   {'F1':<15} {avg_f1:>6.1f}%")
            print(f"   {'-'*25}")
            print(f"   {'Cycles':<15} {len(results_df)}")
            print(f"   {'Time range':<15} {results_df['EoR_s'].min():.0f}-{results_df['EoR_s'].max():.0f}s")
            
            return results_df
            
        except Exception as e:
            print(f"   ✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    
    def generate_figure7_baseline(self, results_df: pd.DataFrame):
        """
        Generate Figure 7 baseline visualization
        Shows estimated vs ground truth positions for a sample cycle (AFTER WARMUP)
        
        Args:
            results_df: Results dataframe from evaluation
        """
        print(f"\n📊 Generating Figure 7 Baseline Visualization...")
        
        try:
            # Load data
            A_pos = pd.read_csv(self.files['A_positions'])
            B_pos = pd.read_csv(self.files['B_positions'])
            eor_df = pd.read_csv(self.files['eor'])
            gt_df = self.load_ground_truth()
            
            if gt_df.empty:
                print("   ⚠️  Cannot generate figure without ground truth")
                return
            
            # ⏰ Filter to post-warmup period
            eor_df_filtered = eor_df[eor_df['EoR_s'] > self.warmup_time].copy()
            
            if len(eor_df_filtered) == 0:
                print("   ⚠️  No EoR times after warmup period")
                return
            
            # Select a representative time point (middle of post-warmup period)
            sample_time = eor_df_filtered['EoR_s'].iloc[len(eor_df_filtered)//2]
            print(f"   → Visualizing cycle at t = {sample_time:.1f}s")
            
            # Get estimated positions
            est_A = A_pos[np.isclose(A_pos['EoR_s'], sample_time, atol=0.1)]
            est_B = B_pos[np.isclose(B_pos['EoR_s'], sample_time, atol=0.1)]
            
            estimated_positions = []
            if not est_A.empty:
                estimated_positions.extend(est_A['est_pos_m'].tolist())
            if not est_B.empty:
                estimated_positions.extend(est_B['est_pos_m'].tolist())
            
            # Get ground truth
            target_lane = 'lane7_0'
            gt_time = gt_df[(gt_df['lane'] == target_lane) & 
                           np.isclose(gt_df['time'], sample_time, atol=0.5)]
            
            gt_cv = gt_time[gt_time['type'].str.upper().str.contains('CV', na=False)]
            gt_nc = gt_time[~gt_time['type'].str.upper().str.contains('CV', na=False)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Plot layout with three rows
            y_ground_truth = 2.5
            y_cvvls = 1.5
            
            # Ground truth vehicles
            if not gt_cv.empty:
                ax.scatter(gt_cv['pos'], [y_ground_truth]*len(gt_cv), 
                          marker='o', s=100, c='red', label='Ground truth CV',
                          edgecolors='darkred', linewidths=1.5, zorder=3)
            
            if not gt_nc.empty:
                ax.scatter(gt_nc['pos'], [y_ground_truth]*len(gt_nc),
                          marker='s', s=100, c='black', label='Ground truth NC',
                          edgecolors='black', linewidths=1.5, zorder=3)
            
            # CVVL-S estimated positions
            if estimated_positions:
                ax.scatter(estimated_positions, [y_cvvls]*len(estimated_positions),
                          marker='d', s=100, c='green', label='CVVL-S',
                          edgecolors='darkgreen', linewidths=1.5, zorder=3, alpha=0.8)
            
            # Stop bar
            ax.axvline(x=self.params['STOP_POS'], color='orange', 
                      linewidth=3, linestyle='-', label='Stop bar', zorder=2)
            
            # Entrance
            ax.axvline(x=0, color='gray', linewidth=2, 
                      linestyle='--', label='Entrance', zorder=1)
            
            # Formatting
            ax.set_xlim(-50, self.params['STOP_POS'] + 50)
            ax.set_ylim(-0.5, 3.5)
            ax.set_xlabel('Location (m)', fontsize=12, fontweight='bold')
            ax.set_yticks([y_cvvls, y_ground_truth])
            ax.set_yticklabels(['CVVL-S', 'Ground truth'], fontsize=11)
            ax.grid(True, axis='x', alpha=0.3)
            ax.legend(loc='upper left', fontsize=10, ncol=5, framealpha=0.9)
            
            # Title with metrics
            if not results_df.empty:
                row = results_df[np.isclose(results_df['EoR_s'], sample_time, atol=0.5)]
                if not row.empty:
                    metrics = row.iloc[0]
                    title = f"Baseline (Post-Warmup) - Time: {sample_time:.1f}s (Cycle {int(metrics['cycle'])}) | " \
                           f"Precision: {metrics['precision']:.1f}% | " \
                           f"Recall: {metrics['recall']:.1f}% | " \
                           f"F1: {metrics['f1']:.1f}%"
                    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.output_dir / 'figure7_baseline.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Figure saved: {output_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"   ✗ Figure generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    
    def save_results(self, results_df: pd.DataFrame):
        """
        Save evaluation results to CSV and summary
        
        Args:
            results_df: Results dataframe
        """
        print(f"\n💾 Saving Results...")
        
        try:
            # Save detailed results
            results_path = self.output_dir / 'baseline_results_detailed.csv'
            results_df.to_csv(results_path, index=False)
            print(f"   ✓ Detailed results: {results_path}")
            
            # Create summary in Table 1 format
            summary = {
                'Experiment': ['Baseline'],
                'r (s)': [self.params['r']],
                'V/C': [self.params['V/C']],
                'p': [self.params['p']],
                'ToI': [self.params['ToI']],
                'Warmup_cycles': [self.params['WARMUP_CYCLES']],
                'Eval_cycles': [len(results_df)],
                'Precision (%)': [results_df['precision'].mean()],
                'Recall (%)': [results_df['recall'].mean()],
                'F1 (%)': [results_df['f1'].mean()]
            }
            
            summary_df = pd.DataFrame(summary)
            summary_path = self.output_dir / 'table1_baseline_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"   ✓ Table 1 summary: {summary_path}")
            
            # Print summary to console
            print(f"\n" + "="*70)
            print("TABLE 1 BASELINE RESULTS (POST-WARMUP)")
            print("="*70)
            print(summary_df.to_string(index=False))
            print("="*70)
            
        except Exception as e:
            print(f"   ✗ Failed to save results: {e}")
    
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete baseline pipeline
        
        Returns:
            True if pipeline completed successfully
        """
        print("\n" + "="*70)
        print("STARTING COMPLETE BASELINE PIPELINE")
        print("="*70)
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites not met. Please ensure input files exist.")
            return False
        
        # Run pipeline steps
        steps = [
            (self.step1_segment_classification, "AB Segment Classification"),
            (self.step2_estimate_dynamic_params, "Dynamic Parameters Check"),
            (self.step3_compute_Q, "Total Vehicles Q"),
            (self.step4_compute_A_segments, "Type A Segments"),
            (self.step5_compute_B_segments, "Type B Segments")
        ]
        
        for step_func, step_name in steps:
            success = step_func()
            if not success:
                print(f"\n❌ Pipeline failed at: {step_name}")
                return False
        
        # Evaluate results
        results_df = self.evaluate_baseline()
        
        if results_df.empty:
            print(f"\n⚠️  No evaluation results generated")
            return False
        
        # Generate visualizations
        self.generate_figure7_baseline(results_df)
        
        # Save results
        self.save_results(results_df)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"  - table1_baseline_summary.csv")
        print(f"  - baseline_results_detailed.csv")
        print(f"  - figure7_baseline.png")
        print("="*70)
        
        return True


def main():
    """
    Main execution function
    """
    # Initialize pipeline with warmup period
    # Change warmup_cycles parameter if needed (default: 30)
    pipeline = BaselinePipeline(warmup_cycles=30)
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ All tasks completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline encountered errors")
        sys.exit(1)


if __name__ == "__main__":
    main()