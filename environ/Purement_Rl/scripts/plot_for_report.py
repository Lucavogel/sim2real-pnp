"""
Script to generate professional graphs for reports.
Creates publication-optimized versions (high-resolution PDF/PNG).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison(openloop_npz: str, deploy_npz: str, output_dir: str):
    """Generate comparison graphs for the report."""
    
    # Load data
    d_ol = np.load(openloop_npz)
    d_dp = np.load(deploy_npz)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration for publication-ready graphs
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    
    env_idx = 0
    
    # =========================================================================
    # GRAPH 1: Error Norm Comparison (MOST IMPORTANT)
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    t_ol = d_ol['t']
    t_dp = d_dp['t']
    err_ol = d_ol['e_true_clean_norm'][:, env_idx] * 1000  # Convert to mm
    err_dp = d_dp['e_true_clean_norm'][:, env_idx] * 1000
    
    ax.plot(t_ol, err_ol, label='Open-Loop (without RL)', linewidth=2, color='#d62728', alpha=0.8)
    ax.plot(t_dp, err_dp, label='Closed-Loop (with RL)', linewidth=2, color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Position Error (mm)', fontweight='bold')
    ax.set_title('Tracking Accuracy Comparison: Open-Loop vs Closed-Loop RL', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add statistics
    stats_text = (
        f"Open-Loop:  mean={err_ol.mean():.1f}mm, max={err_ol.max():.1f}mm\n"
        f"Closed-Loop: mean={err_dp.mean():.1f}mm, max={err_dp.max():.1f}mm\n"
        f"Improvement: {(1 - err_dp.mean()/err_ol.mean())*100:.0f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig1_error_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_path / 'fig1_error_comparison.pdf', bbox_inches='tight')
    print(f"✓ Graph 1 saved: {output_path / 'fig1_error_comparison.png'}")
    plt.close()
    
    # =========================================================================
    # GRAPH 2: EE Position X (single axis for clarity)
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    ee_true = d_dp['ee_true_local'][:, env_idx, 0] * 1000  # Convert to mm
    ee_des_clean = d_dp['ee_des_clean_local'][:, env_idx, 0] * 1000
    ee_des_obs = d_dp['ee_des_obs_local'][:, env_idx, 0] * 1000
    
    ax.plot(t_dp, ee_des_clean, label='Desired trajectory', linewidth=2.5, 
            color='#1f77b4', linestyle='-', alpha=0.9)
    ax.plot(t_dp, ee_des_obs, label='Observed trajectory (noisy)', linewidth=2, 
            color='red', linestyle='-', alpha=0.9)
    ax.plot(t_dp, ee_true, label='Actual trajectory', linewidth=2, 
            color='#2ca02c', alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('End-effector X position (mm)', fontweight='bold')
    ax.set_title('Trajectory Tracking on X-axis', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig2_ee_position_x.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_path / 'fig2_ee_position_x.pdf', bbox_inches='tight')
    print(f"✓ Graph 2 saved: {output_path / 'fig2_ee_position_x.png'}")
    plt.close()
    
    # =========================================================================
    # GRAPH 3: EE Speed (optional but useful)
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    speed = d_dp['ee_speed'][:, env_idx] * 1000  # Convert to mm/s
    
    ax.plot(t_dp, speed, linewidth=2, color='#9467bd', alpha=0.8)
    ax.fill_between(t_dp, 0, speed, alpha=0.3, color='#9467bd')
    
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('End-effector speed (mm/s)', fontweight='bold')
    ax.set_title('End-effector Speed', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    stats_text = f"Average speed: {speed.mean():.1f} mm/s\nMax speed: {speed.max():.1f} mm/s"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig3_ee_speed.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_path / 'fig3_ee_speed.pdf', bbox_inches='tight')
    print(f"✓ Graph 3 saved: {output_path / 'fig3_ee_speed.png'}")
    plt.close()
    
    # =========================================================================
    # GRAPH 4: Joint J4 (interesting behavior)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    q4_meas_ol = np.degrees(d_ol['q_meas'][:, env_idx, 3])
    q4_ref_ol = np.degrees(d_ol['q_ref'][:, env_idx, 3])
    q4_meas_dp = np.degrees(d_dp['q_meas'][:, env_idx, 3])
    q4_ref_dp = np.degrees(d_dp['q_ref'][:, env_idx, 3])
    q4_target_dp = np.degrees(d_dp['q_target'][:, env_idx, 3])
    
    # Subplot 1: Open-Loop
    ax1.plot(t_ol, q4_ref_ol, label='Reference', linewidth=2.5, linestyle='--', color='#1f77b4', alpha=0.9)
    ax1.plot(t_ol, q4_meas_ol, label='Measured', linewidth=2, color='#d62728')
    ax1.set_ylabel('Joint J4 position (°)', fontweight='bold')
    ax1.set_title('Joint J4 (wrist_1) - Open-Loop', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best')
    
    # Subplot 2: Closed-Loop
    ax2.plot(t_dp, q4_ref_dp, label='Reference', linewidth=2, linestyle='--', color='#1f77b4', alpha=0.7)
    # ax2.plot(t_dp, q4_target_dp, label='Target (ref + RL action)', linewidth=2.5, linestyle='-', color='orange', alpha=0.9)
    ax2.plot(t_dp, q4_meas_dp, label='Measured', linewidth=2, color='#2ca02c')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Joint J4 position (°)', fontweight='bold')
    ax2.set_title('Joint J4 (wrist_1) - Closed-Loop with RL', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best')
    
    # Add note
    action_mean = np.degrees((d_dp['q_target'][:, env_idx, 3] - d_dp['q_ref'][:, env_idx, 3]).mean())
    note_text = f"Note: RL policy applies a bias of +{action_mean:.2f}° on J4\nto improve end-effector accuracy"
    ax2.text(0.02, 0.02, note_text, transform=ax2.transAxes, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontsize=8)
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig4_joint_j4_strategy.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_path / 'fig4_joint_j4_strategy.pdf', bbox_inches='tight')
    print(f"✓ Graph 4 saved: {output_path / 'fig4_joint_j4_strategy.png'}")
    plt.close()
    
    # =========================================================================
    # SUMMARY TABLE (as image)
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    err_ol_stats = d_ol['e_true_clean_norm'][:, env_idx] * 1000
    err_dp_stats = d_dp['e_true_clean_norm'][:, env_idx] * 1000
    
    table_data = [
        ['Metric', 'Open-Loop', 'Closed-Loop RL', 'Improvement'],
        ['Mean error (mm)', f'{err_ol_stats.mean():.2f}', f'{err_dp_stats.mean():.2f}', 
         f'{(1-err_dp_stats.mean()/err_ol_stats.mean())*100:.1f}%'],
        ['Max error (mm)', f'{err_ol_stats.max():.2f}', f'{err_dp_stats.max():.2f}', 
         f'{(1-err_dp_stats.max()/err_ol_stats.max())*100:.1f}%'],
        ['Std error (mm)', f'{err_ol_stats.std():.2f}', f'{err_dp_stats.std():.2f}', 
         f'{(1-err_dp_stats.std()/err_ol_stats.std())*100:.1f}%'],
        ['Avg speed (mm/s)', '-', f'{(d_dp["ee_speed"][:, env_idx].mean()*1000):.1f}', '-'],
        ['Max speed (mm/s)', '-', f'{(d_dp["ee_speed"][:, env_idx].max()*1000):.1f}', '-'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            # Highlight improvement column in green
            if j == 3 and i < 4:
                table[(i, j)].set_facecolor('#C6EFCE')
    
    plt.title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(output_path / 'table_summary.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_path / 'table_summary.pdf', bbox_inches='tight')
    print(f"✓ Table saved: {output_path / 'table_summary.png'}")
    plt.close()
    
    print("\n" + "="*80)
    print("ALL REPORT GRAPHS ARE READY!")
    print("="*80)
    print(f"\nFiles generated in: {output_path.absolute()}")
    print("\nAvailable formats:")
    print("  • High-resolution PNG (300 DPI) - for Word/PowerPoint")
    print("  • Vector PDF - for LaTeX/professional documents")
    print("\nGenerated graphs:")
    print("  1. fig1_error_comparison    - ESSENTIAL")
    print("  2. fig2_ee_position_x       - IMPORTANT")
    print("  3. fig3_ee_speed            - OPTIONAL")
    print("  4. fig4_joint_j4_strategy   - OPTIONAL (technical)")
    print("  5. table_summary            - ESSENTIAL")

def main():
    parser = argparse.ArgumentParser(
        description="Generate professional graphs for academic/technical reports"
    )
    parser.add_argument(
        "--openloop", 
        type=str, 
        required=True,
        help="Path to the openloop_log_*.npz file"
    )
    parser.add_argument(
        "--deploy", 
        type=str, 
        required=True,
        help="Path to the deploy_log_*.npz file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="report_figures",
        help="Output directory for graphs (default: report_figures)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING REPORT GRAPHS")
    print("="*80)
    print(f"\nOpen-Loop: {args.openloop}")
    print(f"Deploy RL: {args.deploy}")
    print(f"Output:    {args.output}")
    print()
    
    plot_comparison(args.openloop, args.deploy, args.output)

if __name__ == "__main__":
    main()
