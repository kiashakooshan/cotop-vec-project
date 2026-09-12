import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# تغییر مهم: فقط Seed 0 را می‌خوانیم
SEEDS = [0]

def aggregate_metric(prefix, column, tail=50):
    """Read {prefix}_seed{N}.csv for every seed, return (mean, std) of 'column'."""
    values = []
    for seed in SEEDS:
        path = f"../results/{prefix}_seed{seed}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            if column in df.columns:
                values.append(df[column].tail(tail).mean())
    
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))

def plot_baselines_with_error_bars():
    print("\n" + "="*50)
    print("📊 Generating Reward Comparison...")
    methods = {"CoTOP (Ours)": "cotop_log", "DDQN": "ddqn_log", 
               "QRMP-DQN": "qrmpdqn_log", "Greedy": "greedy_log", "Local": "local_log"}
    
    means, stds = {}, {}
    for name, prefix in methods.items():
        m, s = aggregate_metric(prefix, "reward")
        if m is not None:
            means[name] = m
            stds[name] = s

    if means:
        plt.figure(figsize=(10, 6))
        # چون فقط ۱ اجرا داریم، yerr صفر خواهد بود اما کد بدون مشکل کار می‌کند
        bars = plt.bar(means.keys(), means.values(), 
                       yerr=[stds[k] for k in means.keys()], capsize=6,
                       color=['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', '#d62728'])
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval - (abs(yval) * 0.05), f'{int(yval)}', 
                     ha='center', va='top', color='white', fontweight='bold')
            
        plt.title('Average Reward Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Average Reward')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/baseline_comparison.png")

def plot_energy_and_makespan_with_error_bars():
    print("\n" + "="*50)
    print("⚡ Generating Detailed Energy & TOTAL Makespan Reports...")
    methods = {"CoTOP": "cotop_metrics", "DDQN": "ddqn_metrics", 
               "QRMP-DQN": "qrmpdqn_metrics", "Greedy": "greedy_metrics", "Local": "local_metrics"}
    
    energy_means, energy_stds = {}, {}
    makespan_means, makespan_stds = {}, {}
    
    for name, prefix in methods.items():
        e_m, e_s = aggregate_metric(prefix, "energy")
        if e_m is not None:
            energy_means[name] = e_m
            energy_stds[name] = e_s
            
        m_m, m_s = aggregate_metric(prefix, "total_makespan")
        if m_m is not None:
            makespan_means[name] = m_m
            makespan_stds[name] = m_s
            
    if energy_means:
        print("\n--- Total Energy Consumption (Joules) ---")
        for method, en in energy_means.items():
            print(f"🔹 {method}: {en:.2f} J")
            
        plt.figure(figsize=(10, 6))
        bars = plt.bar(energy_means.keys(), energy_means.values(), 
                       yerr=[energy_stds[k] for k in energy_means.keys()], capsize=6,
                       color=['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', '#d62728'])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
        plt.title('Total System Energy Consumption (Lower is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Energy (Joules)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/energy_comparison.png")

    if makespan_means:
        print("\n--- TOTAL Makespan of All Cars (Seconds) ---")
        for method, ms in makespan_means.items():
            print(f"⏱️ {method}: {ms:.2f} s")
            
        plt.figure(figsize=(10, 6))
        bars = plt.bar(makespan_means.keys(), makespan_means.values(), 
                       yerr=[makespan_stds[k] for k in makespan_means.keys()], capsize=6,
                       color='#17becf')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel('Total Makespan (Seconds)')
        plt.title('TOTAL DAG Makespan Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/makespan_comparison.png")

def plot_controlled_energy_with_error_bars():
    methods = {"CoTOP": "cotop_metrics", "DDQN": "ddqn_metrics", 
               "QRMP-DQN": "qrmpdqn_metrics", "Greedy": "greedy_metrics", "Local": "local_metrics"}
    
    controlled_energy_means, controlled_energy_stds = {}, {}
    for name, prefix in methods.items():
        m, s = aggregate_metric(prefix, "controlled_energy")
        if m is not None:
            controlled_energy_means[name] = m
            controlled_energy_stds[name] = s
                
    if controlled_energy_means:
        plt.figure(figsize=(10, 6))
        plt.bar(controlled_energy_means.keys(), controlled_energy_means.values(), 
                yerr=[controlled_energy_stds[k] for k in controlled_energy_means.keys()], capsize=6,
                color=['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', '#d62728'])
        plt.title("Energy Consumed by the Algorithm's Own Decision")
        plt.ylabel('Energy (Joules)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/controlled_energy_comparison.png")

def generate_mobility_report():
    print("\n" + "="*50)
    print("📍 GAT-GRU Transformer Trajectory Detection Performance")
    print("="*50)
    path = "../results/mobility_predictions.csv"
    if not os.path.exists(path):
        print("⚠️ Waiting for mobility_predictions.csv to be generated...")
        return
        
    df = pd.read_csv(path)
    report = []
    
    for veh_id, group in df.groupby("veh_id"):
        preds_x, preds_y = group["pred_x"].tolist(), group["pred_y"].tolist()
        acts_x, acts_y = group["actual_x"].tolist(), group["actual_y"].tolist()
        
        dists = [math.dist((px, py), (ax, ay)) for px, py, ax, ay in zip(preds_x, preds_y, acts_x, acts_y)]
        
        if dists:
            ade = sum(dists) / len(dists)  
            fde = dists[-1]                
            report.append({"Vehicle": veh_id, "ADE (m)": round(ade, 3), "FDE (m)": round(fde, 3)})
            
    report_df = pd.DataFrame(report)
    
    print(report_df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    plot_baselines_with_error_bars()
    plot_energy_and_makespan_with_error_bars()
    plot_controlled_energy_with_error_bars()
    generate_mobility_report()
    plt.show()