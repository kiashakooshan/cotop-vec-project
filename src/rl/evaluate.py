import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# --- 1. توابع قبلی برای پاداش و Ablation ---
def plot_baselines():
    print("📊 Generating Baselines Comparison Chart...")
    methods = {"CoTOP (Ours)": "cotop_log.csv", "DDQN": "ddqn_log.csv", 
               "Greedy": "greedy_log.csv", "Local": "local_log.csv"}
    avg_rewards = {}
    for name, file in methods.items():
        path = f"../results/{file}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            avg_rewards[name] = df["reward"].tail(50).mean()

    if avg_rewards:
        plt.figure(figsize=(9, 6))
        bars = plt.bar(avg_rewards.keys(), avg_rewards.values(), color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval - (abs(yval) * 0.05), f'{int(yval)}', 
                     ha='center', va='top', color='white', fontweight='bold')
        plt.title('Average Reward Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Average Reward')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/baseline_comparison.png")

# --- 2. توابع جدید برای انرژی و Makespan ---
def plot_energy_and_makespan():
    print("📊 Generating Energy and Makespan Charts...")
    methods = {"CoTOP": "cotop_metrics.csv", "DDQN": "ddqn_metrics.csv", 
               "Greedy": "greedy_metrics.csv", "Local": "local_metrics.csv"}
    
    energy_data, makespan_avg, makespan_max = {}, {}, {}
    
    for name, file in methods.items():
        path = f"../results/{file}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            energy_data[name] = df["energy"].mean()
            makespan_avg[name] = df["avg_makespan"].mean()
            makespan_max[name] = df["max_makespan"].mean()
            
    # رسم نمودار مصرف انرژی کل 시스템
    if energy_data:
        plt.figure(figsize=(9, 6))
        bars = plt.bar(energy_data.keys(), energy_data.values(), color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
        plt.title('Total System Energy Consumption (Lower is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Energy (Units)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/energy_comparison.png")

    # رسم نمودار مقایسه‌ای Makespan (میانگین و بیشینه)
    if makespan_avg:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - width/2, makespan_avg.values(), width, label='Avg Makespan', color='#17becf')
        ax.bar(x + width/2, makespan_max.values(), width, label='Max Makespan', color='#9467bd')
        
        ax.set_ylabel('Makespan (Seconds)')
        ax.set_title('DAG Makespan Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods.keys())
        ax.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/makespan_comparison.png")

# --- 3. تابع جدید برای ارزیابی دقت تحرک به تفکیک خودرو ---
def generate_mobility_report():
    print("📊 Generating ADE/FDE Mobility Accuracy Report...")
    path = "../results/mobility_predictions.csv"
    if not os.path.exists(path):
        return
        
    df = pd.read_csv(path)
    report = []
    
    for veh_id, group in df.groupby("veh_id"):
        preds_x, preds_y = group["pred_x"].tolist(), group["pred_y"].tolist()
        acts_x, acts_y = group["actual_x"].tolist(), group["actual_y"].tolist()
        
        # محاسبه فاصله اقلیدسی
        dists = [math.dist((px, py), (ax, ay)) for px, py, ax, ay in zip(preds_x, preds_y, acts_x, acts_y)]
        
        if dists:
            ade = sum(dists) / len(dists)  # Average Displacement Error
            fde = dists[-1]                # Final Displacement Error
            report.append({"veh_id": veh_id, "ADE": round(ade, 3), "FDE": round(fde, 3), "num_predictions": len(dists)})
            
    report_df = pd.DataFrame(report)
    report_df.to_csv("../results/mobility_accuracy_per_vehicle.csv", index=False)
    print("✅ Mobility report saved to 'results/mobility_accuracy_per_vehicle.csv'")

if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    plot_baselines()
    plot_energy_and_makespan()
    generate_mobility_report()
    # اگر فایل ablation را هم دارید می‌توانید تابع قبلی آن را اضافه کنید
    plt.show()