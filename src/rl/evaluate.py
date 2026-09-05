import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def plot_baselines():
    print("\n" + "="*50)
    print("📊 Generating Reward Comparison...")
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

def plot_energy_and_makespan():
    print("\n" + "="*50)
    print("⚡ Generating Detailed Energy & TOTAL Makespan Reports...")
    methods = {"CoTOP": "cotop_metrics.csv", "DDQN": "ddqn_metrics.csv", 
               "Greedy": "greedy_metrics.csv", "Local": "local_metrics.csv"}
    
    energy_data, makespan_total = {}, {}
    
    for name, file in methods.items():
        path = f"../results/{file}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            energy_data[name] = df["energy"].mean()
            makespan_total[name] = df["avg_makespan"].mean() # در فایل‌های ارزیاب ما اسم این ستون را نگه داشتیم اما مقدارش Total است
            
    if energy_data:
        print("\n--- Total Energy Consumption (Joules) ---")
        for method, en in energy_data.items():
            print(f"🔹 {method}: {en:.2f} J")
            
        plt.figure(figsize=(9, 6))
        bars = plt.bar(energy_data.keys(), energy_data.values(), color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
        plt.title('Total System Energy Consumption (Lower is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Energy (Joules)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/energy_comparison.png")

    if makespan_total:
        print("\n--- TOTAL Makespan of All Cars (Seconds) ---")
        for method, ms in makespan_total.items():
            print(f"⏱️ {method}: {ms:.2f} s")
            
        plt.figure(figsize=(9, 6))
        bars = plt.bar(makespan_total.keys(), makespan_total.values(), color='#17becf')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel('Total Makespan (Seconds)')
        plt.title('TOTAL DAG Makespan Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("../results/makespan_comparison.png")

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
    
    # چاپ مستقیم و زیبای دیتا فریم در ترمینال برای مشاهده تو
    print(report_df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    plot_baselines()
    plot_energy_and_makespan()
    generate_mobility_report()
    plt.show()