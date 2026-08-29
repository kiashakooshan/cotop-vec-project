import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_baselines():
    print("📊 Generating Baselines Comparison Chart...")
    methods = {"CoTOP (Ours)": "cotop_log.csv", "DDQN": "ddqn_log.csv", 
               "Greedy": "greedy_log.csv", "Local": "local_log.csv"}
    
    avg_rewards = {}
    for name, file in methods.items():
        path = f"../results/{file}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            avg_rewards[name] = df["reward"].mean()
        else:
            print(f"⚠️ Warning: {file} not found.")

    if avg_rewards:
        plt.figure(figsize=(8, 5))
        plt.bar(avg_rewards.keys(), avg_rewards.values(), color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
        plt.title('Average Reward Comparison (Higher is Better)', fontweight='bold')
        plt.ylabel('Average Reward')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("../results/baseline_comparison.png")
        print("✅ Baseline chart saved to 'results/baseline_comparison.png'")

def plot_ablation():
    print("📊 Generating Ablation Study Chart...")
    path = "../results/ablation_log.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        plt.figure(figsize=(8, 5))
        plt.bar(df["Test_Condition"], df["Total_Reward"], color='#9467bd')
        plt.title('Ablation Study (Impact of Modules)', fontweight='bold')
        plt.ylabel('Total Reward')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("../results/ablation_chart.png")
        print("✅ Ablation chart saved to 'results/ablation_chart.png'")
    else:
        print("⚠️ Warning: ablation_log.csv not found.")

if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    plot_baselines()
    plot_ablation()
    plt.show()