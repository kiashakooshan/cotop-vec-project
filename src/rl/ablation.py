import sys
import os
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

MAX_STEPS = 300

def log_ablation_result(test_name, total_reward):
    os.makedirs("../results", exist_ok=True)
    file_path = "../results/ablation_log.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Test_Condition", "Total_Reward"])
        writer.writerow([test_name, total_reward])

def run_single_test(test_name, use_md, use_tp, use_co, agent):
    print(f"\\nRunning: {test_name}")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json",
                 use_mobility_detector=use_md,
                 use_priority=use_tp,
                 use_collaboration=use_co)

    states = env.reset(render=False)
    total_reward = 0

    for step in range(MAX_STEPS):
        actions = []
        for s in states:
            state_t = torch.FloatTensor(s)
            with torch.no_grad():
                action_probs, _ = agent.forward(state_t)
                action = torch.argmax(action_probs).item()
            actions.append(action)

        next_states, reward, done, _ = env.step(actions)
        total_reward += reward
        states = next_states
        if done:
            break

    env.close()
    print(f"Result for {test_name}: {total_reward:.2f}")
    log_ablation_result(test_name, total_reward)

def plot_ablation_results():
    file_path = "../results/ablation_log.csv"
    if not os.path.exists(file_path):
        return
        
    df = pd.read_csv(file_path)
    # برداشتن ۴ تست آخر در صورتی که فایل حاوی اجراهای قدیمی باشد
    df = df.tail(4) 
    
    plt.figure(figsize=(9, 6))
    bars = plt.bar(df["Test_Condition"], df["Total_Reward"], 
                   color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
    
    for bar in bars:
        yval = bar.get_height()
        # نمایش عدد پاداش روی هر ستون
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', 
                 ha='center', va='bottom' if yval > 0 else 'top', fontweight='bold')
        
    plt.title('Ablation Study: Impact of Each Module on Reward', fontsize=14, fontweight='bold')
    plt.ylabel('Total Reward')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../results/ablation_chart.png")
    plt.show()

def run_ablation_studies(seed=0):
    print("Starting Ablation Studies...")
    if os.path.exists("../results/ablation_log.csv"):
        os.remove("../results/ablation_log.csv")

    agent = ActorCritic(11, 6)
    model_path = f"cotop_model_seed{seed}.pth"
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))
    agent.eval()

    run_single_test("Complete_CoTOP", True, True, True, agent)
    run_single_test("w/o_MD", False, True, True, agent)
    run_single_test("w/o_TP", True, False, True, agent)
    run_single_test("w/o_CO", True, True, False, agent)

if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_ablation_studies(seed=seed_arg)
    plot_ablation_results()  # این خط اضافه شد