import sys
import os
import csv
import torch

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
    print(f"\n🔬 Running: {test_name}")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json", 
                 use_mobility_detector=use_md, 
                 use_priority=use_tp, 
                 use_collaboration=use_co)
    
    state = env.reset(render=False)
    total_reward = 0
    
    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs, _ = agent.forward(state_tensor)
            action = torch.argmax(action_probs).item()
            
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done: break
        
    env.close()
    print(f"📊 Result for {test_name}: {total_reward:.2f}")
    log_ablation_result(test_name, total_reward)

def run_ablation_studies():
    print("🚀 Starting Smart Ablation Studies...")
    if os.path.exists("../results/ablation_log.csv"):
        os.remove("../results/ablation_log.csv")
        
    agent = ActorCritic(10, 6)
    if os.path.exists("cotop_trained_model.pth"):
        agent.load_state_dict(torch.load("cotop_trained_model.pth"))
    agent.eval()
    
    # تست کردن سیستم در شرایط مختلف
    run_single_test("Complete_CoTOP", True, True, True, agent)
    run_single_test("w/o_MD", False, True, True, agent)
    run_single_test("w/o_TP", True, False, True, agent)
    run_single_test("w/o_CO", True, True, False, agent)

if __name__ == "__main__":
    run_ablation_studies()