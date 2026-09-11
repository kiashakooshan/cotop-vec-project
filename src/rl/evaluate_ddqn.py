import sys
import os
import torch
import torch.nn as nn
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

MAX_EPISODES = 50 
MAX_STEPS_PER_EPISODE = 300

def log_episode(method_name, episode, total_reward):
    os.makedirs("../results", exist_ok=True)
    file_path = f"../results/{method_name}_log.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward"])
        writer.writerow([episode, total_reward])

def log_advanced_metrics(method_name, energy, controlled_energy, total_makespan, max_makespan):
    file_path = f"../results/{method_name}_metrics.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["energy", "controlled_energy", "total_makespan", "max_makespan"])
        writer.writerow([energy, controlled_energy, total_makespan, max_makespan])

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

def evaluate_ddqn_model():
    print(f"🚀 Evaluating FULLY TRAINED 11D DDQN ({MAX_EPISODES} Episodes)...")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    
    state_dim = 11
    action_dim = 6
    
    agent = QNetwork(state_dim, action_dim)
    
    # لود کردن مدل نهایی
    if os.path.exists("ddqn_model_final.pth"):
        agent.load_state_dict(torch.load("ddqn_model_final.pth"))
    else:
        print("⚠️ Warning: 'ddqn_model_final.pth' not found! Evaluating with random weights.")
    agent.eval()
    
    if os.path.exists("../results/ddqn_log.csv"):
        os.remove("../results/ddqn_log.csv")
    if os.path.exists("../results/ddqn_metrics.csv"):
        os.remove("../results/ddqn_metrics.csv")
        
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                # ارزیابی قطعی: همیشه argmax، بدون هیچ اکتشافی
                action = agent(state_t).argmax().item()
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done: break
            
        print(f"✅ DDQN (Eval) - Episode {episode+1} | Reward: {episode_reward:.2f}")
        log_episode("ddqn", episode + 1, episode_reward)
        
        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics("ddqn", env.episode_energy, env.episode_controlled_energy, total_makespan, max_makespan)
            
        env.close()

if __name__ == "__main__":
    evaluate_ddqn_model()