import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
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

# --- توابع ثبت گزارش‌های پیشرفته برای DDQN ---
def log_advanced_metrics(method_name, energy, avg_makespan, max_makespan):
    file_path = f"../results/{method_name}_metrics.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["energy", "avg_makespan", "max_makespan"])
        writer.writerow([energy, avg_makespan, max_makespan])
# ---------------------------------------------

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

def train_ddqn():
    config_file = "../sumo/osm.sumocfg" 
    rsus_file = "../sumo/rsus.json"
    env = VECEnv(config_file, rsus_file)
    
    # آپدیت ابعاد فضای حالت به 8 
    state_dim = 8
    action_dim = 6
    
    online_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)
    buffer = deque(maxlen=5000)
    
    # پاک کردن لاگ‌های قدیمی
    if os.path.exists("../results/ddqn_log.csv"):
        os.remove("../results/ddqn_log.csv")
    if os.path.exists("../results/ddqn_metrics.csv"):
        os.remove("../results/ddqn_metrics.csv")
        
    print(f"🚀 Starting DDQN Baseline Training ({MAX_EPISODES} Episodes)...")
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        total_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            state_t = torch.FloatTensor(state)
            
            if random.random() < max(0.01, 0.1 - 0.01*(episode/10)):
                action = random.randint(0, action_dim - 1)
            else:
                action = online_net(state_t).argmax().item()
                
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state
            
            if len(buffer) > 32:
                batch = random.sample(buffer, 32)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                
                states_t = torch.FloatTensor(np.array(states))
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards_b)
                next_states_t = torch.FloatTensor(np.array(next_states))
                dones_t = torch.FloatTensor(dones)
                
                next_actions = online_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = target_net(next_states_t).gather(1, next_actions).squeeze(1)
                target_q = rewards_t + 0.95 * next_q * (1 - dones_t)
                current_q = online_net(states_t).gather(1, actions_t).squeeze(1)
                
                loss = F.mse_loss(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if done: 
                break
        
        if episode % 5 == 0:
            target_net.load_state_dict(online_net.state_dict())
            
        print(f"✅ DDQN Agent - Episode {episode + 1}/{MAX_EPISODES} | Total Reward: {total_reward:.2f}")
        log_episode("ddqn", episode + 1, total_reward)
        
        # ثبت گزارش‌های پیشرفته در پایان اپیزود
        avg_makespan = sum(env.episode_makespans) / len(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics("ddqn", env.episode_energy, avg_makespan, max_makespan)
        
        env.close()

if __name__ == "__main__":
    train_ddqn()