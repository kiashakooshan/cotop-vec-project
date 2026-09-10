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

TRAIN_EPISODES = 200
EVAL_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300
TAU = 0.005

def log_episode(method_name, episode, total_reward):
    os.makedirs("../results", exist_ok=True)
    file_path = f"../results/{method_name}_log.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward"])
        writer.writerow([episode, total_reward])

def log_advanced_metrics(method_name, energy, total_makespan, max_makespan):
    file_path = f"../results/{method_name}_metrics.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["energy", "total_makespan", "max_makespan"])
        writer.writerow([energy, total_makespan, max_makespan])

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

def run_ddqn():
    config_file = "../sumo/osm.sumocfg" 
    rsus_file = "../sumo/rsus.json"
    env = VECEnv(config_file, rsus_file)
    
    state_dim = 11
    action_dim = 6
    
    online_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)
    buffer = deque(maxlen=10000)
    
    if os.path.exists("../results/ddqn_log.csv"):
        os.remove("../results/ddqn_log.csv")
    if os.path.exists("../results/ddqn_metrics.csv"):
        os.remove("../results/ddqn_metrics.csv")
        
    print(f"🚀 Phase 1: Training DDQN ({TRAIN_EPISODES} Episodes)...")
    for episode in range(TRAIN_EPISODES):
        state = env.reset(render=False)
        total_reward = 0
        epsilon = max(0.01, 0.5 - (episode / (TRAIN_EPISODES * 0.8)))
        
        for step in range(MAX_STEPS_PER_EPISODE):
            state_t = torch.FloatTensor(state)
            
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                action = online_net(state_t).argmax().item()
                
            next_state, reward, done, _ = env.step(action)
            scaled_reward = reward / 1000.0
            buffer.append((state, action, scaled_reward, next_state, done))
            total_reward += reward
            state = next_state
            
            if len(buffer) > 64:
                batch = random.sample(buffer, 64)
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
                
                for target_param, local_param in zip(target_net.parameters(), online_net.parameters()):
                    target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
                
            if done: break
        print(f"⚙️ Train Ep {episode+1}/{TRAIN_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
        env.close()

    print(f"\n🚀 Phase 2: Evaluating DDQN Deterministically ({EVAL_EPISODES} Episodes)...")
    for episode in range(EVAL_EPISODES):
        state = env.reset(render=False)
        total_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                action = online_net(state_t).argmax().item()
                
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            
            if done: break
            
        print(f"✅ Eval Ep {episode+1}/{EVAL_EPISODES} | Reward: {total_reward:.2f}")
        log_episode("ddqn", episode + 1, total_reward)
        
        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics("ddqn", env.episode_energy, total_makespan, max_makespan)
        
        env.close()

if __name__ == "__main__":
    run_ddqn()