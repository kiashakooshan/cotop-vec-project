import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

MAX_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300

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
    
    # آپدیت ابعاد برای 6 عدد RSU
    state_dim = 10
    action_dim = 6
    
    online_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)
    buffer = deque(maxlen=5000)
    
    print(f"🚀 Starting DDQN Baseline Training ({MAX_EPISODES} Episodes)...")
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        total_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            state_t = torch.FloatTensor(state)
            
            # Epsilon-Greedy با کاهش تدریجی نویز
            if random.random() < max(0.01, 0.1 - 0.01*(episode/10)):
                action = random.randint(0, action_dim - 1)
            else:
                action = online_net(state_t).argmax().item()
                
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state
            
            # اصلاح 4: آموزش واقعی شبکه با نمونه‌گیری از بافر
            if len(buffer) > 32:
                batch = random.sample(buffer, 32)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                
                states_t = torch.FloatTensor(np.array(states))
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards_b)
                next_states_t = torch.FloatTensor(np.array(next_states))
                dones_t = torch.FloatTensor(dones)
                
                # منطق Double DQN
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
        
        # همگام‌سازی دوره‌ای شبکه هدف (Target Network)
        if episode % 5 == 0:
            target_net.load_state_dict(online_net.state_dict())
            
        print(f"✅ DDQN Agent - Episode {episode + 1}/{MAX_EPISODES} | Total Reward: {total_reward:.2f}")
        env.close()

if __name__ == "__main__":
    train_ddqn()