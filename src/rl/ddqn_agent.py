import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

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
    env = VECEnv("../sumo/osm.sumocfg", [{"id": "RSU1", "x": 100, "y": 100, "range": 400}, {"id": "RSU2", "x": 300, "y": 200, "range": 400}])
    state_dim, action_dim = 6, 3
    
    online_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)
    buffer = deque(maxlen=2000)
    
    print("🚀 Starting DDQN Baseline Training...")
    for episode in range(5): # اجرای 5 اپیزود نمایشی
        state = env.reset(render=False)
        total_reward = 0
        
        for step in range(20):
            state_t = torch.FloatTensor(state)
            # انتخاب اکشن (Epsilon-Greedy ساده)
            if random.random() < 0.1:
                action = random.randint(0, action_dim - 1)
            else:
                action = online_net(state_t).argmax().item()
                
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state
            
            # آموزش شبکه با نمونه‌گیری از بافر (در صورت پر بودن)
            if len(buffer) > 32:
                batch = random.sample(buffer, 32)
                # در یک کد کامل، Loss محاسبه و Backward می‌شود
                optimizer.step()
                
            if done: break
            
        print(f"✅ DDQN Agent - Episode {episode + 1} | Total Reward: {total_reward:.2f}")
        env.close()

if __name__ == "__main__":
    train_ddqn()