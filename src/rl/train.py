import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

def log_episode(method_name, episode, total_reward):
    os.makedirs("../results", exist_ok=True)
    file_path = f"../results/{method_name}.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward"])
        writer.writerow([episode, total_reward])

def train_cotop(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    epochs = 50
    print(f"Starting CoTOP training (seed={seed}, {epochs} episodes) -- ALL 40 vehicles controlled")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    agent = ActorCritic(11, 6)
    optimizer = optim.Adam(agent.parameters(), lr=0.0002)
    log_name = f"cotop_train_seed{seed}"
    
    if os.path.exists(f"../results/{log_name}.csv"):
        os.remove(f"../results/{log_name}.csv")
        
    for episode in range(epochs):
        states = env.reset(render=False)
        total_reward = 0
        epsilon = max(0.01, 0.5 - (episode / (epochs * 0.8)))
        
        for step in range(300):
            actions, log_probs, values = [], [], []
            for s in states:
                state_t = torch.FloatTensor(s)
                action_probs, state_value = agent(state_t)
                
                if random.random() < epsilon:
                    action = random.randint(0, 5)
                else:
                    action = torch.argmax(action_probs).item()
                    
                actions.append(action)
                log_probs.append(torch.log(action_probs[action] + 1e-10))
                values.append(state_value)
                
            next_states, reward, done, _ = env.step(actions)
            total_reward += reward
            scaled_reward = reward / 1000.0
            
            # فقط زمانی شبکه را آپدیت کن که حداقل ۱ ماشین در محیط وجود داشته باشد
            if len(log_probs) > 0:
                loss = 0.0
                target = torch.tensor(scaled_reward, dtype=torch.float32)
                for log_prob, value in zip(log_probs, values):
                    advantage = scaled_reward - value.item()
                    actor_loss = -log_prob * advantage
                    critic_loss = F.mse_loss(value.squeeze(), target)
                    loss = loss + actor_loss + critic_loss
                
                loss = loss / len(log_probs)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0) 
                optimizer.step()
                
            states = next_states
            if done:
                break
                
        print(f"Ep {episode+1}/{epochs} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
        log_episode(log_name, episode + 1, total_reward)
        env.close()
        
    torch.save(agent.state_dict(), f"cotop_model_seed{seed}.pth")
    print(f"Saved cotop_model_seed{seed}.pth")

if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_cotop(seed=seed_arg)