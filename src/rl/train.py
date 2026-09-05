import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

def log_episode(method_name, episode, total_reward):
    os.makedirs("../results", exist_ok=True)
    file_path = f"../results/{method_name}_log.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward"])
        writer.writerow([episode, total_reward])

def train_cotop():
    print("🚀 Starting Final CoTOP Training (Advantage + Exploration)...")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    
    agent = ActorCritic(10, 6)
    optimizer = optim.Adam(agent.parameters(), lr=0.002)
    
    epochs = 50 
    
    # پاک کردن لاگ قدیمی آموزش تا داده‌ها قاطی نشوند
    if os.path.exists("../results/cotop_train_log.csv"):
        os.remove("../results/cotop_train_log.csv")
        
    for episode in range(epochs):
        state = env.reset(render=False)
        total_reward = 0
        
        # اکتشاف: از 50% کارهای تصادفی شروع می‌شود و کم‌کم به 1% می‌رسد
        epsilon = max(0.01, 0.5 - (episode / (epochs * 0.8)))
        
        for step in range(300):
            state_tensor = torch.FloatTensor(state)
            
            action_probs, state_value = agent(state_tensor)
            
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                action = torch.argmax(action_probs).item()
                
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # --- جادوی ریاضی: استفاده از Advantage به جای پاداش خام ---
            advantage = reward - state_value.item()
            
            log_prob = torch.log(action_probs[action] + 1e-10)
            actor_loss = -log_prob * advantage
            
            # خطای شبکه Critic (چقدر پیش‌بینی‌اش با واقعیت فاصله داشت)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            critic_loss = F.mse_loss(state_value.squeeze(), reward_tensor.squeeze())
            
            loss = actor_loss + critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ---------------------------------------------------------
            
            state = next_state
            if done: break
            
        print(f"✅ Final Train - Ep {episode+1}/{epochs} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
        log_episode("cotop_train", episode + 1, total_reward)
        env.close()
        
    # ذخیره با یک نام کاملاً جدید و اختصاصی
    torch.save(agent.state_dict(), "cotop_model_final.pth")
    print("💾 Ultimate brain saved as 'cotop_model_final.pth'!")

if __name__ == "__main__":
    train_cotop()