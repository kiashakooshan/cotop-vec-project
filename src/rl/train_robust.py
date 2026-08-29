import sys
import os
import torch
import torch.optim as optim
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

def train_robust_cotop():
    print("🚀 Training CoTOP with Forced Exploration (Epsilon-Greedy)...")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    
    agent = ActorCritic(10, 6)
    optimizer = optim.Adam(agent.parameters(), lr=0.002)
    
    # 100 اپیزود برای یادگیری عمیق‌تر
    epochs = 100 
    
    for episode in range(epochs):
        state = env.reset(render=False)
        total_reward = 0
        
        # نرخ اکتشاف: از 50% کارهای تصادفی شروع می‌شود و کم‌کم به 1% می‌رسد
        epsilon = max(0.01, 0.5 - (episode / (epochs * 0.8)))
        
        for step in range(300):
            state_tensor = torch.FloatTensor(state)
            
            action_probs, state_value = agent(state_tensor)
            
            # تزریق شجاعت! گاهی اوقات به جای خروجی شبکه، تصادفی انتخاب کن
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                action = torch.argmax(action_probs).item()
                
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # یک آپدیت ساده و سریع برای وزن‌ها (Loss = -Reward)
            # در یک کد پیشرفته‌تر اینجا محاسبه Advantage قرار می‌گیرد
            loss = -torch.log(action_probs[action] + 1e-10) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            if done: break
            
        print(f"✅ Robust Train - Ep {episode+1}/{epochs} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
        env.close()
        
    torch.save(agent.state_dict(), "cotop_trained_model.pth")
    print("💾 New, smarter brain saved!")

if __name__ == "__main__":
    train_robust_cotop()