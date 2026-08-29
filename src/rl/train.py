import sys
import os
import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

LEARNING_RATE = 0.0002 
GAMMA = 0.95
MAX_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300  # حالا ماشین ها 5 دقیقه در نقشه حرکت می کنند

def train():
    config_file = "../sumo/osm.sumocfg" 
    rsus_file = "../sumo/rsus.json"
    env = VECEnv(config_file, rsus_file)
    
    state_dim = 10  # 4 ویژگی ماشین + صف 6 عدد RSU
    action_dim = 6  # انتخاب بین 6 دستگاه RSU
    
    agent = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    print(f"🚀 Training Started (LR={LEARNING_RATE}, Episodes={MAX_EPISODES})...")

    all_episode_rewards = []
    
    for episode in range(MAX_EPISODES):
        # باز کردن نقشه گرافیکی فقط برای اپیزود اول
        if episode == 0:
            state = env.reset(render=True)
        else:
            state = env.reset(render=False)
            
        # تبدیل state Numpy به Tensor برای شبکه عصبی
        state = torch.FloatTensor(state)
        
        log_probs = []
        values = []
        rewards = []
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # 1. عامل RL بر اساس State واقعی تصمیم‌گیری می‌کند
            action, log_prob = agent.get_action(state)
            _, state_value = agent(state)
            
            # 2. ارسال اکشن به محیط و دریافت نتایج جدید
            next_state, reward, done, step_data = env.step(action)
            next_state = torch.FloatTensor(next_state)

            if episode == 0:
                time.sleep(0.3)  # 0.3 ثانیه توقف بین هر فریم
            
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                break
            
        # 3. محاسبه ضرر (Loss) و آپدیت وزن‌ها
        Q_val = 0
        policy_loss = 0
        value_loss = 0
        
        for i in reversed(range(len(rewards))):
            Q_val = rewards[i] + GAMMA * Q_val
            advantage = Q_val - values[i].item()
            
            policy_loss = policy_loss - log_probs[i] * advantage
            value_loss = value_loss + torch.pow(values[i] - Q_val, 2)
            
        total_loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"✅ Episode {episode + 1}/{MAX_EPISODES} | Reward: {sum(rewards):.2f} | Loss: {total_loss.item():.4f}")
        
        env.close()
        all_episode_rewards.append(sum(rewards))
        
    print("🏁 Training Finished Successfully!")

    # --- بخش جدید: ذخیره مدل ---
    torch.save(agent.state_dict(), "cotop_trained_model.pth")
    print("💾 Model saved successfully as 'cotop_trained_model.pth'")
    
    # --- بخش جدید: رسم نمودار یادگیری ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, MAX_EPISODES + 1), all_episode_rewards, marker='o', linestyle='-', color='b')
    plt.title('A3C Learning Curve (Total Reward over Episodes)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward (Higher is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    print("📊 Learning curve plot saved as 'learning_curve.png'")
    plt.show()

if __name__ == "__main__":
    train()